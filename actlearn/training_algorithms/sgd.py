import time
import numpy as np
import theano
import theano.tensor as T


def sgd_train(model, data, num_data, batch_size, learning_rate_array, params=None, label=None, monitors=None):
    """
    Train model with stochastic gradient descent
    :param model: The learning structure
    :param data: Tensor array contained data
    :param num_data: total number of data points given
    :param batch_size: batch size
    :param learning_rate_array: learning rate for each params
    :param label: Tensor array contained labels for each data point
    :param params: Parameters in the model that needs to be updated
    :param monitors: Equations of value to be monitored
    :return: averaged monitored value
    """
    num_training_data = num_data
    num_batch = num_training_data / batch_size
    model.logger.info("Start Training Model using Stochastic Gradient Descent")
    model.logger.info("Parameters:")
    model.logger.info("Training Data: %d" % num_training_data)
    model.logger.info("batch size: %d" % batch_size)
    model.logger.info("Number of Batches: %d" % num_batch)
    model.logger.info("learning rate:")
    model.logger.info(learning_rate_array)
    start_time = time.clock()
    index = T.lscalar()
    givens = {}
    if monitors is None:
        active_monitors = model.monitors
    else:
        active_monitors = monitors
    if label is None:
        cost = model.cost()
        outputs = [monitor.tensor() for monitor in active_monitors]
    else:
        y = T.ivector('y')
        cost = model.cost(y)
        outputs = [monitor.tensor(y) for monitor in active_monitors]
        givens[y] = label[index * batch_size: (index + 1) * batch_size]
    if params is None:
        params = model.params
    grad_params = [T.grad(cost, param) for param in params]
    updates = [
        (param, param - learning_rate * grad_param)
        for param, learning_rate, grad_param in zip(params, learning_rate_array, grad_params)
    ]
    givens[model.x] = data[index * batch_size: (index + 1) * batch_size]
    train_model = theano.function(
        inputs=[index],
        outputs=outputs,
        updates=updates,
        givens=givens
    )
    monitor_result = np.zeros((num_batch, len(model.monitors)))
    for i in range(num_batch):
        monitor_result[i, :] = train_model(i)
        model.logger.info("batch %d" % i)
        for monitor_index, monitor in enumerate(model.monitors):
            model.logger.info("\t%10s: %f" % (monitor.name, monitor_result[monitor_index]))
    end_time = time.clock()
    total_time = end_time - start_time
    model.logger.info('total time consumed %.1fs' % total_time)
    return np.mean(monitor_result, axis=0)
