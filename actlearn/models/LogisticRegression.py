"""
Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.
References:
    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

Part of the code is referenced from tutorials on deeplearning.net
http://deeplearning.net/tutorial/logreg.html
"""

from actlearn.models.model import Model
from actlearn.monitor.ModelParamMonitor import ModelParamMonitor
from actlearn.training_algorithms.sgd import sgd_train
from actlearn.log.logger import actlearn_logger
import cPickle as pickle
import os
import time
import numpy as np
import theano
import theano.tensor as T


class LogisticRegression(Model):
    """Multi-class Logistic Regression Class
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, n_in, n_out, input=None, training_algorithm=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        :type training_algorithm: actlearn.training_algorithm.training_algorithm
        :param training_algorithm: training algorithm class to train the model

        """
        Model.__init__(self, 'LogisticRegression')
        # Get Input Tensor
        if input is None:
            self.x = T.matrix('x')
        else:
            self.x = input
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(self.x, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # Cost Function
        self.cost = self.negative_log_likelihood

        # Prevalence matrix for use of class-skewed cost calculation
        self.prevalence = np.ones((n_out,), dtype=theano.config.floatX)

        # parameters of the model
        self.params = [self.W, self.b]

        # Monitored parameters during training
        self.monitors = [
            ModelParamMonitor('avg_batch_cost', self.cost)
        ]

    def clear(self):
        """
        Clear learned model to initial value
        :return:
        """
        self.W.set_value(np.zeros(self.W.get_value().shape, dtype=theano.config.floatX))
        self.b.set_value(np.zeros(self.b.get_value().shape, dtype=theano.config.floatX))

    def negative_log_likelihood(self, y):
        """
        Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y] * T.inv(self.prevalence)[y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y)), self.y_pred
        else:
            raise NotImplementedError()

    def save(self, filename):
        """
        :param filename: name of the file to save the parameters to
        :return:
        """
        out_file = open(filename, 'w')
        pickle.dump(self.params, out_file, protocol=-1)
        out_file.close()

    def load(self, filename):
        """
        :param filename:
        :return:
        """
        if os.path.isfile(filename):
            in_file = open(filename, 'r')
            loaded_params = pickle.load(in_file)
            for param, loaded_param in zip(self.params, loaded_params):
                param.set_value(loaded_param.get_value())
            in_file.close()

    def classify(self, data):
        """
        Get Prediction Result given x
        :return: array of predicted class
        """
        start_time = time.clock()
        predict_model = theano.function(
            inputs=[],
            outputs=[self.y_pred],
            givens={
                self.x: data
            }
        )
        prediction = predict_model()
        end_time = time.clock()
        self.logger.info("Classification Done in %.1fs" % (end_time - start_time))
        return prediction

    def do_training_sgd(self, data, label, num_data, batch_size, learning_rate_array, num_epochs=1, prevalence=None):
        """
        Use Stochastic Gradient Descent to train the model
        :param data: Training Data Tensor
        :param label: Corresponding Class Tensor
        :param num_data: The total amount of data
        :param batch_size: The size per batch
        :param learning_rate_array: Learning Rate for each parameter
        :param num_epochs: Epochs
        :return: None
        """
        self.prevalence = prevalence
        for i in range(num_epochs):
            self.logger.info("Epoch %d" % i)
            monitor_results = sgd_train(model=self, data=data, label=label,
                                        num_data=num_data, batch_size=batch_size,
                                        learning_rate_array=learning_rate_array)
            for monitor_index, monitor in enumerate(self.monitors):
                self.logger.info("\t%10s: %f" % (monitor.name, monitor_results[monitor_index]))
        self.prevalence = np.ones((self.W.get_value().shape[1],), dtype=theano.config.floatX)
