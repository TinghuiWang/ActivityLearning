"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""

from actlearn.models.LogisticRegression import LogisticRegression
from actlearn.monitor.ModelParamMonitor import ModelParamMonitor
from actlearn.models.model import Model
from actlearn.training_algorithms.sgd import sgd_train
import cPickle as pickle
import numpy as np
import theano
import theano.tensor as T
import os
import time


class HiddenLayer(object):
    def __init__(self, numpy_rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.numpy_rng = numpy_rng
        self.n_in = n_in
        self.n_out = n_out
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                    numpy_rng.uniform(
                            low=-np.sqrt(6. / (n_in + n_out)),
                            high=np.sqrt(6. / (n_in + n_out)),
                            size=(n_in, n_out)
                    ),
                    dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

    def clear(self):
        """
        Re-initialize the internal parameters
        :return:
        """
        self.W.set_value(
            np.asarray(
                self.numpy_rng.uniform(
                    low=-np.sqrt(6. / (self.n_in + self.n_out)),
                    high=np.sqrt(6. / (self.n_in + self.n_out)),
                    size=(self.n_in, self.n_out)
                ),
                dtype=theano.config.floatX
            )
        )
        self.b.set_value(
            np.zeros((self.n_out,), dtype=theano.config.floatX)
        )


class MultiLayerPerceptron(Model):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, numpy_rng, n_in, n_hidden, n_out, input=None):
        """Initialize the parameters for the multilayer perceptron

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        # Call parent initialization procedure
        Model.__init__(self, 'MultiLayerPerceptron')
        # Get Input Tensor
        if input is None:
            self.x = T.matrix('x')
        else:
            self.x = input
        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
                numpy_rng=numpy_rng,
                input=self.x,
                n_in=n_in,
                n_out=n_hidden,
                activation=T.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
                input=self.hiddenLayer.output,
                n_in=n_hidden,
                n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.cost = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        # Monitored parameters during training
        self.monitors = [
            ModelParamMonitor('avg_batch_cost', self.cost)
        ]

    def clear(self):
        """
        Re-initialize all layers in MLP
        :return:
        """
        self.hiddenLayer.clear()
        self.logRegressionLayer.clear()

    def save(self, filename):
        """
        :param filename:
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
                outputs=[self.logRegressionLayer.y_pred],
                givens={
                    self.x: data
                }
        )
        prediction = predict_model()
        end_time = time.clock()
        print("Classification Done in %.1fs" % (end_time - start_time))
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
        :param prevalence: inverse of Percentage of each class in the training samples
        :return: None
        """
        if prevalence is not None:
            self.logRegressionLayer.prevalence = prevalence
        for i in range(num_epochs):
            self.logger.info("Epoch %d" % i)
            monitor_results = sgd_train(model=self, data=data, label=label,
                                        num_data=num_data, batch_size=batch_size,
                                        learning_rate_array=learning_rate_array)
            for monitor_index, monitor in enumerate(self.monitors):
                self.logger.info("\t%10s: %f" % (monitor.name, monitor_results[monitor_index]))
        self.logRegressionLayer.prevalence = np.ones(
                (self.logRegressionLayer.W.get_value().shape[1],), dtype=theano.config.floatX)
