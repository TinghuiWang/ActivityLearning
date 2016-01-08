"""
 This tutorial introduces denoising auto-encoders (dA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""

from actlearn.models.model import Model
from actlearn.monitor.ModelParamMonitor import ModelParamMonitor
from actlearn.training_algorithms.sgd import sgd_train
import cPickle as pickle
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import time
import os


class DenoisingAutoencoder(Model):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None,
        corruption_level = 0.1
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives sy mbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type corruption_level: float
        :param corruption_level: Corrupted data percentage
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.corruption_level = corruption_level

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = np.asarray(
                numpy_rng.uniform(
                    low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=np.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.matrix(name='x')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

        self.cost = self.cross_entropy_error

        self.monitors = [
            ModelParamMonitor('avg_batch_cost', self.cost)
        ]

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer
        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def cross_entropy_error(self):
        """ This function computes the cross entropy error for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, self.corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        return cost

    def mean_square_error(self):
        """
        This function computes the least mean square error for one training step of the dA
        :param corruption_level:
        :return:
        """
        tilde_x = self.get_corrupted_input(self.x, self.corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # Calculate the difference between reconstructed input vs original input
        cost = T.mean(T.sum(T.pow(self.x - z, 2), axis=1)/self.n_visible)
        return cost

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

    def do_reconstruction(self, data):
        """
        Get Prediction Result given x
        :return: array of predicted class
        """
        start_time = time.clock()
        hidden_output = self.get_hidden_values(self.x)
        reconstruct_input = self.get_reconstructed_input(hidden_output)
        predict_model = theano.function(
            inputs=[],
            outputs=[reconstruct_input],
            givens={
                self.x: data
            }
        )
        prediction = predict_model()
        end_time = time.clock()
        print("Classification Done in %.1fs" % (end_time - start_time))
        return prediction

    def do_train_sgd(self, data, label, num_data, batch_size, learning_rate_array):
        return sgd_train(model=self, data=data, label=label,
                         num_data=num_data, batch_size=batch_size,
                         learning_rate_array=learning_rate_array)

    def opt_missing_inputs(self, data, mask, learning_rate=0.01, iterations=200):
        init_missing = np.asarray(
                np.random.uniform(
                    low=0,
                    high=1,
                    size=data.get_value().shape
                ),
                dtype=theano.config.floatX
            ) * mask.get_value()
        missing_t = theano.shared(value=init_missing, name='Missing', borrow=True)
        current_input = data + missing_t * mask
        hidden_output = self.get_hidden_values(current_input)
        reconstruct_input = self.get_reconstructed_input(hidden_output)
        cost = T.mean(T.sum(T.pow(reconstruct_input - current_input, 2), axis=1))
        grad = T.grad(cost, missing_t)
        updates = {(missing_t, missing_t - learning_rate * grad)}
        reconstruct_function = theano.function(
            inputs=[],
            outputs=[cost, grad],
            updates=updates,
            givens={}
        )
        theano.printing.pydotprint(reconstruct_function, outfile='./mnist_sda_multiview_reconstruct.png', var_with_name_simple=True)
        for i in range(iterations):
            avg_cost = reconstruct_function()
            print('avg_cost: %f' % avg_cost[0])
        return missing_t.get_value() * mask.get_value() + data.get_value()
