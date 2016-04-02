"""
 This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.

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
import os
import time
import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle
from actlearn.models.model import Model
from actlearn.models.LogisticRegression import LogisticRegression
from actlearn.models.MultiLayerPerceptron import HiddenLayer
from actlearn.models.DenoisingAutoencoder import DenoisingAutoencoder
from actlearn.training_algorithms.sgd import sgd_train
from theano.tensor.shared_randomstreams import RandomStreams


class StackedDenoisingAutoencoder(Model):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_ins=784,
        hidden_layers_sizes=[500, 500],
        n_outs=10,
        corruption_levels=[0.1, 0.1]
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """
        Model.__init__(self, 'StackedDenoisingAutoencoder')
        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        # Get Input Tensor
        if input is None:
            self.x = T.matrix('x')
        else:
            self.x = input

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        # start-snippet-2
        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(numpy_rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = DenoisingAutoencoder(numpy_rng=numpy_rng,
                                            theano_rng=theano_rng,
                                            input=layer_input,
                                            n_visible=input_size,
                                            n_hidden=hidden_layers_sizes[i],
                                            W=sigmoid_layer.W,
                                            bhid=sigmoid_layer.b,
                                            corruption_level=corruption_levels[i])
            self.dA_layers.append(dA_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)

        self.pretrain_monitor_array = [layer.monitors for layer in self.dA_layers]

        self.monitors = self.logLayer.monitors

    def clear(self):
        """
        Re-initialize all the parameters
        :return:
        """
        for layer in self.sigmoid_layers:
            layer.clear()
        self.logLayer.clear()

    def export_to_dict(self):
        """
        Save the parameters of sda to a dict structure that can be saved in a project structure
        :return:
        """
        data = {
            'type': 'model',
            'name': 'StackedDenoisingAutoencoder',
            'data': [np.array(param.get_value()) for param in self.params],
            'modified': time.time()
        }
        return data

    def load_from_dict(self, data):
        """
        Load the parameters from a dict structure
        :type data: dict
        :param data: model data saved in a dict structure
        :return:
        """
        # Check Dict Type, Name
        assert({'type', 'name', 'data', 'modified'}.issubset(data.keys()))
        assert(data['type'] == 'model')
        assert(data['name'] == 'StackedDenoisingAutoencoder')
        # Load Model Data
        loaded_params = data['data']
        for param, loaded_param in zip(self.params, loaded_params):
            param.set_value(loaded_param)

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
            outputs=[self.logLayer.y_pred],
            givens={
                self.x: data
            }
        )
        prediction = predict_model()
        end_time = time.clock()
        print("Classification Done in %.1fs" % (end_time - start_time))
        return prediction

    def do_pretraining(self, data, num_data, batch_size, learning_rate_array, num_epochs=1):
        for i in range(len(self.dA_layers)):
            # Training auto-encoder layer by layer
            self.logger.info("Pre-training layer %d" % i)
            self.cost = self.dA_layers[i].cost
            for j in range(num_epochs):
                self.logger.info("Epoch %d" % j)
                monitor_results = sgd_train(
                    model=self, data=data, label=None,
                    num_data=num_data, batch_size=batch_size,
                    learning_rate_array=learning_rate_array,
                    params=self.dA_layers[i].params,
                    monitors=self.pretrain_monitor_array[i])
                for monitor_index, monitor in enumerate(self.monitors):
                    self.logger.info("\t%10s: %f" % (monitor.name, monitor_results[monitor_index]))
        return

    def do_reconstruction(self, data):
        """
        Get Prediction Result given x
        :return: array of predicted class
        """
        start_time = time.clock()
        da_inputs = []
        reconstruct_input = []
        num_dA_layers = len(self.dA_layers)
        for i in range(num_dA_layers):
            if i == 0:
                da_inputs.append(self.dA_layers[num_dA_layers-i-1].get_hidden_values(self.dA_layers[num_dA_layers-i-1].x))
            else:
                da_inputs.append(reconstruct_input[i-1])
            reconstruct_input.append(self.dA_layers[num_dA_layers-i-1].get_reconstructed_input(da_inputs[i]))
        reconstruct_function = theano.function(
            inputs=[],
            outputs=[reconstruct_input[num_dA_layers-1]],
            givens={
                self.x: data
            }
        )
        theano.printing.pydotprint(reconstruct_function, outfile='./mnist_sda_reconstruction.png', var_with_name_simple=True)
        prediction = reconstruct_function()
        end_time = time.clock()
        print("Reconstruction Done in %.1fs" % (end_time - start_time))
        return prediction

    def do_fine_tuning(self, data, label, num_data, batch_size, learning_rate_array, num_epochs=1):
        monitor_results = None
        self.cost = self.logLayer.cost
        for i in range(num_epochs):
            self.logger.info('epoch %d' % i)
            monitor_results = sgd_train(
                model=self, data=data, label=label,
                num_data=num_data, batch_size=batch_size,
                learning_rate_array=learning_rate_array)
            for monitor_index, monitor in enumerate(self.monitors):
                self.logger.info("\t%10s: %f" % (monitor.name, monitor_results[monitor_index]))
        return monitor_results

    def do_log_layer_training_only(self, data, label, num_data, batch_size, learning_rate_array, num_epochs=1):
        self.cost = self.logLayer.cost
        for i in range(num_epochs):
            print('epoch %d' % i)
            monitor_results = sgd_train(
                model=self, data=data, label=label,
                num_data=num_data, batch_size=batch_size,
                learning_rate_array=learning_rate_array,
                params=self.logLayer.params)
            for monitor_index, monitor in enumerate(self.monitors):
                print("\t%10s: %f" % (monitor.name, monitor_results[monitor_index]))

    def get_log_layer_input(self, data):
        """
        Get Inputs for Logistic Regression Classification Layer
        :param data: test tensor
        :return:
        """
        last_layer_fn = theano.function(
            inputs=[],
            outputs=[self.sigmoid_layers[self.n_layers - 1].output],
            givens={
                self.x: data
            }
        )
        result = last_layer_fn()
        return result[0]

    def opt_missing_input(self, data, mask, batch_size=10, learning_rate=0.5, iterations=10000):
        """
        Optmizing for the missing input masked by mask
        :type data: numpy.array
        :param data:
        :param mask:
        :return:
        """
        # init_missing = np.asarray(
        #     np.random.uniform(
        #         low=0,
        #         high=1,
        #         size=data.get_value().shape
        #     ),
        #     dtype=theano.config.floatX
        # )
        result = np.zeros(data.get_value().shape, dtype=theano.config.floatX)
        init_missing = np.zeros((batch_size, data.get_value().shape[1]), dtype=theano.config.floatX)
        missing_t = theano.shared(value=init_missing, name='Missing', borrow=True)
        input_view = T.matrix('view')
        mask_view = T.matrix('mask')
        hidden_values = []
        reconstruct_input = []
        num_dA_layers = len(self.dA_layers)
        hidden_values.append(input_view + missing_t * mask_view)
        for i in range(num_dA_layers):
            hidden_values.append(self.dA_layers[i].get_hidden_values(hidden_values[i]))
        reconstruct_input.append(hidden_values[num_dA_layers])
        for i in range(num_dA_layers):
            reconstruct_input.append(self.dA_layers[num_dA_layers-i-1].get_reconstructed_input(reconstruct_input[i]))
        # Two options for cost function:
        # 1. Overall Output is as close as input
        # 2. Known part is close
        input_data = hidden_values[0]
        output_data = reconstruct_input[num_dA_layers]
        cost = T.sum(T.pow(output_data - input_data, 2))
        monitor = cost / batch_size
        # Create Update Matrix
        grad = T.grad(cost, missing_t)
        updates = {(missing_t, missing_t - learning_rate * grad)}
        index = T.lscalar()
        reconstruct_function = theano.function(
            inputs=[index],
            outputs=[monitor],
            updates=updates,
            givens={
                input_view: data[index * batch_size:(index + 1) * batch_size],
                mask_view: mask[index * batch_size:(index + 1) * batch_size],
            }
        )
        # Run Reconstruction
        data_num = data.get_value().shape[0]
        num_batch = data_num / batch_size
        for batch in range(num_batch):
            init_missing[:][:] = 0
            missing_t.set_value(init_missing)
            for i in range(iterations):
                avg_cost = reconstruct_function(batch)
                print('avg_cost: %f' % avg_cost[0])
            result[batch * batch_size:(batch + 1) * batch_size] = \
                missing_t.get_value() * mask.get_value()[batch * batch_size : (batch + 1) * batch_size] + \
                data.get_value()[batch * batch_size : (batch + 1) * batch_size]
        return result

    def export_to_graphviz(self, feature_list=None):
        """
        export to graphviz dot string
        :type feature_list: list of str
        :param feature_list: Feature Lists
        :return: String
        """
        output = "digraph FNN {\noverlap=false\n"
        # Create Input Nodes
        num_ins = self.sigmoid_layers[0].W.get_value().shape[0]
        if feature_list is None:
            for i in range(num_ins):
                output += "%u [label=\"%d\" shape=\"box\"]\n" % (i, i)
        else:
            for i in range(num_ins):
                output += "%u [label=\"%s\" shape=\"box\"]\n" % (i, feature_list[i])
        node_num = num_ins
        # Create Intermediate Nodes
        for layer in self.sigmoid_layers:
            # Get parameter values
            W_value = layer.W.get_value()
            b_value = layer.b.get_value()
            # Create Intermediate Nodes
            num_outs = W_value.shape[1]
            for i in range(num_outs):
                output += "%u [label=\"%d\" shape=\"circle\"]\n" % (node_num + i, i)
            # Create Edges
            for i in range(num_ins):
                for j in range(num_outs):
                    if i == 0:
                        output += "%u -> %u [headlabel=\"%.5f\" taillabel=\"%.5f\"]\n" % \
                                  (node_num - num_ins + i, node_num + j, W_value[i][j], b_value[j])
                    else:
                        output += "%u -> %u [headlabel=\"%.5f\"]\n" % \
                                  (node_num - num_ins + i, node_num + j, W_value[i][j])
            node_num += num_outs
            num_ins = num_outs
        # Draw The Final Output Layer
        W_value = self.logLayer.W.get_value()
        b_value = self.logLayer.b.get_value()
        num_outs = W_value.shape[1]
        for i in range(num_outs):
            output += "%u [label=\"Class\\n%d\" shape=\"doublecircle\"]\n" % (node_num + i, i)
        for i in range(num_ins):
            for j in range(num_outs):
                if i == 0:
                    output += "%u -> %u [headlabel=\"%.5f\" taillabel=\"%.5f\"]\n" % \
                              (node_num - num_ins + i, node_num + j, W_value[i][j], b_value[j])
                else:
                    output += "%u -> %u [headlabel=\"%.5f\"]\n" % \
                              (node_num - num_ins + i, node_num + j, W_value[i][j])
        # Create Edges
        output += "}\n"
        return output
