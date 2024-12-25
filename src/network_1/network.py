"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import time

# Third-party libraries
import numpy as np

NeuronCountPerLayer = int
NeuralNetworkLayers = list[NeuronCountPerLayer] # A list of neuron counts, one for each layer of the network

class Network(object):

    def __init__(self, neural_network_layers: NeuralNetworkLayers):
        """The list ``neural_network_layers`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(neural_network_layers)
        self.neural_network_layers = neural_network_layers
        receiver_layers = neural_network_layers[1:]
        self.biases = [np.random.randn(receiver_layer, 1) for receiver_layer in receiver_layers]
        self.weights = [np.random.randn(receiver_layer, input_layer)
                        for input_layer, receiver_layer in zip(neural_network_layers[:-1], receiver_layers)]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.add(np.dot(weight, a), bias))
        return a

    def SGD(self,
            training_data,
            epochs: int,
            mini_batch_size: int,
            learning_rate: float,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        for epoch in range(epochs):
            start = time.process_time_ns()
            np.random.shuffle(training_data)
            shuffle_end = time.process_time_ns()
            print(f"Shuffling took {shuffle_end-start:.2f} seconds")

            mini_training_data_batches = self._create_mini_batches(training_data, mini_batch_size)
            for mini_training_data_batch in mini_training_data_batches:
                self.update_network_using_mini_batch(mini_training_data_batch, learning_rate)
            end = time.process_time_ns()
            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {len(test_data)}, took {end-start:.2f} seconds")
            else:
                print(f"Epoch {epoch} complete in {end-start:.2f} seconds")

    def update_network_using_mini_batch(self, mini_training_data_batch, learning_rate: float):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        accumulated_weight_adjustments = [np.zeros(weights_for_layer.shape) for weights_for_layer in self.weights]
        accumulated_bias_adjustments = [np.zeros(biases_for_layer.shape) for biases_for_layer in self.biases]

        for mnist_image, expected_mnist_value in mini_training_data_batch:
            delta_bias, delta_weight = self.backprop(mnist_image, expected_mnist_value)

            accumulated_weight_adjustments = [nabla_w_for_layer + delta_nabla_w_for_layer
                       for nabla_w_for_layer, delta_nabla_w_for_layer in zip(accumulated_weight_adjustments, delta_weight)]

            accumulated_bias_adjustments = [nabla_b_for_layer + delta_nabla_b_for_layer
                       for nabla_b_for_layer, delta_nabla_b_for_layer in zip(accumulated_bias_adjustments, delta_bias)]

        adjustment_rate = (learning_rate / len(mini_training_data_batch))
        self.weights = [weights_for_layer - adjustment_rate * weight_adjustments_for_layer
                        for weights_for_layer, weight_adjustments_for_layer in zip(self.weights, accumulated_weight_adjustments)]
        self.biases = [biases_for_layer - adjustment_rate * bias_adjustments_for_layer
                       for biases_for_layer, bias_adjustments_for_layer in zip(self.biases, accumulated_bias_adjustments)]

    def backprop(self, mnist_image, expected_mnist_value):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        # feed forward
        activation_for_layer = mnist_image # first activation is the input image
        activations = [mnist_image] # list to store all the activations, layer by layer
        raw_results_in_all_layers = [] # list to store all the z vectors, layer by layer
        for weights_for_layer, biases_for_layer in zip(self.weights, self.biases):
            raw_layer_result = np.dot(weights_for_layer, activation_for_layer) + biases_for_layer
            raw_results_in_all_layers.append(raw_layer_result)
            activation_for_layer = sigmoid(raw_layer_result) # activation for next layer
            activations.append(activation_for_layer)

        # backward pass
        delta_weights = [np.zeros(weights_for_layer.shape) for weights_for_layer in self.weights]
        delta_bias = [np.zeros(biases_for_layer.shape) for biases_for_layer in self.biases]

        last_item = -1
        delta = self.cost_derivative(activations[last_item], expected_mnist_value) * \
                sigmoid_prime(raw_results_in_all_layers[last_item])

        delta_bias[last_item] = delta

        second_last_item = -2
        transposed_second_last_activation = activations[second_last_item].transpose()
        delta_weights[last_item] = np.dot(delta, transposed_second_last_activation)
        # Note that the variable `layer` in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # layer = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        second_layer = 2
        all_layers_except_input = range(second_layer, self.num_layers)
        for layer in all_layers_except_input:
            raw_layer_result = raw_results_in_all_layers[-layer]
            sp = sigmoid_prime(raw_layer_result)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            delta_bias[-layer] = delta
            delta_weights[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        return delta_bias, delta_weights

    def backprop_bias(self, mnist_image, expected_mnist_value):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        # feed forward
        activation_for_layer = mnist_image # first activation is the input image
        activations = [mnist_image] # list to store all the activations, layer by layer
        raw_results_in_all_layers = [] # list to store all the z vectors, layer by layer
        for weights_for_layer, biases_for_layer in zip(self.weights, self.biases):
            raw_layer_result = np.dot(weights_for_layer, activation_for_layer) + biases_for_layer
            raw_results_in_all_layers.append(raw_layer_result)
            activation_for_layer = sigmoid(raw_layer_result) # activation for next layer
            activations.append(activation_for_layer)

        # backward pass
        delta_bias = [np.zeros(biases_for_layer.shape) for biases_for_layer in self.biases]

        last_item = -1
        delta = self.cost_derivative(activations[last_item], expected_mnist_value) * \
                sigmoid_prime(raw_results_in_all_layers[last_item])

        delta_bias[last_item] = delta

        second_last_item = -2
        transposed_second_last_activation = activations[second_last_item].transpose()
        # Note that the variable `layer` in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # layer = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        second_layer = 2
        all_layers_except_input = range(second_layer, self.num_layers)
        for layer in all_layers_except_input:
            raw_layer_result = raw_results_in_all_layers[-layer]
            sp = sigmoid_prime(raw_layer_result)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            delta_bias[-layer] = delta
        return delta_bias

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(training_input)), expected_result)
                        for (training_input, expected_result) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    @staticmethod
    def cost_derivative(output_activations, expected_result):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return output_activations - expected_result

    @staticmethod
    def _create_mini_batches(training_data, mini_batch_size):
        training_data_size = len(training_data)
        return (
            training_data[batch_start_index:batch_start_index + mini_batch_size]
            for batch_start_index in range(0, training_data_size, mini_batch_size)
        )

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return np.divide(1.0, np.add(1.0, np.exp(-z)))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    value = sigmoid(z)
    return value * (1 - value)
