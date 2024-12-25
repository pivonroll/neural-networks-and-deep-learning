"""network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""
#### Libraries
# Standard library
import json
import random
import sys
import time
from typing import Protocol

# Third-party libraries
import numpy as np
from numpy.typing import ArrayLike


#### Define the quadratic and cross-entropy cost functions
class Cost(Protocol):
    def fn(self, feed_forward_result, expected_result) -> ArrayLike:
        ...

    def delta(self, z, feed_forward_result, expected_result) -> ArrayLike:
        ...


class QuadraticCost(object):

    @staticmethod
    def fn(feed_forward_result, expected_result) -> ArrayLike:
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(feed_forward_result - expected_result)**2

    @staticmethod
    def delta(z, feed_forward_result, expected_result) -> ArrayLike:
        """Return the error delta from the output layer."""
        return (feed_forward_result - expected_result) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(feed_forward_result, expected_result) -> ArrayLike:
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-expected_result * np.log(feed_forward_result) - (1 - expected_result) * np.log(1 - feed_forward_result)))

    @staticmethod
    def delta(z, feed_forward_result, expected_result) -> ArrayLike:
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return feed_forward_result - expected_result


#### Main Network class
class Network(object):

    def __init__(self, neural_network_layers, cost: Cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layers = len(neural_network_layers)
        self.neural_network_layers = neural_network_layers
        self.default_weight_initializer()
        self.cost=cost

    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        receiver_layers = self.neural_network_layers[1:]

        self.weights = [np.random.randn(receiver_layer, input_layer) / np.sqrt(input_layer)
                        for input_layer, receiver_layer in zip(self.neural_network_layers[:-1], receiver_layers)]
        self.biases = [np.random.randn(receiver_layer, 1) for receiver_layer in receiver_layers]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        receiver_layers = self.neural_network_layers[1:]

        self.weights = [np.random.randn(receiver_layer, input_layer)
                        for input_layer, receiver_layer in zip(self.neural_network_layers[:-1], receiver_layers)]

        self.biases = [np.random.randn(receiver_layer, 1) for receiver_layer in receiver_layers]

    def feedforward(self, result):
        """Return the output of the network if ``a`` is input."""
        for bias_for_layer, weights_for_layer in zip(self.biases, self.weights):
            result = sigmoid(np.add(np.dot(weights_for_layer, result), bias_for_layer))
        return result

    def SGD(self,
            training_data,
            epochs,
            mini_batch_size,
            eta,
            regularization_coefficient = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``regularization_coefficient``.
        The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        training_data_size = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for epoch in range(epochs):
            np.random.shuffle(training_data)

            mini_batches = (
                training_data[mini_batch_index_splitter: mini_batch_index_splitter + mini_batch_size]
                for mini_batch_index_splitter in range(0, training_data_size, mini_batch_size)
            )

            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, eta, regularization_coefficient, len(training_data)
                )

            print(f"Epoch {epoch} training complete")

            if monitor_training_cost:
                cost = self.total_cost(training_data, regularization_coefficient)
                training_cost.append(cost)
                print(f"Cost on training data: {cost}")
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print(f"Accuracy on training data: {accuracy} / {training_data_size}")
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, regularization_coefficient, convert=True)
                evaluation_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")
            if monitor_evaluation_accuracy and evaluation_data:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"Accuracy on evaluation data: {self.accuracy(evaluation_data)} / {len(evaluation_data)}")

        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self,
                          mini_batch,
                          eta,
                          regularization_coefficient,
                          training_data_size):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``regularization_coefficient`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        accumulated_weight_delta = [np.zeros(weights_for_layer.shape) for weights_for_layer in self.weights]
        accumulated_bias_delta = [np.zeros(bias_for_layer.shape) for bias_for_layer in self.biases]

        for training_input, expected_result in mini_batch:
            delta_bias, delta_weight = self.backprop(training_input, expected_result)
            accumulated_weight_delta = [np.add(accumulated_weights_for_layer, delta_weights_for_layer)
                                              for accumulated_weights_for_layer, delta_weights_for_layer in
                                              zip(accumulated_weight_delta, delta_weight)]
            accumulated_bias_delta = [np.add(accumulated_bias_for_layer, delta_bias_for_layer)
                                            for accumulated_bias_for_layer, delta_bias_for_layer in
                                            zip(accumulated_bias_delta, delta_bias)]

        self.weights = [(1 - eta * (regularization_coefficient / training_data_size)) *
                        weights_for_layer - (eta / len(mini_batch)) * accumulated_weight_delta_for_layer
                        for weights_for_layer, accumulated_weight_delta_for_layer in zip(self.weights, accumulated_weight_delta)]

        self.biases = [bias_for_layer - (eta/len(mini_batch)) * delta_bias_for_layer
                       for bias_for_layer, delta_bias_for_layer in zip(self.biases, accumulated_bias_delta)]

    def backprop(self, training_input, expected_result):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        # feedforward
        activation = training_input
        activations = [training_input] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.add(np.dot(w, activation), b)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta_weights = [np.zeros(weights_for_layer.shape) for weights_for_layer in self.weights]
        delta_bias = [np.zeros(bias_for_layer.shape) for bias_for_layer in self.biases]

        delta = self.cost.delta(zs[-1], activations[-1], expected_result)
        delta_bias[-1] = delta
        delta_weights[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp
            delta_bias[-layer] = delta
            delta_weights[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        return delta_bias, delta_weights

    def backprop_bias(self, training_input, expected_result):
        """Return a tuple ``delta_bias`` representing the
        delta_bias part of the gradient for the cost function C_x.  ``delta_bias`` is
        a layer-by-layer lists of numpy arrays, similar
        to ``self.biases``."""

        # feedforward
        activation = training_input
        activations = [training_input] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.add(np.dot(w, activation), b)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta_bias = [np.zeros(bias_for_layer.shape) for bias_for_layer in self.biases]

        delta = self.cost.delta(zs[-1], activations[-1], expected_result)
        delta_bias[-1] = delta
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sigmoid_prime(z)
            delta_bias[-layer] = delta
        return delta_bias

    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """
        if convert:
            results = [(np.argmax(self.feedforward(training_input)), np.argmax(expected_result))
                       for (training_input, expected_result) in data]
        else:
            results = [(np.argmax(self.feedforward(training_input)), expected_result)
                        for (training_input, expected_result) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data, regularization_coefficient, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for training_input, expected_result in data:
            feed_forward_result = self.feedforward(training_input)
            if convert: expected_result = vectorized_result(expected_result)
            cost += self.cost.fn(feed_forward_result, expected_result) / len(data)
        cost += 0.5 * (regularization_coefficient / len(data)) * sum(
            np.linalg.norm(weights_for_layer)**2 for weights_for_layer in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.neural_network_layers,
                "weights": [weights_for_layer.tolist() for weights_for_layer in self.weights],
                "biases": [bias_for_layer.tolist() for bias_for_layer in self.biases],
                "cost": str(self.cost.__name__)}
        with open(filename, "w") as f:
            json.dump(data, f)

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    with open(filename, "r") as f:
        data = json.load(f)

    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(weights_for_layer) for weights_for_layer in data["weights"]]
    net.biases = [np.array(bias_for_layer) for bias_for_layer in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return np.divide(1.0, np.add(1.0, np.exp(-z)))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    value = sigmoid(z)
    return value * (1-value)
