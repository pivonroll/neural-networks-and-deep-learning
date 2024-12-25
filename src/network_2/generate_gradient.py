"""generate_gradient.py
~~~~~~~~~~~~~~~~~~~~~~~

Use network2 to figure out the average starting values of the gradient
error terms \delta^l_j = \partial C / \partial z^l_j = \partial C /
\partial b^l_j.

"""

#### Libraries
# Standard library
import json
import math
import functools
import time

# My library
from src.mnist_loader import load_data_wrapper
from network2 import Network

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Load the data
    full_td, _, _ = load_data_wrapper()
    td = full_td[:1000]  # Just use the first 1000 items of training data
    epochs = 500  # Number of epochs to train for

    print("\nTwo hidden layers:")
    net = Network([784, 30, 30, 10])
    initial_norms(td, net)
    abbreviated_gradient = [
        ag[:6] for ag in get_average_gradient(net, td)[:-1]]
    print("Saving the averaged gradient for the top six neurons in each " + \
          "layer.\nWARNING: This will affect the look of the book, so be " + \
          "sure to check the\nrelevant material (early chapter 5).")
    with open("initial_gradient.json", "w") as f:
        json.dump(abbreviated_gradient, f)
    training(td, net, epochs, "norms_during_training_2_layers.json")
    plot_training(
        epochs, "norms_during_training_2_layers.json", 2)

    print("\nThree hidden layers:")
    net = Network([784, 30, 30, 30, 10])
    initial_norms(td, net)
    training(td, net, epochs, "norms_during_training_3_layers.json")
    plot_training(
        epochs, "norms_during_training_3_layers.json", 3)

    print("\nFour hidden layers:")
    net = Network([784, 30, 30, 30, 30, 10])
    initial_norms(td, net)
    training(td, net, epochs,
             "norms_during_training_4_layers.json")
    plot_training(
        epochs, "norms_during_training_4_layers.json", 4)


def initial_norms(training_data, net):
    average_gradient = get_average_gradient(net, training_data)
    norms = [list_norm(avg) for avg in average_gradient[:-1]]
    print("Average gradient for the hidden layers: " + str(norms))


def training(training_data, net, epochs, filename):
    norms = []
    for epoch in range(epochs):
        average_gradient = get_average_gradient(net, training_data)
        norms.append([list_norm(avg) for avg in average_gradient[:-1]])
        print(f"Epoch: {epoch}")
        start = time.process_time_ns()
        net.SGD(
            training_data=training_data,
            epochs=1,
            mini_batch_size=1000,
            eta=0.1,
            regularization_coefficient=5.0
        )
        end = time.process_time_ns()
        print(f"Time to train for one epoch: {end - start} ns")
    with open(filename, "w") as f:
        json.dump(norms, f)


def plot_training(epochs, filename, num_layers):
    with open(filename, "r") as f:
        norms = json.load(f)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    blue = "#2A6EA6"
    orange = "#FFA933"
    red = "#FF5555"
    green = "#55FF55"
    purple = "#5555FF"
    colors = [blue, orange, red, green, purple]
    for layer_index in range(num_layers):
        ax.plot(np.arange(epochs),
                [n[layer_index] for n in norms],
                color=colors[layer_index],
                label="Hidden layer %s" % (layer_index + 1,))
    ax.set_xlim([0, epochs])
    ax.grid(True)
    ax.set_xlabel('Number of epochs of training')
    ax.set_title('Speed of learning: %s hidden layers' % num_layers)
    ax.set_yscale('log')
    plt.legend(loc="upper right")
    fig_filename = "training_speed_%s_layers.png" % num_layers
    plt.savefig(fig_filename)
    plt.show()


def get_average_gradient(net, training_data):
    start = time.process_time_ns()
    nabla_b_results = (net.backprop_bias(training_input, expected_result)
                       for training_input, expected_result in training_data)
    gradient = list_sum(nabla_b_results)
    result = [(np.reshape(g, len(g)) / len(training_data)).tolist()
            for g in gradient]
    end = time.process_time_ns()
    print(f"Time to calculate average gradient: {end - start} ns")
    return result


def zip_sum(a, b):
    return [np.add(x, y) for (x, y) in zip(a, b)]


def list_sum(l):
    return functools.reduce(zip_sum, l)


def list_norm(l):
    return math.sqrt(sum([x * x for x in l]))


if __name__ == "__main__":
    main()
