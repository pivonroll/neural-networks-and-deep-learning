"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip
from typing import Tuple, List

# Third-party libraries
import numpy as np
import numpy.typing as npt

MNISTImage = npt.ArrayLike # A training image is a numpy array with 784 pixels representing a 28x28 image
MNISTImages = npt.ArrayLike # A list of training images, 50,000 in total
MNISTImageValue = int
MNISTImageValues = npt.ArrayLike # A list of training image values, 50,000 in total

MNISTData = Tuple[MNISTImages, MNISTImageValues] # A tuple containing the training images and their corresponding values

def load_data() -> Tuple[MNISTData, MNISTData, MNISTData]:
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.


print(load_data_wrapper()[0])
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('../../data/mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    training_data, validation_data, test_data = u.load()
    f.close()
    return training_data, validation_data, test_data

MNISTTrainingResultVector = npt.ArrayLike # A 10-dimensional unit vector with a 1.0 in the jth position and zeroes elsewhere
MNISTDataWrapper = Tuple[MNISTImage, MNISTTrainingResultVector]
MNISTImageAndResult = Tuple[MNISTImage, MNISTImageValue]

def load_data_wrapper() -> Tuple[List[MNISTDataWrapper], List[MNISTImageAndResult], List[MNISTImageAndResult]]:
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    mnist_training_data, mnist_validation_data, mnist_test_data = load_data()
    mnist_training_images = mnist_training_data[0]
    training_inputs = [np.reshape(mnist_training_image, (784, 1)) for mnist_training_image in mnist_training_images]
    mnist_training_values = mnist_training_data[1]
    training_results = [vectorized_result(mnist_training_value) for mnist_training_value in mnist_training_values]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in mnist_validation_data[0]]
    validation_data = list(zip(validation_inputs, mnist_validation_data[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in mnist_test_data[0]]
    test_data = list(zip(test_inputs, mnist_test_data[1]))
    return training_data, validation_data, test_data

def vectorized_result(j) -> MNISTTrainingResultVector:
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
