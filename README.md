# Parallelized-Neural-Network-for-MNIST
 
This program presents a parallel implementation for a neural network, following closely the Python code found here: https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/notebook. Our neural network classifies handwritten digits from the MNIST dataset (https://yann.lecun.com/exdb/mnist/). We use the MNIST.go package to load this data. Each image is 28x28 pixels and is represented as a 784 element long array. Our network contains 3 layers - one input, one hidden, and one output. We write a set of matrix operations (mimicking operations found in NumPy, albeit less optimized), and operations for gradient descent, forward propagation, and back propagation - the tools that ultimately allow our network to learn a set of weights and biases that can make accurate inferences on unseen test data.

Our implementation includes:

An input/output component that allows the program to read in data.

A sequential implementation.

Two parallel implementations: A work-stealing and work-balancing algorithm using a unbounded-dequeue implemented as linked-list. The neural network is parallelized in an ensemble fashion, i.e. training multiple models and then averaging the weights and biases.

Matrix operations written from scratch for adding, substracting, multiplying matrices to matrices and matrices to and from scalars.