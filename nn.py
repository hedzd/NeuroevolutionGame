import numpy as np
import math

class NeuralNetwork:

    def __init__(self, layer_sizes):
        """
        Neural Network initialization.
        Given layer_sizes as an input, you have to design a Fully Connected Neural Network architecture here.
        :param layer_sizes: A list containing neuron numbers in each layers. For example [3, 10, 2] means that there are
        3 neurons in the input layer, 10 neurons in the hidden layer, and 2 neurons in the output layer.
        """
        center = 0
        margin = 1

        self.w = []
        self.b = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.normal(center, margin, size=(layer_sizes[i + 1], layer_sizes[i]))
            self.w.append(w)
            b = np.zeros((layer_sizes[i + 1], 1))
            self.b.append(b)

        self.activation = np.vectorize(self.activation)

    def activation(self, x):
    #     """
    #     The activation function of our neural network, e.g., Sigmoid, ReLU.
    #     :param x: Vector of a layer in our network.
    #     :return: Vector after applying activation function.
    #     """
        if x<0:
            return 1 - 1/(1 + math.exp(x))
        else:
            return 1/(1 + math.exp(-x))





    def forward(self, x):
        """
        Receives input vector as a parameter and calculates the output vector based on weights and biases.
        :param x: Input vector which is a numpy array.
        :return: Output vector
        """
        a = x
        for i in range(len(self.w)):
            z = (self.w[i] @ a) + self.b[i]
            a = self.activation(z)
        
        return a
    