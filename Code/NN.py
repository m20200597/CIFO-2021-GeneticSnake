import numpy as np

inputUnits = 8
hiddenLayer1Units = 9
hiddenLayer2Units = 15
outputUnits = 3

Weight1_shape = (hiddenLayer1Units,inputUnits)
Weight2_shape = (hiddenLayer2Units,hiddenLayer1Units)
Weight3_shape = (outputUnits,hiddenLayer2Units)

nWeights = inputUnits*hiddenLayer1Units + hiddenLayer1Units*hiddenLayer2Units + hiddenLayer2Units*outputUnits

def get_weights_from_encoded(individual):
    ''' Decodes the full array of weights into each layer'''
    W1 = individual[0:Weight1_shape[0] * Weight1_shape[1]]
    W2 = individual[Weight1_shape[0] * Weight1_shape[1]:Weight2_shape[0] * Weight2_shape[1] + Weight1_shape[0] * Weight1_shape[1]]
    W3 = individual[Weight2_shape[0] * Weight2_shape[1] + Weight1_shape[0] * Weight1_shape[1]:]

    return (
        W1.reshape(Weight1_shape[0], Weight1_shape[1]), 
        W2.reshape(Weight2_shape[0], Weight2_shape[1]), 
        W3.reshape(Weight3_shape[0], Weight3_shape[1])
        )


def softmax(z):
    ''' Implements the softmax activation function '''
    s = np.exp(z.T) / np.sum(np.exp(z.T), axis=1).reshape(-1, 1)

    return s


def sigmoid(z):
    ''' Implements the sigmoid activation function '''
    s = 1 / (1 + np.exp(-z))

    return s


def forward_propagation(X, individual):
    ''' Performs the forward propagation algorithm with the input and the
    weights received for this particular individual '''
    W1, W2, W3 = get_weights_from_encoded(individual)

    Z1 = np.matmul(W1, X.T)
    A1 = np.tanh(Z1)
    Z2 = np.matmul(W2, A1)
    A2 = np.tanh(Z2)
    Z3 = np.matmul(W3, A2)
    A3 = softmax(Z3)
    return A3