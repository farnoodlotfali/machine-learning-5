import numpy as np
from sigmoid import sigmoid


def predict(Theta1, Theta2, X):
    #PREDICT Predict the label of an input given a trained neural network
    #   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
    #   trained weights of a neural network (Theta1, Theta2)

    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # You need to return the following variable correctly
    p = np.zeros((X.shape[0], 1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the following code to make predictions using
    #               your learned neural network. You should set p to a
    #               vector containing labels between 1 to num_labels.
    #
    # Hint: The numpy argmax function might come in useful to return the index of the max element. In particular, the max

    # Add bias term to input layer
    a1 = np.hstack((np.ones((m, 1)), X))

    # Calculate activations of hidden layer
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)

    # Add bias term to hidden layer
    a2 = np.hstack((np.ones((m, 1)), a2))

    # Calculate activations of output layer
    z3 = np.dot(a2, Theta2.T)
    h = sigmoid(z3)

    # Find index of maximum value in each row
    p = np.argmax(h, axis=1)
    p = p.reshape((m, 1))
    p = p + 1

    # =========================================================================

    return p