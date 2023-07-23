import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient




def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_par):
    # NNCOSTFUNCTION Implements the neural network cost function for a two layer
    # neural network which performs classification
    #   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    #   X, y, lambda_par) computes the cost and gradient of the neural network. The
    #   parameters for the neural network are "unrolled" into the vector
    #   nn_params and need to be converted back into the weight matrices.
    #
    #   The returned parameter grad should be a "unrolled" vector of the
    #   partial derivatives of the neural network.
    #
    
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    # Setup some useful variables
    m = X.shape[0]

    # You need to return the following variables correctly
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # ====================== YOUR CODE HERE ======================
    # Part 1: Feedforward the neural network and return the cost in the
    # variable J.

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

    # Convert y to a binary matrix
    y_matrix = np.eye(num_labels)[y.astype(int)-1]

    # Calculate cost function without regularization
    J = (-1/m) * np.sum(y_matrix * np.log(h) + (1-y_matrix) * np.log(1-h))

    # Regularize cost function
    reg = (lambda_par/(2*m)) * (np.sum(np.square(Theta1[:,1:])) + np.sum(np.square(Theta2[:,1:])))
    J = J + reg

    # Part 2: Implement the backpropagation algorithm to compute the gradients
    # Theta1_grad and Theta2_grad.

    delta3 = h - y_matrix
    delta2 = np.dot(delta3, Theta2[:,1:]) * sigmoidGradient(z2)

    Delta2 = np.dot(delta3.T, a2)
    Delta1 = np.dot(delta2.T, a1)

    Theta2_grad = Delta2/m
    Theta1_grad = Delta1/m

    # Part 3: Implement regularization with the cost function and gradients.
    Theta2_grad[:,1:] = Theta2_grad[:,1:] + (lambda_par/m) * Theta2[:,1:]
    Theta1_grad[:,1:] = Theta1_grad[:,1:] + (lambda_par/m) * Theta1[:,1:]

    # =========================================================================

    # Unroll gradients
    grad = np.concatenate((Theta1_grad.ravel(), Theta2_grad.ravel()), axis=0)

    return J, grad


def nnCostFunction2(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_par):
    # NNCOSTFUNCTION Implements the neural network cost function for a two layer
    # neural network which performs classification
    #   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    #   X, y, lambda_par) computes the cost and gradient of the neural network. The
    #   parameters for the neural network are "unrolled" into the vector
    #   nn_params and need to be converted back into the weight matrices.
    #
    #   The returned parameter grad should be a "unrolled" vector of the
    #   partial derivatives of the neural network.
    #
    
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    # Setup some useful variables
    m = X.shape[0]

    # You need to return the following variables correctly
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # ====================== YOUR CODE HERE ======================
    # Part 1: Feedforward the neural network and return the cost in the
    # variable J.

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

    # Convert y to a binary matrix
    y_matrix = np.eye(num_labels)[y.astype(int)-1]

    # Calculate cost function without regularization
    J = (-1/m) * np.sum(y_matrix * np.log(h) + (1-y_matrix) * np.log(1-h))

    # Regularize cost function
    reg = (lambda_par/(2*m)) * (np.sum(np.square(Theta1[:,1:])) + np.sum(np.square(Theta2[:,1:])))
    J = J + reg

    # Part 2: Implement the backpropagation algorithm to compute the gradients
    # Theta1_grad and Theta2_grad.

    delta3 = h - y_matrix
    delta2 = np.dot(delta3, Theta2[:,1:]) * sigmoidGradient(z2)

    Delta2 = np.dot(delta3.T, a2)
    Delta1 = np.dot(delta2.T, a1)

    Theta2_grad = Delta2/m
    Theta1_grad = Delta1/m

    # Part 3: Implement regularization with the cost function and gradients.
    Theta2_grad[:,1:] = Theta2_grad[:,1:] + (lambda_par/m) * Theta2[:,1:]
    Theta1_grad[:,1:] = Theta1_grad[:,1:] + (lambda_par/m) * Theta1[:,1:]

    # =========================================================================

    # Unroll gradients
    grad = np.concatenate((Theta1_grad.ravel(), Theta2_grad.ravel()), axis=0)

    return J, grad
