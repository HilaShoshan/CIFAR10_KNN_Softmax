from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]  # each minibatch size
    C = W.shape[1]  # the number of classes

    # Softmax Loss
    for i in range(N):
        example = X[i]
        classes_scores = np.dot(example, W)  # predicted scores
        classes_scores -= np.max(classes_scores)
        true_classes_scores = classes_scores[y[i]]  # true scores
        numerator_pred = np.exp(classes_scores)  # h_i = e^(x_i) / Σe^(x_k)
        denominator = np.sum(numerator_pred)
        numerator_true = np.exp(true_classes_scores)
        loss -= np.log(numerator_true / denominator)  # update loss: L = -Σy_i * log(h_i)
        for j in range(C):
            dW[:, j] += (numerator_pred[j] * example) / denominator
            if j == y[i]:
                dW[:, y[i]] -= example

    loss /= N
    dW /= N

    # Regularization
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Softmax Loss
    N = X.shape[0]  # each minibatch size
    classes_scores = np.dot(X, W)  # matrix multiplication
    classes_scores -= np.max(classes_scores, axis=1)[:, np.newaxis]
    numerator = np.exp(classes_scores)
    denominator = np.sum(numerator, axis=1, keepdims=True)
    h = numerator / denominator
    loss = np.sum(-np.log(h[np.arange(N), y]))

    # Regularization

    loss /= N
    loss += 0.5 * reg * np.sum(W * W)

    ind = np.zeros_like(classes_scores)
    ind[np.arange(N), y] = 1
    dW = np.dot(X.T, numerator / denominator - ind)
    dW /= N
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
