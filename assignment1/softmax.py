from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg, regtype='L2'):
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
    - regtype: Regularization type: L1 or L2

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
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****



    for i in xrange(X.shape[0]):

        scores = X[i].dot(W)
        # First of all, as stated in lecture notes, we can shift them to the max.
        scores -= max(scores)

        # We know that Li= -fyi + log(sum(exp fj))
        # Since our f is our softmax function and its output is "scores"
        fyi= scores[y[i]]
        sum_exp_scores= np.sum(np.exp(scores))

        Li= (-1 * fyi) + np.log(sum_exp_scores)

        # Updating the loss
        loss = loss + Li

        # Probabilities for given classes are given by expfyi/sumj(expfj)
        # So we need to iterate over classes
        for c in range(W.shape[1]):
            score = scores[c]
            prob = np.exp(score) / sum_exp_scores

            dW[:,c] += (prob * X[i])

            # True class
            if y[i] == c:
                dW[:,c] -= X[i]

    # For regularization, take average and add regularization loss
    W_sqrt = W * W

    loss = (loss / X.shape[0])+ (reg * np.sum(W_sqrt))
    dW = dW / X.shape[0] + (reg * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg, regtype='L2'):
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
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    # First of all, as stated in lecture notes, we can shift them to the max.
    scores -= np.max(scores , axis=1).reshape(-1,1)
    divisor = np.sum(np.exp(scores) , axis=1).reshape(-1,1)

    # Our f is our softmax function and its output is "scores"
    softmax = np.exp(scores) / divisor
    loss = -np.sum(np.log(softmax[range(X.shape[0]), list(y)]))

    # Regularization
    W_sqrt = W * W
    loss = (loss / X.shape[0])+ (reg * np.sum(W_sqrt))

    softmax[range(X.shape[0]), list(y)] += -1
    dW = (X.T).dot(softmax)
    dW = dW / X.shape[0] + (reg * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
