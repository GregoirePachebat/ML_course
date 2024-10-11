# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""
import sys
sys.path.append("../")
import numpy as np
from ex02.template.costs import compute_loss


def least_squares(y, tx):
    """calculate the least squares.
    
    input :
    y : np.array (N,1)
    The training output
    tx : np.array (N, D)
    The training input expanded or not
    
    return :
    w : np.array
    the weights of the model
    mse : float
    The training error"""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # least squares: TODO
    # returns mse, and optimal weights
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    mse = compute_loss(y, tx, w)
    return w, mse
    # ***************************************************

