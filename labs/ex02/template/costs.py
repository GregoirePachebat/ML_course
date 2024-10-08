# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np
from typing_extensions import Literal



def compute_loss(y, tx, w, type: Literal["MSE", "MAE"] = 'MSE' ):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    if type == "MSE":
        return np.sum((y-tx@w).T @ (y-tx@w))
    if type == 'MAE' : 
        return np.sum(np.sqrt((y-tx@w).T @ (y-tx@w)))
    # ***************************************************

