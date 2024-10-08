import numpy as np


def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute subgradient gradient vector for MAE
    e = y - tx@w
    sign_e = np.sign(e)
    sign_e.reshape(-1, 1)

    return np.sum(sign_e * tx, axis=0)/sign_e.shape[0]
    # ***************************************************

