"""Kerne functions for the MMD."""

import tensorflow as tf


# pylint: disable=invalid-name
def squared_pairwise_dist(X, Y):
    """Computes the squared pairwise Euclidean distances between x and y.

    Args:
        X: a tensor of shape [num_x_samples, num_features]
        Y: a tensor of shape [num_y_samples, num_features]

    Returns:
        a distance matrix of dimensions [num_x_samples, num_y_samples]

    Raises:
        ValueError: if the inputs do no matched the specified dimensions.
    """

    if not len(X.get_shape()) == len(Y.get_shape()) == 2:
        raise ValueError("Both inputs should be matrices.")
    if X.get_shape().as_list()[1] != Y.get_shape().as_list()[1]:
        raise ValueError("The number of features should be the same.")

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.linalg.diag_part(XX)
    Y_sqnorms = tf.linalg.diag_part(YY)

    def r(x):
        return tf.expand_dims(x, 0)

    def c(x):
        return tf.expand_dims(x, 1)

    return c(X_sqnorms) + r(Y_sqnorms) - 2 * XY


def mix_gaussian_kernel(X, Y, sigmas, wts=None, K_XY_only=False):
    """Computes a Gaussian Radial Basis Kernel between the samples of X and Y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.

    Args:
        X: a tensor of shape [num_samples, num_features]
        Y: a tensor of shape [num_samples, num_features]
        sigmas: a list of floats which denote the widths of each of the
            gaussians in the kernel.
    """

    # Define unit weights if not given. Check length otherwise
    if wts is None:
        wts = [1] * len(sigmas)
    else:
        if not len(wts) == len(sigmas):
            raise ValueError(
                "The number of weights need to be equal to number of sigmas."
            )

    K_XX, K_XY, K_YY = 0.0, 0.0, 0.0

    for sigma, wt in zip(sigmas, wts):
        gamma = 1.0 / (2.0 * sigma ** 2)
        K_XY += wt * tf.exp(-gamma * squared_pairwise_dist(X, Y))

    if K_XY_only:
        return K_XY

    for sigma, wt in zip(sigmas, wts):
        gamma = 1.0 / (2.0 * sigma ** 2)
        K_XX += wt * tf.exp(-gamma * squared_pairwise_dist(X, X))
        K_YY += wt * tf.exp(-gamma * squared_pairwise_dist(Y, Y))

    return K_XX, K_XY, K_YY


# Cauchy distribution
#


def mix_cauchy_kernel(X, Y, alphas, wts=None, K_XY_only=False):
    """Computes a mixture of cauchy kernels between
    the samples of x and y. We create a sum of multiple inverse multiquadratic
    kernels each having width alpha.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        alphas: list of floats which denote the widths of the kernels.
    """
    # Define unit weights if not given. Check length otherwise
    if wts is None:
        wts = [1] * len(alphas)
    else:
        if not len(wts) == len(alphas):
            raise ValueError(
                "The number of weights need to be equal to number of sigmas."
            )

    K_XX, K_XY, K_YY = 0, 0, 0

    for alpha, wt in zip(alphas, wts):
        gamma = 1 / (alpha ** 2)
        K_XY += wt * 1 / (1 + gamma * squared_pairwise_dist(X, Y))

    if K_XY_only:
        return K_XY

    for alpha, wt in zip(alphas, wts):
        gamma = 1 / (alpha ** 2)
        K_XX += wt * 1 / (1 + gamma * squared_pairwise_dist(X, X))
        K_YY += wt * 1 / (1 + gamma * squared_pairwise_dist(Y, Y))

    return K_XX, K_XY, K_YY


# Breit-Wigner distribution
#


def mix_breit_wigner_kernel(X, Y, alphas, wts=None, K_XY_only=False):
    """Computes a mixture of Breit-Wigner kernels between
    the samples of x and y. We create a sum of multiple inverse multiquadratic
    kernels each having width alpha.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        alphas: list of floats which denote the widths of the kernels.
    """
    # Define unit weights if not given. Check length otherwise
    if wts is None:
        wts = [1] * len(alphas)
    else:
        if not len(wts) == len(alphas):
            raise ValueError(
                "The number of weights need to be equal to number of sigmas."
            )

    K_XX, K_XY, K_YY = 0, 0, 0

    for alpha, wt in zip(alphas, wts):
        gamma = 4 / (alpha ** 2)
        K_XY += wt * 1 / (1 + gamma * squared_pairwise_dist(X, Y))

    if K_XY_only:
        return K_XY

    for alpha, wt in zip(alphas, wts):
        gamma = 4 / (alpha ** 2)
        K_XX += wt * 1 / (1 + gamma * squared_pairwise_dist(X, X))
        K_YY += wt * 1 / (1 + gamma * squared_pairwise_dist(Y, Y))

    return K_XX, K_XY, K_YY
