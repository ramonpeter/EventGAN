"""Loss functions."""

from typing import List, Tuple
import tensorflow as tf


def discriminator_regularizer(
    discriminator: tf.keras.Model,
    real_in: tf.Tensor,
    gen_in: tf.Tensor,
    batch_size: int,
):
    """
    Regularizer for the discriminator network. It regularizes
    the discriminator with respect to the gradients for both real
    and fake events.
    The implementation is from Roth et al. (arxiv:1705.09367 [cs.LG])
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(real_in)
        tape.watch(gen_in)
        logit_real = discriminator(real_in)
        logit_gen = discriminator(gen_in)

    grad_logit_real = tf.reshape(tape.gradient(logit_real, real_in), [batch_size, -1])
    grad_logit_gen = tf.reshape(tape.gradient(logit_gen, gen_in), [batch_size, -1])

    grad_logits_real_norm = tf.norm(grad_logit_real, axis=1, keepdims=True)
    grad_logits_gen_norm = tf.norm(grad_logit_gen, axis=1, keepdims=True)

    reg_real = tf.square(1.0 - tf.math.sigmoid(logit_real)) * tf.square(
        grad_logits_real_norm
    )
    reg_gen = tf.square(tf.math.sigmoid(logit_gen)) * tf.square(grad_logits_gen_norm)
    disc_regularizer = tf.reduce_mean(reg_real + reg_gen)

    return disc_regularizer


def resonance_loss(
    gen: tf.Tensor,
    real: tf.Tensor,
    kernel: callable,
    kernel_widths: List[Tuple[float]],
    resonances: int,
):
    """
    Calculate individual MMD2 losses for all
    appearing resonances in the respective scattering
    process.
    """
    mmd_loss = 0.0
    for k in range(resonances):
        mmd_gen = gen[:, k : k + 1]
        mmd_real = real[:, k : k + 1]
        mmd_loss += squared_mmd(mmd_real, mmd_gen, kernel, kernel_widths[k])

    return mmd_loss


def squared_mmd(  # pylint: disable=invalid-name
    x: tf.Tensor, y: tf.Tensor, kernel: callable, kernel_widths: Tuple[float], wts=None
):
    r"""
    Computes the Squared Maximum Mean Discrepancy (MMD2) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the
    samples of the distributions of x and y.

    Here we use the kernel-two-sample estimate using the empirical mean of
    the two distributions:

    MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
                = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },

    where K = <\phi(x), \phi(y)>, is the desired kernel function.

    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        kernel: a function which computes the kernel in MMD. Defaults
                to the GaussianKernelMatrix.
    Returns:
        a scalar denoting the squared maximum mean discrepancy loss.
    """
    k_xx, k_xy, k_yy = kernel(x, y, kernel_widths, wts=wts)

    mmd2 = tf.reduce_mean(k_xx) + tf.reduce_mean(k_yy) - 2 * tf.reduce_mean(k_xy)

    # We do not allow the loss to become negative.
    mmd2 = tf.where(mmd2 > 0, mmd2, 0, name="value")

    return mmd2
