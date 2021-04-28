"""Custom Resonances layer."""

import tensorflow as tf
from eventgan.modules.backend import safer_sqrt


class Resonances(tf.keras.layers.Layer):
    """Resonance Layer.
    This function gives the invariant mass of n particles.

    # Arguments
        input: Array with input data
            that will be used to calculate the invariant mass from.
        particle_id: Integers, particle IDs of n particles.
            If there is only one momentum that has to be considered
            the shape is:
            `particle_id = [particle_1]`.

            If there are more then one momenta (q = p_1 + p_2 + ..)
            the shape is: `particle_id = [particle_1, particle_2,..]`.
    """

    def __init__(self, topology, scaler=1.0, **kwargs):

        self.topology = topology
        self.resonances = len(self.topology)
        self.scaler = scaler

        super().__init__(**kwargs)

    def call(self, inputs):  # pylint: disable=arguments-differ
        # Build the actual logic
        res = [0] * self.resonances
        for i in range(self.resonances):
            res[i] = self._invariant_mass(inputs, self.topology[i])

        result = tf.concat(res, -1)

        return result

    def _invariant_mass(self, x, particle_id):
        """Calculates the invariant mass of n particles."""
        # pylint: disable=invalid-name
        Es = 0
        PXs = 0
        PYs = 0
        PZs = 0
        for particle in particle_id:
            Es += x[:, 0 + particle * 4]
            PXs += x[:, 1 + particle * 4]
            PYs += x[:, 2 + particle * 4]
            PZs += x[:, 3 + particle * 4]

        m2 = tf.square(Es) - tf.square(PXs) - tf.square(PYs) - tf.square(PZs)

        # Clipping is crucial to stabilize the gradient (-> NaNs during backprop)
        m = tf.sign(m2) * safer_sqrt(m2)

        # Normalize with respect to data
        m = self.scaler * m
        m = tf.expand_dims(m, -1)
        return m

    # pylint: enable=invalid-name
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.resonances)

    def get_config(self):
        config = {"topology": self.topology, "scaler": self.scaler}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
