"""Lorentzvector Layer."""

import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K


# pylint: disable=C0103, W0221
class LorentzVector(tf.keras.layers.Layer):
    """
    A layer to create different representations
    of a lorentzvector.
    neural network structures.
    """

    def __init__(
        self,
        n_features=4,
        n_particles=2,
        order=1,  # 0: (E1,px1,py1,pz1,E2,...), 1: (E1,..,En,px1..,pxn,py1..,pyn,pz1,...,pzn)
        mass_in=1,  #
        E=1,
        Px=1,
        Py=1,
        Pz=1,
        M=0,
        Pt=0,
        Eta=0,
        Y=0,
        Phi=0,
        logE=0,
        logPt=0,
        logPtcut=0,
        ptcuts=0.0,
        dscaler=1.0,
        **kwargs
    ):
        """
        Args:
            n_features:     int, number of input features.
            n_particles:    int, number of input particles.
            order:          bool, whether the input is in C (0) or Fortran (1) ordering.
            mass_in:        bool, whether the input contains the masses instead of energies.

            out_features:   various combinations of output features. Standard is
                            (E,px,py,pz). Other options are:
                            M, pt, eta, y, phi, logE, logPt, logPtcut.

            ptcuts:         float, where the ptcuts are set.
            dscaler:        float, rescaler the features.
        """
        super().__init__(**kwargs)

        self.n_features = n_features
        self.n_particles = n_particles

        self.order = order
        self.mass_in = mass_in

        self.E = E
        self.Px = Px
        self.Py = Py
        self.Pz = Pz

        self.M = M
        self.Pt = Pt
        self.Eta = Eta
        self.Y = Y
        self.Phi = Phi
        self.logE = logE
        self.logPt = logPt
        self.logPtcut = logPtcut

        if isinstance(ptcuts, list):
            assert len(ptcuts) == self.n_particles
            for idx, cut in enumerate(ptcuts):
                if cut < 0.0:
                    ptcuts[idx] = 0.0
            self.ptcuts = K.constant(ptcuts)
        else:
            self.ptcuts = tf.zeros(self.n_particles)

        self.dscaler = dscaler

        self.nout = (
            int(self.E)
            + int(self.Px)
            + int(self.Py)
            + int(self.Pz)
            + int(self.M)
            + int(self.Pt)
            + int(self.Eta)
            + int(self.Y)
            + int(self.Phi)
            + int(self.logE)
            + int(self.logPt)
            + int(self.logPtcut)
        )

        self.shape_dim = self.nout * self.n_particles

    def call(self, x):
        """Build the actual logic."""

        # (None, n_features_in * n_particles) -> (None, n_features_out * n_particles)
        out_features = []

        if self.mass_in:
            metric_vector = [1.0, 1.0, 1.0, 1.0]
        else:
            metric_vector = [1.0, -1.0, -1.0, -1.0]

        metric = K.variable(np.array(metric_vector))

        # rescale variables
        x = self.dscaler * x

        # Our input is of the form
        # (b,f,p)
        # -> (batch_size, features, particles)
        if self.order:
            x = tf.reshape(x, (tf.shape(input=x)[0], self.n_features, self.n_particles))
        else:
            x = tf.transpose(
                a=tf.reshape(
                    x, (tf.shape(input=x)[0], self.n_particles, self.n_features)
                ),
                perm=(0, 2, 1),
            )

        # Let's build a few helpful matrices

        # All the individual dimensions
        # bp
        Xs = x[:, 1, :]
        Ys = x[:, 2, :]
        Zs = x[:, 3, :]

        # Element wise square of x
        # bfp
        x2 = tf.square(x)

        if self.mass_in:
            Ms = x[:, 0, :]
            Es = tf.sqrt(tf.maximum(tf.tensordot(x2, metric, axes=[1, 0]), K.epsilon()))
        else:
            Es = x[:, 0, :]
            Ms = tf.sqrt(tf.maximum(tf.tensordot(x2, metric, axes=[1, 0]), K.epsilon()))

        # Momentum
        Ps = tf.sqrt(tf.maximum(x2[:, 1, :] + x2[:, 2, :] + x2[:, 3, :], K.epsilon()))
        # Transverse Momentum
        Pts = tf.sqrt(tf.maximum(x2[:, 1, :] + x2[:, 2, :], K.epsilon()))
        # Rapidity
        ys = 0.5 * (
            K.log(tf.maximum(Es + Zs, K.epsilon()))
            - K.log(tf.maximum(Es - Zs, K.epsilon()))
        )
        # Pseudo-Rapidity
        Etas = 0.5 * (
            K.log(tf.maximum(Ps + Zs, K.epsilon()))
            - K.log(tf.maximum(Ps - Zs, K.epsilon()))
        )
        # Phi
        Phis = tf.atan2(Ys, Xs)
        # log E
        logEs = K.log(x[:, 0, :])
        # log Pt
        logPts = K.log(tf.maximum(Pts, K.epsilon()))
        # log Pt cuts
        logPtscut = K.log(tf.maximum(Pts - self.ptcuts, K.epsilon()))

        # Construct Lorentz-Vector
        if self.M:
            out_features.append(Ms)
        if self.E:
            out_features.append(Es)
        if self.Px:
            out_features.append(Xs)
        if self.Py:
            out_features.append(Ys)
        if self.Pz:
            out_features.append(Zs)

        if self.Pt:
            out_features.append(Pts)
        if self.Eta:
            out_features.append(Etas)
        if self.Y:
            out_features.append(ys)
        if self.Phi:
            out_features.append(Phis)
        if self.logE:
            out_features.append(logEs)
        if self.logPt:
            out_features.append(logPts)
        if self.logPtcut:
            out_features.append(logPtscut)

        y = K.stack(out_features, axis=1)
        y = tf.transpose(a=y, perm=(0, 2, 1))
        results = tf.reshape(y, (tf.shape(input=y)[0], self.shape_dim))

        return results

    def compute_output_shape(self, input_shape):
        """
        Input shape:
            2D tensor with shape:
            `(batch_size, features_in * particles_in)`

        Output shape:
            2D tensor with shape:
            `(batch_size, features_out * particles_in)`,
            where features_out != features_in in general.
        """
        return (input_shape[0], self.nout * self.n_particles)

    def get_config(self):
        config = {
            "n_features": self.n_features,
            "n_particles": self.n_particles,
            "order": self.order,
            "mass_in": self.mass_in,
            "E": self.E,
            "Px": self.Px,
            "Py": self.Py,
            "Pz": self.Pz,
            "M": self.M,
            "Pt": self.Pt,
            "Eta": self.Eta,
            "Y": self.Y,
            "Phi": self.Phi,
            "logE": self.logE,
            "logPt": self.logPt,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
