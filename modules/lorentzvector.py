# basic modules
import sys
import numpy as np

# tensorflow modules
import tensorflow as tf
from tensorflow.keras import backend as K

class LorentzVector(tf.keras.layers.Layer):
    """
    A layer to create different representations
    of a lorentzvector.
    neural network structures. 
    """
    def __init__(
        self,
        n_features  = 4,
        n_particles = 2,
        debug       = False,

        order = 1, # 0: (E1,px1,py1,pz1,E2,...), 1: (E1,..,En,px1..,pxn,py1..,pyn,pz1,...,pzn)
        mass_in = 1,  # 

        E    = 1,
        Px   = 1,
        Py   = 1,
        Pz   = 1,

        M   = 0,
        Pt  = 0,
        Eta = 0,
        Y   = 0,
        Phi = 0,
        logE = 0,
        logPt = 0,
        logPtcut=0,

        ptcuts=0.0,

        dscaler = 1.0,

        **kwargs
    ):
        """
        Args:
            n_features:     int, number of input features.
            n_particles:    int, number of input particles.
            debug:          bool, whether the layer is run in debug mode
            order:          bool, whether the input is in C (0) or Fortran (1) ordering.
            mass_in:        bool, whether the input contains the masses instead of energies.

            out_features:   various combinations of output features. Standard is
                            (E,px,py,pz). Other options are:
                            M, pt, eta, y, phi, logE, logPt, logPtcut.

            ptcuts:         float, where the ptcuts are set.
            dscaler:        float, rescaler the features.
        """
        super().__init__(**kwargs)

        self.debug = debug

        if self.debug:
            self.n_features  = 4
            self.n_particles = 2
        else:
            self.n_features  = n_features
            self.n_particles = n_particles

        self.order = order
        self.mass_in = mass_in

        self.E   = E
        self.Px  = Px
        self.Py  = Py
        self.Pz  = Pz

        self.M   = M
        self.Pt  = Pt
        self.Eta = Eta
        self.Y   = Y
        self.Phi = Phi
        self.logE = logE
        self.logPt = logPt
        self.logPtcut = logPtcut

        if isinstance(ptcuts, list):
            assert(len(ptcuts) == self.n_particles)
            for idx, cut in enumerate(ptcuts):
                if cut < 0.0:
                    ptcuts[idx] = 0.0
            self.ptcuts = K.constant(ptcuts)
        else:
            self.ptcuts = tf.zeros(self.n_particles)

        self.dscaler = dscaler

        self.nout = (int(self.E)   +
                     int(self.Px)  +
                     int(self.Py)  +
                     int(self.Pz)  +
                     int(self.M)   +
                     int(self.Pt)  +
                     int(self.Eta) +
                     int(self.Y)   +
                     int(self.Phi) +
                     int(self.logE)+
                     int(self.logPt)+
                     int(self.logPtcut)
                 )

    def call(self, x):
        """Build the actual logic."""


        # (None, n_features_in * n_particles) -> (None, n_features_out * n_particles)

        if self.debug: # (b-quark und top quark)
            if self.mass_in and self.order:
                x= K.variable(np.array([[ 4.7, 173.0, 23.0714, -23.0714, -5.95763, 5.95763, 144.559, 70.157   ],
                                        [ 4.7, 173.0, -44.2193, 44.2193, 22.6458, -22.6458, 66.5098, 156.411  ],
                                        [ 4.7, 173.0, -145.893, 145.893, -7.78709, 7.78709, -6.34396, 973.583 ],
                                        [ 4.7, 173.0, -5.84513, 5.84513, 11.6187, -11.6187, 12.7253, -277.621 ],
                                        [ 4.7, 173.0, 16.8124, -16.8124, 84.6426, -84.6426, -226.115, -1651.57]]))

                energy = K.variable(np.array([[ 146.585, 188.199],
                                        [ 83.1494, 238.457],
                                        [ 146.314, 999.569],
                                        [ 18.7932, 327.371],
                                        [ 242.069, 1662.84]]))

            else:
                x= K.variable(np.array([[ 146.585, 23.0714, -5.95763, 144.559, 188.199, -23.0714, 5.95763, 70.157],
                                        [ 83.1494, -44.2193, 22.6458, 66.5098, 238.457, 44.2193, -22.6458, 156.411],
                                        [ 146.314, -145.893, -7.78709, -6.34396, 999.569, 145.893, 7.78709, 973.583],
                                        [ 18.7932, -5.84513, 11.6187, 12.7253, 327.371, 5.84513, -11.6187, -277.621],
                                        [ 242.069, 16.8124, 84.6426, -226.115, 1662.84, -16.8124, -84.6426, -1651.57]]))

                mb = K.variable(np.array([4.69839237, 4.70005045, 4.70473675, 4.70014593, 4.73085695]))
                mt = K.variable(np.array([173.00021994, 173.00046583, 172.99977356, 173.00055568, 172.93459507]))

        out_features = []

        if self.mass_in:
            metric_vector = [ 1., 1., 1., 1.]
        else:
            metric_vector = [ 1., -1., -1., -1.]

        metric = K.variable(np.array(metric_vector))

        # rescale variables
        x = self.dscaler * x


        # Our input is of the form
        # (b,f,p)
        # -> (batch_size, features, particles)
        if self.order:
            x =  tf.reshape(x,(tf.shape(input=x)[0], self.n_features, self.n_particles))
        else:
            x =  tf.transpose(a=tf.reshape(x,(tf.shape(input=x)[0], self.n_particles, self.n_features)),perm=(0,2,1))

        # Let's build a few helpful matrices

        # All the individual dimensions
        # bp
        Xs = x[:,1,:]
        Ys = x[:,2,:]
        Zs = x[:,3,:]

        # Element wise square of x
        # bfp
        x2 = tf.square(x)

        if self.mass_in:
            Ms = x[:,0,:]
            Es = tf.sqrt(tf.maximum(tf.tensordot(x2, metric, axes=[1,0]), K.epsilon()))
        else:
            Es = x[:,0,:]
            Ms = tf.sqrt(tf.maximum(tf.tensordot(x2, metric, axes=[1,0]), K.epsilon()))

        # Momentum
        Ps = tf.sqrt(tf.maximum(x2[:, 1, :] + x2[:, 2, :] + x2[:, 3, :], K.epsilon()))
        # Transverse Momentum
        Pts = tf.sqrt(tf.maximum(x2[:, 1, :] + x2[:, 2, :], K.epsilon()))
        # Rapidity
        ys = 0.5 * (K.log(tf.maximum(Es + Zs, K.epsilon())) - K.log(tf.maximum(Es - Zs, K.epsilon())))
        # Pseudo-Rapidity
        Etas = 0.5 * (K.log(tf.maximum(Ps + Zs, K.epsilon())) - K.log(tf.maximum(Ps - Zs, K.epsilon())))
        # Phi
        Phis = tf.atan2(Ys,Xs)
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


        y = K.stack(out_features, axis = 1)

        self.shape_dim = self.nout * self.n_particles
        y = tf.transpose(a=y,perm=(0,2,1))
        results = tf.reshape(y,(tf.shape(input=y)[0], self.shape_dim))


        if self.debug:
            if self.mass_in and self.order:
                print ("Momenta:")
                print (K.eval(results))
                print ("Shape:")
                print (K.eval(tf.shape(input=results)))
                print ("Energys:")
                print (K.eval(energy))
            else:
                print ("results:")
                print (K.eval(mt))
                print (K.eval(mb))
                print (K.eval(results))
                print (K.eval(tf.shape(input=results)))

        if self.debug:
            sys.exit()

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
            'n_features' : self.n_features,
            'n_particles' : self.n_particles,
            'debug': self.debug,
            'order': self.order,
            'mass_in' : self.mass_in,
            'E' : self.E,
            'Px' : self.Px,
            'Py' : self.Py,
            'Pz' : self.Pz,
            'M' : self.M,
            'Pt' : self.Pt,
            'Eta' : self.Eta,
            'Y' : self.Y,
            'Phi' : self.Phi,
            'logE' : self.logE,
            'logPt' : self.logPt
        }
        base_config = super(LorentzVector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
