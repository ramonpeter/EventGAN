"""EventGAN class."""

from typing import List, Union, Tuple
import os
import math

import numpy as np
import pandas as pd

# import tensorflow
import tensorflow as tf
from sklearn.model_selection import train_test_split

# import custom modules
import eventgan.utils.kernels as kf
from eventgan.modules.lorentzvector import LorentzVector
from eventgan.modules.resonances import Resonances
from eventgan.modules.losses import resonance_loss, discriminator_regularizer


# pylint: disable=C0103, W0212
class EventGAN:
    """
    EventGAN to generate LHC events parametrized as
    4 vectors. To specify the process being trained on
    it requires the number of final state particles,
    the topology (indicate possible s-channels),
    and the external masses [GeV]. Optionally,
    a MMD can be used to resolve the resonances
    appearing in the s-channels.
    """

    def __init__(
        self,
        n_particles: int,
        latent_dim: int,
        topology: List[Tuple[int]],
        input_masses: list,
        train_data_path: str,
        train_fraction: float,
        test_data_path: str,
        scaler: float,
        save_path: str,
        g_units: int,
        g_layers: int,
        d_units: int,
        d_layers: int,
        reg_weight: float,
        use_mmd_loss: bool = False,
        mmd_weight: float = 1.0,
        mmd_kernel: Union[str, callable] = "BREIT-WIGNER",
        mmd_kernel_widths: List[Tuple[float]] = None,
    ):
        r"""
        Args:
            n_particles:
                int specifying the number of external particles.
            latent_dim:
                int specifying dimension of the latent space.
            topology:
                a list of integer tuples indicating the possible s-channels
                of intermediate on-shell resonances. Eg.,

                    ``topology = [(0,1),(2,3,4)]``

                if the invariants s_{01} and s_{234} belongt to resonances.
            input_masses:
                list of external final-state masses in GeV. Required
                to guarantee on-shell conditions for generated particles.
            train_data_path:
                path to the file used for training.
            train_fraction:
                defines the fraction of data which is actually used
                for training. Useful, to check stability.
            test_data_path:
                path to the file containing independent
                test/validation events.
            scaler:
                scaler factor to preprocess the events.
                Network performs best when scaler is choosen
                such that stdev(data/scaler)~1.
            save_path:
                path where logs, plots and weights of the run
                are saved.
            g_units:
                number of units in each hidden Dense layer
                in the generator.
            g_layers:
                number of hidden layers in the generator.
            d_units:
                number of units in each hidden Dense layer
                in the discriminator.
            d_layers:
                number of hidden layers in the discriminator.
            reg_weight:
                float defining the effect of the disc
                regularizer.
            use_mmd_loss:
                switch to use a MMD loss to resolve the resonances.
                Default is ``False``.
            mmd_weight: float=None,
                relative contribution of the MMD loss to the total
                generator loss. Needs to be fixed by hyperparameter
                tuning. Default is ``1.0``.
            mmd_kernel: Union[str, callable]=None,
                kernel function which is used to evaluate the MMD.
                It expects one of ``'BREIT-WIGNER'``, ``'CAUCHY'``,
                ``'GAUSS'`` or a callable kernel of the format:

                    ``kernel(tensor1, tensor2, kernel_widths)``.

                Default is ``'BREIT-WIGNER'``.
            mmd_kernel_widths:
                list of float tuples, where len(list) is the number of
                MMDs added to the loss (usually one MMD per resonance)
                and len(tuple) is the number of kernel_widths per
                resonance. Default is `None`. A valid input would
                look like ``mmd_kernel_widths = [(2.05,), (1.49,), (1.0, 10.)]``.
                In the case of resonances and when using the "BREIT-WIGNER"
                kernel it is a good starting point to use the physical
                widths :math:`\Gamma` of the assumed resonant particle in GeV.

                For instance, lets assume we have a Drell-Yan process,
                like :math:`e^+ e^- -> \mu^+ \mu^-`:

                   ---->----|           |---->---- [0]
                       e-   |           |    mu-
                            |~~~~~~~~~~~|
                            |    y/Z    |
                   ----<----|           |----<---- [1]
                       e+                   mu+

                Then the topology should be set to:

                    ``topology = [(0,1)]``

                And the kernel to "BREIT-WIGNER" with

                    ``mmd_kernel_widths = [(2.44,)]``

                since :math:`\Gamma_{Z} = 2.44 GeV`.

                **Note**: It is required that

                    ``len(topology) ==  len(mmd_kernel_widths)``
        """
        # Set process paramaters
        self.n_particles = n_particles
        self.latent_dim = latent_dim
        self.topology = list(topology)
        self.resonances = len(topology)

        # MMD
        self.use_mmd_loss = use_mmd_loss
        self.mmd_weight = mmd_weight
        self.mmd_kernel_widths = list(mmd_kernel_widths)
        if isinstance(mmd_kernel, str):
            if mmd_kernel == "BREIT-WIGNER":
                self.mmd_kernel = kf.mix_breit_wigner_kernel
            elif mmd_kernel == "CAUCHY":
                self.mmd_kernel = kf.mix_cauchy_kernel
            elif mmd_kernel == "GAUSS":
                self.mmd_kernel = kf.mix_gaussian_kernel
            else:
                raise ValueError(f'Unknown kernel function "{mmd_kernel}"')
        else:
            self.mmd_kernel = mmd_kernel

        # make sure the number of resonances and kernels match
        assert len(self.topology) == len(
            self.mmd_kernel_widths
        ), "Number of kernel_widths and number of resonances do not match."

        # I/O parameters
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.scaler = scaler
        self.save_path = save_path

        # Preprosess and load data
        self.initialize_data(train_fraction, input_masses)

        # Define networks
        self.generator = self.get_generator(g_units, g_layers)
        self.discriminator = self.get_discriminator(d_units, d_layers)

        # Define loss functions
        self.bc_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.mmd_loss = resonance_loss
        self.disc_regularizer = discriminator_regularizer
        self.reg_weight = reg_weight

    def get_generator(self, n_units: int, n_layers: int):
        """
        Generator model for the EventGAN framework
        ----------
        Args:
        n_units     : number of nodes per layer.
        n_layers    : number of layers.
        ----------
        Input:
        (noise, masses): (tf.Tensor of shape (batch_size, 3 * n_particles),
                        tf.Tensor of shape (batch_size, n_particles )) --
                        the noise input and the masses of the outgoing particles
        ----------
        Output:
        four_momenta : tf.Tensor of shape (batch_size, 4 * n_particles) --
                    the 4-momenta of the events as (E1,px1,py1,pz1,E2,...)
        """
        # Input
        noise = tf.keras.layers.Input(shape=(3 * self.n_particles,), name="noise")
        masses = tf.keras.layers.Input(shape=(self.n_particles,), name="masses")

        # hidden layers
        x = noise
        for _ in range(n_layers):
            x = tf.keras.layers.Dense(n_units, activation=tf.keras.layers.LeakyReLU())(
                x
            )

        # 3-momenta
        three_momenta = tf.keras.layers.Dense(3 * self.n_particles, name="3-momenta")(x)
        mass_combined_momenta = tf.keras.layers.concatenate(
            [masses, three_momenta], name="mass_and_momenta"
        )

        # Get 4-momenta
        four_momenta = LorentzVector(n_particles=self.n_particles, name="4-momenta")(
            mass_combined_momenta
        )

        return tf.keras.Model([noise, masses], four_momenta, name="Generator")

    def get_discriminator(self, n_units: int, n_layers: int):
        """
        Discriminator model for the EventGAN framework
        ----------
        Args:
        n_units     : number of nodes per layer.
        n_layers    : number of layers.
        ----------
        Input:
        events : tf.Tensor of shape (batch_size, 4 * n_particles) --
                4-momenta of the real/fake events
        ----------
        Output:
        logit : tf.Tensor of shape (batch_size, 1) --
                the logit of the class probability
        """
        # Input
        events = tf.keras.layers.Input(shape=(4 * self.n_particles,), name="events")

        # 3-momenta
        x = events
        for _ in range(n_layers):
            x = tf.keras.layers.Dense(n_units, activation=tf.keras.layers.LeakyReLU())(
                x
            )

        logit = tf.keras.layers.Dense(1, name="logit")(x)

        return tf.keras.Model(events, logit, name="Discriminator")

    def save_weights(self, suffix: str = ""):
        """Save the model weights"""
        self.generator.save_weights(
            f"{self.save_path}/generator_weights{suffix}.h5", save_format="h5"
        )
        self.discriminator.save_weights(
            f"{self.save_path}/discriminator_weights{suffix}.h5", save_format="h5"
        )

    def load_weights(self, suffix: str = ""):
        """Load the model weights"""
        self.generator.load_weights(f"{self.save_path}/generator_weights{suffix}.h5")
        self.discriminator.load_weights(
            f"{self.save_path}/discriminator_weights{suffix}.h5"
        )

    def initialize_data(self, training_fraction: float, input_masses: list):
        """Get input data and preprocess"""

        data = pd.read_hdf(self.train_data_path)
        data = data.iloc[:, :].values
        train_data, _ = train_test_split(data, train_size=training_fraction)

        # define different validation/test set
        try:
            data = pd.read_hdf(self.test_data_path)
            data = data.iloc[:, :].values
            test_data = data
        except FileNotFoundError:
            test_data = np.copy(train_data)

        # check if the mass-input and data match
        masses = tf.constant([input_masses])
        assert (
            4 * masses.shape[1] == train_data.shape[1]
        ), "Masses do not match input data"

        # scale data
        self.train_data = train_data / self.scaler
        self.test_data = test_data / self.scaler
        self.masses = masses / self.scaler

    @staticmethod
    def sample_data(data, batch_size):
        """Get array of samples from loaded data"""
        index = np.arange(data.shape[0])
        if batch_size <= data.shape[0]:
            choice = np.random.choice(index, batch_size, replace=False)
        else:
            choice = np.random.choice(index, batch_size, replace=True)

        batch = data[choice]
        return tf.convert_to_tensor(batch, dtype=tf.float32)

    def get_events(self, n_samples):
        """Sample events"""
        noise = tf.random.normal(shape=(n_samples, self.latent_dim))
        mass_batch = tf.repeat(self.masses, n_samples, axis=0)
        events = self.scaler * self.generator([noise, mass_batch])
        return events.numpy()

    @tf.function
    def train_step(
        self,
        real_batch: tf.Tensor,
        mass_batch: tf.Tensor,
        d_optimizer: tf.keras.optimizers.Optimizer,
        g_optimizer: tf.keras.optimizers.Optimizer,
        batch_size: int,
    ):
        """Do a single train step"""

        # Sample random points in the latent space
        random_noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        # Decode them to fake images
        gen_batch = self.generator([random_noise, mass_batch])

        # Assemble labels discriminating real from fake images
        ones = tf.ones((batch_size, 1))
        zeros = tf.zeros((batch_size, 1))

        # Train the discriminator
        with tf.GradientTape() as tape:
            logit_real = self.discriminator(real_batch)
            logit_fake = self.discriminator(gen_batch)

            # Add gen and real loss
            d_loss = self.bc_loss(ones, logit_real)
            d_loss += self.bc_loss(zeros, logit_fake)

            # Add regularization
            disc_reg = self.disc_regularizer(
                logit_real, real_batch, logit_fake, gen_batch
            )

            d_loss += self.reg_weight / 2 * disc_reg

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # =========================================================#

        # Sample random points in the latent space
        random_noise = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Train the generator
        with tf.GradientTape() as tape:
            gen_out = self.generator(random_noise)
            logit_fake = self.discriminator(gen_out)
            g_loss = self.bc_loss(ones, logit_fake)

            # Get the masses
            res_gen = Resonances(
                self.topology, dscaler=self.scaler, name="GenResonances"
            )(gen_out)

            res_real = Resonances(
                self.topology, dscaler=self.scaler, name="RealResonances"
            )(real_batch)

            # Define the mmd floss function
            mmd_loss = self.mmd_loss(
                res_gen,
                res_real,
                self.mmd_kernel,
                self.mmd_kernel_widths,
                resonances=self.resonances,
            )

            g_loss += self.mmd_weight * mmd_loss

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return d_loss, g_loss

    def train(
        self,
        optimizer_args: dict,
        epochs=1000,
        iterations=1000,
        batch_size=1024,
        safe_weights: bool = False,
        safe_epochs: list = None,
    ):
        """ Train the Model """

        # Optimizer and scheduler
        lr_schedule_g = tf.keras.optimizers.schedules.InverseTimeDecay(
            optimizer_args["lr_g"],
            iterations,
            optimizer_args["decay_g"],
            staircase=True,
        )

        lr_schedule_d = tf.keras.optimizers.schedules.InverseTimeDecay(
            optimizer_args["lr_d"],
            iterations,
            optimizer_args["decay_d"],
            staircase=True,
        )

        # Instantiate one optimizer for the discriminator and another for the generator.
        d_optimizer = tf.keras.optimizers.Adam(
            lr_schedule_d,
            beta_1=optimizer_args["d_beta_1"],
            beta_2=optimizer_args["d_beta_2"],
        )

        g_optimizer = tf.keras.optimizers.Adam(
            lr_schedule_g,
            beta_1=optimizer_args["g_beta_1"],
            beta_2=optimizer_args["g_beta_2"],
        )

        mass_batch = tf.repeat(self.masses, batch_size, axis=0)

        for step in range(epochs):
            epoch = math.floor(step / iterations)

            # Online GAN: fetch new batch every step
            real_batch = tf.convert_to_tensor(
                self.sample_data(self.train_data, batch_size), np.float32
            )

            # Train the discriminator & generator on one batch of real images.
            d_loss, g_loss = self.train_step(
                real_batch, mass_batch, d_optimizer, g_optimizer, batch_size=batch_size
            )

            # Logging.
            if step % iterations == 0:
                # Print metrics
                print(
                    "Epoch #{}: Generative Loss: {}, Discriminator Loss: {}, Learning Rate: {}".format(
                        step, g_loss, d_loss, g_optimizer._decayed_lr(tf.float32)
                    )
                )
                if epoch in safe_epochs:
                    if safe_weights:
                        if not os.path.exists(
                            f"{self.save_path}/intermediate/epoch_{epoch}"
                        ):
                            os.makedirs(f"{self.save_path}/intermediate/epoch_{epoch}")

                    self.generator.save_weights(
                        f"{self.save_path}/intermediate/epoch_{epoch}/weights_c_model.h5"
                    )
                    self.discriminator.save_weights(
                        f"{self.save_path}/intermediate/epoch_{epoch}/weights_c_model.h5"
                    )

        return d_loss, g_loss
