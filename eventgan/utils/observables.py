"""Observables for plotting."""

import numpy as np

# pylint: disable=C0103
class Observables:
    """
    Contains different functions to calculate 1-dim observables.
    """

    def __init__(self, ):
        self.epsilon = 1e-8

    @staticmethod
    def identity(x: np.array):
        """Simply gives the input back"""
        return x

    @staticmethod
    def _momentum(x: np.array, entry: int, particle_id: list):
        """Parent function giving the ith
        momentum entry of n particles.

        # Arguments
            input: Array with input data
            entry: the momentum entry which should be returned
            particle_id: Integers, particle IDs of n particles
                If there is only one momentum that has to be considered
                the shape is:
                `particle_id = [particle_1]`.

                If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
                `particle_id = [particle_1, particle_2,..]`.
        """
        Ps = 0
        for particle in particle_id:
            Ps += x[:, entry + particle * 4]

        return Ps

    def energy(self, x: np.array, particle_id: list):
        """Returns the energy"""
        return self._momentum(x, 0, particle_id)

    def x_momentum(self, x: np.array, particle_id: list):
        """Returns p_x"""
        return self._momentum(x, 1, particle_id)

    def y_momentum(self, x: np.array, particle_id: list):
        """Returns p_y"""
        return self._momentum(x, 2, particle_id)

    def z_momentum(self, x: np.array, particle_id: list):
        """Returns p_z"""
        return self._momentum(x, 3, particle_id)

    def x_momentum_over_abs(self, x: np.array, particle_id: list, n_particles: int):
        """Returns p_x/(Sum |p_x|)"""
        momentum = self.x_momentum(x, particle_id)

        abs_sum = 0
        for i in range(n_particles):
            abs_sum += np.abs(self.x_momentum(x, [i]))

        return momentum / abs_sum

    def y_momentum_over_abs(self, x: np.array, particle_id: list, n_particles: int):
        """Returns p_y/(Sum |p_y|)"""
        momentum = self.y_momentum(x, particle_id)

        abs_sum = 0
        for i in range(n_particles):
            abs_sum += np.abs(self.y_momentum(x, [i]))

        return momentum / abs_sum

    @staticmethod
    def invariant_mass_square(x: np.array, particle_id: list):
        """Squared Invariant Mass.
        This function gives the squared invariant mass of n particles.

        # Arguments
            input: Array with input data
                that will be used to calculate the invariant mass from.
            particle_id: Integers, particle IDs of n particles.
                If there is only one momentum that has to be considered
                the shape is:
                `particle_id = [particle_1]`.

                If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
                `particle_id = [particle_1, particle_2,..]`.
        """
        Es = 0
        PXs = 0
        PYs = 0
        PZs = 0
        for particle in particle_id:
            Es += x[:, 0 + particle * 4]
            PXs += x[:, 1 + particle * 4]
            PYs += x[:, 2 + particle * 4]
            PZs += x[:, 3 + particle * 4]

        m2 = np.square(Es) - np.square(PXs) - np.square(PYs) - np.square(PZs)
        return m2

    def invariant_mass(self, x: np.array, particle_id: list):
        """Invariant Mass.
        This function gives the invariant mass of n particles.

        # Arguments
            input: Array with input data
                that will be used to calculate the invariant mass from.
            particle_id: Integers, particle IDs of n particles.
                If there is only one momentum that has to be considered
                the shape is:
                `particle_id = [particle_1]`.

                If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
                `particle_id = [particle_1, particle_2,..]`.
        """
        m = np.sqrt(np.clip(self.invariant_mass_square(x, particle_id), self.epsilon, None))
        return m

    @staticmethod
    def transverse_momentum(x: np.array, particle_id: list):
        """This function gives the transverse momentum of n particles.

        # Arguments
            particle_id: Integers, particle IDs of n particles
                If there is only one momentum that has to be considered
                the shape is:
                `particle_id = [particle_1]`.

                If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
                `particle_id = [particle_1, particle_2,..]`.
        """
        PXs = 0
        PYs = 0
        for particle in particle_id:
            PXs += x[:, 1 + particle * 4]
            PYs += x[:, 2 + particle * 4]

        PXs2 = np.square(PXs)
        PYs2 = np.square(PYs)

        pTs = PXs2 + PYs2

        pT= np.sqrt(pTs)
        return pT

    def transverse_momentum_cut(self, x: np.array, particle_id: list, cut: float=20.):
        """This function gives a smoothed out observable of
        the transverse momentum which is affected by a cut.

        # Arguments
            particle_id: Integers, particle IDs of n particles
                If there is only one momentum that has to be considered
                the shape is:
                `particle_id = [particle_1]`.

                If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
                `particle_id = [particle_1, particle_2,..]`.
        """
        pT = self.transverse_momentum(x, particle_id)
        trans = np.log(pT - cut)
        return trans

    def transverse_momentum_cut2(self, x: np.array, particle_id: list, cut: float=20.):
        """This function gives smoothed out observable of
        the transverse momentum which is affected by a cut.

        # Arguments
            particle_id: Integers, particle IDs of n particles
                If there is only one momentum that has to be considered
                the shape is:
                `particle_id = [particle_1]`.

                If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
                `particle_id = [particle_1, particle_2,..]`.
        """
        pT = self.transverse_momentum(x, particle_id)
        trans1 = np.exp(cut - pT)
        trans = cut - pT - np.log(1 - trans1)
        return trans

    def rapidity(self, x: np.array, particle_id: list):
        """Rapidity.
        This function gives the rapidity of n particles.

        # Arguments
            particle_id: Integers, particle IDs of n particles
                If there is only one momentum that has to be considered
                the shape is:
                `particle_id = [particle_1]`.

                If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
                `particle_id = [particle_1, particle_2,..]`.
        """
        Es = 0
        PZs = 0
        for particle in particle_id:
            Es += x[:, 0 + particle * 4]
            PZs += x[:, 3 + particle * 4]

        y = 0.5 * (
            np.log(np.clip(Es + PZs, self.epsilon, None))
            - np.log(np.clip(Es - PZs, self.epsilon, None))
        )

        return y

    @staticmethod
    def phi(x: np.array, particle_id: list):
        """Azimuthal angle phi.
        This function gives the azimuthal angle oftthe particle.

        # Arguments
            particle_id: Integers, particle IDs of two particles given in
                the shape:
                `particle_id = [particle_1, particle_2]`.
        """
        PX1s = 0
        PY1s = 0
        for particle in particle_id:
            PX1s += x[:, 1 + particle * 4]
            PY1s += x[:, 2 + particle * 4]

        phi = np.arctan2(PY1s, PX1s)

        return phi

    def pseudo_rapidity(self, x: np.array, particle_id: list):
        """Psudo Rapidity.
        This function gives the pseudo rapidity of n particles.

        # Arguments
            particle_id: Integers, particle IDs of n particles
                If there is only one momentum that has to be considered
                the shape is:
                `particle_id = [particle_1]`.

                If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
                `particle_id = [particle_1, particle_2,..]`.
        """
        Es = 0
        PXs = 0
        PYs = 0
        PZs = 0
        for particle in particle_id:
            Es += x[:, 0 + particle * 4]
            PXs += x[:, 1 + particle * 4]
            PYs += x[:, 2 + particle * 4]
            PZs += x[:, 3 + particle * 4]

        Ps = np.sqrt(np.square(PXs) + np.square(PYs) + np.square(PZs))
        eta = 0.5 * (
            np.log(np.clip(Ps + PZs, self.epsilon, None))
            - np.log(np.clip(Ps - PZs, self.epsilon, None))
        )

        return eta

    def pseudo_rapidity_cut(self, x: np.array, particle_id: list, cut: float=6.):
        """
        This function gives a smoothed out observable of
        the pseudo rapidity which is affected by a cut.

        # Arguments
            particle_id: Integers, particle IDs of n particles
                If there is only one momentum that has to be considered
                the shape is:
                `particle_id = [particle_1]`.

                If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
                `particle_id = [particle_1, particle_2,..]`.
        """
        eta = self.pseudo_rapidity(x, particle_id)

        trans = (cut - eta) / (cut + eta)

        return np.array(trans)

    @staticmethod
    def delta_phi(x: np.array, particle_id: list):
        """Delta Phi.
        This function gives the difference in the azimuthal angle of 2 particles.

        # Arguments
            particle_id: Integers, particle IDs of two particles given in
                the shape:
                `particle_id = [particle_1, particle_2]`.
        """

        PX1s = x[:, 1 + particle_id[0] * 4]
        PY1s = x[:, 2 + particle_id[0] * 4]

        PX2s = x[:, 1 + particle_id[1] * 4]
        PY2s = x[:, 2 + particle_id[1] * 4]

        phi1s = np.arctan2(PY1s, PX1s)
        phi2s = np.arctan2(PY2s, PX2s)

        dphi = np.fabs(phi1s - phi2s)
        dphimin = np.where(dphi > np.pi, 2.0 * np.pi - dphi, dphi)

        return dphimin

    def delta_rapidity(self, x: np.array, particle_id: list):
        """Delta Rapidity.
        This function gives the rapidity difference of 2 particles.

        # Arguments
            particle_id: Integers, particle IDs of two particles given in
                the shape:
                `particle_id = [particle_1, particle_2]`.
        """

        E1s = x[:, 0 + particle_id[0] * 4]
        PZ1s = x[:, 3 + particle_id[0] * 4]

        E2s = x[:, 0 + particle_id[1] * 4]
        PZ2s = x[:, 3 + particle_id[1] * 4]

        y1 = 0.5 * (np.log(np.clip(E1s + PZ1s, self.epsilon, None)) - np.log(np.clip(E1s - PZ1s, self.epsilon, None)))
        y2 = 0.5 * (np.log(np.clip(E2s + PZ2s, self.epsilon, None)) - np.log(np.clip(E2s - PZ2s, self.epsilon, None)))
        dy = np.abs(y1 - y2)

        return dy

    def delta_R(self, x: np.array, particle_id: list):
        """Delta R.
        This function gives the Delta R of 2 particles.

        # Arguments
            particle_id: Integers, particle IDs of two particles given in
                the shape:
                `particle_id = [particle_1, particle_2]`.
        """

        dy = self.delta_rapidity(x, particle_id=particle_id)
        dphi = self.delta_phi(x, particle_id=particle_id)

        dR = np.sqrt(dphi ** 2 + dy ** 2)

        return dR

    def delta_R_cut(self, x: np.array, particle_id: list, cut: float=0.4):
        """
        This function gives a smoothed out observable of
        delta R which is affected by a cut.

        # Arguments
            particle_id: Integers, particle IDs of two particles given in
                the shape:
                `particle_id = [particle_1, particle_2]`.
        """
        dR = self.delta_R(x, particle_id)

        T = np.log(dR - cut)

        return T
