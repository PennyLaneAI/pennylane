# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions related to the construction of the taylor form Hamiltonian."""
import numpy as np
import h5py
from scipy.special import factorial
import itertools
import subprocess
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def _cform_onemode_kinetic(freqs, n_states):
    """Calculates the kinetic energy part of the one body integrals to correct the integrals
    for localized modes

    Args:
        freqs(int): the harmonic frequencies
        n_states(int): maximum number of bosonic states per mode

    Returns:
        the kinetic energy part of the one body integrals
    """
    # action of kinetic energy operator for m=1,...,M modes with frequencies freqs[m]
    nmodes = len(freqs)
    all_mode_combos = []
    for aa in range(nmodes):
        all_mode_combos.append([aa])
    all_bos_combos = list(itertools.product(range(n_states), range(n_states)))

    rank = comm.Get_rank()
    size = comm.Get_size()
    boscombos_on_rank = np.array_split(all_bos_combos, size)[rank]
    chunksize = len(boscombos_on_rank)

    local_K_mat = np.zeros(len(all_mode_combos) * chunksize)

    nn = 0
    for [ii] in all_mode_combos:

        ii = int(ii)

        mm = 0
        for [ki, hi] in boscombos_on_rank:
            m_const = freqs[ii] / 4
            ind = nn * len(boscombos_on_rank) + mm
            if ki == hi:
                local_K_mat[ind] += m_const * (2 * ki + 1)
            if ki == hi + 2:
                local_K_mat[ind] -= m_const * np.sqrt((hi + 2) * (hi + 1))
            if ki == hi - 2:
                local_K_mat[ind] -= m_const * np.sqrt((hi - 1) * hi)
            mm += 1
        nn += 1
    return local_K_mat


def _cform_twomode_kinetic(pes, n_states):
    """Calculates the kinetic energy part of the two body integrals to correct the integrals
    for localized modes

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode

    Returns:
        the kinetic energy part of the two body integrals
    """
    nmodes = len(pes.freqs)

    all_mode_combos = []
    for aa in range(nmodes):
        for bb in range(nmodes):
            all_mode_combos.append([aa, bb])
    all_bos_combos = list(itertools.product(range(n_states), range(n_states), range(n_states), range(n_states)))

    rank = comm.Get_rank()
    size = comm.Get_size()
    boscombos_on_rank = np.array_split(all_bos_combos, size)[rank]
    chunksize = len(boscombos_on_rank)

    local_kin_cform_twobody = np.zeros(len(all_mode_combos) * chunksize)

    nn = 0
    for [ii, jj] in all_mode_combos:

        ii, jj = int(ii), int(jj)
        # skip through the things that are not needed
        if jj >= ii:
            nn += 1
            continue

        Usum = np.einsum("i,i->", pes.uloc[:, ii], pes.uloc[:, jj])
        m_const = Usum * np.sqrt(pes.freqs[ii] * pes.freqs[jj]) / 4

        mm = 0
        for [ki, kj, hi, hj] in boscombos_on_rank:
            ind = nn * len(boscombos_on_rank) + mm
            ki, kj, hi, hj = int(ki), int(kj), int(hi), int(hj)

            if ki == hi + 1 and kj == hj + 1:
                local_kin_cform_twobody[ind] -= m_const * np.sqrt((hi + 1) * (hj + 1))
            if ki == hi + 1 and kj == hj - 1:
                local_kin_cform_twobody[ind] += m_const * np.sqrt((hi + 1) * hj)
            if ki == hi - 1 and kj == hj + 1:
                local_kin_cform_twobody[ind] += m_const * np.sqrt(hi * (hj + 1))
            if ki == hi - 1 and kj == hj - 1:
                local_kin_cform_twobody[ind] -= m_const * np.sqrt(hi * hj)

            mm += 1
        nn += 1

    return local_kin_cform_twobody


def _cform_onemode(pes, n_states):
    """
    Calculates the one-body integrals from the given potential energy surface data for the
    Christiansen Hamiltonian.

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode

    Returns:
        the one-body integrals for the Christiansen Hamiltonian
    """

    nmodes = len(pes.freqs)
    all_mode_combos = []
    for aa in range(nmodes):
        all_mode_combos.append([aa])
    all_bos_combos = list(itertools.product(range(n_states), range(n_states)))

    rank = comm.Get_rank()
    size = comm.Get_size()
    boscombos_on_rank = np.array_split(all_bos_combos, size)[rank]
    chunksize = len(boscombos_on_rank)

    local_ham_cform_onebody = np.zeros(len(all_mode_combos) * chunksize)

    nn = 0
    for [ii] in all_mode_combos:

        ii = int(ii)

        mm = 0
        for [ki, hi] in boscombos_on_rank:

            sqrt = (2 ** (ki + hi) * factorial(ki) * factorial(hi) * np.pi) ** (-0.5)
            order_k = np.zeros(n_states)
            order_k[ki] = 1.0
            order_h = np.zeros(n_states)
            order_h[hi] = 1.0
            hermite_ki = np.polynomial.hermite.Hermite(order_k, [-1, 1])(pes.grid)
            hermite_hi = np.polynomial.hermite.Hermite(order_h, [-1, 1])(pes.grid)
            quadrature = np.sum(
                pes.gauss_weights * pes.pes_onemode[ii, :] * hermite_ki * hermite_hi
            )
            full_coeff = sqrt * quadrature  # * 219475 for converting into cm^-1
            ind = nn * len(boscombos_on_rank) + mm
            local_ham_cform_onebody[ind] += full_coeff
            mm += 1
        nn += 1
    return local_ham_cform_onebody + _cform_onemode_kinetic(pes.freqs, n_states)


def _cform_onemode_dipole(pes, n_states):
    """
    Calculates the one-body integrals from the given potential energy surface data for the
    Christiansen dipole operator

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode

    Returns:
        the one-body integrals for the Christiansen dipole operator
    """

    nmodes = pes.dipole_onemode.shape[0]
    all_mode_combos = []
    for aa in range(nmodes):
        all_mode_combos.append([aa])
    all_bos_combos = list(itertools.product(range(n_states), range(n_states)))

    rank = comm.Get_rank()
    size = comm.Get_size()
    boscombos_on_rank = np.array_split(all_bos_combos, size)[rank]
    chunksize = len(boscombos_on_rank)

    local_dipole_cform_onebody = np.zeros((len(all_mode_combos) * chunksize, 3))

    nn = 0
    for [ii] in all_mode_combos:

        ii = int(ii)

        mm = 0
        for [ki, hi] in boscombos_on_rank:

            ki, hi = int(ki), int(hi)
            sqrt = (2 ** (ki + hi) * factorial(ki) * factorial(hi) * np.pi) ** (-0.5)
            order_k = np.zeros(n_states)
            order_k[ki] = 1.0
            order_h = np.zeros(n_states)
            order_h[hi] = 1.0
            hermite_ki = np.polynomial.hermite.Hermite(order_k, [-1, 1])(pes.grid)
            hermite_hi = np.polynomial.hermite.Hermite(order_h, [-1, 1])(pes.grid)
            ind = nn * len(boscombos_on_rank) + mm
            for alpha in range(3):
                quadrature = np.sum(
                    pes.gauss_weights * pes.dipole_onemode[ii, :, alpha] * hermite_ki * hermite_hi
                )
                full_coeff = sqrt * quadrature  # * 219475 for converting into cm^-1
                local_dipole_cform_onebody[ind, alpha] += full_coeff
            mm += 1
        nn += 1

    return local_dipole_cform_onebody


def _cform_twomode(pes, n_states):
    """
    Calculates the two-body integrals from the given potential energy surface data for the
    Christiansen Hamiltonian

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode

    Returns:
        the two-body integrals for the Christiansen Hamiltonian
    """

    nmodes = pes.pes_twomode.shape[0]

    all_mode_combos = []
    for aa in range(nmodes):
        for bb in range(nmodes):
            all_mode_combos.append([aa, bb])
    all_bos_combos = list(itertools.product(range(n_states), range(n_states), range(n_states), range(n_states)))

    rank = comm.Get_rank()
    size = comm.Get_size()
    boscombos_on_rank = np.array_split(all_bos_combos, size)[rank]
    chunksize = len(boscombos_on_rank)

    local_ham_cform_twobody = np.zeros(len(all_mode_combos) * chunksize)

    nn = 0
    for [ii, jj] in all_mode_combos:

        ii, jj = int(ii), int(jj)
        # skip through the things that are not needed
        if jj >= ii:
            nn += 1
            continue

        mm = 0
        for [ki, kj, hi, hj] in boscombos_on_rank:

            ki, kj, hi, hj = int(ki), int(kj), int(hi), int(hj)

            sqrt = (
                2 ** (ki + kj + hi + hj)
                * factorial(ki)
                * factorial(kj)
                * factorial(hi)
                * factorial(hj)
            ) ** (-0.5) / np.pi
            order_ki = np.zeros(n_states)
            order_ki[ki] = 1.0
            order_kj = np.zeros(n_states)
            order_kj[kj] = 1.0
            order_hi = np.zeros(n_states)
            order_hi[hi] = 1.0
            order_hj = np.zeros(n_states)
            order_hj[hj] = 1.0
            hermite_ki = np.polynomial.hermite.Hermite(order_ki, [-1, 1])(pes.grid)
            hermite_kj = np.polynomial.hermite.Hermite(order_kj, [-1, 1])(pes.grid)
            hermite_hi = np.polynomial.hermite.Hermite(order_hi, [-1, 1])(pes.grid)
            hermite_hj = np.polynomial.hermite.Hermite(order_hj, [-1, 1])(pes.grid)
            quadrature = np.einsum(
                "a,b,a,b,ab,a,b->",
                pes.gauss_weights,
                pes.gauss_weights,
                hermite_ki,
                hermite_kj,
                pes.pes_twomode[ii, jj, :, :],
                hermite_hi,
                hermite_hj,
            )
            full_coeff = sqrt * quadrature  # * 219475 to get cm^-1
            ind = nn * len(boscombos_on_rank) + mm
            local_ham_cform_twobody[ind] += full_coeff
            mm += 1
        nn += 1
    return local_ham_cform_twobody


def _cform_twomode_dipole(pes, n_states):
    """
    Calculates the one-body integrals from the given potential energy surface data for the
    Christiansen dipole operator

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode

    Returns:
        the one-body integrals for the Christiansen dipole operator
    """

    nmodes = pes.dipole_twomode.shape[0]

    all_mode_combos = []
    for aa in range(nmodes):
        for bb in range(nmodes):
            all_mode_combos.append([aa, bb])
    all_bos_combos = list(itertools.product(range(n_states), range(n_states), range(n_states), range(n_states)))

    rank = comm.Get_rank()
    size = comm.Get_size()
    boscombos_on_rank = np.array_split(all_bos_combos, size)[rank]
    chunksize = len(boscombos_on_rank)

    local_dipole_cform_twobody = np.zeros((len(all_mode_combos) * chunksize, 3))

    nn = 0
    for [ii, jj] in all_mode_combos:

        ii, jj = int(ii), int(jj)
        # skip through the things that are not needed
        if jj >= ii:
            nn += 1
            continue

        mm = 0
        for [ki, kj, hi, hj] in boscombos_on_rank:

            ki, kj, hi, hj = int(ki), int(kj), int(hi), int(hj)

            sqrt = (
                2 ** (ki + kj + hi + hj)
                * factorial(ki)
                * factorial(kj)
                * factorial(hi)
                * factorial(hj)
            ) ** (-0.5) / np.pi
            order_ki = np.zeros(n_states)
            order_ki[ki] = 1.0
            order_kj = np.zeros(n_states)
            order_kj[kj] = 1.0
            order_hi = np.zeros(n_states)
            order_hi[hi] = 1.0
            order_hj = np.zeros(n_states)
            order_hj[hj] = 1.0
            hermite_ki = np.polynomial.hermite.Hermite(order_ki, [-1, 1])(pes.grid)
            hermite_kj = np.polynomial.hermite.Hermite(order_kj, [-1, 1])(pes.grid)
            hermite_hi = np.polynomial.hermite.Hermite(order_hi, [-1, 1])(pes.grid)
            hermite_hj = np.polynomial.hermite.Hermite(order_hj, [-1, 1])(pes.grid)
            ind = nn * len(boscombos_on_rank) + mm
            for alpha in range(3):
                quadrature = np.einsum(
                    "a,b,a,b,ab,a,b->",
                    pes.gauss_weights,
                    pes.gauss_weights,
                    hermite_ki,
                    hermite_kj,
                    pes.dipole_twomode[ii, jj, :, :, alpha],
                    hermite_hi,
                    hermite_hj,
                )
                full_coeff = sqrt * quadrature  # * 219475 to get cm^-1
                local_dipole_cform_twobody[ind, alpha] += full_coeff
            mm += 1
        nn += 1

    return local_dipole_cform_twobody


def _cform_threemode(pes, n_states):
    """
    Calculates the three-body integrals from the given potential energy surface data for the
    Christiansen Hamiltonian

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode

    Returns:
        the three-body integrals for the Christiansen Hamiltonian
    """
    nmodes = pes.pes_threemode.shape[0]

    all_mode_combos = []
    for aa in range(nmodes):
        for bb in range(nmodes):
            for cc in range(nmodes):
                all_mode_combos.append([aa, bb, cc])

    all_bos_combos = list(
        itertools.product(
            range(n_states), range(n_states), range(n_states), range(n_states), range(n_states), range(n_states)
        )
    )
    rank = comm.Get_rank()
    size = comm.Get_size()
    boscombos_on_rank = np.array_split(all_bos_combos, size)[rank]
    chunksize = len(boscombos_on_rank)

    local_ham_cform_threebody = np.zeros(len(all_mode_combos) * chunksize)

    nn = 0
    for [ii1, ii2, ii3] in all_mode_combos:

        ii1, ii2, ii3 = int(ii1), int(ii2), int(ii3)
        # skip the objects that are not needed
        if ii2 >= ii1 or ii3 >= ii2:
            nn += 1
            continue

        mm = 0
        for [k1, k2, k3, h1, h2, h3] in boscombos_on_rank:

            k1, k2, k3, h1, h2, h3 = int(k1), int(k2), int(k3), int(h1), int(h2), int(h3)
            sqrt = (
                2 ** (k1 + k2 + k3 + h1 + h2 + h3)
                * factorial(k1)
                * factorial(k2)
                * factorial(k3)
                * factorial(h1)
                * factorial(h2)
                * factorial(h3)
            ) ** (-0.5) / (np.pi**1.5)
            order_k1 = np.zeros(n_states)
            order_k1[k1] = 1.0
            order_k2 = np.zeros(n_states)
            order_k2[k2] = 1.0
            order_k3 = np.zeros(n_states)
            order_k3[k3] = 1.0
            order_h1 = np.zeros(n_states)
            order_h1[h1] = 1.0
            order_h2 = np.zeros(n_states)
            order_h2[h2] = 1.0
            order_h3 = np.zeros(n_states)
            order_h3[h3] = 1.0
            hermite_k1 = np.polynomial.hermite.Hermite(order_k1, [-1, 1])(pes.grid)
            hermite_k2 = np.polynomial.hermite.Hermite(order_k2, [-1, 1])(pes.grid)
            hermite_k3 = np.polynomial.hermite.Hermite(order_k3, [-1, 1])(pes.grid)
            hermite_h1 = np.polynomial.hermite.Hermite(order_h1, [-1, 1])(pes.grid)
            hermite_h2 = np.polynomial.hermite.Hermite(order_h2, [-1, 1])(pes.grid)
            hermite_h3 = np.polynomial.hermite.Hermite(order_h3, [-1, 1])(pes.grid)

            quadrature = np.einsum(
                "a,b,c,a,b,c,abc,a,b,c->",
                pes.gauss_weights,
                pes.gauss_weights,
                pes.gauss_weights,
                hermite_k1,
                hermite_k2,
                hermite_k3,
                pes.pes_threemode[ii1, ii2, ii3, :, :, :],
                hermite_h1,
                hermite_h2,
                hermite_h3,
            )
            full_coeff = sqrt * quadrature  # * 219475 to get cm^-1
            ind = nn * len(boscombos_on_rank) + mm
            local_ham_cform_threebody[ind] = full_coeff
            mm += 1
        nn += 1
    return local_ham_cform_threebody


def _cform_threemode_dipole(pes, n_states):
    """
    Calculates the one-body integrals from the given potential energy surface data for the
    Christiansen dipole operator

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode

    Returns:
        the one-body integrals for the Christiansen dipole operator
    """
    nmodes = pes.dipole_threemode.shape[0]

    all_mode_combos = []
    for aa in range(nmodes):
        for bb in range(nmodes):
            for cc in range(nmodes):
                all_mode_combos.append([aa, bb, cc])
    all_bos_combos = list(
        itertools.product(
            range(n_states), range(n_states), range(n_states), range(n_states), range(n_states), range(n_states)
        )
    )
    rank = comm.Get_rank()
    size = comm.Get_size()
    boscombos_on_rank = np.array_split(all_bos_combos, size)[rank]
    chunksize = len(boscombos_on_rank)

    local_dipole_cform_threebody = np.zeros((len(all_mode_combos) * chunksize, 3))

    nn = 0
    for [ii1, ii2, ii3] in all_mode_combos:

        ii1, ii2, ii3 = int(ii1), int(ii2), int(ii3)
        # skip the objects that are not needed
        if ii2 >= ii1 or ii3 >= ii2:
            nn += 1
            continue

        mm = 0
        for [k1, k2, k3, h1, h2, h3] in boscombos_on_rank:

            k1, k2, k3, h1, h2, h3 = int(k1), int(k2), int(k3), int(h1), int(h2), int(h3)
            sqrt = (
                2 ** (k1 + k2 + k3 + h1 + h2 + h3)
                * factorial(k1)
                * factorial(k2)
                * factorial(k3)
                * factorial(h1)
                * factorial(h2)
                * factorial(h3)
            ) ** (-0.5) / (np.pi**1.5)
            order_k1 = np.zeros(n_states)
            order_k1[k1] = 1.0
            order_k2 = np.zeros(n_states)
            order_k2[k2] = 1.0
            order_k3 = np.zeros(n_states)
            order_k3[k3] = 1.0
            order_h1 = np.zeros(n_states)
            order_h1[h1] = 1.0
            order_h2 = np.zeros(n_states)
            order_h2[h2] = 1.0
            order_h3 = np.zeros(n_states)
            order_h3[h3] = 1.0
            hermite_k1 = np.polynomial.hermite.Hermite(order_k1, [-1, 1])(pes.grid)
            hermite_k2 = np.polynomial.hermite.Hermite(order_k2, [-1, 1])(pes.grid)
            hermite_k3 = np.polynomial.hermite.Hermite(order_k3, [-1, 1])(pes.grid)
            hermite_h1 = np.polynomial.hermite.Hermite(order_h1, [-1, 1])(pes.grid)
            hermite_h2 = np.polynomial.hermite.Hermite(order_h2, [-1, 1])(pes.grid)
            hermite_h3 = np.polynomial.hermite.Hermite(order_h3, [-1, 1])(pes.grid)
            ind = nn * len(boscombos_on_rank) + mm
            for alpha in range(3):
                quadrature = np.einsum(
                    "a,b,c,a,b,c,abc,a,b,c->",
                    pes.gauss_weights,
                    pes.gauss_weights,
                    pes.gauss_weights,
                    hermite_k1,
                    hermite_k2,
                    hermite_k3,
                    pes.dipole_threemode[ii1, ii2, ii3, :, :, :, alpha],
                    hermite_h1,
                    hermite_h2,
                    hermite_h3,
                )
                full_coeff = sqrt * quadrature  # * 219475 to get cm^-1
                local_dipole_cform_threebody[ind, alpha] = full_coeff
            mm += 1
        nn += 1

    return local_dipole_cform_threebody


def _load_cform_onemode(num_proc, nmodes, quad_order):
    """
    Loader to collect and combine pes_onemode from multiple processors.

    Args:
        num_proc (int): number of processors
        nmodes (int): number of normal modes
        quad_order (int): order for Gauss-Hermite quadratures

    Returns:
        TensorLike[float]: one-body integrals for Christiansen Hamiltonian
    """
    final_shape = (nmodes, quad_order, quad_order)
    nmode_combos = int(nmodes)

    ham_cform_onebody = np.zeros(np.prod(final_shape))
    r0 = 0
    r1 = 0
    for mode_combo in range(nmode_combos):
        local_chunk = np.zeros(quad_order**2)

        l0 = 0
        l1 = 0
        for rank in range(num_proc):
            f = h5py.File("cform_H1data" + f"_{rank}" + ".hdf5", "r+")
            local_ham_cform_onebody = f["H1"][()]
            chunk = np.array_split(local_ham_cform_onebody, nmode_combos)[mode_combo]  #
            l1 += len(chunk)
            local_chunk[l0:l1] = chunk
            l0 += len(chunk)

        r1 += len(local_chunk)
        ham_cform_onebody[r0:r1] = local_chunk
        r0 += len(local_chunk)

    ham_cform_onebody = ham_cform_onebody.reshape(final_shape)

    return ham_cform_onebody


def _load_cform_twomode(num_proc, nmodes, quad_order):
    """
    Loader to collect and combine pes_twomode from multiple processors.

    Args:
        num_proc (int): number of processors
        nmodes (int): number of normal modes
        quad_order (int): order for Gauss-Hermite quadratures

    Returns:
        TensorLike[float]: two-body integrals for Christiansen Hamiltonian
    """
    final_shape = (nmodes, nmodes, quad_order, quad_order, quad_order, quad_order)
    nmode_combos = nmodes**2

    ham_cform_twobody = np.zeros(np.prod(final_shape))
    r0 = 0
    r1 = 0
    for mode_combo in range(nmode_combos):
        local_chunk = np.zeros(quad_order**4)

        l0 = 0
        l1 = 0
        for rank in range(num_proc):
            f = h5py.File("cform_H2data" + f"_{rank}" + ".hdf5", "r+")
            local_ham_cform_twobody = f["H2"][()]
            chunk = np.array_split(local_ham_cform_twobody, nmode_combos)[mode_combo]  #
            l1 += len(chunk)
            local_chunk[l0:l1] = chunk
            l0 += len(chunk)

        r1 += len(local_chunk)
        ham_cform_twobody[r0:r1] = local_chunk
        r0 += len(local_chunk)

    ham_cform_twobody = ham_cform_twobody.reshape(final_shape)

    return ham_cform_twobody


def _load_cform_threemode(num_proc, nmodes, quad_order):
    """
    Loader to collect and combine pes_threemode from multiple processors.

    Args:
        num_proc (int): number of processors
        nmodes (int): number of normal modes
        quad_order (int): order for Gauss-Hermite quadratures

    Returns:
        TensorLike[float]: three-body integrals for Christiansen Hamiltonian
    """
    final_shape = (
        nmodes,
        nmodes,
        nmodes,
        quad_order,
        quad_order,
        quad_order,
        quad_order,
        quad_order,
        quad_order,
    )
    nmode_combos = nmodes**3

    ham_cform_threebody = np.zeros(np.prod(final_shape))
    r0 = 0
    r1 = 0
    for mode_combo in range(nmode_combos):
        local_chunk = np.zeros(quad_order**6)

        l0 = 0
        l1 = 0
        for rank in range(num_proc):
            f = h5py.File("cform_H3data" + f"_{rank}" + ".hdf5", "r+")
            local_ham_cform_threebody = f["H3"][()]  # 64 * 4096
            chunk = np.array_split(local_ham_cform_threebody, nmode_combos)[mode_combo]  #
            l1 += len(chunk)
            local_chunk[l0:l1] = chunk
            l0 += len(chunk)

        r1 += len(local_chunk)
        ham_cform_threebody[r0:r1] = local_chunk
        r0 += len(local_chunk)

    ham_cform_threebody = ham_cform_threebody.reshape(final_shape)

    return ham_cform_threebody


def _load_cform_onemode_dipole(num_proc, nmodes, quad_order):
    """
    Loader to collect and combine dipole_onemode from multiple processors.

    Args:
        num_proc (int): number of processors
        nmodes (int): number of normal modes
        quad_order (int): order for Gauss-Hermite quadratures

    Returns:
        TensorLike[float]: one-body integrals for Christiansen dipole operator
    """
    final_shape = (nmodes, quad_order, quad_order)
    nmode_combos = int(nmodes)

    dipole_cform_onebody = np.zeros((np.prod(final_shape), 3))
    r0 = 0
    r1 = 0
    for mode_combo in range(nmode_combos):
        local_chunk = np.zeros((quad_order**2, 3))

        l0 = 0
        l1 = 0
        for rank in range(num_proc):
            f = h5py.File("cform_D1data" + f"_{rank}" + ".hdf5", "r+")
            local_dipole_cform_onebody = f["D1"][()]
            chunk = np.array_split(local_dipole_cform_onebody, nmode_combos, axis=0)[mode_combo]  #
            l1 += chunk.shape[0]
            local_chunk[l0:l1, :] = chunk
            l0 += chunk.shape[0]

        r1 += local_chunk.shape[0]
        dipole_cform_onebody[r0:r1, :] = local_chunk
        r0 += local_chunk.shape[0]

    dipole_cform_onebody = dipole_cform_onebody.reshape(final_shape + (3,))

    return dipole_cform_onebody.transpose(3, 0, 1, 2)


def _load_cform_twomode_dipole(num_proc, nmodes, quad_order):
    """
    Loader to collect and combine dipole_twomode from multiple processors.

    Args:
        num_proc (int): number of processors
        nmodes (int): number of normal modes
        quad_order (int): order for Gauss-Hermite quadratures

    Returns:
        TensorLike[float]: two-body integrals for Christiansen dipole operator
    """
    final_shape = (nmodes, nmodes, quad_order, quad_order, quad_order, quad_order)
    nmode_combos = int(nmodes**2)

    dipole_cform_twobody = np.zeros((np.prod(final_shape), 3))
    r0 = 0
    r1 = 0
    for mode_combo in range(nmode_combos):
        local_chunk = np.zeros((quad_order**4, 3))

        l0 = 0
        l1 = 0
        for rank in range(num_proc):
            f = h5py.File("cform_D2data" + f"_{rank}" + ".hdf5", "r+")
            local_dipole_cform_twobody = f["D2"][()]
            chunk = np.array_split(local_dipole_cform_twobody, nmode_combos, axis=0)[mode_combo]  #
            l1 += chunk.shape[0]
            local_chunk[l0:l1, :] = chunk
            l0 += chunk.shape[0]

        r1 += local_chunk.shape[0]
        dipole_cform_twobody[r0:r1, :] = local_chunk
        r0 += local_chunk.shape[0]

    dipole_cform_twobody = dipole_cform_twobody.reshape(final_shape + (3,))

    return dipole_cform_twobody.transpose(6, 0, 1, 2, 3, 4, 5)


def _load_cform_threemode_dipole(num_proc, nmodes, quad_order):
    """
    Loader to collect and combine dipole_threemode from multiple processors.

    Args:
        num_proc (int): number of processors
        nmodes (int): number of normal modes
        quad_order (int): order for Gauss-Hermite quadratures

    Returns:
        TensorLike[float]: three-body integrals for Christiansen dipole operator
    """
    final_shape = (
        nmodes,
        nmodes,
        nmodes,
        quad_order,
        quad_order,
        quad_order,
        quad_order,
        quad_order,
        quad_order,
    )
    nmode_combos = int(nmodes**3)

    dipole_cform_threebody = np.zeros((np.prod(final_shape), 3))
    r0 = 0
    r1 = 0
    for mode_combo in range(nmode_combos):
        local_chunk = np.zeros((quad_order**6, 3))

        l0 = 0
        l1 = 0
        for rank in range(num_proc):
            f = h5py.File("cform_D3data" + f"_{rank}" + ".hdf5", "r+")
            local_dipole_cform_threebody = f["D3"][()]
            chunk = np.array_split(local_dipole_cform_threebody, nmode_combos, axis=0)[
                mode_combo
            ]  #
            l1 += chunk.shape[0]
            local_chunk[l0:l1, :] = chunk
            l0 += chunk.shape[0]

        r1 += local_chunk.shape[0]
        dipole_cform_threebody[r0:r1, :] = local_chunk
        r0 += local_chunk.shape[0]

    dipole_cform_threebody = dipole_cform_threebody.reshape(final_shape + (3,))

    return dipole_cform_threebody.transpose(9, 0, 1, 2, 3, 4, 5, 6, 7, 8)


def christiansen_integrals(pes, n_states=16, cubic=False):
    r"""Generates the vibrational Hamiltonian integrals in Christiansen form.

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode
        cubic(bool): Flag to include three-mode couplings. Default is ``False``.

    Returns:
        TensorLike[float]: the integrals for the Christiansen Hamiltonian
    """

    local_ham_cform_onebody = _cform_onemode(pes, n_states)
    comm.Barrier()

    f = h5py.File("cform_H1data" + f"_{rank}" + ".hdf5", "w")
    f.create_dataset("H1", data=local_ham_cform_onebody)
    f.close()
    comm.Barrier()

    ham_cform_onebody = None
    if rank == 0:
        ham_cform_onebody = _load_cform_onemode(size, len(pes.freqs), n_states)
        process = subprocess.Popen("rm " + "cform_H1data*", stdout=subprocess.PIPE, shell=True)
        output, error = process.communicate()

    comm.Barrier()
    ham_cform_onebody = comm.bcast(ham_cform_onebody, root=0)

    local_ham_cform_twobody = _cform_twomode(pes, n_states)
    if pes.localized:
        local_ham_cform_twobody += _cform_twomode_kinetic(pes, n_states)
    comm.Barrier()

    f = h5py.File("cform_H2data" + f"_{rank}" + ".hdf5", "w")
    f.create_dataset("H2", data=local_ham_cform_twobody)
    f.close()
    comm.Barrier()

    ham_cform_twobody = None
    if rank == 0:
        ham_cform_twobody = _load_cform_twomode(size, len(pes.freqs), n_states)
        process = subprocess.Popen("rm " + "cform_H2data*", stdout=subprocess.PIPE, shell=True)
        output, error = process.communicate()

    comm.Barrier()
    ham_cform_twobody = comm.bcast(ham_cform_twobody, root=0)

    if cubic:
        local_ham_cform_threebody = _cform_threemode(pes, n_states)

        f = h5py.File("cform_H3data" + f"_{rank}" + ".hdf5", "w")
        f.create_dataset("H3", data=local_ham_cform_threebody)
        f.close()
        comm.Barrier()

        ham_cform_threebody = None
        if rank == 0:
            ham_cform_threebody = _load_cform_threemode(size, len(pes.freqs), n_states)
            process = subprocess.Popen("rm " + "cform_H3data*", stdout=subprocess.PIPE, shell=True)
            output, error = process.communicate()

        comm.Barrier()
        ham_cform_threebody = comm.bcast(ham_cform_threebody, root=0)

        H_arr = [ham_cform_onebody, ham_cform_twobody, ham_cform_threebody]
    else:
        H_arr = [ham_cform_onebody, ham_cform_twobody]

    return H_arr


def christiansen_integrals_dipole(pes, n_states=16, cubic=False):
    r"""Generates the vibrational dipole integrals in Christiansen form.

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode
        cubic(bool): Flag to include three-mode couplings. Default is ``False``.

    Returns:
        TensorLike[float]: the integrals for the Christiansen dipole operator
    """

    local_dipole_cform_onebody = _cform_onemode_dipole(pes, n_states)
    comm.Barrier()

    f = h5py.File("cform_D1data" + f"_{rank}" + ".hdf5", "w")
    f.create_dataset("D1", data=local_dipole_cform_onebody)
    f.close()
    comm.Barrier()

    dipole_cform_onebody = None
    if rank == 0:
        dipole_cform_onebody = _load_cform_onemode_dipole(size, len(pes.freqs), n_states)
        process = subprocess.Popen("rm " + "cform_D1data*", stdout=subprocess.PIPE, shell=True)
        output, error = process.communicate()

    comm.Barrier()
    dipole_cform_onebody = comm.bcast(dipole_cform_onebody, root=0)

    if pes.localized is True or pes.dipole_level > 1:
        local_dipole_cform_twobody = _cform_twomode_dipole(pes, n_states)
        comm.Barrier()

        f = h5py.File("cform_D2data" + f"_{rank}" + ".hdf5", "w")
        f.create_dataset("D2", data=local_dipole_cform_twobody)
        f.close()
        comm.Barrier()

        dipole_cform_twobody = None
        if rank == 0:
            dipole_cform_twobody = _load_cform_twomode_dipole(size, len(pes.freqs), n_states)
            process = subprocess.Popen("rm " + "cform_D2data*", stdout=subprocess.PIPE, shell=True)
            output, error = process.communicate()
        comm.Barrier()
        dipole_cform_twobody = comm.bcast(dipole_cform_twobody, root=0)

    if pes.localized is True or pes.dipole_level > 2:
        local_dipole_cform_threebody = _cform_threemode_dipole(pes, n_states)
        comm.Barrier()

        f = h5py.File("cform_D3data" + f"_{rank}" + ".hdf5", "w")
        f.create_dataset("D3", data=local_dipole_cform_threebody)
        f.close()

        dipole_cform_threebody = None
        if rank == 0:
            dipole_cform_threebody = _load_cform_threemode_dipole(size, len(pes.freqs), n_states)
            process = subprocess.Popen("rm " + "cform_D3data*", stdout=subprocess.PIPE, shell=True)
            output, error = process.communicate()
        comm.Barrier()

        dipole_cform_threebody = comm.bcast(dipole_cform_threebody, root=0)

        D_arr = [dipole_cform_onebody, dipole_cform_twobody, dipole_cform_threebody]
    else:
        D_arr = [dipole_cform_onebody, dipole_cform_twobody]

    return D_arr
