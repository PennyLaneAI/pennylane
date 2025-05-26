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
import itertools
from pathlib import Path

import h5py
import numpy as np
from scipy.special import factorial

from pennylane import concurrency

# pylint: disable = redefined-outer-name,


def _cform_onemode_kinetic(freqs, n_states, num_workers=1, backend="serial"):
    """Calculates the kinetic energy part of the one body integrals to correct the integrals
    for localized modes

    Args:
        freqs(int): the harmonic frequencies
        n_states(int): maximum number of bosonic states per mode
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".
    Returns:
        TensorLike[float]: the kinetic energy part of the one body integrals
    """
    # action of kinetic energy operator for m=1,...,M modes with frequencies freqs[m]
    nmodes = len(freqs)
    all_mode_combos = [[aa] for aa in range(nmodes)]
    all_bos_combos = list(itertools.product(range(n_states), range(n_states)))

    executor_class = concurrency.backends.get_executor(backend)

    with executor_class(max_workers=num_workers) as executor:
        boscombos_on_ranks = np.array_split(all_bos_combos, num_workers)
        args = [
            (rank, boscombos_on_rank, freqs, all_mode_combos)
            for rank, boscombos_on_rank in enumerate(boscombos_on_ranks)
        ]
        executor.starmap(_local_onemode_kinetic, args)

    return _load_cform_onemode_kinetic(num_workers, nmodes, n_states)


def _local_onemode_kinetic(rank, boscombos_on_rank, freqs, all_mode_combos):
    """Worker function to calculate the kinetic energy part of the one body integrals to correct the integrals
    for localized modes. The result are written to a hdf5 file.

    Args:
        rank(int) : the rank of the process
        boscombos_on_rank (int) : list of the combination of bosonic states handled by this process
        freqs(int): the harmonic frequencies
        all_mode_combos [int] : list of the combination of nmodes (the length of the list of harmonic frequencies)
    """

    local_K_mat = np.zeros(len(all_mode_combos) * len(boscombos_on_rank))
    for nn, [ii] in enumerate(all_mode_combos):
        ii = int(ii)
        m_const = freqs[ii] / 4
        for mm, [ki, hi] in enumerate(boscombos_on_rank):
            ind = nn * len(boscombos_on_rank) + mm

            if ki == hi:
                local_K_mat[ind] += m_const * (2 * ki + 1)
            elif ki == hi + 2:
                local_K_mat[ind] -= m_const * np.sqrt((hi + 2) * (hi + 1))
            elif ki == hi - 2:
                local_K_mat[ind] -= m_const * np.sqrt((hi - 1) * hi)

    # May be clobbered after multiple execution
    file_path = Path(f"cform_H1Kdata_{rank}.hdf5")
    with h5py.File(file_path, "w") as f:
        f.create_dataset("H1", data=local_K_mat)


def _cform_twomode_kinetic(pes, n_states, num_workers=1, backend="serial"):
    """Calculates the kinetic energy part of the two body integrals to correct the integrals
    for localized modes

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".
    Returns:
        TensorLike[float]: the kinetic energy part of the two body integrals
    """
    nmodes = len(pes.freqs)

    all_mode_combos = [[aa, bb] for aa in range(nmodes) for bb in range(nmodes)]
    all_bos_combos = list(
        itertools.product(range(n_states), range(n_states), range(n_states), range(n_states))
    )
    executor_class = concurrency.backends.get_executor(backend)

    with executor_class(max_workers=num_workers) as executor:
        boscombos_on_ranks = np.array_split(all_bos_combos, num_workers)
        args = [
            (rank, boscombos_on_rank, pes, all_mode_combos)
            for rank, boscombos_on_rank in enumerate(boscombos_on_ranks)
        ]
        executor.starmap(_local_cform_twomode_kinetic, args)

    return _load_cform_twomode_kinetic(num_workers, nmodes, n_states)


def _local_cform_twomode_kinetic(rank, boscombos_on_rank, pes, all_mode_combos):
    """Worker function to calculate the kinetic energy part of the two body integrals to correct the integrals
    for localized modes. The result are written to a hdf5 file.

    Args:
        rank(int) : the rank of the process
        boscombos_on_rank (int) : list of the combination of bosonic states handled by this process
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        all_mode_combos (int) : list of the combination of nmodes (the length of the list of harmonic frequencies)
    """
    local_kin_cform_twobody = np.zeros(len(all_mode_combos) * len(boscombos_on_rank))
    for nn, (ii, jj) in enumerate(all_mode_combos):
        # TODO: Skip unnecessary combinations
        if jj >= ii:
            continue
        Usum = np.einsum("i,i->", pes.uloc[:, ii], pes.uloc[:, jj])
        m_const = Usum * np.sqrt(pes.freqs[ii] * pes.freqs[jj]) / 4

        for mm, (ki, kj, hi, hj) in enumerate(boscombos_on_rank):
            ki, kj, hi, hj = int(ki), int(kj), int(hi), int(hj)

            conditions = {
                (ki == hi + 1 and kj == hj + 1): -m_const * np.sqrt((hi + 1) * (hj + 1)),
                (ki == hi + 1 and kj == hj - 1): +m_const * np.sqrt((hi + 1) * hj),
                (ki == hi - 1 and kj == hj + 1): +m_const * np.sqrt(hi * (hj + 1)),
                (ki == hi - 1 and kj == hj - 1): -m_const * np.sqrt(hi * hj),
            }
            ind = nn * len(boscombos_on_rank) + mm
            for condition, value in conditions.items():
                if condition:
                    local_kin_cform_twobody[ind] += value
    # May be clobbered after multiple execution
    file_path = Path(f"cform_H2Kdata_{rank}.hdf5")
    with h5py.File(file_path, "w") as f:
        f.create_dataset("H2", data=local_kin_cform_twobody)


def _cform_onemode(pes, n_states, num_workers=1, backend="serial"):
    """
    Calculates the one-body integrals from the given potential energy surface data for the
    Christiansen Hamiltonian.

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".

    Returns:
        TensorLike[float]: the one-body integrals for the Christiansen Hamiltonian
    """

    nmodes = len(pes.freqs)
    all_mode_combos = [[aa] for aa in range(nmodes)]
    all_bos_combos = list(itertools.product(range(n_states), range(n_states)))

    executor_class = concurrency.backends.get_executor(backend)
    with executor_class(max_workers=num_workers) as executor:
        boscombos_on_ranks = np.array_split(all_bos_combos, num_workers)
        args = [
            (rank, boscombos_on_rank, n_states, pes, all_mode_combos)
            for rank, boscombos_on_rank in enumerate(boscombos_on_ranks)
        ]
        executor.starmap(_local_cform_onemode, args)

    return _load_cform_onemode(num_workers, nmodes, n_states) + _cform_onemode_kinetic(
        pes.freqs, n_states
    )


def _local_cform_onemode(rank, boscombos_on_rank, n_states, pes, all_mode_combos):
    """Worker function to calculate the one body integrals. The result are written to a hdf5 file.

    Args:
        rank(int) : the rank of the process
        boscombos_on_rank (int) : list of the combination of bosonic states handled by this process
        n_states(int): maximum number of bosonic states per mode
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        all_mode_combos (int) : list of the combination of nmodes (the length of the list of harmonic frequencies)
    """
    local_ham_cform_onebody = np.zeros(len(all_mode_combos) * len(boscombos_on_rank))

    for nn, ii in enumerate(all_mode_combos):

        for mm, (ki, hi) in enumerate(boscombos_on_rank):
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
            full_coeff = sqrt * quadrature
            ind = nn * len(boscombos_on_rank) + mm
            local_ham_cform_onebody[ind] += full_coeff
    # May be clobbered after multiple execution
    file_path = Path(f"cform_H1data_{rank}.hdf5")
    with h5py.File(file_path, "w") as f:
        f.create_dataset("H1", data=local_ham_cform_onebody)


def _cform_onemode_dipole(pes, n_states, num_workers=1, backend="serial"):
    """
    Calculates the one-body integrals from the given potential energy surface data for the
    Christiansen dipole operator

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".

    Returns:
        TensorLike[float]: the one-body integrals for the Christiansen dipole operator
    """

    nmodes = pes.dipole_onemode.shape[0]
    all_mode_combos = [[aa] for aa in range(nmodes)]
    all_bos_combos = list(itertools.product(range(n_states), range(n_states)))

    executor_class = concurrency.backends.get_executor(backend)

    with executor_class(max_workers=num_workers) as executor:
        boscombos_on_ranks = np.array_split(all_bos_combos, num_workers)
        args = [
            (rank, boscombos_on_rank, n_states, pes, all_mode_combos)
            for rank, boscombos_on_rank in enumerate(boscombos_on_ranks)
        ]
        executor.starmap(_local_cform_onemode_dipole, args)

    return _load_cform_onemode_dipole(num_workers, nmodes, n_states)


def _local_cform_onemode_dipole(rank, boscombos_on_rank, n_states, pes, all_mode_combos):
    """Worker function to calculate the one-body integrals from the given potential energy surface data for the
    Christiansen dipole operator. The result are written to a hdf5 file.

    Args:
        rank(int) : the rank of the process
        boscombos_on_rank (int) : list of the combination of bosonic states handled by this process
        n_states(int): maximum number of bosonic states per mode
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        all_mode_combos (int) : list of the combination of nmodes (the length of the list of harmonic frequencies)
    """
    local_dipole_cform_onebody = np.zeros((len(all_mode_combos) * len(boscombos_on_rank), 3))

    for nn, ii in enumerate(all_mode_combos):

        for mm, (ki, hi) in enumerate(boscombos_on_rank):
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
    # May be clobbered after multiple execution
    file_path = Path(f"cform_D1data_{rank}.hdf5")
    with h5py.File(file_path, "w") as f:
        f.create_dataset("D1", data=local_dipole_cform_onebody)


def _cform_twomode(pes, n_states, num_workers=1, backend="serial"):
    """
    Calculates the two-body integrals from the given potential energy surface data for the
    Christiansen Hamiltonian

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".


    Returns:
        TensorLike[float]: the two-body integrals for the Christiansen Hamiltonian
    """

    nmodes = pes.pes_twomode.shape[0]

    all_mode_combos = [[aa, bb] for aa in range(nmodes) for bb in range(nmodes)]
    all_bos_combos = list(
        itertools.product(range(n_states), range(n_states), range(n_states), range(n_states))
    )
    executor_class = concurrency.backends.get_executor(backend)
    with executor_class(max_workers=num_workers) as executor:
        boscombos_on_ranks = np.array_split(all_bos_combos, num_workers)
        args = [
            (rank, boscombos_on_rank, n_states, pes, all_mode_combos)
            for rank, boscombos_on_rank in enumerate(boscombos_on_ranks)
        ]
        executor.starmap(_local_cform_twomode, args)

    return _load_cform_twomode(num_workers, nmodes, n_states)


def _local_cform_twomode(rank, boscombos_on_rank, n_states, pes, all_mode_combos):
    """Worker function to calculate the two-body integrals from the given potential energy surface data for the
    Christiansen Hamiltonian. The result are written to a hdf5 file.

    Args:
        rank(int) : the rank of the process
        boscombos_on_rank (int) : list of the combination of bosonic states handled by this process
        n_states(int): maximum number of bosonic states per mode
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        all_mode_combos (int) : list of the combination of nmodes (the length of the list of harmonic frequencies)
    """

    local_ham_cform_twobody = np.zeros(len(all_mode_combos) * len(boscombos_on_rank))
    for nn, (ii, jj) in enumerate(all_mode_combos):
        # TODO: Skip unnecessary combinations
        if jj >= ii:
            continue

        for mm, (ki, kj, hi, hj) in enumerate(boscombos_on_rank):
            ki, kj, hi, hj = int(ki), int(kj), int(hi), int(hj)

            sqrt = (
                2 ** (ki + kj + hi + hj)
                * factorial(ki)
                * factorial(kj)
                * factorial(hi)
                * factorial(hj)
            ) ** (-0.5) / np.pi

            orders = [np.zeros(n_states) for _ in range(4)]
            orders[0][ki] = 1.0
            orders[1][kj] = 1.0
            orders[2][hi] = 1.0
            orders[3][hj] = 1.0
            hermite_polys = [
                np.polynomial.hermite.Hermite(order, [-1, 1])(pes.grid) for order in orders
            ]

            quadrature = np.einsum(
                "a,b,a,b,ab,a,b->",
                pes.gauss_weights,
                pes.gauss_weights,
                hermite_polys[0],
                hermite_polys[1],
                pes.pes_twomode[ii, jj, :, :],
                hermite_polys[2],
                hermite_polys[3],
            )
            full_coeff = sqrt * quadrature  # * 219475 for cm^-1 conversion
            ind = nn * len(boscombos_on_rank) + mm
            local_ham_cform_twobody[ind] += full_coeff
    # May be clobbered after multiple execution
    file_path = Path(f"cform_H2data_{rank}.hdf5")
    with h5py.File(file_path, "w") as f:
        f.create_dataset("H2", data=local_ham_cform_twobody)


def _cform_twomode_dipole(pes, n_states, num_workers=1, backend="serial"):
    """
    Calculates the two-body integrals from the given potential energy surface data for the
    Christiansen dipole operator

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".


    Returns:
        TensorLike[float]: the one-body integrals for the Christiansen dipole operator
    """

    nmodes = pes.dipole_twomode.shape[0]

    all_mode_combos = [[aa, bb] for aa in range(nmodes) for bb in range(nmodes)]
    all_bos_combos = list(
        itertools.product(range(n_states), range(n_states), range(n_states), range(n_states))
    )
    executor_class = concurrency.backends.get_executor(backend)

    with executor_class(max_workers=num_workers) as executor:
        boscombos_on_ranks = np.array_split(all_bos_combos, num_workers)
        args = [
            (rank, boscombos_on_rank, n_states, pes, all_mode_combos)
            for rank, boscombos_on_rank in enumerate(boscombos_on_ranks)
        ]
        executor.starmap(_local_cform_twomode_dipole, args)

    return _load_cform_twomode_dipole(num_workers, nmodes, n_states)


def _local_cform_twomode_dipole(rank, boscombos_on_rank, n_states, pes, all_mode_combos):
    """Worker function to calculate the two-body integrals from the given potential energy surface data for the
    Christiansen dipole operator. The result are written to a hdf5 file.

    Args:
        rank(int) : the rank of the process
        boscombos_on_rank (int) : list of the combination of bosonic states handled by this process
        n_states(int): maximum number of bosonic states per mode
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        all_mode_combos (int) : list of the combination of nmodes (the length of the list of harmonic frequencies)
    """

    local_dipole_cform_twobody = np.zeros((len(all_mode_combos) * len(boscombos_on_rank), 3))

    for nn, (ii, jj) in enumerate(all_mode_combos):
        # TODO: Skip unnecessary combinations
        if jj >= ii:
            continue
        for mm, (ki, kj, hi, hj) in enumerate(boscombos_on_rank):
            ki, kj, hi, hj = int(ki), int(kj), int(hi), int(hj)
            sqrt = (
                2 ** (ki + kj + hi + hj)
                * factorial(ki)
                * factorial(kj)
                * factorial(hi)
                * factorial(hj)
            ) ** (-0.5) / np.pi
            orders = [np.zeros(n_states) for _ in range(4)]
            orders[0][ki], orders[1][kj], orders[2][hi], orders[3][hj] = 1.0, 1.0, 1.0, 1.0
            hermite_polys = [
                np.polynomial.hermite.Hermite(order, [-1, 1])(pes.grid) for order in orders
            ]
            ind = nn * len(boscombos_on_rank) + mm
            for alpha in range(3):
                quadrature = np.einsum(
                    "a,b,a,b,ab,a,b->",
                    pes.gauss_weights,
                    pes.gauss_weights,
                    hermite_polys[0],
                    hermite_polys[1],
                    pes.dipole_twomode[ii, jj, :, :, alpha],
                    hermite_polys[2],
                    hermite_polys[3],
                )
                full_coeff = sqrt * quadrature
                local_dipole_cform_twobody[ind, alpha] += full_coeff
    # May be clobbered after multiple execution
    file_path = Path(f"cform_D2data_{rank}.hdf5")
    with h5py.File(file_path, "w") as f:
        f.create_dataset("D2", data=local_dipole_cform_twobody)


def _cform_threemode(pes, n_states, num_workers=1, backend="serial"):
    """
    Calculates the three-body integrals from the given potential energy surface data for the
    Christiansen Hamiltonian

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".

    Returns:
        TensorLike[float]: the three-body integrals for the Christiansen Hamiltonian
    """
    nmodes = pes.pes_threemode.shape[0]

    all_mode_combos = [
        [aa, bb, cc] for aa in range(nmodes) for bb in range(nmodes) for cc in range(nmodes)
    ]

    all_bos_combos = list(
        itertools.product(
            range(n_states),
            range(n_states),
            range(n_states),
            range(n_states),
            range(n_states),
            range(n_states),
        )
    )
    executor_class = concurrency.backends.get_executor(backend)

    with executor_class(max_workers=num_workers) as executor:
        boscombos_on_ranks = np.array_split(all_bos_combos, num_workers)
        args = [
            (rank, boscombos_on_rank, n_states, pes, all_mode_combos)
            for rank, boscombos_on_rank in enumerate(boscombos_on_ranks)
        ]
        executor.starmap(_local_cform_threemode, args)

    return _load_cform_threemode(num_workers, nmodes, n_states)


def _local_cform_threemode(rank, boscombos_on_rank, n_states, pes, all_mode_combos):
    """Worker function to calculate the three-body integrals from the given potential energy surface data for the
    Christiansen Hamiltonian. The result are written to a hdf5 file.

    Args:
        rank(int) : the rank of the process
        boscombos_on_rank (int) : list of the combination of bosonic states handled by this process
        n_states(int): maximum number of bosonic states per mode
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        all_mode_combos (int) : list of the combination of nmodes (the length of the list of harmonic frequencies)
    """

    local_ham_cform_threebody = np.zeros(len(all_mode_combos) * len(boscombos_on_rank))
    for nn, (ii1, ii2, ii3) in enumerate(all_mode_combos):
        # TODO: Skip unnecessary combinations
        if ii2 >= ii1 or ii3 >= ii2:
            continue
        for mm, (k1, k2, k3, h1, h2, h3) in enumerate(boscombos_on_rank):
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
            orders = [np.zeros(n_states) for _ in range(6)]
            orders[0][k1], orders[1][k2], orders[2][k3] = 1.0, 1.0, 1.0
            orders[3][h1], orders[4][h2], orders[5][h3] = 1.0, 1.0, 1.0
            hermite_polys = [
                np.polynomial.hermite.Hermite(order, [-1, 1])(pes.grid) for order in orders
            ]
            quadrature = np.einsum(
                "a,b,c,a,b,c,abc,a,b,c->",
                pes.gauss_weights,
                pes.gauss_weights,
                pes.gauss_weights,
                hermite_polys[0],
                hermite_polys[1],
                hermite_polys[2],
                pes.pes_threemode[ii1, ii2, ii3, :, :, :],
                hermite_polys[3],
                hermite_polys[4],
                hermite_polys[5],
            )
            ind = nn * len(boscombos_on_rank) + mm
            full_coeff = sqrt * quadrature
            local_ham_cform_threebody[ind] = full_coeff
    # May be clobbered after multiple execution
    file_path = Path(f"cform_H3data_{rank}.hdf5")
    with h5py.File(file_path, "w") as f:
        f.create_dataset("H3", data=local_ham_cform_threebody)


def _cform_threemode_dipole(pes, n_states, num_workers=1, backend="serial"):
    """
    Calculates the three-body integrals from the given potential energy surface data for the
    Christiansen dipole operator

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".


    Returns:
        TensorLike[float]: the one-body integrals for the Christiansen dipole operator
    """
    nmodes = pes.dipole_threemode.shape[0]

    all_mode_combos = [
        [aa, bb, cc] for aa in range(nmodes) for bb in range(nmodes) for cc in range(nmodes)
    ]
    all_bos_combos = list(
        itertools.product(
            range(n_states),
            range(n_states),
            range(n_states),
            range(n_states),
            range(n_states),
            range(n_states),
        )
    )

    executor_class = concurrency.backends.get_executor(backend)

    with executor_class(max_workers=num_workers) as executor:
        boscombos_on_ranks = np.array_split(all_bos_combos, num_workers)
        args = [
            (rank, boscombos_on_rank, n_states, pes, all_mode_combos)
            for rank, boscombos_on_rank in enumerate(boscombos_on_ranks)
        ]
        executor.starmap(_local_cform_threemode_dipole, args)

    dipole_cform_threebody = _load_cform_threemode_dipole(num_workers, nmodes, n_states)

    return dipole_cform_threebody


def _local_cform_threemode_dipole(rank, boscombos_on_rank, n_states, pes, all_mode_combos):
    """Worker function to calculate the three-body integrals from the given potential energy surface data for the
    Christiansen dipole operator. The result are written to a hdf5 file.

    Args:
        rank(int) : the rank of the process
        boscombos_on_rank (int) : list of the combination of bosonic states handled by this process
        n_states(int): maximum number of bosonic states per mode
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        all_mode_combos (int) : list of the combination of nmodes (the length of the list of harmonic frequencies)
    """

    local_dipole_cform_threebody = np.zeros((len(all_mode_combos) * len(boscombos_on_rank), 3))

    for nn, (ii1, ii2, ii3) in enumerate(all_mode_combos):
        # TODO: Skip unnecessary combinations
        if ii2 >= ii1 or ii3 >= ii2:
            continue
        for mm, (k1, k2, k3, h1, h2, h3) in enumerate(boscombos_on_rank):
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
            orders = [np.zeros(n_states) for _ in range(6)]
            orders[0][k1], orders[1][k2], orders[2][k3] = 1.0, 1.0, 1.0
            orders[3][h1], orders[4][h2], orders[5][h3] = 1.0, 1.0, 1.0
            hermite_polys = [
                np.polynomial.hermite.Hermite(order, [-1, 1])(pes.grid) for order in orders
            ]
            ind = nn * len(boscombos_on_rank) + mm
            for alpha in range(3):
                quadrature = np.einsum(
                    "a,b,c,a,b,c,abc,a,b,c->",
                    pes.gauss_weights,
                    pes.gauss_weights,
                    pes.gauss_weights,
                    hermite_polys[0],
                    hermite_polys[1],
                    hermite_polys[2],
                    pes.dipole_threemode[ii1, ii2, ii3, :, :, :, alpha],
                    hermite_polys[3],
                    hermite_polys[4],
                    hermite_polys[5],
                )
                full_coeff = sqrt * quadrature
                local_dipole_cform_threebody[ind, alpha] = full_coeff
    # May be clobbered after multiple execution
    file_path = Path(f"cform_D3data_{rank}.hdf5")
    with h5py.File(file_path, "w") as f:
        f.create_dataset("D3", data=local_dipole_cform_threebody)


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
            file_path = Path("cform_H1data" + f"_{rank}" + ".hdf5")
            with h5py.File(file_path, "r+") as f:
                local_ham_cform_onebody = f["H1"][()]
            chunk = np.array_split(local_ham_cform_onebody, nmode_combos)[mode_combo]
            l1 += len(chunk)
            local_chunk[l0:l1] = chunk
            l0 += len(chunk)

        r1 += len(local_chunk)
        ham_cform_onebody[r0:r1] = local_chunk
        r0 += len(local_chunk)

    ham_cform_onebody = ham_cform_onebody.reshape(final_shape)

    return ham_cform_onebody


def _load_cform_onemode_kinetic(num_proc, nmodes, quad_order):
    """
    Loader to collect and combine pes_onemode_kinetic from multiple processors.

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
            file_path = Path("cform_H1Kdata" + f"_{rank}" + ".hdf5")
            with h5py.File(file_path, "r+") as f:
                local_ham_cform_onebody = f["H1"][()]
            chunk = np.array_split(local_ham_cform_onebody, nmode_combos)[mode_combo]
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
            file_path = Path("cform_H2data" + f"_{rank}" + ".hdf5")
            with h5py.File(file_path, "r+") as f:
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


def _load_cform_twomode_kinetic(num_proc, nmodes, quad_order):
    """
    Loader to collect and combine pes_twomode_kinetic from multiple processors.

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
            file_path = Path("cform_H2Kdata" + f"_{rank}" + ".hdf5")
            with h5py.File(file_path, "r+") as f:
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
            file_path = Path("cform_H3data" + f"_{rank}" + ".hdf5")
            with h5py.File(file_path, "r+") as f:
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
            file_path = Path("cform_D1data" + f"_{rank}" + ".hdf5")
            with h5py.File(file_path, "r+") as f:
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
            file_path = Path("cform_D2data" + f"_{rank}" + ".hdf5")
            with h5py.File(file_path, "r+") as f:
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
            file_path = Path("cform_D3data" + f"_{rank}" + ".hdf5")
            with h5py.File(file_path, "r+") as f:
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


def christiansen_integrals(pes, n_states=16, cubic=False, num_workers=1, backend="serial"):
    r"""Compute Christiansen vibrational Hamiltonian integrals.

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode
        cubic(bool): Flag to include three-mode couplings. Default is ``False``.
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".


    Returns:
        TensorLike[float]: the integrals for the Christiansen Hamiltonian
    """

    ham_cform_onebody = _cform_onemode(pes, n_states, num_workers=num_workers, backend=backend)
    for path in Path.cwd().glob("cform_H1data*"):
        path.unlink()

    ham_cform_twobody = _cform_twomode(pes, n_states, num_workers=num_workers, backend=backend)
    if pes.localized:
        ham_cform_twobody += _cform_twomode_kinetic(
            pes, n_states, num_workers=num_workers, backend=backend
        )
    for path in Path.cwd().glob("cform_H2data*"):
        path.unlink()

    if cubic:
        ham_cform_threebody = _cform_threemode(
            pes, n_states, num_workers=num_workers, backend=backend
        )
        for path in Path.cwd().glob("cform_H3data*"):
            path.unlink()

        H_arr = [ham_cform_onebody, ham_cform_twobody, ham_cform_threebody]
    else:
        H_arr = [ham_cform_onebody, ham_cform_twobody]

    return H_arr


def christiansen_integrals_dipole(pes, n_states=16, num_workers=1, backend="serial"):
    r"""Compute Christiansen vibrational dipole integrals.

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".


    Returns:
        TensorLike[float]: the integrals for the Christiansen dipole operator
    """
    dipole_cform_onebody = _cform_onemode_dipole(
        pes, n_states, num_workers=num_workers, backend=backend
    )
    for path in Path.cwd().glob("cform_D1data*"):
        path.unlink()

    dipole_cform_twobody = None

    if pes.localized is True or pes.dipole_level > 1:
        dipole_cform_twobody = _cform_twomode_dipole(
            pes, n_states, num_workers=num_workers, backend=backend
        )
        for path in Path.cwd().glob("cform_D2data*"):
            path.unlink()

    if pes.localized is True or pes.dipole_level > 2:
        dipole_cform_threebody = _cform_threemode_dipole(
            pes, n_states, num_workers=num_workers, backend=backend
        )
        for path in Path.cwd().glob("cform_D3data*"):
            path.unlink()

        D_arr = [dipole_cform_onebody, dipole_cform_twobody, dipole_cform_threebody]
    else:
        D_arr = [dipole_cform_onebody, dipole_cform_twobody]

    return D_arr
