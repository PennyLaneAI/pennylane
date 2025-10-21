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
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import factorial

from pennylane import concurrency

# pylint: disable = redefined-outer-name, too-many-positional-arguments

try:
    import h5py

    has_h5py = True
except ImportError:
    has_h5py = False

def _cform_onemode_placzek(pes, n_states, num_workers=1, backend="serial", path=None):
    """
    Calculates the one-body integrals from the given potential energy surface data for the
    Christiansen placzek operator

    Args:
        pes(PolarizablePES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".
        path (string): the path to the directory where results will be saved. Default value is set to None.

    Returns:
        TensorLike[float]: the one-body integrals for the Christiansen placzek operator
    """

    nmodes = pes.placzek_onemode.shape[0]
    all_mode_combos = [[aa] for aa in range(nmodes)]
    all_bos_combos = list(itertools.product(range(n_states), range(n_states)))

    executor_class = concurrency.backends.get_executor(backend)

    with executor_class(max_workers=num_workers) as executor:
        boscombos_on_ranks = np.array_split(all_bos_combos, num_workers)
        args = [
            (rank, boscombos_on_rank, n_states, pes, all_mode_combos, path)
            for rank, boscombos_on_rank in enumerate(boscombos_on_ranks)
        ]
        executor.starmap(_local_cform_onemode_placzek, args)

    result = _load_cform_onemode_placzek(num_workers, nmodes, n_states, path)

    return result


# pylint: disable=too-many-arguments
def _local_cform_onemode_placzek(rank, boscombos_on_rank, n_states, pes, all_mode_combos, path):
    """Worker function to calculate the one-body integrals from the given potential energy surface data for the
    Christiansen placzek operator. The result are written to a hdf5 file.

    Args:
        rank(int) : the rank of the process
        boscombos_on_rank (int) : list of the combination of bosonic states handled by this process
        n_states(int): maximum number of bosonic states per mode
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        all_mode_combos (int) : list of the combination of nmodes (the length of the list of harmonic frequencies)
        path (string): the path to the directory where results will be saved.
    """
    local_placzek_cform_onebody = np.zeros((len(all_mode_combos) * len(boscombos_on_rank), 6))

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
            for alpha in range(6):
                quadrature = np.sum(
                    pes.gauss_weights * pes.placzek_onemode[ii, :, alpha] * hermite_ki * hermite_hi
                )
                full_coeff = sqrt * quadrature  # * 219475 for converting into cm^-1
                local_placzek_cform_onebody[ind, alpha] += full_coeff
    _write_data(path, rank, "cform_P1data", "P1", local_placzek_cform_onebody)


def _cform_twomode_placzek(pes, n_states, num_workers=1, backend="serial", path=None):
    """
    Calculates the two-body integrals from the given potential energy surface data for the
    Christiansen placzek operator

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".
        path (string): the path to the directory where results will be saved. Default value is set to None.


    Returns:
        TensorLike[float]: the one-body integrals for the Christiansen placzek operator
    """

    nmodes = pes.placzek_twomode.shape[0]

    all_mode_combos = [[aa, bb] for aa in range(nmodes) for bb in range(nmodes)]
    all_bos_combos = list(
        itertools.product(range(n_states), range(n_states), range(n_states), range(n_states))
    )
    executor_class = concurrency.backends.get_executor(backend)

    with executor_class(max_workers=num_workers) as executor:
        boscombos_on_ranks = np.array_split(all_bos_combos, num_workers)
        args = [
            (rank, boscombos_on_rank, n_states, pes, all_mode_combos, path)
            for rank, boscombos_on_rank in enumerate(boscombos_on_ranks)
        ]
        executor.starmap(_local_cform_twomode_placzek, args)
    result = _load_cform_twomode_placzek(num_workers, nmodes, n_states, path)

    return result


# pylint: disable=too-many-arguments
def _local_cform_twomode_placzek(rank, boscombos_on_rank, n_states, pes, all_mode_combos, path):
    """Worker function to calculate the two-body integrals from the given potential energy surface data for the
    Christiansen placzek operator. The result are written to a hdf5 file.

    Args:
        rank(int) : the rank of the process
        boscombos_on_rank (int) : list of the combination of bosonic states handled by this process
        n_states(int): maximum number of bosonic states per mode
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        all_mode_combos (int) : list of the combination of nmodes (the length of the list of harmonic frequencies)
        path (string): the path to the directory where results will be saved.
    """

    local_placzek_cform_twobody = np.zeros((len(all_mode_combos) * len(boscombos_on_rank), 6))

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
            for alpha in range(6):
                quadrature = np.einsum(
                    "a,b,a,b,ab,a,b->",
                    pes.gauss_weights,
                    pes.gauss_weights,
                    hermite_polys[0],
                    hermite_polys[1],
                    pes.placzek_twomode[ii, jj, :, :, alpha],
                    hermite_polys[2],
                    hermite_polys[3],
                )
                full_coeff = sqrt * quadrature
                local_placzek_cform_twobody[ind, alpha] += full_coeff
    _write_data(path, rank, "cform_P2data", "P2", local_placzek_cform_twobody)


def _cform_threemode_placzek(pes, n_states, num_workers=1, backend="serial", path=None):
    """
    Calculates the three-body integrals from the given potential energy surface data for the
    Christiansen placzek operator

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode
        num_workers (int): the number of concurrent units used for the computation. Default value is set to 1.
        backend (string): the executor backend from the list of supported backends.
            Available options : "mp_pool", "cf_procpool", "cf_threadpool", "serial", "mpi4py_pool", "mpi4py_comm". Default value is set to "serial".
        path (string): the path to the directory where results will be saved. Default value is set to None.


    Returns:
        TensorLike[float]: the one-body integrals for the Christiansen placzek operator
    """

    nmodes = pes.placzek_threemode.shape[0]

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
            (rank, boscombos_on_rank, n_states, pes, all_mode_combos, path)
            for rank, boscombos_on_rank in enumerate(boscombos_on_ranks)
        ]
        executor.starmap(_local_cform_threemode_placzek, args)

    result = _load_cform_threemode_placzek(num_workers, nmodes, n_states, path)

    return result


# pylint: disable=too-many-arguments
def _local_cform_threemode_placzek(rank, boscombos_on_rank, n_states, pes, all_mode_combos, path):
    """Worker function to calculate the three-body integrals from the given potential energy surface data for the
    Christiansen placzek operator. The result are written to a hdf5 file.

    Args:
        rank(int) : the rank of the process
        boscombos_on_rank (int) : list of the combination of bosonic states handled by this process
        n_states(int): maximum number of bosonic states per mode
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        all_mode_combos (int) : list of the combination of nmodes (the length of the list of harmonic frequencies)
        path (string): the path to the directory where results will be saved.
    """

    local_placzek_cform_threebody = np.zeros((len(all_mode_combos) * len(boscombos_on_rank), 6))

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
            for alpha in range(6):
                quadrature = np.einsum(
                    "a,b,c,a,b,c,abc,a,b,c->",
                    pes.gauss_weights,
                    pes.gauss_weights,
                    pes.gauss_weights,
                    hermite_polys[0],
                    hermite_polys[1],
                    hermite_polys[2],
                    pes.placzek_threemode[ii1, ii2, ii3, :, :, :, alpha],
                    hermite_polys[3],
                    hermite_polys[4],
                    hermite_polys[5],
                )
                full_coeff = sqrt * quadrature
                local_placzek_cform_threebody[ind, alpha] = full_coeff
    _write_data(path, rank, "cform_P3data", "P3", local_placzek_cform_threebody)


def _load_cform_onemode_placzek(num_proc, nmodes, quad_order, path):
    """
    Loader to collect and combine placzek_onemode from multiple processors.

    Args:
        num_proc (int): number of processors
        nmodes (int): number of normal modes
        quad_order (int): order for Gauss-Hermite quadratures
        path (string): the path to the directory where results are saved.

    Returns:
        TensorLike[float]: one-body integrals for Christiansen placzek operator
    """
    final_shape = (nmodes, quad_order, quad_order)
    nmode_combos = int(nmodes)

    placzek_cform_onebody = np.zeros((np.prod(final_shape), 6))
    r0 = 0
    r1 = 0
    for mode_combo in range(nmode_combos):
        local_chunk = np.zeros((quad_order**2, 6))

        l0 = 0
        l1 = 0
        for rank in range(num_proc):
            local_placzek_cform_onebody = _read_data(path, rank, "cform_P1data", "P1")
            chunk = np.array_split(local_placzek_cform_onebody, nmode_combos, axis=0)[mode_combo]  #
            l1 += chunk.shape[0]
            local_chunk[l0:l1, :] = chunk
            l0 += chunk.shape[0]

        r1 += local_chunk.shape[0]
        placzek_cform_onebody[r0:r1, :] = local_chunk
        r0 += local_chunk.shape[0]

    placzek_cform_onebody = placzek_cform_onebody.reshape(final_shape + (6,))

    return placzek_cform_onebody.transpose(3, 0, 1, 2)


def _load_cform_twomode_placzek(num_proc, nmodes, quad_order, path):
    """
    Loader to collect and combine placzek_twomode from multiple processors.

    Args:
        num_proc (int): number of processors
        nmodes (int): number of normal modes
        quad_order (int): order for Gauss-Hermite quadratures
        path (string): the path to the directory where results are saved.

    Returns:
        TensorLike[float]: two-body integrals for Christiansen placzek operator
    """
    final_shape = (nmodes, nmodes, quad_order, quad_order, quad_order, quad_order)
    nmode_combos = int(nmodes**2)

    placzek_cform_twobody = np.zeros((np.prod(final_shape), 6))
    r0 = 0
    r1 = 0
    for mode_combo in range(nmode_combos):
        local_chunk = np.zeros((quad_order**4, 6))

        l0 = 0
        l1 = 0
        for rank in range(num_proc):
            local_placzek_cform_twobody = _read_data(path, rank, "cform_P2data", "P2")
            chunk = np.array_split(local_placzek_cform_twobody, nmode_combos, axis=0)[mode_combo]  #
            l1 += chunk.shape[0]
            local_chunk[l0:l1, :] = chunk
            l0 += chunk.shape[0]

        r1 += local_chunk.shape[0]
        placzek_cform_twobody[r0:r1, :] = local_chunk
        r0 += local_chunk.shape[0]

    placzek_cform_twobody = placzek_cform_twobody.reshape(final_shape + (6,))

    return placzek_cform_twobody.transpose(6, 0, 1, 2, 3, 4, 5)


def _load_cform_threemode_placzek(num_proc, nmodes, quad_order, path):
    """
    Loader to collect and combine placzek_threemode from multiple processors.

    Args:
        num_proc (int): number of processors
        nmodes (int): number of normal modes
        quad_order (int): order for Gauss-Hermite quadratures
        path (string): the path to the directory where results are saved.

    Returns:
        TensorLike[float]: three-body integrals for Christiansen placzek operator
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

    placzek_cform_threebody = np.zeros((np.prod(final_shape), 6))
    r0 = 0
    r1 = 0
    for mode_combo in range(nmode_combos):
        local_chunk = np.zeros((quad_order**6, 6))

        l0 = 0
        l1 = 0
        for rank in range(num_proc):
            local_placzek_cform_threebody = _read_data(path, rank, "cform_P3data", "P3")
            chunk = np.array_split(local_placzek_cform_threebody, nmode_combos, axis=0)[
                mode_combo
            ]  #
            l1 += chunk.shape[0]
            local_chunk[l0:l1, :] = chunk
            l0 += chunk.shape[0]

        r1 += local_chunk.shape[0]
        placzek_cform_threebody[r0:r1, :] = local_chunk
        r0 += local_chunk.shape[0]

    placzek_cform_threebody = placzek_cform_threebody.reshape(final_shape + (6,))

    return placzek_cform_threebody.transpose(9, 0, 1, 2, 3, 4, 5, 6, 7, 8)


def christiansen_integrals_placzek(pes, n_states=16, num_workers=1, backend="serial"):
    r"""Computes Christiansen vibrational placzek integrals.

    The Christiansen placzek operator is constructed similar to the vibrational Hamiltonian operator
    defined in Eqs. D4-D7 of `arXiv:2504.10602 <https://arxiv.org/abs/2504.10602>`_. The placzek
    operator is defined as

    .. math::

        \mu = \sum_{i}^M \sum_{k_i, l_i}^{N_i} C_{k_i, l_i}^{(i)} b_{k_i}^{\dagger} b_{l_i} +
        \sum_{i<j}^{M} \sum_{k_i,l_i}^{N_i} \sum_{k_j,l_j}^{N_j} C_{k_i k_j, l_i l_j}^{(i,j)}
        b_{k_i}^{\dagger} b_{k_j}^{\dagger} b_{l_i} b_{l_j},


    where :math:`b^{\dagger}` and :math:`b` are the creation and annihilation
    operators, :math:`M` represents the number of normal modes and :math:`N` is the number of
    modals. The coefficients :math:`C` represent the one-mode and two-mode integrals defined as

    .. math::

        C_{k_i, l_i}^{(i)} = \int \phi_i^{k_i}(Q_i) \left( D_1^{(i)}(Q_i) \right) \phi_i^{h_i}(Q_i),

    and

    .. math::

        C_{k_i, k_j, l_i, l_j}^{(i,j)} = \int \int \phi_i^{k_i}(Q_i) \phi_j^{k_j}(Q_j)
        D_2^{(i,j)}(Q_i, Q_j) \phi_i^{l_i}(Q_i) \phi_j^{l_j}(Q_j) \; \text{d} Q_i \text{d} Q_j,

    where :math:`\phi` represents a modal, :math:`Q` represents a normal coordinate and :math:`D`
    represents the placzek function obtained from the expansion

    .. math::

        D({Q}) = \sum_i D_1(Q_i) + \sum_{i>j} D_2(Q_i,Q_j) + ....

    Similarly, the three-mode integrals can be obtained
    following Eq. D7 of `arXiv:2504.10602 <https://arxiv.org/abs/2504.10602>`_.

    Args:
        pes(VibrationalPES): object containing the vibrational potential energy surface data
        n_states(int): maximum number of bosonic states per mode
        num_workers (int): the number of concurrent units used for the computation. Default value is
            set to 1.
        backend (string): the executor backend from the list of supported backends. Available
            options are ``mp_pool``, ``cf_procpool``, ``cf_threadpool``, ``serial``,
            ``mpi4py_pool``, ``mpi4py_comm``. Default value is set to ``serial``. See Usage Details
            for more information.

    Returns:
        List[TensorLike[float]]: the integrals for the Christiansen placzek operator

    .. note::

        This function requires the ``h5py`` package to be installed.
        It can be installed with ``pip install h5py``.

    **Example**

    >>> symbols  = ['H', 'F']
    >>> geometry = np.array([[0.0, 0.0, -0.40277116], [0.0, 0.0, 1.40277116]])
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> pes = qml.qchem.vibrational_pes(mol, optimize = False, placzek_level = 3, cubic=True)
    >>> integrals = qml.qchem.vibrational.christiansen_integrals_placzek(pes, n_states = 2)
    >>> print(integrals[0][2])
    [[[-0.00074107 -0.02287269]
    [-0.02287269 -0.00216419]]]

    .. details::
        :title: Usage Details

        The ``backend`` options allow to run calculations using multiple threads or multiple
        processes.

        - ``serial``: This executor wraps Python standard library calls without support for
            multithreaded or multiprocess execution. Any calls to external libraries that utilize
            threads, such as BLAS through numpy, can still use multithreaded calls at that layer.

        - ``mp_pool``: This executor wraps Python standard library `multiprocessing.Pool <https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing.pool>`_
            interface, and provides support for execution using multiple processes.

        - ``cf_procpool``: This executor wraps Python standard library `concurrent.futures.ProcessPoolExecutor <https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor>`_
            interface, and provides support for execution using multiple processes.

        - ``cf_threadpool``: This executor wraps Python standard library `concurrent.futures.ThreadPoolExecutor <https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor>`_
            interface, and provides support for execution using multiple threads. The threading
            executor may not provide execution speed-ups for tasks when using a GIL-enabled Python.

        - ``mpi4py_pool``: This executor wraps the `mpi4py.futures.MPIPoolExecutor <https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpipoolexecutor>`_
            class, and provides support for execution using multiple processes launched using MPI.

        - ``mpi4py_comm``: This executor wraps the `mpi4py.futures.MPICommExecutor <https://mpi4py.readthedocs.io/en/stable/mpi4py.futures.html#mpicommexecutor>`_
            class, and provides support for execution using multiple processes launched using MPI.
    """
    if not has_h5py:  # pragma: no cover
        raise ImportError(
            "christiansen_integrals_placzek requires the h5py package. "
            "You can install it with pip install h5py"
        )  # pragma: no cover
    with TemporaryDirectory() as path:
        placzek_cform_onebody = _cform_onemode_placzek(
            pes, n_states, num_workers=num_workers, backend=backend, path=path
        )

        placzek_cform_twobody = None

        if pes.localized is True or pes.placzek_level > 1:
            placzek_cform_twobody = _cform_twomode_placzek(
                pes, n_states, num_workers=num_workers, backend=backend, path=path
            )

        if pes.localized is True or pes.placzek_level > 2:
            placzek_cform_threebody = _cform_threemode_placzek(
                pes, n_states, num_workers=num_workers, backend=backend, path=path
            )

            P_arr = [placzek_cform_onebody, placzek_cform_twobody, placzek_cform_threebody]
        else:
            P_arr = [placzek_cform_onebody, placzek_cform_twobody]

    return P_arr