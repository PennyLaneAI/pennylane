# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests the apply_operation functions from devices/qubit
"""

from functools import reduce

import numpy as np
import pytest
from dummy_debugger import Debugger
from gate_data import I, X, Y, Z
from scipy.sparse import csr_matrix, kron
from scipy.stats import unitary_group

import pennylane as qml
from pennylane.devices.qubit.apply_operation import (
    apply_operation,
    apply_operation_csr_matrix,
    apply_operation_einsum,
    apply_operation_tensordot,
)
from pennylane.operation import _UNSET_BATCH_SIZE, Operation

ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]


def apply_operation_sparse_wrapped(op, state, is_state_batched: bool = False):
    """Apply an operation to a state using the sparse matrix method"""
    # Convert op to a CSR matrix
    op = qml.QubitUnitary(csr_matrix(op.matrix()), wires=op.wires)
    # Convert state into numpy
    state = qml.math.asarray(state, like="numpy")
    return apply_operation_csr_matrix(op, state, is_state_batched)


methods = [
    apply_operation_einsum,
    apply_operation_tensordot,
    apply_operation,
]

# pylint: disable=import-outside-toplevel,unsubscriptable-object,arguments-differ


def test_custom_operator_with_matrix():
    """Test that apply_operation works with any operation that defines a matrix."""

    mat = np.array(
        [
            [0.39918205 + 0.3024376j, -0.86421077 + 0.04821758j],
            [0.73240679 + 0.46126509j, 0.49576832 - 0.07091251j],
        ]
    )

    # pylint: disable=too-few-public-methods
    class CustomOp(Operation):
        """Custom Operation"""

        num_wires = 1

        def matrix(self):
            return mat

    state = np.array([-0.30688912 - 0.4768824j, 0.8100052 - 0.14931113j])

    new_state = apply_operation(CustomOp(0), state)
    assert qml.math.allclose(new_state, mat @ state)


class TestSparseOperation:
    """Test the sparse matrix application method"""

    ops_to_sparsify = [
        qml.PauliX(0),
        qml.CNOT((0, 1)),
        qml.Toffoli((0, 1, 2)),
        qml.MultiControlledX(wires=[0, 1, 2, 3, 4], control_values=[1, 1, 1, 1]),
        qml.GroverOperator(wires=[0, 1, 2]),
        qml.IsingXX(np.pi / 2, wires=[0, 1]),
        qml.DoubleExcitation(np.pi / 4, wires=[0, 1, 2, 3]),
    ]

    def test_sparse_operation_dense_state(self):
        """Test that apply_operation works with a sparse matrix operation"""

        # Create a random unitary matrix
        U = unitary_group.rvs(2**3)
        U = qml.math.asarray(U, like="numpy")

        # Create a random state vector
        state = np.random.rand(2**3) + 1j * np.random.rand(2**3)
        state = qml.math.asarray(state, like="numpy").reshape([2] * 3)

        # Apply the operation
        U_sp = qml.QubitUnitary(csr_matrix(U), wires=range(3))
        new_state = apply_operation_csr_matrix(U_sp, state)
        expected_state = state.reshape((1, 8)) @ U.reshape((8, 8)).T
        expected_state = expected_state.reshape([2] * 3)

        assert qml.math.allclose(new_state, expected_state)

    def test_sparse_operation_sparse_state(self):
        """Test that apply_operation does not support with a sparse state operation"""

        # Create a random unitary matrix
        U = unitary_group.rvs(2**3)
        U = qml.math.asarray(U, like="numpy")

        # Create a random state vector
        state = np.random.rand(2**3) + 1j * np.random.rand(2**3)
        state = csr_matrix(state)
        U_sp = qml.QubitUnitary(csr_matrix(U), wires=range(3))

        # Apply the operation
        with pytest.raises(
            TypeError,
            match="State should not be sparse",
        ):
            apply_operation_csr_matrix(U_sp, state)

    @pytest.mark.parametrize("N", range(4, 10, 2))
    def test_sparse_operation_large_N(self, N):
        """Test that apply_operation_csr_matrix works with multiple wires
        with operators composed of random I X Y Z tensored together"""
        # Make a sparse unitary matrix by tensor producting several smaller unitaries

        U_list = [I, X, Y, Z]
        U_list = [csr_matrix(op) for op in U_list]
        # Pick random indices, then choose unitaries
        indices = np.random.choice(len(U_list), size=N)
        Us = [U_list[idx] for idx in indices]

        U = Us[0]
        for i in range(1, N):
            U = kron(U, Us[i], format="csr")

        state_shape = (2,) * N
        state_size = 2**N
        state = np.random.rand(state_size) + 1j * np.random.rand(state_size)
        state = state / np.linalg.norm(state)
        state = state.reshape(state_shape)

        U_sp = qml.QubitUnitary(csr_matrix(U), wires=range(N))
        new_state = apply_operation_csr_matrix(U_sp, state)

        # Don't waste time constructing dense U to test, instead we just check that the U^Dagger @ state is correct
        final_state = apply_operation_csr_matrix(U_sp, new_state)
        assert qml.math.allclose(final_state, state)

    @pytest.mark.parametrize("N", range(4, 10, 2))
    @pytest.mark.parametrize(
        "op",
        [
            qml.QubitUnitary(
                csr_matrix(X),
                wires=[0],
            ),
            qml.QubitUnitary(
                csr_matrix(Y),
                wires=[0],
            ),
            qml.QubitUnitary(
                csr_matrix(Z),
                wires=[0],
            ),
        ],
    )
    def test_sparse_operation_dispatch(self, op, N):
        """Test that the operators dispatch correctly for sparse or dense states."""

        expected_shape = (2,) * N
        # Create a dense state, shape (2,2,2)
        state = np.random.rand(*(2,) * N) + 1j * np.random.rand(*(2,) * N)

        new_state = apply_operation(op, state)

        # Confirm the return type and shape
        assert isinstance(new_state, np.ndarray)
        assert new_state.shape == expected_shape

    @pytest.mark.parametrize("op", ops_to_sparsify)
    def test_sparse_operation_wrapper(self, op):
        """Test that apply_operation_sparse_wrapped correctly handles larger quantum operations
        by converting them to sparse matrices"""

        # Get a compatible state
        wires = op.wires
        system_size = len(wires) + 1
        state = np.random.rand(2**system_size) + 1j * np.random.rand(2**system_size)
        state = state / np.linalg.norm(state)
        state = state.reshape([2] * system_size)

        new_state = apply_operation_sparse_wrapped(op, state)
        expected_state = apply_operation(op, state)
        assert qml.math.allclose(new_state, expected_state)


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("wire", (0, 1))
class TestTwoQubitStateSpecialCases:
    """Test the special cases on a two qubit state.  Also tests the special cases for einsum and tensor application methods
    for additional testing of these generic matrix application methods."""

    def test_paulix(self, method, wire, ml_framework):
        """Test the application of a paulix gate on a two qubit state."""

        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        new_state = method(qml.PauliX(wire), initial_state)

        initial0dim = qml.math.take(initial_state, 0, axis=wire)
        new1dim = qml.math.take(new_state, 1, axis=wire)

        assert qml.math.allclose(initial0dim, new1dim)

        initial1dim = qml.math.take(initial_state, 1, axis=wire)
        new0dim = qml.math.take(new_state, 0, axis=wire)
        assert qml.math.allclose(initial1dim, new0dim)

    def test_pauliz(self, method, wire, ml_framework):
        """Test the application of a pauliz gate on a two qubit state."""
        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        new_state = method(qml.PauliZ(wire), initial_state)

        initial0 = qml.math.take(initial_state, 0, axis=wire)
        new0 = qml.math.take(new_state, 0, axis=wire)
        assert qml.math.allclose(initial0, new0)

        initial1 = qml.math.take(initial_state, 1, axis=wire)
        new1 = qml.math.take(new_state, 1, axis=wire)
        assert qml.math.allclose(initial1, -new1)

    def test_pauliy(self, method, wire, ml_framework):
        """Test the application of a pauliy gate on a two qubit state."""
        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        new_state = method(qml.PauliY(wire), initial_state)

        initial0 = qml.math.take(initial_state, 0, axis=wire)
        new1 = qml.math.take(new_state, 1, axis=wire)
        assert qml.math.allclose(1j * initial0, new1)

        initial1 = qml.math.take(initial_state, 1, axis=wire)
        new0 = qml.math.take(new_state, 0, axis=wire)
        assert qml.math.allclose(-1j * initial1, new0)

    def test_hadamard(self, method, wire, ml_framework):
        """Test the application of a hadamard on a two qubit state."""
        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        new_state = method(qml.Hadamard(wire), initial_state)

        inv_sqrt2 = 1 / np.sqrt(2)

        initial0 = qml.math.take(initial_state, 0, axis=wire)
        initial1 = qml.math.take(initial_state, 1, axis=wire)

        expected0 = inv_sqrt2 * (initial0 + initial1)
        new0 = qml.math.take(new_state, 0, axis=wire)
        assert qml.math.allclose(new0, expected0)

        expected1 = inv_sqrt2 * (initial0 - initial1)
        new1 = qml.math.take(new_state, 1, axis=wire)
        assert qml.math.allclose(new1, expected1)

    def test_phaseshift(self, method, wire, ml_framework):
        """test the application of a phaseshift gate on a two qubit state."""

        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        phase = qml.math.asarray(-2.3, like=ml_framework)
        shift = qml.math.exp(1j * qml.math.cast(phase, np.complex128))

        new_state = method(qml.PhaseShift(phase, wire), initial_state)

        new0 = qml.math.take(new_state, 0, axis=wire)
        initial0 = qml.math.take(initial_state, 0, axis=wire)
        assert qml.math.allclose(new0, initial0)

        initial1 = qml.math.take(initial_state, 1, axis=wire)
        new1 = qml.math.take(new_state, 1, axis=wire)
        assert qml.math.allclose(shift * initial1, new1)

    def test_cnot(self, method, wire, ml_framework):
        """Test the application of a cnot gate on a two qubit state."""

        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        control = wire
        target = int(not control)

        new_state = method(qml.CNOT((control, target)), initial_state)

        initial0 = qml.math.take(initial_state, 0, axis=control)
        new0 = qml.math.take(new_state, 0, axis=control)
        assert qml.math.allclose(initial0, new0)

        initial1 = qml.math.take(initial_state, 1, axis=control)
        new1 = qml.math.take(new_state, 1, axis=control)
        assert qml.math.allclose(initial1[1], new1[0])
        assert qml.math.allclose(initial1[0], new1[1])

    def test_grover(self, method, wire, ml_framework):
        """Test the application of GroverOperator on a two qubit state."""

        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        wires = [wire, 1 - wire]
        op = qml.GroverOperator(wires)
        new_state = method(op, initial_state)

        overlap = qml.math.sum(initial_state) / 2
        ones_state = qml.math.ones_like(initial_state) / 2
        expected_state = 2 * ones_state * overlap - initial_state
        assert qml.math.allclose(new_state, expected_state)
        state_via_mat = qml.math.tensordot(
            op.matrix().reshape([2] * 4), initial_state, axes=[[2, 3], [0, 1]]
        )
        assert qml.math.allclose(new_state, state_via_mat)

    def test_identity(self, method, wire, ml_framework):
        """Test the application of a GlobalPhase gate on a two qubit state."""

        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        new_state = method(qml.Identity(wire), initial_state)

        assert qml.math.allclose(initial_state, new_state)

    def test_globalphase(self, method, wire, ml_framework):
        """Test the application of a GlobalPhase gate on a two qubit state."""
        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        phase = qml.math.asarray(-2.3, like=ml_framework)
        shift = qml.math.exp(-1j * qml.math.cast(phase, np.complex128))

        new_state_with_wire = method(qml.GlobalPhase(phase, wire), initial_state)
        new_state_no_wire = method(qml.GlobalPhase(phase), initial_state)

        assert qml.math.allclose(shift * initial_state, new_state_with_wire)
        assert qml.math.allclose(shift * initial_state, new_state_no_wire)


def time_independent_hamiltonian():
    """Create a time-independent Hamiltonian on two qubits."""
    ops = [qml.PauliX(0), qml.PauliZ(1), qml.PauliY(0), qml.PauliX(1)]

    coeffs = [qml.pulse.constant, qml.pulse.constant, 0.4, 0.9]

    return qml.pulse.ParametrizedHamiltonian(coeffs, ops)


def time_dependent_hamiltonian():
    """Create a time-dependent two-qubit Hamiltonian that takes two scalar parameters."""
    import jax.numpy as jnp

    ops = [qml.PauliX(0), qml.PauliZ(1), qml.PauliY(0), qml.PauliX(1)]

    def f1(params, t):
        return params * t

    def f2(params, t):
        return params * jnp.cos(t)

    coeffs = [f1, f2, 4, 9]
    return qml.pulse.ParametrizedHamiltonian(coeffs, ops)


@pytest.mark.jax
class TestApplyParametrizedEvolution:
    """Test that apply_operation works with ParametrizedEvolution"""

    @pytest.mark.parametrize("method", methods)
    def test_parametrized_evolution_time_independent(self, method):
        """Test that applying a ParametrizedEvolution gives the expected state
        for a time-independent hamiltonian"""

        import jax.numpy as jnp

        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )

        H = time_independent_hamiltonian()
        params = jnp.array([1.0, 2.0])
        t = 0.4

        op = qml.pulse.ParametrizedEvolution(H=H, params=params, t=t)

        true_mat = qml.math.expm(-1j * qml.matrix(H(params, t=t)) * t)
        U = qml.QubitUnitary(U=true_mat, wires=[0, 1])

        new_state = method(op, initial_state)
        new_state_expected = apply_operation(U, initial_state)

        assert np.allclose(new_state, new_state_expected, atol=0.002)

    @pytest.mark.parametrize("method", methods)
    def test_parametrized_evolution_time_dependent(self, method):
        """Test that applying a ParametrizedEvolution gives the expected state
        for a time dependent Hamiltonian"""

        import jax
        import jax.numpy as jnp

        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )

        H = time_dependent_hamiltonian()
        params = jnp.array([1.0, 2.0])
        t = 0.4

        op = qml.pulse.ParametrizedEvolution(H=H, params=params, t=t)

        def generator(params):
            time_step = 1e-3
            times = jnp.arange(0, t, step=time_step)
            for ti in times:
                yield jax.scipy.linalg.expm(-1j * time_step * qml.matrix(H(params, t=ti)))

        true_mat = reduce(lambda x, y: y @ x, generator(params))
        U = qml.QubitUnitary(U=true_mat, wires=[0, 1])

        new_state = method(op, initial_state)
        new_state_expected = apply_operation(U, initial_state)

        assert np.allclose(new_state, new_state_expected, atol=0.002)

    def test_large_state_small_matrix_evolves_matrix(self, mocker):
        """Test that applying a ParametrizedEvolution operating on less
        than half of the wires in the state uses the default function to evolve
        the matrix"""

        import jax.numpy as jnp

        spy = mocker.spy(qml.math, "einsum")

        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )

        H = time_independent_hamiltonian()
        params = jnp.array([1.0, 2.0])
        t = 0.4

        op = qml.pulse.ParametrizedEvolution(H=H, params=params, t=t)

        true_mat = qml.math.expm(-1j * qml.matrix(H(params, t=t)) * t)
        U = qml.QubitUnitary(U=true_mat, wires=[0, 1])

        new_state = apply_operation(op, initial_state)
        new_state_expected = apply_operation(U, initial_state)

        assert np.allclose(new_state, new_state_expected, atol=0.002)

        # seems like _evolve_state_vector_under_parametrized_evolution calls
        # einsum twice, and the default apply_operation only once
        # and it seems that getting the matrix from the hamiltonian calls einsum a few times.
        assert spy.call_count == 6

    def test_small_evolves_state(self, mocker):
        """Test that applying a ParametrizedEvolution operating on less
        than half of the wires in the state uses the default function to evolve
        the matrix"""

        import jax.numpy as jnp

        spy = mocker.spy(qml.math, "einsum")

        initial_state = np.array(
            [
                [
                    [
                        [
                            [-0.02018048 + 0.0j, 0.0 + 0.05690523j],
                            [0.0 + 0.01425524j, 0.04019714 + 0.0j],
                        ],
                        [
                            [0.0 - 0.07174284j, -0.20230159 + 0.0j],
                            [-0.05067824 + 0.0j, 0.0 + 0.14290331j],
                        ],
                    ],
                    [
                        [
                            [0.0 + 0.05690523j, 0.16046226 + 0.0j],
                            [0.04019714 + 0.0j, 0.0 - 0.11334853j],
                        ],
                        [
                            [-0.20230159 + 0.0j, 0.0 + 0.57045322j],
                            [0.0 + 0.14290331j, 0.402961 + 0.0j],
                        ],
                    ],
                ],
                [
                    [
                        [
                            [0.0 + 0.01425524j, 0.04019714 + 0.0j],
                            [0.01006972 + 0.0j, 0.0 - 0.02839476j],
                        ],
                        [
                            [-0.05067824 + 0.0j, 0.0 + 0.14290331j],
                            [0.0 + 0.03579848j, 0.10094511 + 0.0j],
                        ],
                    ],
                    [
                        [
                            [0.04019714 + 0.0j, 0.0 - 0.11334853j],
                            [0.0 - 0.02839476j, -0.08006798 + 0.0j],
                        ],
                        [
                            [0.0 + 0.14290331j, 0.402961 + 0.0j],
                            [0.10094511 + 0.0j, 0.0 - 0.2846466j],
                        ],
                    ],
                ],
            ]
        )

        H = time_independent_hamiltonian()
        params = jnp.array([1.0, 2.0])
        t = 0.4

        op = qml.pulse.ParametrizedEvolution(H=H, params=params, t=t)

        true_mat = qml.math.expm(-1j * qml.matrix(H(params, t=t)) * t)
        U = qml.QubitUnitary(U=true_mat, wires=[0, 1])

        new_state = apply_operation(op, initial_state)
        new_state_expected = apply_operation(U, initial_state)

        assert np.allclose(new_state, new_state_expected, atol=0.002)

        # seems like _evolve_state_vector_under_parametrized_evolution calls
        # einsum twice, and the default apply_operation only once
        # and it seems that getting the matrix from the hamiltonian calls einsum a few times.
        assert spy.call_count == 7

    def test_parametrized_evolution_raises_error(self):
        """Test applying a ParametrizedEvolution without params or t specified raises an error."""
        import jax.numpy as jnp

        state = jnp.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=complex)
        ev = qml.evolve(qml.pulse.ParametrizedHamiltonian([1], [qml.PauliX("a")]))
        with pytest.raises(
            ValueError,
            match="The parameters and the time window are required to compute the matrix",
        ):
            apply_operation(ev, state)

    def test_parametrized_evolution_state_vector_return_intermediate(self, mocker):
        """Test that when executing a ParametrizedEvolution with ``num_wires >= device.num_wires/2``
        and ``return_intermediate=True``, the ``_evolve_state_vector_under_parametrized_evolution``
        method is used."""
        import jax.numpy as jnp

        H = qml.pulse.ParametrizedHamiltonian([1], [qml.PauliX(0)])
        spy = mocker.spy(qml.math, "einsum")

        phi = jnp.linspace(0.3, 0.7, 7)
        phi_for_RX = phi - phi[0]
        state = jnp.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]], dtype=complex)
        ev = qml.evolve(H, return_intermediate=True)(params=[], t=phi / 2)
        state_ev = apply_operation(ev, state)
        state_rx = apply_operation(qml.RX(phi_for_RX, 0), state)

        assert spy.call_count == 2
        assert qml.math.allclose(state_ev, state_rx, atol=1e-6)

    @pytest.mark.parametrize("num_state_wires", [2, 4])
    def test_with_batched_state(self, num_state_wires, mocker):
        """Test that a ParametrizedEvolution is applied correctly to a batched state.
        Note that the branching logic is different for batched input states, because
        evolving the state vector does not support batching of the state. Instead,
        the evolved matrix is used always."""
        spy_einsum = mocker.spy(qml.math, "einsum")
        H = time_independent_hamiltonian()
        params = np.array([1.0, 2.0])
        t = 0.1

        op = qml.pulse.ParametrizedEvolution(H=H, params=params, t=t)

        initial_state = np.array(
            [
                [[0.81677345 + 0.0j, 0.0 + 0.0j], [0.0 - 0.57695852j, 0.0 + 0.0j]],
                [[0.33894597 + 0.0j, 0.0 + 0.0j], [0.0 - 0.94080584j, 0.0 + 0.0j]],
                [[0.33894597 + 0.0j, 0.0 + 0.0j], [0.0 - 0.94080584j, 0.0 + 0.0j]],
            ]
        )
        if num_state_wires == 4:
            zero_state_two_wires = np.eye(4)[0].reshape((2, 2))
            initial_state = np.tensordot(initial_state, zero_state_two_wires, axes=0)

        true_mat = qml.math.expm(-1j * qml.matrix(H(params, t=t)) * t)
        U = qml.QubitUnitary(U=true_mat, wires=[0, 1])

        new_state = apply_operation(op, initial_state, is_state_batched=True)
        new_state_expected = apply_operation(U, initial_state, is_state_batched=True)
        assert np.allclose(new_state, new_state_expected, atol=0.002)

        if num_state_wires == 4:
            # and it seems that getting the matrix from the hamiltonian calls einsum a few times.
            assert spy_einsum.call_count == 7
        else:
            # and it seems that getting the matrix from the hamiltonian calls einsum a few times.
            assert spy_einsum.call_count == 6


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestSnapshot:
    """Test that apply_operation works for Snapshot ops"""

    def test_no_debugger(self, ml_framework):
        """Test nothing happens when there is no debugger"""
        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)
        new_state = apply_operation(qml.Snapshot(), initial_state)

        assert new_state.shape == initial_state.shape
        assert qml.math.allclose(new_state, initial_state)

    def test_empty_tag(self, ml_framework):
        """Test a snapshot is recorded properly when there is no tag"""
        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        debugger = Debugger()
        new_state = apply_operation(qml.Snapshot(), initial_state, debugger=debugger)

        assert new_state.shape == initial_state.shape
        assert qml.math.allclose(new_state, initial_state)

        assert list(debugger.snapshots.keys()) == [0]
        assert debugger.snapshots[0].shape == (4,)
        assert qml.math.allclose(debugger.snapshots[0], qml.math.flatten(initial_state))

    def test_provided_tag(self, ml_framework):
        """Test a snapshot is recorded property when provided a tag"""
        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)

        debugger = Debugger()
        tag = "abcd"
        new_state = apply_operation(qml.Snapshot(tag), initial_state, debugger=debugger)

        assert new_state.shape == initial_state.shape
        assert qml.math.allclose(new_state, initial_state)

        assert list(debugger.snapshots.keys()) == [tag]
        assert debugger.snapshots[tag].shape == (4,)
        assert qml.math.allclose(debugger.snapshots[tag], qml.math.flatten(initial_state))

    def test_measurement(self, ml_framework):
        """Test that an arbitrary measurement is recorded properly when a snapshot is created"""
        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )
        initial_state = qml.math.asarray(initial_state, like=ml_framework)
        measurement = qml.expval(qml.PauliZ(0))

        debugger = Debugger()
        new_state = apply_operation(
            qml.Snapshot(measurement=measurement), initial_state, debugger=debugger
        )

        assert new_state.shape == initial_state.shape
        assert qml.math.allclose(new_state, initial_state)

        assert list(debugger.snapshots.keys()) == [0]
        assert debugger.snapshots[0].shape == ()
        assert debugger.snapshots[0] == qml.devices.qubit.measure(measurement, initial_state)

    def test_override_shots(self, ml_framework):
        """Test that shots can be overridden for one measurement."""

        initial_state = qml.math.asarray(np.array([1.0, 0.0]), like=ml_framework)

        debugger = Debugger()
        op = qml.Snapshot("tag", qml.sample(wires=0), shots=50)
        _ = apply_operation(op, initial_state, debugger=debugger)

        assert debugger.snapshots["tag"].shape == (50, 1)

    def test_batched_state(self, ml_framework):
        """Test that batched states create batched snapshots."""
        initial_state = qml.math.asarray([[1.0, 0.0], [0.0, 0.1]], like=ml_framework)
        debugger = Debugger()
        new_state = apply_operation(
            qml.Snapshot(), initial_state, is_state_batched=True, debugger=debugger
        )
        assert new_state.shape == initial_state.shape
        assert set(debugger.snapshots) == {0}
        assert np.array_equal(debugger.snapshots[0], initial_state)


@pytest.mark.parametrize("method", methods)
class TestRXCalcGrad:
    """Tests the application and differentiation of an RX gate in the different interfaces."""

    state = np.array(
        [
            [
                [-0.22209168 + 0.21687383j, -0.1302055 - 0.06014422j],
                [-0.24033117 + 0.28282153j, -0.14025702 - 0.13125938j],
            ],
            [
                [-0.42373896 + 0.51912421j, -0.01934135 + 0.07422255j],
                [0.22311677 + 0.2245953j, 0.33154166 + 0.20820744j],
            ],
        ]
    )

    def compare_expected_result(self, phi, state, new_state, g):
        """Compares the new state against the expected state"""
        expected0 = np.cos(phi / 2) * state[0, :, :] + -1j * np.sin(phi / 2) * state[1, :, :]
        expected1 = -1j * np.sin(phi / 2) * state[0, :, :] + np.cos(phi / 2) * state[1, :, :]

        assert qml.math.allclose(new_state[0, :, :], expected0)
        assert qml.math.allclose(new_state[1, :, :], expected1)

        g_expected0 = (
            -0.5 * np.sin(phi / 2) * state[0, :, :] - 0.5j * np.cos(phi / 2) * state[1, :, :]
        )
        g_expected1 = (
            -0.5j * np.cos(phi / 2) * state[0, :, :] - 0.5 * np.sin(phi / 2) * state[1, :, :]
        )

        assert qml.math.allclose(g[0], g_expected0)
        assert qml.math.allclose(g[1], g_expected1)

    @pytest.mark.autograd
    def test_rx_grad_autograd(self, method):
        """Test that the application of an rx gate is differentiable with autograd."""

        state = qml.numpy.array(self.state)

        def f(phi):
            op = qml.RX(phi, wires=0)
            return method(op, state)

        phi = qml.numpy.array(0.325 + 0j, requires_grad=True)

        new_state = f(phi)
        g = qml.jacobian(lambda x: qml.math.real(f(x)))(phi)
        self.compare_expected_result(phi, state, new_state, g)

    @pytest.mark.jax
    @pytest.mark.parametrize("use_jit", (True, False))
    def test_rx_grad_jax(self, method, use_jit):
        """Test that the application of an rx gate is differentiable with jax."""

        import jax

        state = jax.numpy.array(self.state)

        def f(phi):
            op = qml.RX(phi, wires=0)
            return method(op, state)

        if use_jit:
            f = jax.jit(f)

        phi = 0.325

        new_state = f(phi)
        g = jax.jacobian(f, holomorphic=True)(phi + 0j)
        self.compare_expected_result(phi, state, new_state, g)

    @pytest.mark.torch
    def test_rx_grad_torch(self, method):
        """Tests the application and differentiation of an rx gate with torch."""

        import torch

        state = torch.tensor(self.state)

        def f(phi):
            op = qml.RX(phi, wires=0)
            return method(op, state)

        phi = torch.tensor(0.325, requires_grad=True)

        new_state = f(phi)
        # forward-mode needed with complex results.
        # See bug: https://github.com/pytorch/pytorch/issues/94397
        g = torch.autograd.functional.jacobian(f, phi + 0j, strategy="forward-mode", vectorize=True)

        self.compare_expected_result(
            phi.detach().numpy(),
            state.detach().numpy(),
            new_state.detach().numpy(),
            g.detach().numpy(),
        )

    @pytest.mark.tf
    def test_rx_grad_tf(self, method):
        """Tests the application and differentiation of an rx gate with tensorflow"""
        import tensorflow as tf

        state = tf.Variable(self.state)
        phi = tf.Variable(0.8589 + 0j)

        with tf.GradientTape() as grad_tape:
            op = qml.RX(phi, wires=0)
            new_state = method(op, state)

        grads = grad_tape.jacobian(new_state, [phi])
        # tf takes gradient with respect to conj(z), so we need to conj the gradient
        phi_grad = tf.math.conj(grads[0])

        self.compare_expected_result(phi, state, new_state, phi_grad)


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
@pytest.mark.parametrize("method", methods)
class TestBroadcasting:  # pylint: disable=too-few-public-methods
    """Tests that broadcasted operations are applied correctly."""

    # include operations both with batch_size==1 and batch_size>1
    broadcasted_ops = [
        qml.RX(np.array([np.pi]), wires=2),
        qml.RX(np.array([np.pi, np.pi / 2, np.pi / 4]), wires=2),
        qml.PhaseShift(np.array([np.pi]), wires=2),
        qml.PhaseShift(np.array([np.pi, np.pi / 2, np.pi / 4]), wires=2),
        qml.IsingXX(np.array([np.pi]), wires=[1, 2]),
        qml.IsingXX(np.array([np.pi, np.pi / 2, np.pi / 4]), wires=[1, 2]),
        qml.QubitUnitary(
            np.array([unitary_group.rvs(8)]),
            wires=[0, 1, 2],
        ),
        qml.QubitUnitary(
            np.array([unitary_group.rvs(8), unitary_group.rvs(8), unitary_group.rvs(8)]),
            wires=[0, 1, 2],
        ),
    ]

    unbroadcasted_ops = [
        qml.PauliX(2),
        qml.PauliZ(2),
        qml.CNOT([1, 2]),
        qml.RX(np.pi, wires=2),
        qml.PhaseShift(np.pi / 2, wires=2),
        qml.IsingXX(np.pi / 2, wires=[1, 2]),
        qml.QubitUnitary(unitary_group.rvs(8), wires=[0, 1, 2]),
    ]

    @pytest.mark.parametrize("op", broadcasted_ops)
    def test_broadcasted_op(self, op, method, ml_framework):
        """Tests that batched operations are applied correctly to an unbatched state."""
        state = np.ones((2, 2, 2), dtype=complex) / np.sqrt(8)

        res = method(op, qml.math.asarray(state, like=ml_framework))
        missing_wires = 3 - len(op.wires)
        mat = op.matrix()
        expanded_mat = (
            [np.kron(np.eye(2**missing_wires), mat[i]) for i in range(op.batch_size)]
            if missing_wires
            else [mat[i] for i in range(op.batch_size)]
        )
        expected = [
            (expanded_mat[i] @ state.flatten()).reshape((2, 2, 2)) for i in range(op.batch_size)
        ]

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize("op", unbroadcasted_ops)
    def test_broadcasted_state(self, op, method, ml_framework):
        """Tests that unbatched operations are applied correctly to a batched state."""
        state = np.ones((3, 2, 2, 2), dtype=complex) / np.sqrt(8)

        res = method(op, qml.math.asarray(state, like=ml_framework), is_state_batched=True)
        missing_wires = 3 - len(op.wires)
        mat = op.matrix()
        expanded_mat = np.kron(np.eye(2**missing_wires), mat) if missing_wires else mat
        expected = [(expanded_mat @ state[i].flatten()).reshape((2, 2, 2)) for i in range(3)]

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize("op", broadcasted_ops)
    def test_broadcasted_op_broadcasted_state(self, op, method, ml_framework):
        """Tests that batched operations are applied correctly to a batched state."""
        if method is apply_operation_tensordot:
            pytest.skip("Tensordot doesn't support batched operator and batched state.")

        state = np.ones((3, 2, 2, 2), dtype=complex) / np.sqrt(8)

        res = method(op, qml.math.asarray(state, like=ml_framework), is_state_batched=True)
        missing_wires = 3 - len(op.wires)
        mat = op.matrix()
        expanded_mat = (
            [np.kron(np.eye(2**missing_wires), mat[i]) for i in range(op.batch_size)]
            if missing_wires
            else [mat[i] for i in range(op.batch_size)]
        )
        expected = [
            (expanded_mat[i] @ state[i].flatten()).reshape((2, 2, 2)) for i in range(op.batch_size)
        ]

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    def test_batch_size_set_if_missing(self, method, ml_framework):
        """Tests that the batch_size is set on an operator if it was missing before.
        Mostly useful for TF-autograph since it may have batch size set to None."""
        param = qml.math.asarray([0.1, 0.2, 0.3], like=ml_framework)
        state = np.ones((2, 2)) / 2
        op = qml.RX(param, 0)
        assert op._batch_size is _UNSET_BATCH_SIZE  # pylint:disable=protected-access
        state = method(op, state)
        assert state.shape == (3, 2, 2)


@pytest.mark.parametrize("method", methods)
class TestLargerOperations:
    """Tests matrix applications on states and operations with larger numbers of wires."""

    state = np.array(
        [
            [
                [
                    [-0.21733955 - 0.01990267j, 0.22960893 - 0.0312392j],
                    [0.21406652 - 0.07552019j, 0.09527143 + 0.01870987j],
                ],
                [
                    [0.05603182 - 0.26879067j, -0.02755183 - 0.03097822j],
                    [-0.43962358 - 0.17435254j, 0.12820737 + 0.06794554j],
                ],
            ],
            [
                [
                    [-0.09270161 - 0.3132961j, -0.03276799 + 0.07557535j],
                    [-0.15712707 - 0.32666969j, -0.00898954 + 0.1324474j],
                ],
                [
                    [-0.17760532 + 0.08415488j, -0.26872752 - 0.05767781j],
                    [0.23142582 - 0.1970496j, 0.15483611 - 0.15100495j],
                ],
            ],
        ]
    )

    @pytest.mark.parametrize("control_values", [[1, 1, 1], [0, 1, 0], None, [1, 0, 0]])
    def test_multicontrolledx(self, method, control_values):
        """Tests a four qubit multi-controlled x gate."""

        op = qml.MultiControlledX(wires=(0, 1, 2, 3), control_values=control_values)
        new_state = method(op, self.state)

        expected_state = np.copy(self.state)
        if control_values is None:
            values = (1, 1, 1)
        else:
            values = tuple(map(int, control_values))
        expected_state[values + (1,)] = self.state[values + (0,)]
        expected_state[values + (0,)] = self.state[values + (1,)]

        assert qml.math.allclose(new_state, expected_state)

    def test_double_excitation(self, method):
        """Tests a double excitation operation compared to its decomposition."""

        op = qml.DoubleExcitation(np.array(2.14), wires=(3, 1, 2, 0))

        state_v1 = method(op, self.state)

        state_v2 = self.state
        for d_op in op.decomposition():
            state_v2 = method(d_op, state_v2)

        assert qml.math.allclose(state_v1, state_v2)

    @pytest.mark.parametrize("apply_wires", ([0, 3], [0, 1, 3, 2], [2, 1], [1, 3]))
    def test_grover(self, method, apply_wires):
        """Tests a four qubit GroverOperator."""
        op = qml.GroverOperator(apply_wires)
        new_state = method(op, self.state)

        expected_state = self.state
        for _op in op.decomposition():
            expected_state = method(_op, expected_state)

        assert qml.math.allclose(expected_state, new_state)


class TestApplyGroverOperator:
    """Test that GroverOperator is applied correctly."""

    def grover_kernel_full_wires(self, state, op_wires, batched):
        """Additional kernel to apply GroverOperator to all state wires."""
        prefactor = 2 ** (1 - len(op_wires))
        sum_axes = tuple(range(batched, np.ndim(state)))
        collapsed = np.sum(state, axis=sum_axes)
        return prefactor * np.expand_dims(collapsed, sum_axes) - state

    def grover_kernel_partial_wires(self, state, op_wires, batched):
        """Additional kernel to apply GroverOperator to some of all state wires."""
        num_wires = len(op_wires)
        sum_axes = [w + batched for w in op_wires]
        collapsed = np.sum(state, tuple(sum_axes))
        prefactor = 2 ** (1 - num_wires)
        bcast_shape = [2] * num_wires + list(state.shape[:-num_wires])
        expanded = np.broadcast_to(prefactor * collapsed, bcast_shape)
        source = list(range(num_wires))
        expanded = np.moveaxis(expanded, source, sum_axes)
        return expanded - state

    @pytest.mark.parametrize(
        "op_wires, state_wires, einsum_called, tensordot_called",
        [
            (2, 2, True, False),
            (3, 3, False, True),
            (9, 9, False, False),
            (2, 13, False, True),
            (3, 9, False, True),
            (9, 13, False, False),
        ],
    )
    def test_dispatching(self, op_wires, state_wires, einsum_called, tensordot_called, mocker):
        """Test that apply_operation dispatches to einsum, tensordot and the kernel correctly."""
        # pylint: disable=too-many-arguments
        state = np.random.random([2] * state_wires) + 1j * np.random.random([2] * state_wires)

        op = qml.GroverOperator(list(range(op_wires)))
        spy_einsum = mocker.spy(qml.math, "einsum")
        spy_tensordot = mocker.spy(qml.math, "argsort")
        apply_operation(op, state, is_state_batched=False, debugger=None)
        assert spy_einsum.call_count == int(einsum_called)
        assert spy_tensordot.call_count == int(tensordot_called)

    @pytest.mark.parametrize("op_wires, state_wires", [(2, 2), (3, 3), (9, 9)])
    @pytest.mark.parametrize("batch_dim", [None, 1, 3])
    def test_correctness_full_wires(self, op_wires, state_wires, batch_dim):
        """Test that apply_operation is correct for GroverOperator for all dispatch branches
        when applying it to all wires of a state."""
        batched = batch_dim is not None
        shape = [batch_dim] + [2] * state_wires if batched else [2] * state_wires
        flat_shape = (batch_dim, 2**state_wires) if batched else (2**state_wires,)
        state = np.random.random(shape) + 1j * np.random.random(shape)

        op = qml.GroverOperator(list(range(op_wires)))
        out = apply_operation(op, state, is_state_batched=batched, debugger=None)
        # Double transpose to accomodate for batching
        expected_via_mat = (op.matrix() @ state.reshape(flat_shape).T).T.reshape(shape)
        expected_via_kernel = self.grover_kernel_full_wires(state, op.wires, batched)
        assert np.allclose(out, expected_via_mat)
        assert np.allclose(out, expected_via_kernel)

    @pytest.mark.parametrize("op_wires, state_wires", [(3, 5), (9, 13)])
    @pytest.mark.parametrize("batch_dim", [None, 1, 3])
    def test_correctness_partial_wires(self, op_wires, state_wires, batch_dim):
        """Test that apply_operation is correct for GroverOperator for all dispatch branches
        but einsum (because Grover can't act on a single wire)
        when applying it only to some of the wires of a state."""
        batched = batch_dim is not None
        shape = [batch_dim] + [2] * state_wires if batched else [2] * state_wires
        state = np.random.random(shape) + 1j * np.random.random(shape)

        for start_wire in [0, 1, state_wires - op_wires]:
            wires = list(range(start_wire, start_wire + op_wires))
            op = qml.GroverOperator(wires)
            out = apply_operation(op, state, is_state_batched=batched, debugger=None)
            expected_via_mat = apply_operation_tensordot(op, state, batched)
            expected_via_kernel = self.grover_kernel_partial_wires(state, wires, batched)
            assert np.allclose(out, expected_via_mat)
            assert np.allclose(out, expected_via_kernel)

    @pytest.mark.autograd
    @pytest.mark.parametrize("op_wires, state_wires", [(2, 2), (3, 3), (9, 9), (3, 5), (9, 13)])
    @pytest.mark.parametrize("batch_dim", [None, 1, 3])
    def test_correctness_autograd(self, op_wires, state_wires, batch_dim):
        """Test that apply_operation is correct for GroverOperator for all dispatch branches
        when applying it to an Autograd state."""
        batched = batch_dim is not None
        shape = [batch_dim] + [2] * state_wires if batched else [2] * state_wires
        # Input state
        state = np.random.random(shape) + 1j * np.random.random(shape)

        wires = list(range(op_wires))
        op = qml.GroverOperator(wires)
        expected_via_mat = apply_operation_tensordot(op, state, batched)
        if op_wires == state_wires:
            expected_via_kernel = self.grover_kernel_full_wires(state, wires, batched)
        else:
            expected_via_kernel = self.grover_kernel_partial_wires(state, wires, batched)

        # Cast to interface and apply operation
        state = qml.numpy.array(state)
        out = apply_operation(op, state, is_state_batched=batched, debugger=None)

        assert qml.math.allclose(out, expected_via_mat)
        assert qml.math.allclose(out, expected_via_kernel)

    @pytest.mark.tf
    @pytest.mark.parametrize("op_wires, state_wires", [(2, 2), (3, 3), (9, 9), (3, 5), (9, 13)])
    @pytest.mark.parametrize("batch_dim", [None, 1, 3])
    def test_correctness_tf(self, op_wires, state_wires, batch_dim):
        """Test that apply_operation is correct for GroverOperator for all dispatch branches
        when applying it to a Tensorflow state."""
        import tensorflow as tf

        batched = batch_dim is not None
        shape = [batch_dim] + [2] * state_wires if batched else [2] * state_wires
        # Input state
        state = np.random.random(shape) + 1j * np.random.random(shape)

        wires = list(range(op_wires))
        op = qml.GroverOperator(wires)
        expected_via_mat = apply_operation_tensordot(op, state, batched)
        if op_wires == state_wires:
            expected_via_kernel = self.grover_kernel_full_wires(state, wires, batched)
        else:
            expected_via_kernel = self.grover_kernel_partial_wires(state, wires, batched)

        # Cast to interface and apply operation
        state = tf.Variable(state)
        out = apply_operation(op, state, is_state_batched=batched, debugger=None)

        assert qml.math.allclose(out, expected_via_mat)
        assert qml.math.allclose(out, expected_via_kernel)

    @pytest.mark.jax
    @pytest.mark.parametrize("op_wires, state_wires", [(2, 2), (3, 3), (9, 9), (3, 5), (9, 13)])
    @pytest.mark.parametrize("batch_dim", [None, 1, 3])
    def test_correctness_jax(self, op_wires, state_wires, batch_dim):
        """Test that apply_operation is correct for GroverOperator for all dispatch branches
        when applying it to a Jax state."""
        import jax

        jax.config.update("jax_enable_x64", True)

        batched = batch_dim is not None
        shape = [batch_dim] + [2] * state_wires if batched else [2] * state_wires
        # Input state
        state = np.random.random(shape) + 1j * np.random.random(shape)

        wires = list(range(op_wires))
        op = qml.GroverOperator(wires)
        expected_via_mat = apply_operation_tensordot(op, state, batched)
        if op_wires == state_wires:
            expected_via_kernel = self.grover_kernel_full_wires(state, wires, batched)
        else:
            expected_via_kernel = self.grover_kernel_partial_wires(state, wires, batched)

        # Cast to interface and apply operation
        state = jax.numpy.array(state)
        out = apply_operation(op, state, is_state_batched=batched, debugger=None)

        assert qml.math.allclose(out, expected_via_mat)
        assert qml.math.allclose(out, expected_via_kernel)

    @pytest.mark.torch
    @pytest.mark.parametrize("op_wires, state_wires", [(2, 2), (3, 3), (9, 9), (3, 5), (9, 13)])
    @pytest.mark.parametrize("batch_dim", [None, 1, 3])
    def test_correctness_torch(self, op_wires, state_wires, batch_dim):
        """Test that apply_operation is correct for GroverOperator for all dispatch branches
        when applying it to a Torch state."""
        import torch

        batched = batch_dim is not None
        shape = [batch_dim] + [2] * state_wires if batched else [2] * state_wires
        # Input state
        state = np.random.random(shape) + 1j * np.random.random(shape)

        wires = list(range(op_wires))
        op = qml.GroverOperator(wires)
        expected_via_mat = apply_operation_tensordot(op, state, batched)
        if op_wires == state_wires:
            expected_via_kernel = self.grover_kernel_full_wires(state, wires, batched)
        else:
            expected_via_kernel = self.grover_kernel_partial_wires(state, wires, batched)

        # Cast to interface and apply operation
        state = torch.tensor(state, requires_grad=True)
        out = apply_operation(op, state, is_state_batched=batched, debugger=None)

        assert qml.math.allclose(out, expected_via_mat)
        assert qml.math.allclose(out, expected_via_kernel)


class TestMultiControlledXKernel:
    """Test the specialized kernel for MultiControlledX and its dispatching."""

    # pylint: disable=too-many-arguments
    @pytest.mark.parametrize(
        "num_op_wires, num_state_wires, einsum_called, tdot_called",
        [
            # state small and matrix huge -> not possible because num_op_wires<=num_state_wires
            # matrix large -> kernel
            (9, 9, 0, 0),
            # matrix large, state huge -> still kernel, not tensordot
            (9, 9, 0, 0),
            # matrix tiny, state not huge -> einsum
            (2, 12, 1, 0),
            # matrix small, state not huge -> tensordot
            (5, 12, 0, 1),
            # matrix tiny, state huge -> tensordot
            (2, 13, 0, 1),
            # matrix small, state huge -> tensordot
            (5, 13, 0, 1),
        ],
    )
    def test_multicontrolledx_dispatching(
        self, num_op_wires, num_state_wires, einsum_called, tdot_called, mocker
    ):
        """Test that apply_multicontrolledx dispatches to the right method and is correct."""
        op = qml.MultiControlledX(wires=list(range(num_op_wires)))
        state = np.random.random([2] * num_state_wires).astype(complex)
        spies = [mocker.spy(qml.math, "einsum"), mocker.spy(qml.math, "tensordot")]
        out = apply_operation(op, state, is_state_batched=False, debugger=None)
        # Compute expected output
        exp_out = state.copy()
        idx = (1,) * (num_op_wires - 1)
        exp_out[idx] = np.roll(exp_out[idx], 1, 0)
        assert spies[0].call_count == einsum_called
        assert spies[1].call_count == tdot_called
        assert np.allclose(out, exp_out)

    @pytest.mark.jax
    @pytest.mark.parametrize("batch_dim", [None, 1, 3])
    def test_with_jax(self, batch_dim):
        """Test that the custom kernel works with JAX."""
        from jax import numpy as jnp

        op = qml.MultiControlledX(wires=[0, 4, 3, 1])
        state_shape = ([batch_dim] if batch_dim is not None else []) + [2] * 5
        state = np.random.random(state_shape).astype(complex)
        jax_state = jnp.array(state)
        out = apply_operation(op, jax_state, is_state_batched=batch_dim is not None, debugger=None)
        # Compute expected output
        exp_out = state.copy()
        exp_out[..., 1, :, :, 1, 1] = np.roll(exp_out[..., 1, :, :, 1, 1], 1, -2)
        assert qml.math.allclose(out, exp_out)

    @pytest.mark.tf
    @pytest.mark.parametrize("batch_dim", [None, 1, 3])
    def test_with_tf(self, batch_dim):
        """Test that the custom kernel works with Tensorflow."""
        import tensorflow as tf

        op = qml.MultiControlledX(wires=[0, 4, 3, 1])
        state_shape = ([batch_dim] if batch_dim is not None else []) + [2] * 5
        state = np.random.random(state_shape).astype(complex)
        tf_state = tf.Variable(state)
        out = apply_operation(op, tf_state, is_state_batched=batch_dim is not None, debugger=None)
        # Compute expected output
        exp_out = state.copy()
        exp_out[..., 1, :, :, 1, 1] = np.roll(exp_out[..., 1, :, :, 1, 1], 1, -2)
        assert qml.math.allclose(out, exp_out)

    @pytest.mark.autograd
    @pytest.mark.parametrize("batch_dim", [None, 1, 3])
    def test_with_autograd(self, batch_dim):
        """Test that the custom kernel works with Autograd."""
        op = qml.MultiControlledX(wires=[0, 4, 3, 1])
        state_shape = ([batch_dim] if batch_dim is not None else []) + [2] * 5
        state = np.random.random(state_shape).astype(complex)
        ag_state = qml.numpy.array(state)
        out = apply_operation(op, ag_state, is_state_batched=batch_dim is not None, debugger=None)
        # Compute expected output
        exp_out = state.copy()
        exp_out[..., 1, :, :, 1, 1] = np.roll(exp_out[..., 1, :, :, 1, 1], 1, -2)
        assert qml.math.allclose(out, exp_out)

    @pytest.mark.torch
    @pytest.mark.parametrize("batch_dim", [None, 1, 3])
    def test_with_torch(self, batch_dim):
        """Test that the custom kernel works with Torch."""
        import torch

        op = qml.MultiControlledX(wires=[0, 4, 3, 1])
        state_shape = ([batch_dim] if batch_dim is not None else []) + [2] * 5
        state = np.random.random(state_shape).astype(complex)
        torch_state = torch.tensor(state, requires_grad=True)
        out = apply_operation(
            op, torch_state, is_state_batched=batch_dim is not None, debugger=None
        )
        # Compute expected output
        exp_out = state.copy()
        exp_out[..., 1, :, :, 1, 1] = np.roll(exp_out[..., 1, :, :, 1, 1], 1, -2)
        assert qml.math.allclose(out, exp_out)


@pytest.mark.tf
class TestLargeTFCornerCases:
    """Test large corner cases for tensorflow."""

    @pytest.mark.parametrize(
        "op", (qml.PauliZ(8), qml.PhaseShift(1.0, 8), qml.S(8), qml.T(8), qml.CNOT((5, 6)))
    )
    def test_tf_large_state(self, op):
        """Tests that custom kernels that use slicing fall back to a different method when
        the state has a large number of wires."""
        import tensorflow as tf

        state = np.zeros([2] * 10, dtype=complex)
        state = tf.Variable(state)
        new_state = apply_operation(op, state)

        # still all zeros.  Mostly just making sure error not raised
        assert qml.math.allclose(state, new_state)

    def test_cnot_large_batched_state_tf(self):
        """Test that CNOT with large batched states works as expected."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=8)

        @qml.qnode(dev, interface="tf")
        def auxiliary_qcnn_circuit(inputs):
            qml.AmplitudeEmbedding(features=inputs, wires=range(4), normalize=True)
            qml.CNOT(wires=[0, 1])
            qml.PauliZ(1)
            qml.Toffoli(wires=[0, 2, 4])
            qml.Toffoli(wires=[0, 2, 5])
            qml.Toffoli(wires=[0, 2, 6])
            qml.Toffoli(wires=[0, 2, 7])
            return [qml.expval(qml.PauliZ(i)) for i in range(4, 8)]

        batch_size = 3
        params = np.random.rand(batch_size, 16)
        result = auxiliary_qcnn_circuit(tf.Variable(params))
        assert qml.math.shape(result) == (4, batch_size)

    def test_pauliz_large_batched_state_tf(self):
        """Test that PauliZ with large batched states works as expected."""
        import tensorflow as tf

        @qml.qnode(qml.device("default.qubit"), interface="tf")
        def circuit(init_state):
            qml.StatePrep(init_state, wires=range(8))
            qml.PauliX(0)
            qml.PauliZ(0)
            return qml.state()

        states = np.zeros((3, 256))
        states[:, 0] = 1.0
        results = circuit(tf.Variable(states))
        assert qml.math.shape(results) == (3, 256)
        assert np.array_equal(results[:, 128], [-1.0 + 0.0j] * 3)


# pylint: disable=too-few-public-methods
class TestConditionalsAndMidMeasure:
    """Test dispatching for mid-circuit measurements and conditionals."""

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("ml_framework", ml_frameworks_list)
    @pytest.mark.parametrize("batched", (False, True))
    @pytest.mark.parametrize("unitary", (qml.CRX, qml.CRZ))
    @pytest.mark.parametrize("wires", ([0, 1], [1, 0]))
    def test_conditional(self, wires, unitary, batched, ml_framework):
        """Test the application of a Conditional on an arbitrary state."""

        n_states = int(batched) + 1
        initial_state = np.array(
            [
                [
                    0.3541035 + 0.05231577j,
                    0.6912382 + 0.49474503j,
                    0.29276263 + 0.06231887j,
                    0.10736635 + 0.21947607j,
                ],
                [
                    0.09803567 + 0.47557068j,
                    0.4427561 + 0.13810454j,
                    0.26421703 + 0.5366283j,
                    0.03825933 + 0.4357423j,
                ],
            ][:n_states]
        )

        rotated_state = qml.math.dot(
            initial_state, qml.matrix(unitary(-0.238, wires), wire_order=[0, 1]).T
        )
        rotated_state = qml.math.asarray(rotated_state, like=ml_framework)
        rotated_state = qml.math.squeeze(qml.math.reshape(rotated_state, (n_states, 2, 2)))

        m0 = qml.measure(0)
        op = qml.ops.op_math.Conditional(m0, unitary(0.238, wires))

        mid_meas = {m0.measurements[0]: 0}
        old_state = apply_operation(
            op, rotated_state, batched, interface=ml_framework, mid_measurements=mid_meas
        )
        assert qml.math.allclose(rotated_state, old_state)

        mid_meas[m0.measurements[0]] = 1
        new_state = apply_operation(
            op, rotated_state, batched, interface=ml_framework, mid_measurements=mid_meas
        )
        assert qml.math.allclose(
            qml.math.squeeze(initial_state), qml.math.reshape(new_state, (n_states, 4))
        )

    @pytest.mark.parametrize("m_res", [(0, 0), (1, 1)])
    def test_mid_measure(self, m_res, monkeypatch):
        """Test the application of a MidMeasureMP on an arbitrary state to give a basis state."""

        initial_state = np.array(
            [
                [0.09068964 + 0.36775595j, 0.37578343 + 0.4786927j],
                [0.3537292 + 0.27214766j, 0.01928256 + 0.53536021j],
            ]
        )

        mid_state, end_state = np.zeros((2, 2), dtype=complex), np.zeros((2, 2), dtype=complex)
        mid_state[m_res[0]] = initial_state[m_res[0]] / np.linalg.norm(initial_state[m_res[0]])
        end_state[m_res] = mid_state[m_res] / np.abs(mid_state[m_res])

        m0, m1 = qml.measure(0).measurements[0], qml.measure(1).measurements[0]
        mid_meas = {}

        monkeypatch.setattr(np.random, "binomial", lambda *args: m_res[0])

        res_state = apply_operation(m0, initial_state, mid_measurements=mid_meas)
        assert qml.math.allclose(mid_state, res_state)

        res_state = apply_operation(m1, res_state, mid_measurements=mid_meas)
        assert qml.math.allclose(end_state, res_state)

        assert mid_meas == {m0: m_res[0], m1: m_res[1]}

    def test_error_bactched_mid_measure(self):
        """Test that an error is raised when mid_measure is applied to a batched input state."""

        with pytest.raises(ValueError, match="MidMeasureMP cannot be applied to batched states."):
            m0, input_state = qml.measure(0).measurements[0], qml.math.array([[1, 0], [1, 0]])
            apply_operation(m0, state=input_state, is_state_batched=True)
