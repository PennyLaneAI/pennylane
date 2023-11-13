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
import pytest

import numpy as np
from scipy.stats import unitary_group
import pennylane as qml


from pennylane.devices.qubit.apply_operation import (
    apply_operation,
    apply_operation_einsum,
    apply_operation_tensordot,
)

ml_frameworks_list = [
    "numpy",
    pytest.param("autograd", marks=pytest.mark.autograd),
    pytest.param("jax", marks=pytest.mark.jax),
    pytest.param("torch", marks=pytest.mark.torch),
    pytest.param("tensorflow", marks=pytest.mark.tf),
]


methods = [apply_operation_einsum, apply_operation_tensordot, apply_operation]


def test_custom_operator_with_matrix():
    """Test that apply_operation works with any operation that defines a matrix."""

    mat = np.array(
        [
            [0.39918205 + 0.3024376j, -0.86421077 + 0.04821758j],
            [0.73240679 + 0.46126509j, 0.49576832 - 0.07091251j],
        ]
    )

    # pylint: disable=too-few-public-methods
    class CustomOp(qml.operation.Operation):
        num_wires = 1

        def matrix(self):
            return mat

    state = np.array([-0.30688912 - 0.4768824j, 0.8100052 - 0.14931113j])

    new_state = apply_operation(CustomOp(0), state)
    assert qml.math.allclose(new_state, mat @ state)


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
        """Test the application of a cnot gate on a two qubit state."""

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


# pylint:disable = unused-argument
def time_independent_hamiltonian():
    """Create a time-independent Hamiltonian on two qubits."""
    ops = [qml.PauliX(0), qml.PauliZ(1), qml.PauliY(0), qml.PauliX(1)]

    def f1(params, t):
        return params  # constant

    def f2(params, t):
        return params  # constant

    coeffs = [f1, f2, 4, 9]

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
class TestApplyParameterizedEvolution:
    @pytest.mark.parametrize("method", methods)
    def test_parameterized_evolution_time_independent(self, method):
        """Test that applying a ParameterizedEvolution gives the expected state
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
        t = 4

        op = qml.pulse.ParametrizedEvolution(H=H, params=params, t=t)

        true_mat = qml.math.expm(-1j * qml.matrix(H(params, t=t)) * t)
        U = qml.QubitUnitary(U=true_mat, wires=[0, 1])

        new_state = method(op, initial_state)
        new_state_expected = apply_operation(U, initial_state)

        assert np.allclose(new_state, new_state_expected, atol=0.002)

    @pytest.mark.parametrize("method", methods)
    def test_parameterized_evolution_time_dependent(self, method):
        """Test that applying a ParameterizedEvolution gives the expected state
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
        t = 4

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
        """Test that applying a ParameterizedEvolution operating on less
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
        t = 4

        op = qml.pulse.ParametrizedEvolution(H=H, params=params, t=t)

        true_mat = qml.math.expm(-1j * qml.matrix(H(params, t=t)) * t)
        U = qml.QubitUnitary(U=true_mat, wires=[0, 1])

        new_state = apply_operation(op, initial_state)
        new_state_expected = apply_operation(U, initial_state)

        assert np.allclose(new_state, new_state_expected, atol=0.002)

        # seems like _evolve_state_vector_under_parametrized_evolution calls
        # einsum twice, and the default apply_operation only once
        assert spy.call_count == 1

    def test_small_evolves_state(self, mocker):
        """Test that applying a ParameterizedEvolution operating on less
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
        t = 4

        op = qml.pulse.ParametrizedEvolution(H=H, params=params, t=t)

        true_mat = qml.math.expm(-1j * qml.matrix(H(params, t=t)) * t)
        U = qml.QubitUnitary(U=true_mat, wires=[0, 1])

        new_state = apply_operation(op, initial_state)
        new_state_expected = apply_operation(U, initial_state)

        assert np.allclose(new_state, new_state_expected, atol=0.002)

        # seems like _evolve_state_vector_under_parametrized_evolution calls
        # einsum twice, and the default apply_operation only once
        assert spy.call_count == 2

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

    def test_batched_state_raises_an_error(self):
        """Test that if is_state_batche=True, an error is raised"""
        H = time_independent_hamiltonian()
        params = np.array([1.0, 2.0])
        t = 4

        op = qml.pulse.ParametrizedEvolution(H=H, params=params, t=t)

        initial_state = np.array(
            [
                [[0.81677345 + 0.0j, 0.0 + 0.0j], [0.0 - 0.57695852j, 0.0 + 0.0j]],
                [[0.33894597 + 0.0j, 0.0 + 0.0j], [0.0 - 0.94080584j, 0.0 + 0.0j]],
            ]
        )

        with pytest.raises(RuntimeError, match="does not support standard broadcasting"):
            _ = apply_operation(op, initial_state, is_state_batched=True)


@pytest.mark.parametrize("ml_framework", ml_frameworks_list)
class TestSnapshot:
    """Test that apply_operation works for Snapshot ops"""

    class Debugger:  # pylint: disable=too-few-public-methods
        """A dummy debugger class"""

        def __init__(self):
            self.active = True
            self.snapshots = {}

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

        debugger = self.Debugger()
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

        debugger = self.Debugger()
        tag = "abcd"
        new_state = apply_operation(qml.Snapshot(tag), initial_state, debugger=debugger)

        assert new_state.shape == initial_state.shape
        assert qml.math.allclose(new_state, initial_state)

        assert list(debugger.snapshots.keys()) == [tag]
        assert debugger.snapshots[tag].shape == (4,)
        assert qml.math.allclose(debugger.snapshots[tag], qml.math.flatten(initial_state))


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

    broadcasted_ops = [
        qml.RX(np.array([np.pi, np.pi / 2, np.pi / 4]), wires=2),
        qml.PhaseShift(np.array([np.pi, np.pi / 2, np.pi / 4]), wires=2),
        qml.IsingXX(np.array([np.pi, np.pi / 2, np.pi / 4]), wires=[1, 2]),
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
        state = np.ones((2, 2, 2)) / np.sqrt(8)

        res = method(op, qml.math.asarray(state, like=ml_framework))
        missing_wires = 3 - len(op.wires)
        mat = op.matrix()
        expanded_mat = [
            np.kron(np.eye(2**missing_wires), mat[i]) if missing_wires else mat[i]
            for i in range(3)
        ]
        expected = [(expanded_mat[i] @ state.flatten()).reshape((2, 2, 2)) for i in range(3)]

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    @pytest.mark.parametrize("op", unbroadcasted_ops)
    def test_broadcasted_state(self, op, method, ml_framework):
        """Tests that unbatched operations are applied correctly to a batched state."""
        state = np.ones((3, 2, 2, 2)) / np.sqrt(8)

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

        state = np.ones((3, 2, 2, 2)) / np.sqrt(8)

        res = method(op, qml.math.asarray(state, like=ml_framework), is_state_batched=True)
        missing_wires = 3 - len(op.wires)
        mat = op.matrix()
        expanded_mat = [
            np.kron(np.eye(2**missing_wires), mat[i]) if missing_wires else mat[i]
            for i in range(3)
        ]
        expected = [(expanded_mat[i] @ state[i].flatten()).reshape((2, 2, 2)) for i in range(3)]

        assert qml.math.get_interface(res) == ml_framework
        assert qml.math.allclose(res, expected)

    def test_batch_size_set_if_missing(self, method, ml_framework):
        """Tests that the batch_size is set on an operator if it was missing before.
        Mostly useful for TF-autograph since it may have batch size set to None."""
        param = qml.math.asarray([0.1, 0.2, 0.3], like=ml_framework)
        state = np.ones((2, 2)) / 2
        op = qml.RX(param, 0)
        op._batch_size = None  # pylint:disable=protected-access
        state = method(op, state)
        assert state.shape == (3, 2, 2)
        assert op.batch_size == 3


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

    def test_multicontrolledx(self, method):
        """Tests a four qubit multi-controlled x gate."""

        new_state = method(qml.MultiControlledX(wires=(0, 1, 2, 3)), self.state)

        expected_state = np.copy(self.state)
        expected_state[1, 1, 1, 1] = self.state[1, 1, 1, 0]
        expected_state[1, 1, 1, 0] = self.state[1, 1, 1, 1]

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

        expected_state = np.copy(self.state)
        for _op in op.decomposition():
            expected_state = method(_op, expected_state)

        assert qml.math.allclose(expected_state, new_state)


@pytest.mark.parametrize(
    "num_wires, einsum_called, tensordot_called",
    [(2, True, False), (3, False, True), (9, False, False)],
)
def test_grover_dispatching(num_wires, einsum_called, tensordot_called, mocker):
    """Test that apply_grover dispatches to einsum correctly for small numbers of wires."""
    op = qml.GroverOperator(list(range(num_wires)))
    state = np.zeros([2] * num_wires, dtype=complex)
    spy_einsum = mocker.spy(qml.math, "einsum")
    spy_tensordot = mocker.spy(qml.math, "argsort")
    apply_operation(op, state, is_state_batched=False, debugger=None)
    assert spy_einsum.call_count == int(einsum_called)
    assert spy_tensordot.call_count == int(tensordot_called)


@pytest.mark.tf
@pytest.mark.parametrize("op", (qml.PauliZ(8), qml.CNOT((5, 6))))
def test_tf_large_state(op):
    """Tests that custom kernels that use slicing fall back to a different method when
    the state has a large number of wires."""
    import tensorflow as tf

    state = np.zeros([2] * 10)
    state = tf.Variable(state)
    new_state = apply_operation(op, state)

    # still all zeros.  Mostly just making sure error not raised
    assert qml.math.allclose(state, new_state)
