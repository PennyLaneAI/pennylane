# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the qft template.
"""
import numpy as np
import pytest
from gate_data import QFT

import pennylane as qml
from pennylane.capture import run_autograph


@pytest.mark.jax
def test_standard_validity():
    """Check the operation using the assert_valid function."""
    op = qml.QFT.operator(wires=(0, 1, 2))
    qml.ops.functions.assert_valid(op, skip_pickle=True)


class TestQFT:
    """Tests for the qft operations"""

    @pytest.mark.parametrize("n_qubits", range(2, 6))
    def test_QFT_decomposition(self, n_qubits):
        """Test if the QFT operation is correctly decomposed"""
        decomp = qml.QFT(wires=range(n_qubits))

        dev = qml.device("default.qubit", wires=n_qubits)

        out_states = []
        for state in np.eye(2**n_qubits):
            ops = [qml.StatePrep(state, wires=range(n_qubits))] + decomp
            qs = qml.tape.QuantumScript(ops, [qml.state()])
            out_states.append(dev.execute(qs))

        reconstructed_unitary = np.array(out_states).T
        expected_unitary = qml.QFT(wires=range(n_qubits)).matrix()

        assert np.allclose(reconstructed_unitary, expected_unitary)

    @pytest.mark.parametrize("n_qubits", (3,))
    def test_QFT_decomposition(self, n_qubits):
        """Test if the QFT operation is correctly decomposed"""
        decomp = qml.QFT.operator(wires=range(n_qubits)).decomposition()

        dev = qml.device("default.qubit", wires=n_qubits)

        out_states = []
        for state in np.eye(2**n_qubits):
            ops = [qml.StatePrep(state, wires=range(n_qubits))] + decomp
            qs = qml.tape.QuantumScript(ops, [qml.state()])
            out_states.append(dev.execute(qs))

        reconstructed_unitary = np.array(out_states).T
        expected_unitary = QFT

        assert np.allclose(reconstructed_unitary, expected_unitary)

    @pytest.mark.parametrize("n_qubits", range(2, 10))
    def test_QFT_adjoint_identity(self, n_qubits, tol):
        """Test if using the qml.adjoint transform the resulting operation is
        the inverse of QFT."""

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circ(n_qubits):
            qml.adjoint(qml.QFT)(wires=range(n_qubits))
            qml.QFT(wires=range(n_qubits))
            return qml.state()

        assert np.allclose(1, circ(n_qubits)[0], tol)

        for i in range(1, n_qubits):
            assert np.allclose(0, circ(n_qubits)[i], tol)

    @pytest.mark.jax
    def test_jit(self):
        import jax
        import jax.numpy as jnp

        wires = 3

        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev)
        def circuit_qft(basis_state):
            qml.BasisState(basis_state, wires=range(wires))
            qml.QFT(wires=range(wires))
            return qml.state()

        jit_qft = jax.jit(circuit_qft)

        res = circuit_qft(jnp.array([1.0, 0.0, 0.0]))
        res2 = jit_qft(jnp.array([1.0, 0.0, 0.0]))

        assert qml.math.allclose(res, res2)


@pytest.mark.jax
@pytest.mark.capture
# pylint:disable=protected-access
class TestDynamicDecomposition:
    """Tests that dynamic decomposition via compute_qfunc_decomposition works correctly."""

    @pytest.mark.parametrize("autograph", [True, False])
    @pytest.mark.parametrize("n_wires", [4, 5])
    @pytest.mark.parametrize("wires", [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]])
    @pytest.mark.parametrize("max_expansion", [1, 2, 3, 4, None])
    @pytest.mark.parametrize(
        "gate_set", [[qml.Hadamard, qml.CNOT, qml.PhaseShift, qml.GlobalPhase], None]
    )
    def test_qft(
        self, max_expansion, gate_set, n_wires, wires, autograph
    ):  # pylint:disable=too-many-arguments, too-many-positional-arguments
        """Test that QFT gives correct result after dynamic decomposition."""

        import jax

        from pennylane.transforms.decompose import DecomposeInterpreter

        @DecomposeInterpreter(max_expansion=max_expansion, gate_set=gate_set)
        @qml.qnode(device=qml.device("default.qubit", wires=n_wires))
        def circuit(wires):
            qml.QFT(wires=wires)
            return qml.state()

        if autograph:
            circuit = run_autograph(circuit)
        jaxpr = jax.make_jaxpr(circuit)(wires=wires)
        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *wires)

        with qml.capture.pause():

            @qml.transforms.decompose(max_expansion=max_expansion, gate_set=gate_set)
            @qml.qnode(device=qml.device("default.qubit", wires=n_wires))
            def circuit_comparison():
                qml.QFT(wires=wires)
                return qml.state()

            result_comparison = circuit_comparison()

        assert qml.math.allclose(*result, result_comparison)
