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
from pennylane.capture.autograph import run_autograph


@pytest.mark.jax
def test_standard_validity():
    """Check the operation using the assert_valid function."""
    op = qml.QFT(wires=(0, 1, 2))
    qml.ops.functions.assert_valid(op)


class TestQFT:
    """Tests for the qft operations"""

    def test_QFT(self):
        """Test if the QFT matrix is equal to a manually-calculated version for 3 qubits"""
        op = qml.QFT(wires=range(3))
        res = op.matrix()
        exp = QFT
        assert np.allclose(res, exp)

    @pytest.mark.parametrize("n_qubits", range(2, 6))
    def test_QFT_compute_decomposition(self, n_qubits):
        """Test if the QFT operation is correctly decomposed"""
        decomp = qml.QFT.compute_decomposition(wires=range(n_qubits))

        dev = qml.device("default.qubit", wires=n_qubits)

        out_states = []
        for state in np.eye(2**n_qubits):
            ops = [qml.StatePrep(state, wires=range(n_qubits))] + decomp
            qs = qml.tape.QuantumScript(ops, [qml.state()])
            out_states.append(dev.execute(qs))

        reconstructed_unitary = np.array(out_states).T
        expected_unitary = qml.QFT(wires=range(n_qubits)).matrix()

        assert np.allclose(reconstructed_unitary, expected_unitary)

    @pytest.mark.parametrize("n_qubits", range(2, 6))
    def test_QFT_decomposition(self, n_qubits):
        """Test if the QFT operation is correctly decomposed"""
        op = qml.QFT(wires=range(n_qubits))
        decomp = op.decomposition()

        dev = qml.device("default.qubit", wires=n_qubits)

        out_states = []
        for state in np.eye(2**n_qubits):
            ops = [qml.StatePrep(state, wires=range(n_qubits))] + decomp
            qs = qml.tape.QuantumScript(ops, [qml.state()])
            out_states.append(dev.execute(qs))

        reconstructed_unitary = np.array(out_states).T
        expected_unitary = qml.QFT(wires=range(n_qubits)).matrix()

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

    def test_matrix(self, tol):
        """Test that the matrix representation is correct."""

        res_static = qml.QFT.compute_matrix(2)
        res_dynamic = qml.QFT(wires=[0, 1]).matrix()
        res_reordered = qml.QFT(wires=[0, 1]).matrix([1, 0])

        expected = np.array(
            [
                [0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j],
                [0.5 + 0.0j, 0.0 + 0.5j, -0.5 + 0.0j, -0.0 - 0.5j],
                [0.5 + 0.0j, -0.5 + 0.0j, 0.5 - 0.0j, -0.5 + 0.0j],
                [0.5 + 0.0j, -0.0 - 0.5j, -0.5 + 0.0j, 0.0 + 0.5j],
            ]
        )

        assert np.allclose(res_static, expected, atol=tol, rtol=0)
        assert np.allclose(res_dynamic, expected, atol=tol, rtol=0)

        expected_permuted = [
            [0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j, 0.5 + 0.0j],
            [0.5 + 0.0j, 0.5 - 0.0j, -0.5 + 0.0j, -0.5 + 0.0j],
            [0.5 + 0.0j, -0.5 + 0.0j, 0.0 + 0.5j, -0.0 - 0.5j],
            [0.5 + 0.0j, -0.5 + 0.0j, -0.0 - 0.5j, 0.0 + 0.5j],
        ]
        assert np.allclose(res_reordered, expected_permuted, atol=tol, rtol=0)

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

    def test_qft_plxpr(self):
        """Test that the dynamic decomposition of QFT has the correct plxpr"""
        import jax

        from pennylane.capture.primitives import for_loop_prim
        from pennylane.tape.plxpr_conversion import CollectOpsandMeas
        from pennylane.transforms.decompose import DecomposeInterpreter

        wires = [0, 1, 2, 3]
        gate_set = None
        max_expansion = 1

        @DecomposeInterpreter(max_expansion=max_expansion, gate_set=gate_set)
        def circuit(wires):
            qml.QFT(wires=wires)
            return qml.state()

        jaxpr = jax.make_jaxpr(circuit)(wires=wires)

        # Validate Jaxpr
        jaxpr_eqns = jaxpr.eqns
        outer_loop, swap_loop = (eqn for eqn in jaxpr_eqns if eqn.primitive == for_loop_prim)
        assert outer_loop.primitive == for_loop_prim
        assert swap_loop.primitive == for_loop_prim
        outer_loop_eqn = outer_loop.params["jaxpr_body_fn"].eqns
        swap_loop_eqn = swap_loop.params["jaxpr_body_fn"].eqns

        hadamards = [eqn for eqn in outer_loop_eqn if eqn.primitive == qml.Hadamard._primitive]
        assert len(hadamards) == 1

        cphaseshift_loop = [eqn for eqn in outer_loop_eqn if eqn.primitive == for_loop_prim]
        assert cphaseshift_loop[0].primitive == for_loop_prim
        cphaseshift_eqns = cphaseshift_loop[0].params["jaxpr_body_fn"].eqns
        assert cphaseshift_eqns[-1].primitive == qml.ControlledPhaseShift._primitive

        assert swap_loop_eqn[-1].primitive == qml.SWAP._primitive

        # Validate Ops
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *wires)
        ops_list = collector.state["ops"]
        tape = qml.tape.QuantumScript([qml.QFT(wires=wires)])
        [decomp_tape], _ = qml.transforms.decompose(
            tape, max_expansion=max_expansion, gate_set=gate_set
        )
        for op1, op2 in zip(ops_list, decomp_tape.operations):
            assert qml.equal(op1, op2, check_interface=False)

    @pytest.mark.parametrize("autograph", [True, False])
    @pytest.mark.parametrize("n_wires", [4, 5])
    @pytest.mark.parametrize("wires", [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]])
    @pytest.mark.parametrize("max_expansion", [1, 2, 3, 4, None])
    @pytest.mark.parametrize("gate_set", [[qml.Hadamard, qml.CNOT, qml.PhaseShift], None])
    def test_qft(
        self, max_expansion, gate_set, n_wires, wires, autograph
    ):  # pylint:disable=too-many-arguments, too-many-positional-arguments
        """Test that QFT gives correct result after dynamic decomposition."""

        from functools import partial

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

            @partial(qml.transforms.decompose, max_expansion=max_expansion, gate_set=gate_set)
            @qml.qnode(device=qml.device("default.qubit", wires=n_wires))
            def circuit_comparison():
                qml.QFT(wires=wires)
                return qml.state()

            result_comparison = circuit_comparison()

        assert qml.math.allclose(*result, result_comparison)

    @pytest.mark.usefixtures("enable_graph_decomposition")
    @pytest.mark.parametrize("wires", [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]])
    def test_qft_new_decomposition(self, wires):
        """Test that QFT gives the correct decomposition in the graph-based system."""

        import jax

        from pennylane.tape.plxpr_conversion import CollectOpsandMeas
        from pennylane.transforms.decompose import DecomposeInterpreter

        @DecomposeInterpreter(gate_set={"GlobalPhase", "RX", "RZ", "CNOT"})
        def circuit():
            qml.QFT(wires=wires)

        jaxpr = jax.make_jaxpr(circuit)()
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts)

        graph = qml.decomposition.DecompositionGraph(
            operations=[qml.QFT(wires=wires)],
            gate_set={"GlobalPhase", "RX", "RZ", "CNOT"},
        )
        solution = graph.solve()
        expected_resources = solution.resource_estimate(qml.QFT(wires=wires))

        assert len(collector.state["ops"]) == expected_resources.num_gates
