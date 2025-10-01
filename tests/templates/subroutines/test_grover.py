# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Tests for the Grover Diffusion Operator template
"""
import functools
import itertools

import numpy as np
import pytest

import pennylane as qml
from pennylane.capture.autograph import run_autograph
from pennylane.ops import Hadamard, MultiControlledX, PauliZ
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


def test_repr():
    """Tests the repr method for GroverOperator."""
    op = qml.GroverOperator(wires=(0, 1, 2), work_wires=(3, 4))
    expected = "GroverOperator(wires=[0, 1, 2], work_wires=[3, 4])"
    assert repr(op) == expected


def test_work_wire_property():
    op = qml.GroverOperator(wires=(0, 1, 2), work_wires=(3, 4))
    expected = qml.wires.Wires((3, 4))
    assert op.work_wires == expected


@pytest.mark.jax
def test_standard_validity():
    """Test the standard criteria for a valid operation."""
    work_wires = qml.wires.Wires((3, 4))
    op = qml.GroverOperator(wires=(0, 1, 2), work_wires=work_wires)
    qml.ops.functions.assert_valid(op)


def test_work_wires():
    """Assert work wires get passed to MultiControlledX"""
    wires = ("a", "b")
    work_wire = ("aux",)

    op = qml.GroverOperator(wires=wires, work_wires=work_wire)

    assert op.hyperparameters["work_wires"] == work_wire

    ops = op.decomposition()

    assert ops[2].hyperparameters["work_wires"] == work_wire


def test_work_wires_None():
    """Test that work wires of None are not inpreted as work wires."""
    op = qml.GroverOperator(wires=(0, 1, 2, 3), work_wires=None)
    assert op.hyperparameters["work_wires"] == qml.wires.Wires([])


@pytest.mark.parametrize("bad_wires", [0, (0,), tuple()])
def test_single_wire_error(bad_wires):
    """Assert error raised when called with only a single wire"""

    with pytest.raises(ValueError, match="GroverOperator must have at least"):
        qml.GroverOperator(wires=bad_wires)


def test_id():
    """Assert id keyword works"""

    op = qml.GroverOperator(wires=(0, 1), id="hello")

    assert op.id == "hello"


decomp_3wires = [
    qml.Hadamard,
    qml.Hadamard,
    qml.PauliZ,
    qml.MultiControlledX,
    qml.PauliZ,
    qml.Hadamard,
    qml.Hadamard,
    qml.GlobalPhase,
]


def decomposition_wires(wires):
    wire_order = [
        wires[0],
        wires[1],
        wires[2],
        wires,
        wires[2],
        wires[0],
        wires[1],
        wires,
    ]
    return wire_order


@pytest.mark.parametrize("n_wires", [2, 4, 7])
def test_grover_diffusion_matrix(n_wires):
    """Test that the Grover diffusion matrix is the same as when constructed in a different way"""
    wires = list(range(n_wires))

    # Test-oracle
    oracle = np.identity(2**n_wires)
    oracle[0, 0] = -1

    # s1 = H|0>, Hadamard on a single qubit in the ground state
    s1 = np.array([1, 1]) / np.sqrt(2)

    # uniform superposition state
    s = functools.reduce(np.kron, list(itertools.repeat(s1, n_wires)))
    # Grover matrix
    G_matrix = qml.GroverOperator(wires=wires).matrix()

    amplitudes = G_matrix @ oracle @ s
    probs = amplitudes**2

    # Create Grover diffusion matrix G in alternative way
    oplist = list(itertools.repeat(Hadamard.compute_matrix(), n_wires - 1))
    oplist.append(PauliZ.compute_matrix())

    ctrl_str = [0] * (n_wires - 1)
    CX = MultiControlledX(
        control_values=ctrl_str,
        wires=wires,
        work_wires=None,
    ).matrix()

    M = functools.reduce(np.kron, oplist)
    G = M @ CX @ M

    amplitudes2 = G @ oracle @ s
    probs2 = amplitudes2**2

    assert np.allclose(probs, probs2)


def test_grover_diffusion_matrix_results():
    """Test that the matrix gives the same result as when running the example in the documentation
    `here <https://pennylane.readthedocs.io/en/stable/code/api/pennylane.templates.subroutines.GroverOperator.html>`_
    """
    n_wires = 3
    wires = list(range(n_wires))

    def oracle():
        qml.Hadamard(wires[-1])
        qml.Toffoli(wires=wires)
        qml.Hadamard(wires[-1])

    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def GroverSearch(num_iterations=1):
        for wire in wires:
            qml.Hadamard(wire)

        for _ in range(num_iterations):
            oracle()
            qml.GroverOperator(wires=wires)
        return qml.probs(wires)

    # Get probabilities from example
    probs_example = GroverSearch(num_iterations=1)

    # Grover diffusion matrix
    G_matrix = qml.GroverOperator(wires=wires).matrix()

    oracle_matrix = np.identity(2**n_wires)
    oracle_matrix[-1, -1] = -1

    # s1 = H|0>, Hadamard on a single qubit in the ground state
    s1 = np.array([1, 1]) / np.sqrt(2)

    # uniform superposition state
    s = functools.reduce(np.kron, list(itertools.repeat(s1, n_wires)))

    amplitudes = G_matrix @ oracle_matrix @ s
    # Check that the probabilities are the same
    probs_matrix = amplitudes**2

    assert np.allclose(probs_example, probs_matrix)


@pytest.mark.parametrize("wires", ((0, 1, 2), ("a", "c", "b")))
def test_expand(wires):
    """Asserts decomposition uses expected operations and wires"""
    op = qml.GroverOperator(wires=wires)

    decomp = op.decomposition()

    expected_wires = decomposition_wires(wires)

    assert len(decomp) == len(decomp_3wires) == len(expected_wires)

    for actual_op, expected_class, expected_wire in zip(decomp, decomp_3wires, expected_wires):
        assert isinstance(actual_op, expected_class)
        assert actual_op.wires == qml.wires.Wires(expected_wire)


@pytest.mark.capture
def test_decomposition_new_capture():
    """Tests the decomposition rule implemented with the new system."""
    op = qml.GroverOperator(wires=(0, 1, 2))

    for rule in qml.list_decomps(qml.GroverOperator):
        _test_decomposition_rule(op, rule)


def test_decomposition_new():
    """Tests the decomposition rule implemented with the new system."""
    op = qml.GroverOperator(wires=(0, 1, 2))

    for rule in qml.list_decomps(qml.GroverOperator):
        _test_decomposition_rule(op, rule)


@pytest.mark.parametrize("n_wires", [6, 13])
def test_findstate(n_wires):
    """Asserts can find state marked by oracle, with operation full matrix and decomposition."""
    wires = list(range(n_wires))

    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def circ():
        for wire in wires:
            qml.Hadamard(wire)

        for _ in range(2):
            qml.Hadamard(wires[0])
            qml.MultiControlledX(wires=wires[1:] + wires[0:1])
            qml.Hadamard(wires[0])
            qml.GroverOperator(wires=wires)

        return qml.probs(wires=wires)

    probs = circ()

    assert np.argmax(probs) == len(probs) - 1


def test_matrix(tol):
    """Test that the matrix representation is correct."""

    res_static = qml.GroverOperator.compute_matrix(2, work_wires=None)
    res_dynamic = qml.GroverOperator(wires=[0, 1]).matrix()
    res_reordered = qml.GroverOperator(wires=[0, 1]).matrix([1, 0])

    expected = np.array(
        [[-0.5, 0.5, 0.5, 0.5], [0.5, -0.5, 0.5, 0.5], [0.5, 0.5, -0.5, 0.5], [0.5, 0.5, 0.5, -0.5]]
    )

    assert np.allclose(res_static, expected, atol=tol, rtol=0)
    assert np.allclose(res_dynamic, expected, atol=tol, rtol=0)
    # reordering should not affect this particular matrix
    assert np.allclose(res_reordered, expected, atol=tol, rtol=0)


@pytest.mark.parametrize("n_wires", [2, 3, 5])
def test_decomposition_matrix(n_wires):
    """Test that the decomposition and the matrix match."""
    wires = list(range(n_wires))
    op = qml.GroverOperator(wires)
    mat1 = op.matrix()
    mat2 = qml.matrix(qml.tape.QuantumScript(op.decomposition()), wire_order=wires)
    assert np.allclose(mat1, mat2)


@pytest.mark.jax
def test_jax_jit():
    import jax

    n_wires = 3
    wires = list(range(n_wires))

    def oracle():
        qml.Hadamard(wires[-1])
        qml.Toffoli(wires=wires)
        qml.Hadamard(wires[-1])

    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def circuit():
        for wire in wires:
            qml.Hadamard(wire)

        oracle()
        qml.GroverOperator(wires=wires)

        oracle()
        qml.GroverOperator(wires=wires)

        return qml.probs(wires)

    jit_circuit = jax.jit(circuit)

    assert qml.math.allclose(circuit(), jit_circuit())


@pytest.mark.jax
@pytest.mark.capture
# pylint:disable=protected-access
class TestDynamicDecomposition:
    """Tests that dynamic decomposition via compute_qfunc_decomposition works correctly."""

    def test_grover_plxpr(self):
        """Test that the dynamic decomposition of Grover has the correct plxpr"""
        import jax

        from pennylane.capture.primitives import for_loop_prim
        from pennylane.tape.plxpr_conversion import CollectOpsandMeas
        from pennylane.transforms.decompose import DecomposeInterpreter

        wires = [0, 1, 2]
        work_wires = np.array([3, 4])
        gate_set = None
        max_expansion = 1

        @DecomposeInterpreter(max_expansion=max_expansion, gate_set=gate_set)
        def circuit(wires):
            qml.GroverOperator(wires=wires, work_wires=work_wires)
            return qml.state()

        jaxpr = jax.make_jaxpr(circuit)(wires)

        # Validate Jaxpr
        jaxpr_eqns = jaxpr.eqns
        # 2 Hadamard loops
        hadamard_loops_eqns = [eqn for eqn in jaxpr_eqns if eqn.primitive == for_loop_prim]
        assert len(hadamard_loops_eqns) == 2
        for hadamard_loop in hadamard_loops_eqns:
            assert hadamard_loop.primitive == for_loop_prim
            hadamard_inner_eqns = hadamard_loop.params["jaxpr_body_fn"].eqns
            assert hadamard_inner_eqns[-1].primitive == qml.Hadamard._primitive

        # 4 remaining operations
        remaining_ops = [
            eqn
            for eqn in jaxpr_eqns
            if eqn.primitive
            in (qml.PauliZ._primitive, qml.MultiControlledX._primitive, qml.GlobalPhase._primitive)
        ]
        assert len(remaining_ops) == 4

        # Validate Ops
        collector = CollectOpsandMeas()
        collector.eval(jaxpr.jaxpr, jaxpr.consts, *wires)
        ops_list = collector.state["ops"]

        tape = qml.tape.QuantumScript([qml.GroverOperator(wires=wires, work_wires=work_wires)])
        [decomp_tape], _ = qml.transforms.decompose(
            tape, max_expansion=max_expansion, gate_set=gate_set
        )
        for op1, op2 in zip(ops_list, decomp_tape.operations):
            if op1.name == "GlobalPhase":
                # GlobalPhase applied to single wire instead of all wires
                assert op1.name == op2.name
                assert qml.math.allclose(op1.parameters, op2.parameters)
            elif op1.name == "MultiControlledX":
                # MultiControlledX's work_wire is traced in plxpr but not in tape
                assert op1.name == op2.name
                assert op1.wires == op2.wires
                assert op1.control_wires == op2.control_wires
                assert op1.control_values == op2.control_values
            else:
                assert qml.equal(op1, op2)

    @pytest.mark.parametrize("autograph", [True, False])
    @pytest.mark.parametrize(
        "wires, work_wires", [([0, 1, 2], [3, 4]), ([0, 1, 4], [2, 3]), ([0, 2, 4], [1])]
    )
    @pytest.mark.parametrize("max_expansion", [1, 2, 3, 4, None])
    @pytest.mark.parametrize(
        "gate_set", [[qml.Hadamard, qml.CNOT, qml.PauliX, qml.GlobalPhase, qml.RZ], None]
    )
    def test_grover(
        self, max_expansion, gate_set, wires, work_wires, autograph
    ):  # pylint:disable=too-many-arguments, too-many-positional-arguments
        """Test that Grover gives correct result after dynamic decomposition."""

        from functools import partial

        import jax

        from pennylane.transforms.decompose import DecomposeInterpreter

        @DecomposeInterpreter(max_expansion=max_expansion, gate_set=gate_set)
        @qml.qnode(device=qml.device("default.qubit", wires=5))
        def circuit(wires):
            qml.GroverOperator(wires=wires, work_wires=work_wires)
            return qml.state()

        if autograph:
            circuit = run_autograph(circuit)
        jaxpr = jax.make_jaxpr(circuit)(wires)
        result = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, *wires)

        with qml.capture.pause():

            @partial(qml.transforms.decompose, max_expansion=max_expansion, gate_set=gate_set)
            @qml.qnode(device=qml.device("default.qubit", wires=5))
            def circuit_comparison():
                qml.GroverOperator(wires=wires, work_wires=work_wires)
                return qml.state()

            result_comparison = circuit_comparison()

        assert qml.math.allclose(*result, result_comparison)
