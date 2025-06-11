# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Unit tests for the optimization transform ``pattern_matching_optimization``.
"""

# pylint: disable=too-many-statements
import pytest

import pennylane as qml
import pennylane.numpy as np
from pennylane.exceptions import QuantumFunctionError
from pennylane.transforms.commutation_dag import commutation_dag
from pennylane.transforms.optimization.pattern_matching import (
    BackwardMatch,
    ForwardMatch,
    _update_qubits,
    pattern_matching,
    pattern_matching_optimization,
)


class TestPatternMatchingOptimization:
    """Pattern matching circuit optimization tests."""

    def test_simple_quantum_function_pattern_matching(self):
        """Test pattern matching algorithm for circuit optimization with a CNOTs template."""

        def circuit():
            qml.Toffoli(wires=[3, 4, 0])
            qml.CNOT(wires=[1, 4])
            qml.CNOT(wires=[2, 1])
            qml.Hadamard(wires=3)
            qml.PauliZ(wires=1)
            qml.CNOT(wires=[2, 3])
            qml.Toffoli(wires=[2, 3, 0])
            qml.CNOT(wires=[1, 4])
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_template:
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])

        template = qml.tape.QuantumScript.from_queue(q_template)
        dev = qml.device("default.qubit", wires=5)

        qnode = qml.QNode(circuit, dev)
        qnode_res = qnode()

        optimized_qfunc = pattern_matching_optimization(circuit, pattern_tapes=[template])
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode_res = optimized_qnode()

        cnots_qnode = qml.specs(qnode)()["resources"].gate_types["CNOT"]
        cnots_optimized_qnode = qml.specs(optimized_qnode)()["resources"].gate_types["CNOT"]

        tape = qml.workflow.construct_tape(qnode)()
        assert len(tape.operations) == 8
        assert cnots_qnode == 4

        optimized_tape = qml.workflow.construct_tape(optimized_qnode)()
        assert len(optimized_tape.operations) == 7
        assert cnots_optimized_qnode == 3

        assert qnode_res == optimized_qnode_res
        assert np.allclose(qml.matrix(optimized_qnode)(), qml.matrix(qnode)())

    def test_simple_quantum_function_pattern_matching_qnode(self):
        """Test pattern matching algorithm for circuit optimization with a CNOTs template."""
        dev = qml.device("default.qubit", wires=5)

        @qml.qnode(device=dev)
        def circuit():
            qml.Toffoli(wires=[3, 4, 0])
            qml.CNOT(wires=[1, 4])
            qml.CNOT(wires=[2, 1])
            qml.Hadamard(wires=3)
            qml.PauliZ(wires=1)
            qml.CNOT(wires=[2, 3])
            qml.Toffoli(wires=[2, 3, 0])
            qml.CNOT(wires=[1, 4])
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_template:
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])

        template = qml.tape.QuantumScript.from_queue(q_template)

        optimized_qnode = pattern_matching_optimization(circuit, pattern_tapes=[template])
        optimized_qnode()
        assert np.allclose(qml.matrix(optimized_qnode)(), qml.matrix(circuit)())

    def test_custom_quantum_cost(self):
        """Test pattern matching algorithm for circuit optimization with a CNOTs template with custom quantum dict."""

        def circuit():
            qml.Toffoli(wires=[3, 4, 0])
            qml.CNOT(wires=[1, 4])
            qml.CNOT(wires=[2, 1])
            qml.Hadamard(wires=3)
            qml.PauliZ(wires=1)
            qml.CNOT(wires=[2, 3])
            qml.Toffoli(wires=[2, 3, 0])
            qml.CNOT(wires=[1, 4])
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_template:
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])

        template = qml.tape.QuantumScript.from_queue(q_template)
        dev = qml.device("default.qubit", wires=5)

        qnode = qml.QNode(circuit, dev)
        qnode_res = qnode()

        quantum_cost = {"CNOT": 10}
        optimized_qfunc = pattern_matching_optimization(
            circuit, pattern_tapes=[template], custom_quantum_cost=quantum_cost
        )
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode_res = optimized_qnode()

        cnots_qnode = qml.specs(qnode)()["resources"].gate_types["CNOT"]
        cnots_optimized_qnode = qml.specs(optimized_qnode)()["resources"].gate_types["CNOT"]

        tape = qml.workflow.construct_tape(qnode)()
        assert len(tape.operations) == 8
        assert cnots_qnode == 4

        optimized_tape = qml.workflow.construct_tape(optimized_qnode)()
        assert len(optimized_tape.operations) == 7
        assert cnots_optimized_qnode == 3

        assert qnode_res == optimized_qnode_res
        assert np.allclose(qml.matrix(optimized_qnode)(), qml.matrix(qnode)())

    def test_no_match_not_optimized(self):
        """Test pattern matching algorithm for circuit optimization with no match and therefore no optimization."""

        def circuit():
            qml.Toffoli(wires=[3, 4, 0])
            qml.CNOT(wires=[1, 4])
            qml.CNOT(wires=[2, 1])
            qml.Hadamard(wires=3)
            qml.PauliZ(wires=1)
            qml.CNOT(wires=[2, 3])
            qml.Toffoli(wires=[2, 3, 0])
            qml.CNOT(wires=[1, 4])
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_template:
            qml.PauliX(wires=0)
            qml.PauliX(wires=0)

        template = qml.tape.QuantumScript.from_queue(q_template)
        dev = qml.device("default.qubit", wires=5)

        qnode = qml.QNode(circuit, dev)
        qnode_res = qnode()

        optimized_qfunc = pattern_matching_optimization(circuit, pattern_tapes=[template])
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode_res = optimized_qnode()

        cnots_qnode = qml.specs(qnode)()["resources"].gate_types["CNOT"]
        cnots_optimized_qnode = qml.specs(optimized_qnode)()["resources"].gate_types["CNOT"]

        tape = qml.workflow.construct_tape(qnode)()
        assert len(tape.operations) == 8
        assert cnots_qnode == 4

        optimized_tape = qml.workflow.construct_tape(optimized_qnode)()
        assert len(optimized_tape.operations) == 8
        assert cnots_optimized_qnode == 4

        assert qnode_res == optimized_qnode_res
        assert np.allclose(qml.matrix(optimized_qnode)(), qml.matrix(qnode)())

    def test_adjoint_s(self):
        def circuit():
            qml.S(wires=0)
            qml.PauliZ(wires=0)
            qml.S(wires=1)
            qml.CZ(wires=[0, 1])
            qml.S(wires=1)
            qml.S(wires=2)
            qml.CZ(wires=[1, 2])
            qml.S(wires=2)
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_template:
            qml.S(wires=0)
            qml.S(wires=0)
            qml.PauliZ(wires=0)

        template = qml.tape.QuantumScript.from_queue(q_template)
        dev = qml.device("default.qubit", wires=5)

        qnode = qml.QNode(circuit, dev)
        qnode_res = qnode()

        optimized_qfunc = pattern_matching_optimization(circuit, pattern_tapes=[template])
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode_res = optimized_qnode()

        s_qnode = qml.specs(qnode)()["resources"].gate_types["S"]
        s_adjoint_optimized_qnode = qml.specs(optimized_qnode)()["resources"].gate_types[
            "Adjoint(S)"
        ]

        tape = qml.workflow.construct_tape(qnode)()
        assert len(tape.operations) == 8
        assert s_qnode == 5

        optimized_tape = qml.workflow.construct_tape(optimized_qnode)()
        assert len(optimized_tape.operations) == 5
        assert s_adjoint_optimized_qnode == 1

        assert qnode_res == optimized_qnode_res
        assert np.allclose(qml.matrix(optimized_qnode)(), qml.matrix(qnode)())

    def test_template_with_toffoli(self):
        """Test pattern matching algorithm for circuit optimization with a template having Toffoli gates."""

        def circuit():
            qml.Toffoli(wires=[0, 2, 4])
            qml.CZ(wires=[0, 2])
            qml.RY(0.1, wires=[1])
            qml.PauliZ(wires=[0])
            qml.CNOT(wires=[0, 2])
            qml.PauliZ(wires=[0])
            qml.CNOT(wires=[1, 4])
            qml.PauliY(wires=1)
            qml.PauliZ(wires=0)
            qml.CZ(wires=[0, 2])
            qml.Toffoli(wires=[0, 2, 4])
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_template:
            qml.Toffoli(wires=[0, 1, 2])
            qml.CNOT(wires=[0, 1])
            qml.Toffoli(wires=[0, 1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])

        template = qml.tape.QuantumScript.from_queue(q_template)
        dev = qml.device("default.qubit", wires=5)

        qnode = qml.QNode(circuit, dev)
        qnode_res = qnode()

        optimized_qfunc = pattern_matching_optimization(circuit, pattern_tapes=[template])
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode_res = optimized_qnode()

        toffolis_qnode = qml.specs(qnode)()["resources"].gate_types["Toffoli"]
        toffolis_optimized_qnode = qml.specs(optimized_qnode)()["resources"].gate_types["Toffoli"]

        tape = qml.workflow.construct_tape(qnode)()
        assert len(tape.operations) == 11
        assert toffolis_qnode == 2

        optimized_tape = qml.workflow.construct_tape(optimized_qnode)()
        assert len(optimized_tape.operations) == 10
        assert toffolis_optimized_qnode == 0

        assert qnode_res == optimized_qnode_res
        assert np.allclose(qml.matrix(optimized_qnode)(), qml.matrix(qnode)())

    def test_template_with_swap(self):
        """Test pattern matching algorithm for circuit optimization with a template having swap gates."""

        def circuit():
            qml.PauliZ(wires=1)
            qml.PauliZ(wires=3)
            qml.CNOT(wires=[2, 0])
            qml.SWAP(wires=[1, 3])
            qml.CZ(wires=[0, 2])
            qml.PauliX(wires=1)
            qml.PauliX(wires=3)
            qml.Toffoli(wires=[2, 0, 3])
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_template:
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 0])
            qml.CNOT(wires=[0, 1])
            qml.SWAP(wires=[0, 1])

        template = qml.tape.QuantumScript.from_queue(q_template)
        dev = qml.device("default.qubit", wires=4)

        qnode = qml.QNode(circuit, dev)
        qnode_res = qnode()

        quantum_cost = {"SWAP": 10, "CNOT": 1}
        optimized_qfunc = pattern_matching_optimization(
            circuit, pattern_tapes=[template], custom_quantum_cost=quantum_cost
        )
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode_res = optimized_qnode()

        gate_qnode = qml.specs(qnode)()["resources"].gate_types
        swap_qnode = gate_qnode["SWAP"]
        cnot_qnode = gate_qnode["CNOT"]

        gate_qnode_optimized = qml.specs(optimized_qnode)()["resources"].gate_types
        swap_optimized_qnode = gate_qnode_optimized["SWAP"]
        cnot_optimized_qnode = gate_qnode_optimized["CNOT"]

        tape = qml.workflow.construct_tape(qnode)()
        assert len(tape.operations) == 8
        assert swap_qnode == 1
        assert cnot_qnode == 1

        optimized_tape = qml.workflow.construct_tape(optimized_qnode)()
        assert len(optimized_tape.operations) == 10
        assert swap_optimized_qnode == 0
        assert cnot_optimized_qnode == 4

        assert qnode_res == optimized_qnode_res
        assert np.allclose(qml.matrix(optimized_qnode)(), qml.matrix(qnode)())

    def test_template_with_multiple_swap(self):
        """Test pattern matching algorithm for circuit optimization with a template having multiple swap gates."""

        def circuit():
            qml.PauliZ(wires=1)
            qml.PauliZ(wires=3)
            qml.CNOT(wires=[2, 0])
            qml.SWAP(wires=[1, 3])
            qml.SWAP(wires=[1, 3])
            qml.CZ(wires=[0, 2])
            qml.PauliX(wires=1)
            qml.PauliX(wires=3)
            qml.Toffoli(wires=[2, 0, 3])
            qml.SWAP(wires=[3, 1])
            qml.SWAP(wires=[1, 3])
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_template:
            qml.SWAP(wires=[0, 1])
            qml.SWAP(wires=[0, 1])

        template = qml.tape.QuantumScript.from_queue(q_template)
        dev = qml.device("default.qubit", wires=4)

        qnode = qml.QNode(circuit, dev)
        qnode_res = qnode()

        optimized_qfunc = pattern_matching_optimization(circuit, pattern_tapes=[template])
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode_res = optimized_qnode()

        gate_qnode = qml.specs(qnode)()["resources"].gate_types
        swap_qnode = gate_qnode["SWAP"]
        cnot_qnode = gate_qnode["CNOT"]

        gate_qnode_optimized = qml.specs(optimized_qnode)()["resources"].gate_types
        swap_optimized_qnode = gate_qnode_optimized["SWAP"]
        cnot_optimized_qnode = gate_qnode_optimized["CNOT"]

        tape = qml.workflow.construct_tape(qnode)()
        assert len(tape.operations) == 11
        assert swap_qnode == 4
        assert cnot_qnode == 1

        optimized_tape = qml.workflow.construct_tape(optimized_qnode)()
        assert len(optimized_tape.operations) == 7
        assert swap_optimized_qnode == 0
        assert cnot_optimized_qnode == 1

        assert qnode_res == optimized_qnode_res
        assert np.allclose(qml.matrix(optimized_qnode)(), qml.matrix(qnode)())

    def test_template_with_multiple_control_swap(self):
        """Test pattern matching algorithm for circuit optimization with a template having multiple cswap gates."""

        def circuit():
            qml.PauliZ(wires=1)
            qml.PauliZ(wires=3)
            qml.CNOT(wires=[2, 0])
            qml.CSWAP(wires=[0, 1, 3])
            qml.CSWAP(wires=[0, 1, 3])
            qml.CZ(wires=[0, 2])
            qml.PauliX(wires=1)
            qml.PauliX(wires=3)
            qml.Toffoli(wires=[2, 0, 3])
            qml.CSWAP(wires=[0, 3, 1])
            qml.CSWAP(wires=[0, 1, 3])
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_template:
            qml.CSWAP(wires=[0, 1, 2])
            qml.CSWAP(wires=[0, 1, 2])

        template = qml.tape.QuantumScript.from_queue(q_template)
        dev = qml.device("default.qubit", wires=4)

        qnode = qml.QNode(circuit, dev)
        qnode_res = qnode()

        optimized_qfunc = pattern_matching_optimization(circuit, pattern_tapes=[template])
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode_res = optimized_qnode()

        gate_qnode = qml.specs(qnode)()["resources"].gate_types
        cswap_qnode = gate_qnode["CSWAP"]
        cnot_qnode = gate_qnode["CNOT"]

        gate_qnode_optimized = qml.specs(optimized_qnode)()["resources"].gate_types
        cswap_optimized_qnode = gate_qnode_optimized["CSWAP"]
        cnot_optimized_qnode = gate_qnode_optimized["CNOT"]

        tape = qml.workflow.construct_tape(qnode)()
        assert len(tape.operations) == 11
        assert cswap_qnode == 4
        assert cnot_qnode == 1

        optimized_tape = qml.workflow.construct_tape(optimized_qnode)()
        assert len(optimized_tape.operations) == 7
        assert cswap_optimized_qnode == 0
        assert cnot_optimized_qnode == 1

        assert qnode_res == optimized_qnode_res
        assert np.allclose(qml.matrix(optimized_qnode)(), qml.matrix(qnode)())

    def test_parametrized_pattern_matching(self):
        """Test pattern matching algorithm for circuit optimization with parameters."""

        def circuit(x, y):
            qml.Toffoli(wires=[3, 4, 0])
            qml.RZ(x, wires=[0])
            qml.RZ(-x, wires=[0])
            qml.RZ(-x, wires=[1])
            qml.CNOT(wires=[1, 4])
            qml.RZ(x, wires=[1])
            qml.CNOT(wires=[2, 1])
            qml.Hadamard(wires=3)
            qml.PauliZ(wires=1)
            qml.RX(y, wires=[3])
            qml.CNOT(wires=[2, 3])
            qml.RX(-y, wires=[3])
            qml.Toffoli(wires=[2, 3, 0])
            qml.CNOT(wires=[1, 4])
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_template_rz:
            qml.RZ(0.1, wires=[0])
            qml.RZ(-0.1, wires=[0])

        template_rz = qml.tape.QuantumScript.from_queue(q_template_rz)
        with qml.queuing.AnnotatedQueue() as q_template_rx:
            qml.RX(0.2, wires=[0])
            qml.RX(-0.2, wires=[0])

        template_rx = qml.tape.QuantumScript.from_queue(q_template_rx)
        dev = qml.device("default.qubit", wires=5)

        qnode = qml.QNode(circuit, dev)
        qnode_res = qnode(0.1, 0.2)

        optimized_qfunc = pattern_matching_optimization(
            circuit, pattern_tapes=[template_rx, template_rz]
        )
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode_res = optimized_qnode(0.1, 0.2)

        rx_qnode = qml.specs(qnode)(0.1, 0.2)["resources"].gate_types["RX"]
        rx_optimized_qnode = qml.specs(optimized_qnode)(0.1, 0.2)["resources"].gate_types["RX"]

        rz_qnode = qml.specs(qnode)(0.1, 0.2)["resources"].gate_types["RZ"]
        rz_optimized_qnode = qml.specs(optimized_qnode)(0.1, 0.2)["resources"].gate_types["RZ"]

        tape = qml.workflow.construct_tape(qnode)(0.1, 0.2)
        assert len(tape.operations) == 14
        assert rx_qnode == 2
        assert rz_qnode == 4

        optimized_tape = qml.workflow.construct_tape(optimized_qnode)(0.1, 0.2)
        assert len(optimized_tape.operations) == 8
        assert rx_optimized_qnode == 0
        assert rz_optimized_qnode == 0

        assert qnode_res == optimized_qnode_res
        assert np.allclose(qml.matrix(optimized_qnode)(0.1, 0.2), qml.matrix(qnode)(0.1, 0.2))

    def test_multiple_patterns(self):
        """Test pattern matching algorithm for circuit optimization with three different templates."""

        def circuit():
            qml.CNOT(wires=[0, 1])
            qml.PauliZ(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliZ(wires=0)
            qml.PauliX(wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=1)
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_template_cnot:
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 1])

        template_cnot = qml.tape.QuantumScript.from_queue(q_template_cnot)
        with qml.queuing.AnnotatedQueue() as q_template_x:
            qml.PauliX(wires=[0])
            qml.PauliX(wires=[0])

        template_x = qml.tape.QuantumScript.from_queue(q_template_x)
        with qml.queuing.AnnotatedQueue() as q_template_z:
            qml.PauliZ(wires=[0])
            qml.PauliZ(wires=[0])

        template_z = qml.tape.QuantumScript.from_queue(q_template_z)
        dev = qml.device("default.qubit", wires=5)

        qnode = qml.QNode(circuit, dev)
        qnode_res = qnode()

        optimized_qfunc = pattern_matching_optimization(
            circuit, pattern_tapes=[template_x, template_z, template_cnot]
        )
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode_res = optimized_qnode()

        cnots_qnode = qml.specs(qnode)()["resources"].gate_types["CNOT"]
        cnots_optimized_qnode = qml.specs(optimized_qnode)()["resources"].gate_types["CNOT"]

        tape = qml.workflow.construct_tape(qnode)()
        assert len(tape.operations) == 7
        assert cnots_qnode == 3

        optimized_tape = qml.workflow.construct_tape(optimized_qnode)()
        assert len(optimized_tape.operations) == 1
        assert cnots_optimized_qnode == 1

        assert qnode_res == optimized_qnode_res
        assert np.allclose(qml.matrix(optimized_qnode)(), qml.matrix(qnode)())

    def test_mod_5_4_pattern_matching(self):
        """Test pattern matching algorithm for mod_5_4 with a CNOTs template."""

        def mod_5_4():
            qml.PauliX(wires=4)
            qml.Hadamard(wires=4)
            qml.CNOT(wires=[3, 4])
            qml.CNOT(wires=[0, 4])
            qml.T(wires=4)
            qml.CNOT(wires=[3, 4])
            qml.adjoint(qml.T)(wires=4)
            qml.CNOT(wires=[0, 4])
            qml.CNOT(wires=[0, 3])
            qml.adjoint(qml.T)(wires=3)
            qml.CNOT(wires=[0, 3])
            qml.CNOT(wires=[3, 4])
            qml.CNOT(wires=[2, 4])
            qml.adjoint(qml.T)(wires=4)
            qml.CNOT(wires=[3, 4])
            qml.T(wires=4)
            qml.CNOT(wires=[2, 4])
            qml.CNOT(wires=[2, 3])
            qml.T(wires=3)
            qml.CNOT(wires=[2, 3])
            qml.Hadamard(wires=4)
            qml.CNOT(wires=[3, 4])
            qml.Hadamard(wires=4)
            qml.CNOT(wires=[2, 4])
            qml.adjoint(qml.T)(wires=4)
            qml.CNOT(wires=[1, 4])
            qml.T(wires=4)
            qml.CNOT(wires=[2, 4])
            qml.adjoint(qml.T)(wires=4)
            qml.CNOT(wires=[1, 4])
            qml.T(wires=4)
            qml.CNOT(wires=[1, 2])
            qml.adjoint(qml.T)(wires=2)
            qml.CNOT(wires=[1, 2])
            qml.Hadamard(wires=4)
            qml.CNOT(wires=[2, 4])
            qml.Hadamard(wires=4)
            qml.CNOT(wires=[1, 4])
            qml.T(wires=4)
            qml.CNOT(wires=[0, 4])
            qml.adjoint(qml.T)(wires=4)
            qml.CNOT(wires=[1, 4])
            qml.T(wires=4)
            qml.CNOT(wires=[0, 4])
            qml.adjoint(qml.T)(wires=4)
            qml.CNOT(wires=[0, 1])
            qml.T(wires=1)
            qml.CNOT(wires=[0, 1])
            qml.Hadamard(wires=4)
            qml.CNOT(wires=[1, 4])
            qml.CNOT(wires=[0, 4])
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_template:
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])

        template = qml.tape.QuantumScript.from_queue(q_template)
        dev = qml.device("default.qubit", wires=5)

        qnode = qml.QNode(mod_5_4, dev)
        qnode_res = qnode()

        optimized_qfunc = pattern_matching_optimization(mod_5_4, pattern_tapes=[template])
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode_res = optimized_qnode()

        cnots_qnode = qml.specs(qnode)()["resources"].gate_types["CNOT"]
        cnots_optimized_qnode = qml.specs(optimized_qnode)()["resources"].gate_types["CNOT"]

        tape = qml.workflow.construct_tape(qnode)()
        assert len(tape.operations) == 51
        assert cnots_qnode == 28

        optimized_tape = qml.workflow.construct_tape(optimized_qnode)()
        assert len(optimized_tape.operations) == 49
        assert cnots_optimized_qnode == 26

        assert qnode_res == optimized_qnode_res
        assert np.allclose(qml.matrix(optimized_qnode)(), qml.matrix(qnode)())

    @pytest.mark.slow
    def test_vbe_adder_3_pattern_matching(self):
        """Test pattern matching algorithm for vbe_adder_3 with a CNOTs template."""

        def vbe_adder_3():
            qml.T(wires=7)
            qml.T(wires=8)
            qml.Hadamard(wires=3)
            qml.CNOT(wires=[2, 3])
            qml.adjoint(qml.T)(wires=3)
            qml.CNOT(wires=[1, 3])
            qml.CNOT(wires=[2, 3])
            qml.adjoint(qml.T)(wires=3)
            qml.CNOT(wires=[1, 3])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[0, 3])
            qml.T(wires=3)
            qml.CNOT(wires=[2, 3])
            qml.adjoint(qml.T)(wires=3)
            qml.CNOT(wires=[0, 3])
            qml.S(wires=3)
            qml.Hadamard(wires=3)
            qml.Hadamard(wires=6)
            qml.CNOT(wires=[5, 6])
            qml.adjoint(qml.T)(wires=6)
            qml.CNOT(wires=[4, 6])
            qml.CNOT(wires=[5, 6])
            qml.adjoint(qml.T)(wires=6)
            qml.CNOT(wires=[4, 6])
            qml.CNOT(wires=[4, 5])
            qml.CNOT(wires=[4, 6])
            qml.CNOT(wires=[3, 6])
            qml.T(wires=6)
            qml.CNOT(wires=[5, 6])
            qml.adjoint(qml.T)(wires=6)
            qml.CNOT(wires=[3, 6])
            qml.S(wires=6)
            qml.Hadamard(wires=6)
            qml.T(wires=6)
            qml.Hadamard(wires=9)
            qml.CNOT(wires=[8, 9])
            qml.adjoint(qml.T)(wires=9)
            qml.CNOT(wires=[7, 9])
            qml.CNOT(wires=[8, 9])
            qml.adjoint(qml.T)(wires=9)
            qml.CNOT(wires=[7, 9])
            qml.CNOT(wires=[7, 8])
            qml.CNOT(wires=[8, 9])
            qml.CNOT(wires=[6, 9])
            qml.T(wires=9)
            qml.CNOT(wires=[8, 9])
            qml.adjoint(qml.T)(wires=9)
            qml.CNOT(wires=[6, 9])
            qml.T(wires=9)
            qml.CNOT(wires=[6, 8])
            qml.adjoint(qml.T)(wires=8)
            qml.Hadamard(wires=9)
            qml.Hadamard(wires=6)
            qml.adjoint(qml.T)(wires=6)
            qml.CNOT(wires=[5, 6])
            qml.CNOT(wires=[3, 6])
            qml.adjoint(qml.T)(wires=6)
            qml.CNOT(wires=[5, 6])
            qml.T(wires=6)
            qml.CNOT(wires=[3, 6])
            qml.CNOT(wires=[4, 5])
            qml.CNOT(wires=[5, 6])
            qml.T(wires=6)
            qml.CNOT(wires=[4, 6])
            qml.CNOT(wires=[5, 6])
            qml.T(wires=6)
            qml.CNOT(wires=[4, 6])
            qml.CNOT(wires=[3, 5])
            qml.CNOT(wires=[4, 5])
            qml.Hadamard(wires=6)
            qml.Hadamard(wires=3)
            qml.adjoint(qml.S)(wires=3)
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[0, 3])
            qml.adjoint(qml.T)(wires=3)
            qml.CNOT(wires=[2, 3])
            qml.T(wires=3)
            qml.CNOT(wires=[0, 3])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.T(wires=3)
            qml.CNOT(wires=[1, 3])
            qml.CNOT(wires=[2, 3])
            qml.T(wires=3)
            qml.CNOT(wires=[1, 3])
            qml.CNOT(wires=[0, 2])
            qml.CNOT(wires=[1, 2])
            qml.Hadamard(wires=3)
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_template:
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])

        template = qml.tape.QuantumScript.from_queue(q_template)
        dev = qml.device("default.qubit", wires=10)

        qnode = qml.QNode(vbe_adder_3, dev)
        qnode_res = qnode()

        optimized_qfunc = pattern_matching_optimization(vbe_adder_3, pattern_tapes=[template])
        optimized_qnode = qml.QNode(optimized_qfunc, dev)
        optimized_qnode_res = optimized_qnode()

        cnots_qnode = qml.specs(qnode)()["resources"].gate_types["CNOT"]
        cnots_optimized_qnode = qml.specs(optimized_qnode)()["resources"].gate_types["CNOT"]

        tape = qml.workflow.construct_tape(qnode)()
        assert len(tape.operations) == 89
        assert cnots_qnode == 50

        optimized_tape = qml.workflow.construct_tape(optimized_qnode)()
        assert len(optimized_tape.operations) == 84
        assert cnots_optimized_qnode == 45

        assert qnode_res == optimized_qnode_res
        assert np.allclose(qml.matrix(optimized_qnode)(), qml.matrix(qnode)())

    def test_transform_tape(self):
        """Test that the transform works as expected with a tape."""

        operations = [
            qml.S(0),
            qml.Z(0),
            qml.S(1),
            qml.CZ([0, 1]),
            qml.S(1),
            qml.S(2),
            qml.CZ([1, 2]),
            qml.S(2),
        ]
        measurement = [qml.expval(qml.X(0))]
        tape = qml.tape.QuantumScript(operations, measurement)

        pattern = qml.tape.QuantumScript([qml.S(0), qml.S(0), qml.Z(0)])

        batch, postprocessing_fn = qml.transforms.pattern_matching_optimization(
            tape, pattern_tapes=[pattern]
        )

        dev = qml.device("default.qubit")
        result = dev.execute(batch)

        assert batch[0].measurements == measurement
        assert batch[0].operations == [
            qml.adjoint(qml.S(0)),
            qml.Z(1),
            qml.Z(2),
            qml.CZ([0, 1]),
            qml.CZ([1, 2]),
        ]

        assert np.allclose(result, 0.0)

        # pattern_matching_optimization returns a null postprocessing function
        assert postprocessing_fn(result) == result[0]

    def test_wrong_pattern_type(self):
        """Test that we cannot give a quantum function as pattern."""

        def circuit():
            qml.Toffoli(wires=[3, 4, 0])
            qml.CNOT(wires=[1, 4])
            qml.CNOT(wires=[2, 1])
            qml.Hadamard(wires=3)
            qml.PauliZ(wires=1)
            qml.CNOT(wires=[2, 3])
            qml.Toffoli(wires=[2, 3, 0])
            qml.CNOT(wires=[1, 4])
            return qml.expval(qml.PauliX(wires=0))

        def template():
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])

        dev = qml.device("default.qubit", wires=10)

        with pytest.raises(QuantumFunctionError, match="The pattern is not a valid quantum tape."):
            optimized_qfunc = pattern_matching_optimization(circuit, pattern_tapes=[template])
            optimized_qnode = qml.QNode(optimized_qfunc, dev)
            optimized_qnode()

    def test_not_identity(self):
        """Test that we cannot give a pattern that does not implement the identity as argument."""

        def circuit():
            qml.Toffoli(wires=[3, 4, 0])
            qml.CNOT(wires=[1, 4])
            qml.CNOT(wires=[2, 1])
            qml.Hadamard(wires=3)
            qml.PauliZ(wires=1)
            qml.CNOT(wires=[2, 3])
            qml.Toffoli(wires=[2, 3, 0])
            qml.CNOT(wires=[1, 4])
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_template:
            qml.CNOT(wires=[1, 2])

        template = qml.tape.QuantumScript.from_queue(q_template)
        dev = qml.device("default.qubit", wires=10)

        with pytest.raises(
            QuantumFunctionError,
            match="Pattern is not valid, it does not implement identity.",
        ):
            optimized_qfunc = pattern_matching_optimization(circuit, pattern_tapes=[template])
            optimized_qnode = qml.QNode(optimized_qfunc, dev)
            optimized_qnode()

    def test_less_qubit_circuit(self):
        """Test that we cannot have less qubit in the circuit than in the pattern."""

        def circuit():
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_template:
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 1])

        template = qml.tape.QuantumScript.from_queue(q_template)
        dev = qml.device("default.qubit", wires=1)

        with pytest.raises(QuantumFunctionError, match="Circuit has less qubits than the pattern."):
            optimized_qfunc = pattern_matching_optimization(circuit, pattern_tapes=[template])
            optimized_qnode = qml.QNode(optimized_qfunc, dev)
            optimized_qnode()

    def test_pattern_no_measurements(self):
        """Test that pattern cannot contain measurements."""

        def circuit():
            qml.Toffoli(wires=[3, 4, 0])
            qml.CNOT(wires=[1, 4])
            qml.CNOT(wires=[2, 1])
            qml.Hadamard(wires=3)
            qml.PauliZ(wires=1)
            qml.CNOT(wires=[2, 3])
            qml.Toffoli(wires=[2, 3, 0])
            qml.CNOT(wires=[1, 4])
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_template:
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])
            qml.expval(qml.PauliX(wires=0))

        template = qml.tape.QuantumScript.from_queue(q_template)
        dev = qml.device("default.qubit", wires=10)

        with pytest.raises(QuantumFunctionError, match="The pattern contains measurements."):
            optimized_qfunc = pattern_matching_optimization(circuit, pattern_tapes=[template])
            optimized_qnode = qml.QNode(optimized_qfunc, dev)
            optimized_qnode()


class TestPatternMatching:
    """Pattern matching tests."""

    def test_pattern_matching_paper_example(self):
        """Test results of the pattern matching for the paper example."""

        def circuit():
            qml.CNOT(wires=[6, 7])
            qml.CNOT(wires=[7, 5])
            qml.CNOT(wires=[6, 7])
            qml.Toffoli(wires=[7, 6, 5])
            qml.CNOT(wires=[6, 7])
            qml.CNOT(wires=[1, 4])
            qml.CNOT(wires=[6, 3])
            qml.CNOT(wires=[3, 4])
            qml.CNOT(wires=[4, 5])
            qml.CNOT(wires=[0, 5])
            qml.PauliZ(wires=3)
            qml.PauliX(wires=4)
            qml.CNOT(wires=[4, 3])
            qml.CNOT(wires=[3, 1])
            qml.PauliX(wires=4)
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[3, 1])
            qml.CNOT(wires=[3, 5])
            qml.CNOT(wires=[3, 6])
            qml.PauliX(wires=3)
            qml.CNOT(wires=[4, 5])
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_pattern:
            qml.CNOT(wires=[3, 0])
            qml.PauliX(wires=4)
            qml.PauliZ(wires=0)
            qml.CNOT(wires=[4, 2])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[3, 4])
            qml.CNOT(wires=[1, 2])
            qml.PauliX(wires=1)
            qml.CNOT(wires=[1, 0])
            qml.PauliX(wires=1)
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 3])

        pattern = qml.tape.QuantumScript.from_queue(q_pattern)
        circuit_dag = commutation_dag(circuit)()
        pattern_dag = commutation_dag(pattern)

        wires, target_wires, control_wires = _update_qubits(circuit_dag, [0, 5, 1, 2, 4])

        forward = ForwardMatch(
            circuit_dag,
            pattern_dag,
            6,
            0,
            wires,
            target_wires,
            control_wires,
        )
        forward.run_forward_match()

        forward_match = forward.match
        forward_match.sort()

        forward_match_expected = [
            [0, 6],
            [2, 10],
            [4, 7],
            [6, 8],
            [7, 11],
            [8, 12],
            [9, 14],
            [10, 20],
            [11, 18],
        ]

        assert forward_match_expected == forward_match
        qubits = [0, 5, 1, 2, 4]

        backward = BackwardMatch(
            circuit_dag,
            pattern_dag,
            qubits,
            forward.match,
            forward.circuit_matched_with,
            forward.circuit_blocked,
            forward.pattern_matched_with,
            6,
            0,
            wires,
            control_wires,
            target_wires,
        )
        backward.run_backward_match()

        # Figure 5 in the paper
        backward_match_1 = backward.match_final[0].match
        backward_match_qubit_1 = backward.match_final[0].qubit[0]
        backward_match_1.sort()

        # Figure 6 in the paper
        backward_match_2 = backward.match_final[1].match
        backward_match_qubit_2 = backward.match_final[0].qubit[0]
        backward_match_2.sort()

        backward_match_1_expected = [
            [0, 6],
            [2, 10],
            [4, 7],
            [5, 4],
            [6, 8],
            [7, 11],
            [8, 12],
            [9, 14],
            [10, 20],
            [11, 18],
        ]
        backward_match_2_expected = [
            [0, 6],
            [2, 10],
            [3, 1],
            [4, 7],
            [5, 2],
            [6, 8],
            [7, 11],
            [8, 12],
            [9, 14],
            [10, 20],
        ]

        assert backward_match_1_expected == backward_match_1
        assert backward_match_2_expected == backward_match_2
        assert qubits == backward_match_qubit_1 == backward_match_qubit_2

    def test_forward_diamond_pattern(self):
        """Test for a pattern with a diamond shape graph."""

        def circuit():
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[0, 3])
            qml.CNOT(wires=[0, 3])
            qml.CNOT(wires=[2, 1])
            qml.CNOT(wires=[1, 3])
            qml.PauliX(wires=2)
            qml.PauliX(wires=0)
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_pattern:
            qml.CNOT(wires=[1, 0])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=[0])

        pattern = qml.tape.QuantumScript.from_queue(q_pattern)
        circuit_dag = commutation_dag(circuit)()
        pattern_dag = commutation_dag(pattern)

        max_matches = pattern_matching(circuit_dag, pattern_dag)
        expected_longest_match = [[1, 1], [2, 2], [3, 6]]

        assert expected_longest_match == max_matches[0].match

    def test_forward_diamond_pattern_and_circuit(self):
        """Test for a pattern and circuit with a diamond shape graph."""

        def circuit():
            qml.CNOT(wires=[0, 2])
            qml.S(wires=[1])
            qml.Hadamard(wires=3)
            qml.CNOT(wires=[0, 3])
            qml.S(wires=1)
            qml.Hadamard(wires=2)
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[1, 2])
            qml.Hadamard(wires=3)
            qml.CNOT(wires=[0, 2])
            qml.CNOT(wires=[1, 3])
            qml.S(wires=0)
            qml.S(wires=2)
            qml.Hadamard(wires=3)
            return qml.expval(qml.PauliX(wires=0))

        with qml.queuing.AnnotatedQueue() as q_pattern:
            qml.S(wires=0)
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.T(wires=1)
            qml.T(wires=1)
            qml.CNOT(wires=[0, 1])

        pattern = qml.tape.QuantumScript.from_queue(q_pattern)
        circuit_dag = commutation_dag(circuit)()
        pattern_dag = commutation_dag(pattern)

        max_matches = [x.match for x in pattern_matching(circuit_dag, pattern_dag)]
        assert [[1, 6], [5, 9]] in max_matches
