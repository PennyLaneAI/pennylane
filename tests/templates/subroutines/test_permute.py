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
Tests for the Permute template.
"""
import numpy as np

# pylint: disable=too-many-arguments
import pytest

import pennylane as qml
from pennylane.ops.functions.assert_valid import _test_decomposition_rule


@pytest.mark.jax
def test_standard_validity():
    """Check the operation using the assert_valid function."""
    op = qml.Permute([0, 1, 2, 3], wires=(3, 2, 1, 0))
    qml.ops.functions.assert_valid(op)


def test_repr():
    op = qml.Permute([2, 1, 0], wires=(0, 1, 2))
    assert repr(op) == "Permute((2, 1, 0), wires=[0, 1, 2])"


class TestDecomposition:
    """Tests that the template defines the correct decomposition."""

    @pytest.mark.parametrize(
        "permutation_order,wire_order",
        [
            ([1, 0], [0, 1]),
            ([1, 0, 2], [0, 1, 2]),
            ([1, 0, 2, 3], [0, 1, 2, 3]),
            ([0, 2, 1, 3], [0, 1, 2, 3]),
            ([2, 3, 0, 1], [0, 1, 2, 3]),
        ],
    )
    def test_decomposition_new(self, permutation_order, wire_order):
        """Tests the decomposition rule implemented with the new system."""
        op = qml.Permute(permutation_order, wire_order)

        for rule in qml.list_decomps(qml.Permute):
            _test_decomposition_rule(op, rule)

    @pytest.mark.parametrize(
        "permutation_order,wire_order",
        [
            ([1, 0], [0, 1]),
            ([1, 0, 2], [0, 1, 2]),
            ([1, 0, 2, 3], [0, 1, 2, 3]),
            ([0, 2, 1, 3], [0, 1, 2, 3]),
            ([2, 3, 0, 1], [0, 1, 2, 3]),
        ],
    )
    @pytest.mark.capture
    def test_decomposition_new_capture(self, permutation_order, wire_order):
        """Tests the decomposition rule implemented with the new system."""
        op = qml.Permute(permutation_order, wire_order)

        for rule in qml.list_decomps(qml.Permute):
            _test_decomposition_rule(op, rule)

    def test_identity_permutation_qnode(self, mocker):
        """Test that identity permutations have no effect on QNodes."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev, interface="autograd")
        def identity_permutation():
            qml.Permute([0, 1, 2, 3], wires=dev.wires)
            return qml.expval(qml.PauliZ(0))

        spy = mocker.spy(identity_permutation.device, "execute")
        identity_permutation()

        # expand the Permute operation
        tape = spy.call_args[0][0][0]

        assert len(tape.operations) == 0

    def test_identity_permutation_tape(self):
        """Test that identity permutations have no effect on tapes."""

        with qml.queuing.AnnotatedQueue() as q:
            qml.Permute([0, "a", "c", "d"], wires=[0, "a", "c", "d"])

        tape = qml.tape.QuantumScript.from_queue(q)
        # expand the Permute operation
        tape = tape.expand()

        assert len(tape.operations) == 0

    @pytest.mark.parametrize(
        "permutation_order,expected_wires",
        [
            ([1, 0], [(0, 1)]),
            ([1, 0, 2], [(0, 1)]),
            ([1, 0, 2, 3], [(0, 1)]),
            ([0, 2, 1, 3], [(1, 2)]),
            ([2, 3, 0, 1], [(0, 2), (1, 3)]),
        ],
    )
    def test_two_cycle_permutations_qnode(self, mocker, permutation_order, expected_wires):
        """Test some two-cycles on QNodes."""

        dev = qml.device("default.qubit", wires=len(permutation_order))

        @qml.qnode(dev, interface="autograd")
        def two_cycle():
            qml.Permute(permutation_order, wires=dev.wires)
            return qml.expval(qml.PauliZ(0))

        spy = mocker.spy(two_cycle.device, "execute")
        two_cycle()

        tape = spy.call_args[0][0][0]

        # Check that the Permute operation was expanded to SWAPs when the QNode
        # is evaluated, and that the wires are the same
        assert all(op.name == "SWAP" for op in tape.operations)
        assert [op.wires.labels for op in tape.operations] == expected_wires

    @pytest.mark.parametrize(
        # For tape need to specify the wire labels
        "permutation_order,wire_order,expected_wires",
        [
            ([1, 0], [0, 1], [(0, 1)]),
            ([1, 0, 2], [0, 1, 2], [(0, 1)]),
            ([1, 0, 2, 3], [0, 1, 2, 3], [(0, 1)]),
            ([0, 2, 1, 3], [0, 1, 2, 3], [(1, 2)]),
            ([2, 3, 0, 1], [0, 1, 2, 3], [(0, 2), (1, 3)]),
            (["a", "b", 0, 1], [0, 1, "a", "b"], [(0, "a"), (1, "b")]),
        ],
    )
    def test_two_cycle_permutations_tape(self, permutation_order, wire_order, expected_wires):
        """Test some two-cycles on tapes."""

        with qml.queuing.AnnotatedQueue() as q:
            qml.Permute(permutation_order, wire_order)

        tape = qml.tape.QuantumScript.from_queue(q)
        # expand the Permute operation
        tape = tape.expand()

        # Ensure all operations are SWAPs, and that the wires are the same
        assert all(op.name == "SWAP" for op in tape.operations)
        assert [op.wires.labels for op in tape.operations] == expected_wires

    @pytest.mark.parametrize(
        "permutation_order,expected_wires",
        [
            ([1, 2, 0], [(0, 1), (1, 2)]),
            ([3, 0, 1, 2], [(0, 3), (1, 3), (2, 3)]),
            ([1, 2, 3, 0], [(0, 1), (1, 2), (2, 3)]),
        ],
    )
    def test_cyclic_permutations_qnode(self, mocker, permutation_order, expected_wires):
        """Test more general cycles on QNodes."""

        dev = qml.device("default.qubit", wires=len(permutation_order))

        @qml.qnode(dev, interface="autograd")
        def cycle():
            qml.Permute(permutation_order, wires=dev.wires)
            return qml.expval(qml.PauliZ(0))

        spy = mocker.spy(cycle.device, "execute")
        cycle()

        tape = spy.call_args[0][0][0]

        # Check that the Permute operation was expanded to SWAPs when the QNode
        # is evaluated, and that the wires are the same
        assert all(op.name == "SWAP" for op in tape.operations)
        assert [op.wires.labels for op in tape.operations] == expected_wires

    @pytest.mark.parametrize(
        "permutation_order,wire_order,expected_wires",
        [
            ([1, 2, 0], [0, 1, 2], [(0, 1), (1, 2)]),
            (["d", "a", "b", "c"], ["a", "b", "c", "d"], [("a", "d"), ("b", "d"), ("c", "d")]),
            (["b", 0, "d", "a"], ["a", "b", 0, "d"], [("a", "b"), ("b", 0), (0, "d")]),
        ],
    )
    def test_cyclic_permutations_tape(self, permutation_order, wire_order, expected_wires):
        """Test more general cycles on tapes."""

        with qml.queuing.AnnotatedQueue() as q:
            qml.Permute(permutation_order, wire_order)

        tape = qml.tape.QuantumScript.from_queue(q)
        # expand the Permute operation
        tape = tape.expand()

        # Ensure all operations are SWAPs, and that the wires are the same
        assert all(op.name == "SWAP" for op in tape.operations)
        assert [op.wires.labels for op in tape.operations] == expected_wires

    @pytest.mark.parametrize(
        "permutation_order,expected_wires",
        [
            ([3, 0, 2, 1], [(0, 3), (1, 3)]),
            ([1, 3, 0, 4, 2], [(0, 1), (1, 3), (2, 3), (3, 4)]),
            ([5, 1, 4, 2, 3, 0], [(0, 5), (2, 4), (3, 4)]),
        ],
    )
    def test_arbitrary_permutations_qnode(self, mocker, permutation_order, expected_wires):
        """Test arbitrarily generated permutations on QNodes."""

        dev = qml.device("default.qubit", wires=len(permutation_order))

        @qml.qnode(dev, interface="autograd")
        def arbitrary_perm():
            qml.Permute(permutation_order, wires=dev.wires)
            return qml.expval(qml.PauliZ(0))

        spy = mocker.spy(arbitrary_perm.device, "execute")
        arbitrary_perm()

        tape = spy.call_args[0][0][0]

        # Check that the Permute operation was expanded to SWAPs when the QNode
        # is evaluated, and that the wires are the same
        assert all(op.name == "SWAP" for op in tape.operations)
        assert [op.wires.labels for op in tape.operations] == expected_wires

    @pytest.mark.parametrize(
        "permutation_order,wire_order,expected_wires",
        [
            ([1, 3, 0, 2], [0, 1, 2, 3], [(0, 1), (1, 3), (2, 3)]),
            (
                ["d", "a", "e", "b", "c"],
                ["a", "b", "c", "d", "e"],
                [("a", "d"), ("b", "d"), ("c", "e")],
            ),
            (
                ["p", "f", 4, "q", "z", 0, "c", "d"],
                ["z", 0, "d", "c", 4, "f", "q", "p"],
                [("z", "p"), (0, "f"), ("d", 4), ("c", "q"), (4, "p")],
            ),
        ],
    )
    def test_arbitrary_permutations_tape(self, permutation_order, wire_order, expected_wires):
        """Test arbitrarily generated permutations on tapes."""

        with qml.queuing.AnnotatedQueue() as q:
            qml.Permute(permutation_order, wire_order)

        tape = qml.tape.QuantumScript.from_queue(q)
        # expand the Permute operation
        tape = tape.expand()

        # Ensure all operations are SWAPs, and that the wires are the same
        assert all(op.name == "SWAP" for op in tape.operations)
        assert [op.wires.labels for op in tape.operations] == expected_wires

    @pytest.mark.parametrize(
        "num_wires,permutation_order,wire_subset,expected_wires",
        [
            (3, [1, 0], [0, 1], [(0, 1)]),
            (4, [3, 0, 2], [0, 2, 3], [(0, 3), (2, 3)]),
            (6, [4, 2, 1, 3], [1, 2, 3, 4], [(1, 4), (3, 4)]),
        ],
    )
    def test_subset_permutations_qnode(
        self, mocker, num_wires, permutation_order, wire_subset, expected_wires
    ):
        """Test permutation of wire subsets on QNodes."""

        dev = qml.device("default.qubit", wires=num_wires)

        @qml.qnode(dev, interface="autograd")
        def subset_perm():
            qml.Permute(permutation_order, wires=wire_subset)
            return qml.expval(qml.PauliZ(0))

        spy = mocker.spy(subset_perm.device, "execute")
        subset_perm()

        tape = spy.call_args[0][0][0]

        # Check that the Permute operation was expanded to SWAPs when the QNode
        # is evaluated, and that the wires are the same
        assert all(op.name == "SWAP" for op in tape.operations)
        assert [op.wires.labels for op in tape.operations] == expected_wires

    @pytest.mark.parametrize(
        "wire_labels,permutation_order,wire_subset,expected_wires",
        [
            ([0, 1, 2], [1, 0], [0, 1], [(0, 1)]),
            ([0, 1, 2, 3], [3, 0, 2], [0, 2, 3], [(0, 3), (2, 3)]),
            (
                [0, 2, "a", "c", 1, 4],
                [4, "c", 2, "a"],
                [2, "a", "c", 4],
                [(2, 4), ("a", "c"), ("c", 4)],
            ),
        ],
    )
    def test_subset_permutations_tape(
        self, wire_labels, permutation_order, wire_subset, expected_wires
    ):
        """Test permutation of wire subsets on tapes."""

        with qml.queuing.AnnotatedQueue() as q:
            # Make sure all the wires are actually there
            for wire in wire_labels:
                qml.RZ(0, wires=wire)
            qml.Permute(permutation_order, wire_subset)

        tape = qml.tape.QuantumScript.from_queue(q)
        # expand the Permute operation
        tape = tape.expand()

        # Make sure to start comparison after the set of RZs have been applied
        assert all(op.name == "SWAP" for op in tape.operations[len(wire_labels) :])
        assert [op.wires.labels for op in tape.operations[len(wire_labels) :]] == expected_wires

    def test_custom_wire_labels(self, tol):
        """Test that template can deal with non-numeric, nonconsecutive wire labels."""

        permutation = [3, 0, 2, 1]
        permutation2 = ["o", "z", "k", "a"]
        dev = qml.device("default.qubit", wires=4)
        dev2 = qml.device("default.qubit", wires=["z", "a", "k", "o"])

        @qml.qnode(dev)
        def circuit():
            qml.Permute(permutation, wires=range(4))
            return qml.expval(qml.Identity(0)), qml.state()

        @qml.qnode(dev2)
        def circuit2():
            qml.Permute(permutation2, wires=["z", "a", "k", "o"])
            return qml.expval(qml.Identity("z")), qml.state()

        res1, state1 = circuit()
        res2, state2 = circuit2()

        assert np.allclose(res1, res2, atol=tol, rtol=0)
        assert np.allclose(state1, state2, atol=tol, rtol=0)

    @pytest.mark.jax
    def test_jax_jit(self):
        """Tests the template is correctly compiled with JAX JIT."""
        import jax

        dev = qml.device("default.qubit", wires=5)

        @qml.qnode(dev)
        def apply_perm():
            qml.Permute([4, 2, 0, 1, 3], wires=dev.wires)
            return qml.expval(qml.Z(0))

        jit_perm = jax.jit(apply_perm)

        assert qml.math.allclose(apply_perm(), jit_perm())


class TestInputs:
    """Test inputs and pre-processing."""

    @pytest.mark.parametrize(
        "permutation_order,expected_error_message",
        [
            ([0], "Permutations must involve at least 2 qubits."),
            ([0, 1, 2], "Permutation must specify outcome of all wires."),
            ([0, 1, 1, 3], "Values in a permutation must all be unique"),
            ([4, 3, 2, 1], "not present in wire set"),
        ],
    )
    def test_invalid_inputs_qnodes(self, permutation_order, expected_error_message):
        """Tests if errors are thrown for invalid permutations with QNodes."""

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def permute_qubits():
            qml.Permute(permutation_order, wires=dev.wires)
            return qml.expval(qml.PauliZ(0))

        with pytest.raises(ValueError, match=expected_error_message):
            permute_qubits()

    @pytest.mark.parametrize(
        "permutation_order,expected_error_message",
        [
            ([0], "Permutations must involve at least 2 qubits."),
            ([2, "c", "a", 0], "Permutation must specify outcome of all wires."),
            ([2, "a", "c", "c", 1], "Values in a permutation must all be unique"),
            ([2, "a", "d", "c", 1], r"not present in wire set"),
        ],
    )
    def test_invalid_inputs_tape(self, permutation_order, expected_error_message):
        """Tests if errors are thrown for invalid permutations with tapes."""

        wire_labels = [0, 2, "a", "c", 1]

        with qml.queuing.AnnotatedQueue() as q:
            with pytest.raises(ValueError, match=expected_error_message):
                qml.Permute(permutation_order, wires=wire_labels)

        qml.tape.QuantumScript.from_queue(q)

    def test_id(self):
        """Tests that the id attribute can be set."""
        template = qml.Permute([0, 1, 2], wires=[0, 1, 2], id="a")
        assert template.id == "a"
