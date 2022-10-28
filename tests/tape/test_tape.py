# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the QuantumTape"""
import copy
import warnings
from collections import defaultdict

import numpy as np
import pytest
from this import d

import pennylane as qml
from pennylane import CircuitGraph
from pennylane.measurements import (
    MeasurementProcess,
    MeasurementShapeError,
    counts,
    expval,
    sample,
    var,
    probs,
)
from pennylane.tape import QuantumTape, TapeError


def TestOperationMonkeypatching():
    """Test that operations are monkeypatched only within the quantum tape"""
    with QuantumTape() as tape:
        op = qml.RX(0.432, wires=0)
        obs = qml.PauliX(wires="a")
        measure = qml.expval(qml.PauliX(wires="a"))

    assert tape.operations == [op]
    assert tape.observables == [obs]

    # now create an old QNode
    dev = qml.device("default.qubit", wires=[0, "a"])

    @qml.qnode(dev)
    def func(x):
        global op
        op = qml.RX(x, wires=0)
        return qml.expval(qml.PauliX(wires="a"))

    # this should evaluate without error
    func(0.432)

    assert func.circuit.operations == [op]


class TestConstruction:
    """Test for queuing and construction"""

    @pytest.fixture
    def make_tape(self):
        ops = []
        obs = []

        with QuantumTape() as tape:
            ops += [qml.RX(0.432, wires=0)]
            ops += [qml.Rot(0.543, 0, 0.23, wires=0)]
            ops += [qml.CNOT(wires=[0, "a"])]
            ops += [qml.RX(0.133, wires=4)]
            obs += [qml.PauliX(wires="a")]
            qml.expval(obs[0])
            obs += [qml.probs(wires=[0, "a"])]

        return tape, ops, obs

    def test_qubit_queuing(self, make_tape):
        """Test that qubit quantum operations correctly queue"""
        tape, ops, obs = make_tape

        assert len(tape.queue) == 7
        assert tape.operations == ops
        assert tape.observables == obs
        assert tape.output_dim == 5
        assert tape.batch_size is None
        assert tape.interface is None

        assert tape.wires == qml.wires.Wires([0, "a", 4])
        assert tape._output_dim == len(obs[0].wires) + 2 ** len(obs[1].wires)
        assert tape._batch_size is None

    def test_observable_processing(self, make_tape):
        """Test that observables are processed correctly"""
        tape, ops, obs = make_tape

        # test that the internal tape._measurements list is created properly
        assert isinstance(tape._measurements[0], MeasurementProcess)
        assert tape._measurements[0].return_type == qml.measurements.Expectation
        assert tape._measurements[0].obs == obs[0]

        assert isinstance(tape._measurements[1], MeasurementProcess)
        assert tape._measurements[1].return_type == qml.measurements.Probability

        # test the public observables property
        assert len(tape.observables) == 2
        assert tape.observables[0].name == "PauliX"
        assert tape.observables[1].return_type == qml.measurements.Probability

        # test the public measurements property
        assert len(tape.measurements) == 2
        assert all(isinstance(m, MeasurementProcess) for m in tape.measurements)
        assert tape.observables[0].return_type == qml.measurements.Expectation
        assert tape.observables[1].return_type == qml.measurements.Probability

    def test_tensor_observables_matmul(self):
        """Test that tensor observables are correctly processed from the annotated
        queue. Here, we test multiple tensor observables constructed via matmul."""

        with QuantumTape() as tape:
            op = qml.RX(1.0, wires=0)
            t_obs1 = qml.PauliZ(0) @ qml.PauliX(1)
            t_obs2 = t_obs1 @ qml.PauliZ(3)
            m = qml.expval(t_obs2)

        assert tape.operations == [op]
        assert tape.observables == [t_obs2]
        assert tape.measurements[0].return_type is qml.measurements.Expectation
        assert tape.measurements[0].obs is t_obs2

    def test_tensor_observables_rmatmul(self):
        """Test that tensor observables are correctly processed from the annotated
        queue. Here, we test multiple tensor observables constructed via matmul
        with the observable occuring on the left hand side."""

        with QuantumTape() as tape:
            op = qml.RX(1.0, wires=0)
            t_obs1 = qml.PauliZ(1) @ qml.PauliX(0)
            t_obs2 = qml.Hadamard(2) @ t_obs1
            m = qml.expval(t_obs2)

        assert tape.operations == [op]
        assert tape.observables == [t_obs2]
        assert tape.measurements[0].return_type is qml.measurements.Expectation
        assert tape.measurements[0].obs is t_obs2

    def test_tensor_observables_tensor_init(self):
        """Test that tensor observables are correctly processed from the annotated
        queue. Here, we test multiple tensor observables constructed via explicit
        Tensor creation."""

        with QuantumTape() as tape:
            op = qml.RX(1.0, wires=0)
            t_obs1 = qml.PauliZ(1) @ qml.PauliX(0)
            t_obs2 = qml.operation.Tensor(t_obs1, qml.Hadamard(2))
            m = qml.expval(t_obs2)

        assert tape.operations == [op]
        assert tape.observables == [t_obs2]
        assert tape.measurements[0].return_type is qml.measurements.Expectation
        assert tape.measurements[0].obs is t_obs2

    def test_tensor_observables_tensor_matmul(self):
        """Test that tensor observables are correctly processed from the annotated
        queue". Here, wetest multiple tensor observables constructed via matmul
        between two tensor observables."""

        with QuantumTape() as tape:
            op = qml.RX(1.0, wires=0)
            t_obs1 = qml.PauliZ(0) @ qml.PauliX(1)
            t_obs2 = qml.PauliY(2) @ qml.PauliZ(3)
            t_obs = t_obs1 @ t_obs2
            m = qml.var(t_obs)

        assert tape.operations == [op]
        assert tape.observables == [t_obs]
        assert tape.measurements[0].return_type is qml.measurements.Variance
        assert tape.measurements[0].obs is t_obs

    def test_qubit_diagonalization(self, make_tape):
        """Test that qubit diagonalization works as expected"""
        tape, ops, obs = make_tape

        obs_rotations = [o.diagonalizing_gates() for o in obs]
        obs_rotations = [item for sublist in obs_rotations for item in sublist]

        for o1, o2 in zip(tape.diagonalizing_gates, obs_rotations):
            assert isinstance(o1, o2.__class__)
            assert o1.wires == o2.wires

    def test_tensor_process_queuing(self):
        """Test that tensors are correctly queued"""
        with QuantumTape() as tape:
            A = qml.PauliX(wires=0)
            B = qml.PauliZ(wires=1)
            C = A @ B
            D = qml.expval(C)

        assert len(tape.queue) == 4
        assert not tape.operations
        assert tape.measurements == [D]
        assert tape.observables == [C]
        assert tape.output_dim == 1
        assert tape.batch_size is None

    def test_multiple_contexts(self):
        """Test multiple contexts with a single tape."""
        ops = []
        obs = []

        with QuantumTape() as tape:
            ops += [qml.RX(0.432, wires=0)]

        a = qml.Rot(0.543, 0, 0.23, wires=1)
        b = qml.CNOT(wires=[2, "a"])

        with tape:
            ops += [qml.RX(0.133, wires=0)]
            obs += [qml.PauliX(wires="a")]
            qml.expval(obs[0])
            obs += [qml.probs(wires=[0, "a"])]

        assert len(tape.queue) == 5
        assert tape.operations == ops
        assert tape.observables == obs
        assert tape.output_dim == 5
        assert tape.batch_size is None

        assert a not in tape.operations
        assert b not in tape.operations

        assert tape.wires == qml.wires.Wires([0, "a"])

    def test_state_preparation(self):
        """Test that state preparations are correctly processed"""
        params = [np.array([1, 0, 1, 0]) / np.sqrt(2), 1]

        with QuantumTape() as tape:
            A = qml.QubitStateVector(params[0], wires=[0, 1])
            B = qml.RX(params[1], wires=0)
            qml.expval(qml.PauliZ(wires=1))

        assert tape.operations == [A, B]
        assert tape._prep == [A]
        assert tape.get_parameters() == params

    def test_state_preparation_error(self):
        """Test that an exception is raised if a state preparation comes
        after a quantum operation"""
        with pytest.raises(ValueError, match="must occur prior to ops"):
            with QuantumTape() as tape:
                B = qml.PauliX(wires=0)
                qml.BasisState(np.array([0, 1]), wires=[0, 1])

    def test_measurement_before_operation(self):
        """Test that an exception is raised if a measurement occurs before a operation"""

        with pytest.raises(ValueError, match="must occur prior to measurements"):
            with QuantumTape() as tape:
                qml.expval(qml.PauliZ(wires=1))
                qml.RX(0.5, wires=0)
                qml.expval(qml.PauliZ(wires=1))

    def test_sampling(self):
        """Test that the tape correctly marks itself as returning samples"""
        with QuantumTape() as tape:
            qml.expval(qml.PauliZ(wires=1))

        assert not tape.is_sampled

        with QuantumTape() as tape:
            qml.sample(qml.PauliZ(wires=0))

        assert tape.is_sampled

    def test_repr(self):
        """Test the string representation"""

        with QuantumTape() as tape:
            qml.RX(0.432, wires=0)

        s = tape.__repr__()
        expected = "<QuantumTape: wires=[0], params=1>"
        assert s == expected

    def test_circuit_property(self):
        """Test that the underlying circuit property returns the correct
        operations and measurements making up the circuit."""
        r = 1.234
        terminal_measurement = qml.expval(qml.PauliZ(0))

        def f(x):
            qml.PauliX(1)
            qml.RY(x, wires=1)
            qml.PauliZ(1)

        with qml.tape.QuantumTape() as tape:
            m_0 = qml.measure(0)
            qml.cond(m_0, f)(r)
            qml.apply(terminal_measurement)

        target_wire = qml.wires.Wires(1)

        assert len(tape.circuit) == 5
        assert tape.circuit[0].return_type == qml.measurements.MidMeasure

        assert isinstance(tape.circuit[1], qml.transforms.condition.Conditional)
        assert isinstance(tape.circuit[1].then_op, qml.PauliX)
        assert tape.circuit[1].then_op.wires == target_wire

        assert isinstance(tape.circuit[2], qml.transforms.condition.Conditional)
        assert isinstance(tape.circuit[2].then_op, qml.RY)
        assert tape.circuit[2].then_op.wires == target_wire
        assert tape.circuit[2].then_op.data == [r]

        assert isinstance(tape.circuit[3], qml.transforms.condition.Conditional)
        assert isinstance(tape.circuit[3].then_op, qml.PauliZ)
        assert tape.circuit[3].then_op.wires == target_wire

        assert tape.circuit[4] is terminal_measurement

    @pytest.mark.parametrize(
        "x, rot, exp_batch_size",
        [
            (0.2, [0.1, -0.9, 2.1], None),
            ([0.2], [0.1, -0.9, 2.1], 1),
            ([0.2], [[0.1], [-0.9], 2.1], 1),
            ([0.2] * 3, [0.1, [-0.9] * 3, 2.1], 3),
        ],
    )
    def test_update_batch_size(self, x, rot, exp_batch_size):
        """Test that the batch size is correctly inferred from all operation's
        batch_size, when creating and when using `set_parameters`."""

        # Test with tape construction
        with qml.tape.QuantumTape() as tape:
            qml.RX(x, wires=0)
            qml.Rot(*rot, wires=1)
            qml.apply(qml.expval(qml.PauliZ(0)))
            qml.apply(qml.expval(qml.PauliX(1)))

        assert tape.batch_size == exp_batch_size

        # Test with set_parameters
        with qml.tape.QuantumTape() as tape:
            qml.RX(0.2, wires=0)
            qml.Rot(1.0, 0.2, -0.3, wires=1)
            qml.apply(qml.expval(qml.PauliZ(0)))
            qml.apply(qml.expval(qml.PauliX(1)))

        assert tape.batch_size is None

        tape.set_parameters([x] + rot)
        assert tape.batch_size == exp_batch_size

    @pytest.mark.parametrize(
        "x, rot, y",
        [
            (0.2, [[0.1], -0.9, 2.1], [0.1, 0.9]),
            ([0.2], [0.1, [-0.9] * 2, 2.1], 0.1),
        ],
    )
    def test_error_inconsistent_batch_sizes(self, x, rot, y):
        """Test that the batch size is correctly inferred from all operation's
        batch_size, when creating and when using `set_parameters`."""

        with pytest.raises(
            ValueError, match="batch sizes of the quantum script operations do not match."
        ):
            with qml.tape.QuantumTape() as tape:
                qml.RX(x, wires=0)
                qml.Rot(*rot, wires=1)
                qml.RX(y, wires=1)
                qml.apply(qml.expval(qml.PauliZ(0)))

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.2, wires=0)
            qml.Rot(1.0, 0.2, -0.3, wires=1)
            qml.RX(0.2, wires=1)
            qml.apply(qml.expval(qml.PauliZ(0)))
        with pytest.raises(
            ValueError, match="batch sizes of the quantum script operations do not match."
        ):
            tape.set_parameters([x] + rot + [y])


class TestIteration:
    """Test the capabilities related to iterating over tapes."""

    @pytest.fixture
    def make_tape(self):
        ops = []
        meas = []

        with QuantumTape() as tape:
            ops += [qml.RX(0.432, wires=0)]
            ops += [qml.Rot(0.543, 0, 0.23, wires=0)]
            ops += [qml.CNOT(wires=[0, "a"])]
            ops += [qml.RX(0.133, wires=4)]
            meas += [qml.expval(qml.PauliX(wires="a"))]
            meas += [qml.probs(wires=[0, "a"])]

        return tape, ops, meas

    def test_tape_is_iterable(self, make_tape):
        """Test the iterable protocol: that we can iterate over a tape because
        an iterator object can be obtained using the iter function."""
        tape, ops, meas = make_tape

        expected = ops + meas

        tape_iterator = iter(tape)

        iterating = True

        counter = 0

        while iterating:
            try:
                next_tape_elem = next(tape_iterator)

                assert next_tape_elem is expected[counter]
                counter += 1

            except StopIteration:

                # StopIteration is raised by next when there are no more
                # elements to iterate over
                iterating = False

        assert counter == len(expected)

    def test_tape_is_sequence(self, make_tape):
        """Test the sequence protocol: that a tape is a sequence because its
        __len__ and __getitem__ methods work as expected."""
        tape, ops, meas = make_tape

        expected = ops + meas

        for idx, exp_elem in enumerate(expected):
            tape[idx] is exp_elem

        assert len(tape) == len(expected)

    def test_tape_as_list(self, make_tape):
        """Test that a tape can be converted to a list."""
        tape, ops, meas = make_tape
        tape = list(tape)

        expected = ops + meas
        for op, exp_op in zip(tape, expected):
            assert op is exp_op

        assert len(tape) == len(expected)

    def test_iteration_preserves_circuit(self):
        """Test that iterating through a tape doesn't change the underlying
        list of operations and measurements in the circuit."""

        circuit = [
            qml.RX(0.432, wires=0),
            qml.Rot(0.543, 0, 0.23, wires=0),
            qml.CNOT(wires=[0, "a"]),
            qml.RX(0.133, wires=4),
            qml.expval(qml.PauliX(wires="a")),
            qml.probs(wires=[0, "a"]),
        ]

        with QuantumTape() as tape:
            for op in circuit:
                qml.apply(op)

        # Check that the underlying circuit is as expected
        assert tape.circuit == circuit

        # Iterate over the tape
        for op in tape:
            op

        # Check that the underlying circuit is still as expected
        assert tape.circuit == circuit


class TestGraph:
    """Tests involving graph creation"""

    def test_graph_creation(self, mocker):
        """Test that the circuit graph is correctly created"""
        spy = mocker.spy(CircuitGraph, "__init__")

        with QuantumTape() as tape:
            op = qml.RX(1.0, wires=0)
            obs = qml.PauliZ(1)
            qml.expval(obs)

        # graph has not yet been created
        assert tape._graph is None
        spy.assert_not_called()

        # requesting the graph creates it
        g = tape.graph
        assert g.operations == [op]
        assert g.observables == [obs]
        assert tape._graph is not None
        spy.assert_called_once()

        # calling the graph property again does
        # not reconstruct the graph
        g2 = tape.graph
        assert g2 is g
        spy.assert_called_once()


class TestResourceEstimation:
    """Tests for verifying resource counts and depths of tapes."""

    @pytest.fixture
    def make_empty_tape(self):
        with QuantumTape() as tape:
            qml.probs(wires=[0, 1])

        return tape

    @pytest.fixture
    def make_tape(self):
        params = [0.432, 0.123, 0.546, 0.32, 0.76]

        with QuantumTape() as tape:
            qml.RX(params[0], wires=0)
            qml.Rot(*params[1:4], wires=0)
            qml.CNOT(wires=[0, "a"])
            qml.RX(params[4], wires=4)
            qml.expval(qml.PauliX(wires="a"))
            qml.probs(wires=[0, "a"])

        return tape

    @pytest.fixture
    def make_extendible_tape(self):
        params = [0.432, 0.123, 0.546, 0.32, 0.76]

        with QuantumTape() as tape:
            qml.RX(params[0], wires=0)
            qml.Rot(*params[1:4], wires=0)
            qml.CNOT(wires=[0, "a"])
            qml.RX(params[4], wires=4)

        return tape

    def test_specs_empty_tape(self, make_empty_tape):
        """Test specs attribute on an empty tape"""
        tape = make_empty_tape

        assert tape.specs["gate_sizes"] == defaultdict(int)
        assert tape.specs["gate_types"] == defaultdict(int)

        assert tape.specs["num_operations"] == 0
        assert tape.specs["num_observables"] == 1
        assert tape.specs["num_diagonalizing_gates"] == 0
        assert tape.specs["num_used_wires"] == 2
        assert tape.specs["num_trainable_params"] == 0
        assert tape.specs["depth"] == 0

        assert len(tape.specs) == 8

    def test_specs_tape(self, make_tape):
        """Tests that regular tapes return correct specifications"""
        tape = make_tape

        specs = tape.specs

        assert len(specs) == 8

        assert specs["gate_sizes"] == defaultdict(int, {1: 3, 2: 1})
        assert specs["gate_types"] == defaultdict(int, {"RX": 2, "Rot": 1, "CNOT": 1})
        assert specs["num_operations"] == 4
        assert specs["num_observables"] == 2
        assert specs["num_diagonalizing_gates"] == 1
        assert specs["num_used_wires"] == 3
        assert specs["num_trainable_params"] == 5
        assert specs["depth"] == 3

    def test_specs_add_to_tape(self, make_extendible_tape):
        """Test that tapes return correct specs after adding to them."""

        tape = make_extendible_tape
        specs1 = tape.specs

        assert len(specs1) == 8
        assert specs1["gate_sizes"] == defaultdict(int, {1: 3, 2: 1})
        assert specs1["gate_types"] == defaultdict(int, {"RX": 2, "Rot": 1, "CNOT": 1})
        assert specs1["num_operations"] == 4
        assert specs1["num_observables"] == 0
        assert specs1["num_diagonalizing_gates"] == 0
        assert specs1["num_used_wires"] == 3
        assert specs1["num_trainable_params"] == 5
        assert specs1["depth"] == 3

        with tape as tape:
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.1, wires=3)
            qml.expval(qml.PauliX(wires="a"))
            qml.probs(wires=[0, "a"])

        specs2 = tape.specs

        assert len(specs2) == 8
        assert specs2["gate_sizes"] == defaultdict(int, {1: 4, 2: 2})
        assert specs2["gate_types"] == defaultdict(int, {"RX": 2, "Rot": 1, "CNOT": 2, "RZ": 1})
        assert specs2["num_operations"] == 6
        assert specs2["num_observables"] == 2
        assert specs2["num_diagonalizing_gates"] == 1
        assert specs2["num_used_wires"] == 5
        assert specs2["num_trainable_params"] == 6
        assert specs2["depth"] == 4

    def test_resources_empty_tape(self, make_empty_tape):
        """Test that empty tapes return empty resource counts."""
        tape = make_empty_tape

        assert len(tape.specs["gate_types"]) == 0
        assert tape.specs["depth"] == 0

    def test_resources_tape(self, make_tape):
        """Test that regular tapes return correct number of resources."""
        tape = make_tape

        depth = tape.specs["depth"]
        assert depth == 3

        # Verify resource counts
        resources = tape.specs["gate_types"]
        assert len(resources) == 3
        assert resources["RX"] == 2
        assert resources["Rot"] == 1
        assert resources["CNOT"] == 1

    def test_resources_add_to_tape(self, make_extendible_tape):
        """Test that tapes return correct number of resources after adding to them."""
        tape = make_extendible_tape

        depth = tape.specs["depth"]
        assert depth == 3

        resources = tape.specs["gate_types"]
        assert len(resources) == 3
        assert resources["RX"] == 2
        assert resources["Rot"] == 1
        assert resources["CNOT"] == 1

        with tape as tape:
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.1, wires=3)
            qml.expval(qml.PauliX(wires="a"))
            qml.probs(wires=[0, "a"])

        assert tape.specs["depth"] == 4
        resources = tape.specs["gate_types"]
        assert len(resources) == 4
        assert resources["RX"] == 2
        assert resources["Rot"] == 1
        assert resources["CNOT"] == 2
        assert resources["RZ"] == 1


class TestParameters:
    """Tests for parameter processing, setting, and manipulation"""

    @pytest.fixture
    def make_tape(self):
        params = [0.432, 0.123, 0.546, 0.32, 0.76]

        with QuantumTape() as tape:
            qml.RX(params[0], wires=0)
            qml.Rot(*params[1:4], wires=0)
            qml.CNOT(wires=[0, "a"])
            qml.RX(params[4], wires=4)
            qml.expval(qml.PauliX(wires="a"))
            qml.probs(wires=[0, "a"])

        return tape, params

    @pytest.fixture
    def make_tape_with_hermitian(self):
        params = [0.432, 0.123, 0.546, 0.32, 0.76]
        hermitian = qml.numpy.eye(2, requires_grad=False)

        with QuantumTape() as tape:
            qml.RX(params[0], wires=0)
            qml.Rot(*params[1:4], wires=0)
            qml.CNOT(wires=[0, "a"])
            qml.RX(params[4], wires=4)
            qml.expval(qml.Hermitian(hermitian, wires="a"))

        return tape, params, hermitian

    def test_parameter_processing(self, make_tape):
        """Test that parameters are correctly counted and processed"""
        tape, params = make_tape
        assert tape.num_params == len(params)
        assert tape.trainable_params == list(range(len(params)))
        assert tape.get_parameters() == params

    @pytest.mark.parametrize("operations_only", [False, True])
    def test_parameter_processing_operations_only(self, make_tape_with_hermitian, operations_only):
        """Test the operations_only flag for getting the parameters on a tape with
        qml.Hermitian is measured"""
        tape, circuit_params, hermitian = make_tape_with_hermitian
        num_all_params = len(circuit_params) + 1  # + 1 for hermitian
        assert tape.num_params == num_all_params
        assert tape.trainable_params == list(range(num_all_params))
        assert (
            tape.get_parameters(operations_only=operations_only) == circuit_params
            if operations_only
            else circuit_params + [hermitian]
        )

    def test_set_trainable_params(self, make_tape):
        """Test that setting trainable parameters works as expected"""
        tape, params = make_tape
        trainable = [0, 2, 3]
        tape.trainable_params = trainable
        assert tape._trainable_params == trainable
        assert tape.num_params == 3
        assert tape.get_parameters() == [params[i] for i in tape.trainable_params]

        # add additional trainable parameters
        trainable = {1, 2, 3, 4}
        tape.trainable_params = trainable
        assert tape._trainable_params == [1, 2, 3, 4]
        assert tape.num_params == 4
        assert tape.get_parameters() == [params[i] for i in tape.trainable_params]

        # set trainable_params in wrong order
        trainable = {3, 4, 1}
        tape.trainable_params = trainable
        assert tape._trainable_params == [1, 3, 4]
        assert tape.num_params == 3
        assert tape.get_parameters() == [params[i] for i in tape.trainable_params]

    def test_changing_params(self, make_tape):
        """Test that changing trainable parameters works as expected"""
        tape, params = make_tape
        trainable = (0, 2, 3)
        tape.trainable_params = trainable
        assert tape._trainable_params == list(trainable)
        assert tape.num_params == 3
        assert tape.get_parameters() == [params[i] for i in tape.trainable_params]
        assert tape.get_parameters(trainable_only=False) == params

    def test_set_trainable_params_error(self, make_tape):
        """Test that exceptions are raised if incorrect parameters
        are set as trainable"""
        tape, _ = make_tape

        with pytest.raises(ValueError, match="must be non-negative integers"):
            tape.trainable_params = [-1, 0]

        with pytest.raises(ValueError, match="must be non-negative integers"):
            tape.trainable_params = (0.5,)

        with pytest.raises(ValueError, match="only has 5 parameters"):
            tape.trainable_params = {0, 7}

    def test_setting_parameters(self, make_tape):
        """Test that parameters are correctly modified after construction"""
        tape, params = make_tape
        new_params = [0.6543, -0.654, 0, 0.3, 0.6]

        tape.set_parameters(new_params)

        for pinfo, pval in zip(tape._par_info, new_params):
            assert pinfo["op"].data[pinfo["p_idx"]] == pval

        assert tape.get_parameters() == new_params

        new_params = [0.1, -0.2, 1, 5, 0]
        tape.data = new_params

        for pinfo, pval in zip(tape._par_info, new_params):
            assert pinfo["op"].data[pinfo["p_idx"]] == pval

        assert tape.get_parameters() == new_params

    def test_setting_free_parameters(self, make_tape):
        """Test that free parameters are correctly modified after construction"""
        tape, params = make_tape
        new_params = [-0.654, 0.3]

        tape.trainable_params = [1, 3]
        tape.set_parameters(new_params)

        count = 0
        for idx, pinfo in enumerate(tape._par_info):
            if idx in tape.trainable_params:
                assert pinfo["op"].data[pinfo["p_idx"]] == new_params[count]
                count += 1
            else:
                assert pinfo["op"].data[pinfo["p_idx"]] == params[idx]

        assert tape.get_parameters(trainable_only=False) == [
            params[0],
            new_params[0],
            params[2],
            new_params[1],
            params[4],
        ]

    def test_setting_parameters_unordered(self, make_tape, monkeypatch):
        """Test that an 'unordered' trainable_params set does not affect
        the setting of parameter values"""
        tape, params = make_tape
        new_params = [-0.654, 0.3]

        tape.trainable_params = [3, 1]
        tape.set_parameters(new_params)

        assert tape.get_parameters(trainable_only=True) == new_params
        assert tape.get_parameters(trainable_only=False) == [
            params[0],
            new_params[0],
            params[2],
            new_params[1],
            params[4],
        ]

    def test_setting_all_parameters(self, make_tape):
        """Test that all parameters are correctly modified after construction"""
        tape, params = make_tape
        new_params = [0.6543, -0.654, 0, 0.3, 0.6]

        tape.trainable_params = [1, 3]
        tape.set_parameters(new_params, trainable_only=False)

        for pinfo, pval in zip(tape._par_info, new_params):
            assert pinfo["op"].data[pinfo["p_idx"]] == pval

        assert tape.get_parameters(trainable_only=False) == new_params

    def test_setting_parameters_error(self, make_tape):
        """Test that exceptions are raised if incorrect parameters
        are attempted to be set"""
        tape, _ = make_tape

        with pytest.raises(ValueError, match="Number of provided parameters does not match"):
            tape.set_parameters([0.54])

        with pytest.raises(ValueError, match="Number of provided parameters does not match"):
            tape.trainable_params = [2, 3]
            tape.set_parameters([0.54, 0.54, 0.123])

    def test_array_parameter(self):
        """Test that array parameters integrate properly"""
        a = np.array([1, 1, 0, 0]) / np.sqrt(2)
        params = [a, 0.32, 0.76, 1.0]

        with QuantumTape() as tape:
            op = qml.QubitStateVector(params[0], wires=0)
            qml.Rot(params[1], params[2], params[3], wires=0)

        assert tape.num_params == len(params)
        assert tape.get_parameters() == params

        b = np.array([0, 1, 0, 0])
        new_params = [b, 0.543, 0.654, 0.123]
        tape.set_parameters(new_params)
        assert tape.get_parameters() == new_params

        assert np.all(op.data[0] == b)

    def test_measurement_parameter(self):
        """Test that measurement parameters integrate properly"""
        H = np.array([[1, 0], [0, -1]])
        params = [0.32, 0.76, 1.0, H]

        with QuantumTape() as tape:
            qml.Rot(params[0], params[1], params[2], wires=0)
            obs = qml.Hermitian(params[3], wires=0)
            qml.expval(obs)

        assert tape.num_params == len(params)
        assert tape.get_parameters() == params

        H2 = np.array([[0, 1], [1, 1]])
        new_params = [0.543, 0.654, 0.123, H2]
        tape.set_parameters(new_params)
        assert tape.get_parameters() == new_params

        assert np.all(obs.data[0] == H2)


class TestInverseAdjoint:
    """Tests for tape inversion"""

    def test_inverse(self):
        """Test that inversion works as expected"""
        init_state = np.array([1, 1])
        p = [0.1, 0.2, 0.3, 0.4]

        with QuantumTape() as tape:
            prep = qml.BasisState(init_state, wires=[0, "a"])
            ops = [qml.RX(p[0], wires=0), qml.Rot(*p[1:], wires=0).inv(), qml.CNOT(wires=[0, "a"])]
            m1 = qml.probs(wires=0)
            m2 = qml.probs(wires="a")

        with pytest.warns(UserWarning, match="QuantumTape.inv is now deprecated."):
            tape.inv()

        # check that operation order is reversed
        assert [o.name for o in tape.operations] == ["BasisState", "CNOT", "Rot", "RX"]

        # check that operations are inverted
        assert np.allclose(tape.operations[2].parameters, -np.array(p[-1:0:-1]))

        # check that parameter order has reversed
        assert tape.get_parameters() == [init_state, p[1], p[2], p[3], p[0]]

    def test_adjoint(self):
        """Test that tape.adjoint is a copy of in-place inversion."""

        init_state = np.array([1, 1])
        p = [0.1, 0.2, 0.3, 0.4]

        with QuantumTape() as tape:
            prep = qml.BasisState(init_state, wires=[0, "a"])
            ops = [qml.RX(p[0], wires=0), qml.Rot(*p[1:], wires=0).inv(), qml.CNOT(wires=[0, "a"])]
            m1 = qml.probs(wires=0)
            m2 = qml.probs(wires="a")

        with QuantumTape() as tape2:
            adjoint_tape = tape.adjoint()

        assert tape2[0] is adjoint_tape

        assert id(adjoint_tape) != id(tape)
        assert isinstance(adjoint_tape, QuantumTape)

        tape.inv()

        for op1, op2 in zip(adjoint_tape.circuit, tape.circuit):
            assert op1.__class__ is op2.__class__
            if hasattr(op1, "inverse"):
                assert op1.inverse == op2.inverse
            if hasattr(op1, "data"):
                assert qml.math.allclose(op1.data, op2.data)

    def test_parameter_transforms(self):
        """Test that inversion correctly changes trainable parameters"""
        init_state = np.array([1, 1])
        p = [0.1, 0.2, 0.3, 0.4]

        with QuantumTape() as tape:
            prep = qml.BasisState(init_state, wires=[0, "a"])
            ops = [qml.RX(p[0], wires=0), qml.Rot(*p[1:], wires=0).inv(), qml.CNOT(wires=[0, "a"])]
            m1 = qml.probs(wires=0)
            m2 = qml.probs(wires="a")

        tape.trainable_params = [1, 2]
        tape.inv()

        # check that operation order is reversed
        assert tape.trainable_params == [1, 4]
        assert tape.get_parameters() == [p[1], p[0]]

        # undo the inverse
        tape.inv()
        assert tape.trainable_params == [1, 2]
        assert tape.get_parameters() == [p[0], p[1]]
        assert [o.name for o in tape._ops] == ["RX", "Rot", "CNOT"]


class TestExpand:
    """Tests for tape expansion"""

    def test_decomposition(self):
        """Test expanding a tape with operations that have decompositions"""
        with QuantumTape() as tape:
            qml.Rot(0.1, 0.2, 0.3, wires=0)

        new_tape = tape.expand()

        assert len(new_tape.operations) == 3
        assert new_tape.get_parameters() == [0.1, 0.2, 0.3]
        assert new_tape.trainable_params == [0, 1, 2]

        assert isinstance(new_tape.operations[0], qml.RZ)
        assert isinstance(new_tape.operations[1], qml.RY)
        assert isinstance(new_tape.operations[2], qml.RZ)

        # check that modifying the new tape does not affect the old tape

        new_tape.trainable_params = [0]
        new_tape.set_parameters([10])

        assert tape.get_parameters() == [0.1, 0.2, 0.3]
        assert tape.trainable_params == [0, 1, 2]

    def test_decomposition_removing_parameters(self):
        """Test that decompositions which reduce the number of parameters
        on the tape retain tape consistency."""
        with QuantumTape() as tape:
            qml.BasisState(np.array([1]), wires=0)

        # since expansion calls `BasisStatePreparation` we have to expand twice
        new_tape = tape.expand(depth=2)

        assert len(new_tape.operations) == 1
        assert new_tape.operations[0].name == "PauliX"
        assert new_tape.operations[0].wires.tolist() == [0]
        assert new_tape.num_params == 0
        assert new_tape.get_parameters() == []

        assert isinstance(new_tape.operations[0], qml.PauliX)

    def test_decomposition_adding_parameters(self):
        """Test that decompositions which increase the number of parameters
        on the tape retain tape consistency."""
        with QuantumTape() as tape:
            qml.PauliX(wires=0)

        new_tape = tape.expand()

        assert len(new_tape.operations) == 3

        assert new_tape.operations[0].name == "PhaseShift"
        assert new_tape.operations[1].name == "RX"
        assert new_tape.operations[2].name == "PhaseShift"

        assert new_tape.num_params == 3
        assert new_tape.get_parameters() == [np.pi / 2, np.pi, np.pi / 2]

    def test_nested_tape(self):
        """Test that a nested tape properly expands"""
        with QuantumTape() as tape1:
            with QuantumTape() as tape2:
                op1 = qml.RX(0.543, wires=0)
                op2 = qml.RY(0.1, wires=0)

        assert tape1.num_params == 2
        assert tape1.operations == [tape2]

        new_tape = tape1.expand()
        assert new_tape.num_params == 2
        assert len(new_tape.operations) == 2
        assert isinstance(new_tape.operations[0], qml.RX)
        assert isinstance(new_tape.operations[1], qml.RY)

    def test_nesting_and_decomposition(self):
        """Test an example that contains nested tapes and operation decompositions."""

        with QuantumTape() as tape:
            qml.BasisState(np.array([1, 1]), wires=[0, "a"])

            with QuantumTape() as tape2:
                qml.Rot(0.543, 0.1, 0.4, wires=0)

            qml.CNOT(wires=[0, "a"])
            qml.RY(0.2, wires="a")
            qml.probs(wires=0), qml.probs(wires="a")

        new_tape = tape.expand()
        assert len(new_tape.operations) == 4

    def test_stopping_criterion(self):
        """Test that gates specified in the stop_at
        argument are not expanded."""
        with QuantumTape() as tape:
            qml.U3(0, 1, 2, wires=0)
            qml.Rot(3, 4, 5, wires=0)
            qml.probs(wires=0), qml.probs(wires="a")

        new_tape = tape.expand(stop_at=lambda obj: obj.name in ["Rot"])
        assert len(new_tape.operations) == 4
        assert "Rot" in [i.name for i in new_tape.operations]
        assert not "U3" in [i.name for i in new_tape.operations]

    def test_depth_expansion(self):
        """Test expanding with depth=2"""
        with QuantumTape() as tape:
            # Will be decomposed into PauliX(0), PauliX(0)
            # Each PauliX will then be decomposed into PhaseShift, RX, PhaseShift.
            qml.BasisState(np.array([1, 1]), wires=[0, "a"])

            with QuantumTape() as tape2:
                # will be decomposed into a RZ, RY, RZ
                qml.Rot(0.543, 0.1, 0.4, wires=0)

            qml.CNOT(wires=[0, "a"])
            qml.RY(0.2, wires="a")
            qml.probs(wires=0), qml.probs(wires="a")

        new_tape = tape.expand(depth=3)
        assert len(new_tape.operations) == 11

    def test_stopping_criterion_with_depth(self):
        """Test that gates specified in the stop_at
        argument are not expanded."""
        with QuantumTape() as tape:
            # Will be decomposed into PauliX(0), PauliX(0)
            qml.BasisState(np.array([1, 1]), wires=[0, "a"])

            with QuantumTape() as tape2:
                # will be decomposed into a RZ, RY, RZ
                qml.Rot(0.543, 0.1, 0.4, wires=0)

            qml.CNOT(wires=[0, "a"])
            qml.RY(0.2, wires="a")
            qml.probs(wires=0), qml.probs(wires="a")

        new_tape = tape.expand(depth=2, stop_at=lambda obj: obj.name in ["PauliX"])
        assert len(new_tape.operations) == 7

    def test_measurement_expansion(self):
        """Test that measurement expansion works as expected"""
        with QuantumTape() as tape:
            # expands into 2 PauliX
            qml.BasisState(np.array([1, 1]), wires=[0, "a"])
            qml.CNOT(wires=[0, "a"])
            qml.RY(0.2, wires="a")
            qml.probs(wires=0)
            # expands into RY on wire b
            qml.expval(qml.PauliZ("a") @ qml.Hadamard("b"))
            # expands into QubitUnitary on wire 0
            qml.var(qml.Hermitian(np.array([[1, 2], [2, 4]]), wires=[0]))

        new_tape = tape.expand(expand_measurements=True)

        assert len(new_tape.operations) == 5

        expected = [
            qml.measurements.Probability,
            qml.measurements.Expectation,
            qml.measurements.Variance,
        ]
        assert [m.return_type is r for m, r in zip(new_tape.measurements, expected)]

        expected = [None, None, None]
        assert [m.obs is r for m, r in zip(new_tape.measurements, expected)]

        expected = [None, [1, -1, -1, 1], [0, 5]]
        assert [m.eigvals() is r for m, r in zip(new_tape.measurements, expected)]

    def test_expand_tape_multiple_wires(self):
        """Test the expand() method when measurements with more than one observable on the same
        wire are used"""
        with QuantumTape() as tape1:
            qml.RX(0.3, wires=0)
            qml.RY(0.4, wires=1)
            qml.expval(qml.PauliX(0))
            qml.var(qml.PauliX(0) @ qml.PauliX(1))
            qml.expval(qml.PauliX(2))

        with QuantumTape() as tape2:
            qml.RX(0.3, wires=0)
            qml.RY(0.4, wires=1)
            qml.RY(-np.pi / 2, wires=0)
            qml.RY(-np.pi / 2, wires=1)
            qml.expval(qml.PauliZ(0))
            qml.var(qml.PauliZ(0) @ qml.PauliZ(1))
            qml.expval(qml.PauliX(2))

        tape1_exp = tape1.expand()

        assert tape1_exp.graph.hash == tape2.graph.hash

    @pytest.mark.parametrize("ret", [expval, var])
    def test_expand_tape_multiple_wires_non_commuting(self, ret):
        """Test if a QuantumFunctionError is raised during tape expansion if non-commuting
        observables are on the same wire"""
        with QuantumTape() as tape:
            qml.RX(0.3, wires=0)
            qml.RY(0.4, wires=1)
            qml.expval(qml.PauliX(0))
            ret(op=qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match="Only observables that are qubit-wise"):
            tape.expand(expand_measurements=True)

    @pytest.mark.parametrize("ret", [sample, counts, probs])
    def test_expand_tape_multiple_wires_non_commuting_for_sample_type_measurements(self, ret):
        """Test if a more verbose QuantumFunctionError is raised during tape expansion of non-commuting
        observables on the same wire with sample type measurements present"""
        with QuantumTape() as tape:
            qml.RX(0.3, wires=0)
            qml.RY(0.4, wires=1)
            qml.expval(qml.PauliX(0))
            ret(op=qml.PauliZ(0))

        expected_error_msg = (
            "Only observables that are qubit-wise commuting "
            "Pauli words can be returned on the same wire.\n"
            "Try removing all probability, sample and counts measurements "
            "this will allow for splitting of execution and separate measurements "
            "for each non-commuting observable."
        )

        with pytest.raises(qml.QuantumFunctionError, match=expected_error_msg):
            tape.expand(expand_measurements=True)

    def test_is_sampled_reserved_after_expansion(self, monkeypatch, mocker):
        """Test that the is_sampled property is correctly set when tape
        expansion happens."""
        dev = qml.device("default.qubit", wires=1, shots=10)

        # Remove support for an op to enforce decomposition & tape expansion
        mock_ops = copy.copy(dev.operations)
        mock_ops.remove("T")

        with monkeypatch.context() as m:
            m.setattr(dev, "operations", mock_ops)

            def circuit():
                qml.T(wires=0)
                return sample(qml.PauliZ(0))

            # Choosing parameter-shift not to swap the device under the hood
            qnode = qml.QNode(circuit, dev, diff_method="parameter-shift")
            qnode()

            # Double-checking that the T gate is not supported
            assert "T" not in qnode.device.operations
            assert "T" not in qnode._original_device.operations

            assert qnode.qtape.is_sampled


class TestExecution:
    """Tests for tape execution"""

    def test_execute_parameters(self, tol):
        """Test execution works when parameters are both passed and not passed."""
        dev = qml.device("default.qubit", wires=2)
        params = [0.1, 0.2]

        with QuantumTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        assert tape.output_dim == 1
        assert tape.batch_size is None

        # test execution with no parameters
        res1 = dev.execute(tape)
        assert tape.get_parameters() == params

        # test setting parameters
        tape.set_parameters(params=[0.5, 0.6])
        res3 = dev.execute(tape)
        assert not np.allclose(res1, res3, atol=tol, rtol=0)
        assert tape.get_parameters() == [0.5, 0.6]

    def test_no_output_execute(self):
        """Test that tapes with no measurement process return
        an empty list."""
        dev = qml.device("default.qubit", wires=2)
        params = [0.1, 0.2]

        with QuantumTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])

        res = dev.execute(tape)
        assert res.size == 0
        assert np.all(res == np.array([]))

    def test_single_expectation_value(self, tol):
        """Tests correct output shape and evaluation for a tape
        with a single expval output"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        assert tape.output_dim == 1

        res = dev.execute(tape)
        assert res.shape == (1,)

        expected = np.sin(y) * np.cos(x)
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multiple_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.expval(qml.PauliX(1))

        assert tape.output_dim == 2

        res = dev.execute(tape)
        assert res.shape == (2,)

        expected = [np.cos(x), np.sin(y)]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_var_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with expval and var outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.var(qml.PauliX(1))

        assert tape.output_dim == 2

        res = dev.execute(tape)
        assert res.shape == (2,)

        expected = [np.cos(x), np.cos(y) ** 2]
        assert np.allclose(res, expected, atol=tol, rtol=0)

    @pytest.mark.filterwarnings("ignore:Creating an ndarray from ragged nested sequences")
    def test_prob_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and var outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0, 1])

        assert tape.output_dim == 5

        res = dev.execute(tape)

        assert isinstance(res[0], float)
        assert np.allclose(res[0], np.cos(x), atol=tol, rtol=0)

        assert isinstance(res[1], np.ndarray)
        assert np.allclose(res[1], np.abs(dev.state) ** 2, atol=tol, rtol=0)

    def test_single_mode_sample(self):
        """Test that there is only one array of values returned
        for a single wire qml.sample"""
        dev = qml.device("default.qubit", wires=2, shots=10)
        x = 0.543
        y = -0.654

        with QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.sample(qml.PauliZ(0) @ qml.PauliX(1))

        res = dev.execute(tape)
        assert res.shape == (1, 10)

    def test_multiple_samples(self):
        """Test that there is only one array of values returned
        for multiple samples"""
        dev = qml.device("default.qubit", wires=2, shots=10)
        x = 0.543
        y = -0.654

        with QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.sample(qml.PauliZ(0))
            qml.sample(qml.PauliZ(1))

        res = dev.execute(tape)
        assert res.shape == (2, 10)

    def test_samples_expval(self):
        """Test that multiple arrays of values are returned
        for combinations of samples and statistics"""
        dev = qml.device("default.qubit", wires=2, shots=10)
        x = 0.543
        y = -0.654

        with QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.sample(qml.PauliZ(0))
            qml.expval(qml.PauliZ(1))

        res = dev.execute(tape)
        assert res[0].shape == (10,)
        assert isinstance(res[1], np.ndarray)

    def test_decomposition(self, tol):
        """Test decomposition onto a device's supported gate set"""
        dev = qml.device("default.qubit", wires=1)

        with QuantumTape() as tape:
            qml.U3(0.1, 0.2, 0.3, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = tape.expand(stop_at=lambda obj: obj.name in dev.operations)
        res = dev.execute(tape)
        assert np.allclose(res, np.cos(0.1), atol=tol, rtol=0)


class TestCVExecution:
    """Tests for CV tape execution"""

    def test_single_output_value(self, tol):
        """Tests correct execution and output shape for a CV tape
        with a single expval output"""
        dev = qml.device("default.gaussian", wires=2)
        x = 0.543
        y = -0.654

        with QuantumTape() as tape:
            qml.Displacement(x, 0, wires=[0])
            qml.Squeezing(y, 0, wires=[1])
            qml.Beamsplitter(np.pi / 4, 0, wires=[0, 1])
            qml.expval(qml.NumberOperator(0))

        assert tape.output_dim == 1

        res = dev.batch_execute([tape])[0]
        assert res.shape == (1,)

    def test_multiple_output_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple measurement types"""
        dev = qml.device("default.gaussian", wires=2)
        x = 0.543
        y = -0.654

        with QuantumTape() as tape:
            qml.Displacement(x, 0, wires=[0])
            qml.Squeezing(y, 0, wires=[1])
            qml.Beamsplitter(np.pi / 4, 0, wires=[0, 1])
            qml.expval(qml.PolyXP(np.diag([0, 1, 0]), wires=0))  # X^2
            qml.var(qml.P(1))

        assert tape.output_dim == 2

        res = dev.batch_execute([tape])[0]
        assert res.shape == (2,)


class TestTapeCopying:
    """Test for tape copying behaviour"""

    def test_shallow_copy(self):
        """Test that shallow copying of a tape results in all
        contained data being shared between the original tape and the copy"""
        with QuantumTape() as tape:
            qml.BasisState(np.array([1, 0]), wires=[0, 1])
            qml.RY(0.5, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliY(1))

        copied_tape = tape.copy()

        assert copied_tape is not tape

        # the operations are simply references
        assert copied_tape.operations == tape.operations
        assert copied_tape.observables == tape.observables
        assert copied_tape.measurements == tape.measurements
        assert copied_tape.operations[0] is tape.operations[0]

        # operation data is also a reference
        assert copied_tape.operations[0].wires is tape.operations[0].wires
        assert copied_tape.operations[0].data[0] is tape.operations[0].data[0]

        # check that all tape metadata is identical
        assert tape.get_parameters() == copied_tape.get_parameters()
        assert tape.wires == copied_tape.wires
        assert tape.data == copied_tape.data

        # check that the output dim is identical
        assert tape.output_dim == copied_tape.output_dim

        # since the copy is shallow, mutating the parameters
        # on one tape will affect the parameters on another tape
        new_params = [np.array([0, 0]), 0.2]
        tape.set_parameters(new_params)

        # check that they are the same objects in memory
        for i, j in zip(tape.get_parameters(), new_params):
            assert i is j

        for i, j in zip(copied_tape.get_parameters(), new_params):
            assert i is j

    @pytest.mark.parametrize(
        "copy_fn", [lambda tape: tape.copy(copy_operations=True), lambda tape: copy.copy(tape)]
    )
    def test_shallow_copy_with_operations(self, copy_fn):
        """Test that shallow copying of a tape and operations allows
        parameters to be set independently"""

        with QuantumTape() as tape:
            qml.BasisState(np.array([1, 0]), wires=[0, 1])
            qml.RY(0.5, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliY(1))

        copied_tape = copy_fn(tape)

        assert copied_tape is not tape

        # the operations are not references; they are unique objects
        assert copied_tape.operations != tape.operations
        assert copied_tape.observables != tape.observables
        assert copied_tape.measurements != tape.measurements
        assert copied_tape.operations[0] is not tape.operations[0]

        # however, the underlying operation data *is still shared*
        assert copied_tape.operations[0].wires is tape.operations[0].wires
        assert copied_tape.operations[0].data[0] is tape.operations[0].data[0]

        assert tape.get_parameters() == copied_tape.get_parameters()
        assert tape.wires == copied_tape.wires
        assert tape.data == copied_tape.data

        # check that the output dim is identical
        assert tape.output_dim == copied_tape.output_dim

        # Since they have unique operations, mutating the parameters
        # on one tape will *not* affect the parameters on another tape
        new_params = [np.array([0, 0]), 0.2]
        tape.set_parameters(new_params)

        for i, j in zip(tape.get_parameters(), new_params):
            assert i is j

        for i, j in zip(copied_tape.get_parameters(), new_params):
            assert not np.all(i == j)
            assert i is not j

    def test_deep_copy(self):
        """Test that deep copying a tape works, and copies all constituent data except parameters"""
        with QuantumTape() as tape:
            qml.BasisState(np.array([1, 0]), wires=[0, 1])
            qml.RY(0.5, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliY(1))

        copied_tape = copy.deepcopy(tape)

        assert copied_tape is not tape

        # the operations are not references
        assert copied_tape.operations != tape.operations
        assert copied_tape.observables != tape.observables
        assert copied_tape.measurements != tape.measurements
        assert copied_tape.operations[0] is not tape.operations[0]

        # check that the output dim is identical
        assert tape.output_dim == copied_tape.output_dim

        # The underlying operation data has also been copied
        assert copied_tape.operations[0].wires is not tape.operations[0].wires

        # however, the underlying operation *parameters* are still shared
        # to support PyTorch, which does not support deep copying of tensors
        assert copied_tape.operations[0].data[0] is tape.operations[0].data[0]


class TestStopRecording:
    """Test that the stop_recording function works as expected"""

    deprecation_warning = (
        "QuantumTape.stop_recording has moved to qml.QueuingManager.stop_recording"
    )

    def test_recording_stopped(self):
        """Test that recording is stopped within a tape context"""

        with QuantumTape() as tape:
            op0 = qml.RX(0, wires=0)
            assert qml.queuing.QueuingManager.active_context() is tape

            with pytest.warns(UserWarning, match=self.deprecation_warning):
                with tape.stop_recording():
                    op1 = qml.RY(1.0, wires=1)
                    assert qml.queuing.QueuingManager.active_context() is None

                op2 = qml.RZ(2, wires=1)
                assert qml.queuing.QueuingManager.active_context() is tape

        assert len(tape.operations) == 2
        assert tape.operations[0] == op0
        assert tape.operations[1] == op2

    def test_nested_recording_stopped(self):
        """Test that recording is stopped within a nested tape context"""

        with QuantumTape() as tape1:
            op0 = qml.RX(0, wires=0)
            assert qml.queuing.QueuingManager.active_context() is tape1

            with QuantumTape() as tape2:
                assert qml.queuing.QueuingManager.active_context() is tape2
                op1 = qml.RY(1.0, wires=1)

                with pytest.warns(UserWarning, match=self.deprecation_warning):
                    with tape2.stop_recording():
                        assert qml.queuing.QueuingManager.active_context() is None
                        op2 = qml.RZ(0.6, wires=2)
                        op3 = qml.CNOT(wires=[0, 2])

                op4 = qml.Hadamard(wires=0)

            op5 = qml.RZ(2, wires=1)
            assert qml.queuing.QueuingManager.active_context() is tape1

        assert len(tape1.operations) == 3
        assert tape1.operations[0] == op0
        assert tape1.operations[1] == tape2
        assert tape1.operations[2] == op5

        assert len(tape2.operations) == 2
        assert tape2.operations[0] == op1
        assert tape2.operations[1] == op4

    def test_creating_scratch_tape(self):
        """Test that a tape created inside the 'scratch'
        space is properly created and accessible"""
        with QuantumTape() as tape:
            op0 = qml.RX(0, wires=0)
            assert qml.queuing.QueuingManager.active_context() is tape

            with pytest.warns(UserWarning, match=self.deprecation_warning):
                with tape.stop_recording(), QuantumTape() as temp_tape:
                    assert qml.queuing.QueuingManager.active_context() is temp_tape
                    op1 = qml.RY(1.0, wires=1)

            op2 = qml.RZ(2, wires=1)
            assert qml.queuing.QueuingManager.active_context() is tape

        assert len(tape.operations) == 2
        assert tape.operations[0] == op0
        assert tape.operations[1] == op2

        assert len(temp_tape.operations) == 1
        assert temp_tape.operations[0] == op1

    def test_stop_recording_within_tape_cleans_up(self):
        """Test if some error is raised within a stop_recording context, the previously
        active contexts are still returned to avoid popping from an empty deque"""

        with pytest.raises(ValueError):
            with qml.queuing.AnnotatedQueue() as q:
                with qml.QueuingManager.stop_recording():
                    raise ValueError


def test_get_active_tape():
    """Test that the get_active_tape() function returns the currently
    recording tape, or None if no tape is recording"""
    message = (
        "qml.tape.get_active_tape is now deprecated."
        " Please use qml.QueuingManager.active_context"
    )
    with pytest.warns(UserWarning, match=message):
        assert qml.tape.get_active_tape() is None

    with QuantumTape() as tape1:
        with pytest.warns(UserWarning, match=message):
            assert qml.tape.get_active_tape() is tape1

        with QuantumTape() as tape2:
            with pytest.warns(UserWarning, match=message):
                assert qml.tape.get_active_tape() is tape2

        with pytest.warns(UserWarning, match=message):
            assert qml.tape.get_active_tape() is tape1

    with pytest.warns(UserWarning, match=message):
        assert qml.tape.get_active_tape() is None


class TestHashing:
    """Test for tape hashing"""

    @pytest.mark.parametrize(
        "m",
        [
            qml.expval(qml.PauliZ(0)),
            qml.state(),
            qml.probs(wires=0),
            qml.density_matrix(wires=0),
            qml.var(qml.PauliY(0)),
        ],
    )
    def test_identical(self, m):
        """Tests that the circuit hash of identical circuits are identical"""
        a = 0.3
        b = 0.2

        with qml.tape.QuantumTape() as tape1:
            qml.RX(a, wires=[0])
            qml.RY(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.apply(m)

        with qml.tape.QuantumTape() as tape2:
            qml.RX(a, wires=[0])
            qml.RY(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.apply(m)

        assert tape1.hash == tape2.hash

    def test_identical_numeric(self):
        """Tests that the circuit hash of identical circuits are identical
        even though the datatype of the arguments may differ"""
        a = 0.3
        b = 0.2

        with qml.tape.QuantumTape() as tape1:
            qml.RX(a, wires=[0])
            qml.RY(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(np.array(a), wires=[0])
            qml.RY(np.array(b), wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        assert tape1.hash == tape2.hash

    def test_different_wires(self):
        """Tests that the circuit hash of circuits with the same operations
        on different wires have different hashes"""
        a = 0.3
        b = 0.2

        with qml.tape.QuantumTape() as tape1:
            qml.RX(a, wires=[1])
            qml.RY(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(np.array(a), wires=[0])
            qml.RY(np.array(b), wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        assert tape1.hash != tape2.hash

    def test_different_trainabilities(self):
        """Tests that the circuit hash of identical circuits differ
        if the circuits have different trainable parameters"""
        a = 0.3
        b = 0.2

        with qml.tape.QuantumTape() as tape1:
            qml.RX(a, wires=[0])
            qml.RY(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(a, wires=[0])
            qml.RY(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

        tape1.trainable_params = [0]
        tape2.trainable_params = [0, 1]
        assert tape1.hash != tape2.hash

    def test_different_parameters(self):
        """Tests that the circuit hash of circuits with different
        parameters differs"""
        a = 0.3
        b = 0.2
        c = 0.6

        with qml.tape.QuantumTape() as tape1:
            qml.RX(a, wires=[0])
            qml.RY(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(a, wires=[0])
            qml.RY(c, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        assert tape1.hash != tape2.hash

    def test_different_operations(self):
        """Tests that the circuit hash of circuits with different
        operations differs"""
        a = 0.3
        b = 0.2

        with qml.tape.QuantumTape() as tape1:
            qml.RX(a, wires=[0])
            qml.RZ(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(a, wires=[0])
            qml.RY(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        assert tape1.hash != tape2.hash

    def test_different_measurements(self):
        """Tests that the circuit hash of circuits with different
        measurements differs"""
        a = 0.3
        b = 0.2

        with qml.tape.QuantumTape() as tape1:
            qml.RX(a, wires=[0])
            qml.RY(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(a, wires=[0])
            qml.RY(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.var(qml.PauliZ(0))

        assert tape1.hash != tape2.hash

    def test_different_observables(self):
        """Tests that the circuit hash of circuits with different
        observables differs"""
        a = 0.3
        b = 0.2

        A = np.diag([1.0, 2.0])

        with qml.tape.QuantumTape() as tape1:
            qml.RX(a, wires=[0])
            qml.RY(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0))

        with qml.tape.QuantumTape() as tape2:
            qml.RX(a, wires=[0])
            qml.RY(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.Hermitian(A, wires=0))

        assert tape1.hash != tape2.hash

    def test_rotation_modulo_identical(self):
        """Tests that the circuit hash of circuits with single-qubit
        rotations differing by multiples of 2pi have identical hash"""
        a = np.array(np.pi / 2, dtype=np.float64)
        b = np.array(np.pi / 4, dtype=np.float64)

        H = qml.Hamiltonian([0.1, 0.2], [qml.PauliX(0), qml.PauliZ(0) @ qml.PauliY(1)])

        with qml.tape.QuantumTape() as tape1:
            qml.RX(a, wires=[0])
            qml.RY(b, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(H)

        with qml.tape.QuantumTape() as tape2:
            qml.RX(a - 2 * np.pi, wires=[0])
            qml.RY(b + 2 * np.pi, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(H)

        assert tape1.hash == tape2.hash

    def test_controlled_rotation_modulo_identical(self):
        """Tests that the circuit hash of circuits with controlled
        rotations differing by multiples of 2pi have identical hash"""
        a = np.array(np.pi / 2, dtype=np.float64)
        b = np.array(np.pi / 2, dtype=np.float64)

        H = qml.Hamiltonian([0.1, 0.2], [qml.PauliX(0), qml.PauliZ(0) @ qml.PauliY(1)])

        with qml.tape.QuantumTape() as tape1:
            qml.CRX(a, wires=[0, 1])
            qml.CRY(b, wires=[0, 1])
            qml.CNOT(wires=[0, 1])
            qml.expval(H)

        with qml.tape.QuantumTape() as tape2:
            qml.CRX(a - 4 * np.pi, wires=[0, 1])
            qml.CRY(b + 4 * np.pi, wires=[0, 1])
            qml.CNOT(wires=[0, 1])
            qml.expval(H)

        assert tape1.hash == tape2.hash


def cost(tape, dev):
    return qml.execute([tape], dev, interface="autograd", gradient_fn=qml.gradients.param_shift)


measures = [
    (qml.expval(qml.PauliZ(0)), (1,)),
    (qml.var(qml.PauliZ(0)), (1,)),
    (qml.probs(wires=[0]), (1, 2)),
    (qml.probs(wires=[0, 1]), (1, 4)),
    (qml.state(), (1, 8)),  # Assumes 3-qubit device
    (qml.density_matrix(wires=[0, 1]), (1, 4, 4)),
    (
        qml.sample(qml.PauliZ(0)),
        None,
    ),  # Shape is None because the expected shape is in the test case
    (qml.sample(), None),  # Shape is None because the expected shape is in the test case
]

multi_measurements = [
    ([qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))], (2,)),
    ([qml.probs(wires=[0]), qml.probs(wires=[1])], (2, 2)),
    ([qml.probs(wires=[0]), qml.probs(wires=[1, 2])], (2**1 + 2**2,)),
    ([qml.probs(wires=[0, 2]), qml.probs(wires=[1])], (2**2 + 2**1,)),
    (
        [qml.probs(wires=[0]), qml.probs(wires=[1, 2]), qml.probs(wires=[0, 1, 2])],
        (2**1 + 2**2 + 2**3,),
    ),
]


@pytest.mark.filterwarnings("ignore:Creating an ndarray from ragged nested sequences")
class TestOutputShape:
    """Tests for determining the tape output shape of tapes."""

    @pytest.mark.parametrize("measurement, expected_shape", measures)
    @pytest.mark.parametrize("shots", [None, 1, 10])
    def test_output_shapes_single(self, measurement, expected_shape, shots):
        """Test that the output shape produced by the tape matches the expected
        output shape."""
        if shots is None and measurement.return_type is qml.measurements.Sample:
            pytest.skip("Sample doesn't support analytic computations.")

        num_wires = 3
        dev = qml.device("default.qubit", wires=num_wires, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)

        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.apply(measurement)

        shot_dim = shots if not isinstance(shots, tuple) else len(shots)
        if expected_shape is None:
            expected_shape = shot_dim if shot_dim == 1 else (shot_dim,)

        if measurement.return_type is qml.measurements.Sample:
            if measurement.obs is not None:
                expected_shape = (1, shots)

            else:
                expected_shape = (1, shots, num_wires)

        assert tape.shape(dev) == expected_shape

    @pytest.mark.parametrize("measurement, expected_shape", measures)
    @pytest.mark.parametrize("shots", [None, 1, 10, (1, 2, 5, 3)])
    def test_output_shapes_single_qnode_check(self, measurement, expected_shape, shots):
        """Test that the output shape produced by the tape matches the output
        shape of a QNode for a single measurement."""
        if shots is None and measurement.return_type is qml.measurements.Sample:
            pytest.skip("Sample doesn't support analytic computations.")

        if shots is not None and measurement.return_type is qml.measurements.State:
            pytest.skip("State and density matrix don't support finite shots and raise a warning.")

        # TODO: revisit when qml.sample without an observable has been updated
        # with shot vectors
        if (
            isinstance(shots, tuple)
            and measurement.return_type is qml.measurements.Sample
            and not measurement.obs
        ):
            pytest.skip("qml.sample with no observable is to be updated for shot vectors.")

        dev = qml.device("default.qubit", wires=3, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)

        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.apply(measurement)

        res = qml.execute([tape], dev, gradient_fn=qml.gradients.param_shift)[0]

        if isinstance(res, tuple):
            res_shape = tuple([r.shape for r in res])
        else:
            res_shape = res.shape

        res_shape = res_shape if res_shape != tuple() else 1
        assert tape.shape(dev) == res_shape

    def test_output_shapes_single_qnode_check_cutoff(self):
        """Test that the tape output shape is correct when computing
        probabilities with a dummy device that defines a cutoff value."""

        class CustomDevice(qml.QubitDevice):
            """A dummy device that has a cutoff value specified and returns
            analytic probabilities in a fashion similar to the
            strawberryfields.fock device.

            Note: this device definition is used as PennyLane-SF is not a
            dependency of PennyLane core and there are no CV device in
            PennyLane core using a cutoff value.
            """

            name = "Device with cutoff"
            short_name = "dummy.device"
            pennylane_requires = "0.1.0"
            version = "0.0.1"
            author = "CV quantum"

            operations = {}
            observables = {"Identity"}

            def __init__(self, shots=None, wires=None, cutoff=None):
                super().__init__(wires=wires, shots=shots)
                self.cutoff = cutoff

            def apply(self, operations, **kwargs):
                pass

            def analytic_probability(self, wires=None):
                if wires is None:
                    wires = self.wires
                return np.zeros(self.cutoff ** len(wires))

        dev = CustomDevice(wires=2, cutoff=13)

        # If PennyLane-SF is installed, the following can be checked e.g., locally:
        # dev = qml.device("strawberryfields.fock", wires=2, cutoff_dim=13)

        a = np.array(0.1)
        b = np.array(0.2)

        with qml.tape.QuantumTape() as tape:
            qml.probs(wires=[0])

        @qml.qnode(dev)
        def circuit(a, b):
            return qml.probs(wires=[0])

        res_shape = qml.execute([tape], dev, gradient_fn=qml.gradients.param_shift_cv)[0]
        assert tape.shape(dev) == res_shape.shape

    @pytest.mark.autograd
    @pytest.mark.parametrize("measurements, expected", multi_measurements)
    @pytest.mark.parametrize("shots", [None, 1, 10])
    def test_multi_measure(self, measurements, expected, shots):
        """Test that the expected output shape is obtained when using multiple
        expectation value, variance and probability measurements."""
        dev = qml.device("default.qubit", wires=3, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)

        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            for m in measurements:
                qml.apply(m)

        if measurements[0].return_type is qml.measurements.Sample:
            expected[1] = shots
            expected = tuple(expected)

        res = tape.shape(dev)
        assert res == expected

        execution_results = cost(tape, dev)
        expected = execution_results[0].shape
        assert res == expected

    @pytest.mark.autograd
    @pytest.mark.parametrize("measurements, expected", multi_measurements)
    def test_multi_measure_shot_vector(self, measurements, expected):
        """Test that the expected output shape is obtained when using multiple
        expectation value, variance and probability measurements with a shot
        vector."""
        if measurements[0].return_type is qml.measurements.Probability:
            num_wires = set(len(m.wires) for m in measurements)
            if len(num_wires) > 1:
                pytest.skip(
                    "Multi-probs with varying number of varies when using a shot vector is to be updated in PennyLane."
                )

        shots = (1, 1, 5, 1)
        dev = qml.device("default.qubit", wires=3, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)

        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            for m in measurements:
                qml.apply(m)

        if measurements[0].return_type is qml.measurements.Sample:
            expected[1] = shots
            expected = tuple(expected)

        # Update expected as we're using a shotvector
        expected = (len(shots),) + expected
        res = tape.shape(dev)
        assert res == expected

        execution_results = cost(tape, dev)
        expected = execution_results[0].shape
        assert res == expected

    @pytest.mark.autograd
    @pytest.mark.parametrize("shots", [1, 10])
    def test_multi_measure_sample(self, shots):
        """Test that the expected output shape is obtained when using multiple
        qml.sample measurements."""
        dev = qml.device("default.qubit", wires=3, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)

        num_samples = 3
        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            for i in range(num_samples):
                qml.sample(qml.PauliZ(i))

        expected = (num_samples, shots)

        res = tape.shape(dev)
        assert res == expected

        execution_results = cost(tape, dev)
        expected = execution_results[0].shape
        assert res == expected

    @pytest.mark.autograd
    def test_multi_measure_sample_shot_vector(self):
        """Test that the expected output shape is obtained when using multiple
        qml.sample measurements with a shot vector."""
        shots = (1, 1, 5, 1)
        dev = qml.device("default.qubit", wires=3, shots=shots)

        a = np.array(0.1)
        b = np.array(0.2)

        num_samples = 3
        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            for i in range(num_samples):
                qml.sample(qml.PauliZ(i))

        expected = []
        for s in shots:
            shape = (num_samples,) if s == 1 else (s, num_samples)
            expected.append(shape)

        res = tape.shape(dev)
        for r, e in zip(res, expected):
            assert r == e

        execution_results = cost(tape, dev)[0]
        for r, e in zip(res, execution_results):
            assert r == e.shape

    def test_multi_measure_probs_shot_vector_errors(self):
        """Test that getting the output shape of a tape containing multiple
        probability measurements with different number of wires errors when
        using a device with a shot vector."""
        dev = qml.device("default.qubit", wires=3, shots=(1, 2, 3))

        with qml.tape.QuantumTape() as tape:
            qml.probs(wires=[0])
            qml.probs(wires=[1, 2])

        with pytest.raises(ValueError, match="multiple probability measurements"):
            tape.shape(dev)

    def test_raises_multiple_different_measurements(self):
        """Test that getting the output shape of a tape that contains multiple
        types of measurements raises an error."""
        dev = qml.device("default.qubit", wires=3)

        with qml.tape.QuantumTape() as tape:
            qml.RY(0.3, wires=0)
            qml.RX(0.2, wires=0)
            qml.expval(qml.PauliZ(0))
            qml.sample(qml.PauliZ(0))

        with pytest.raises(
            ValueError,
            match="contains multiple types of measurements is unsupported",
        ):
            tape.shape(dev)

    def test_raises_multi_state(self):
        """Test that getting the output shape of a tape that contains multiple
        state measurements raises an error."""
        dev = qml.device("default.qubit", wires=3)

        with qml.tape.QuantumTape() as tape:
            qml.RY(0.3, wires=0)
            qml.RX(0.2, wires=0)
            qml.state()
            qml.density_matrix(wires=0)

        with pytest.raises(ValueError, match="multiple state measurements is not supported"):
            tape.shape(dev)

    def test_raises_sample_shot_vector(self):
        """Test that getting the output shape of a tape that returns samples
        along with a device with a shot vector raises an error."""
        dev = qml.device("default.qubit", wires=3, shots=(1, 2, 3))

        with qml.tape.QuantumTape() as tape:
            qml.RY(0.3, wires=0)
            qml.RX(0.2, wires=0)
            qml.sample()

        with pytest.raises(
            ValueError,
            match="returning samples along with a device with a shot vector",
        ):
            tape.shape(dev)

    def test_raises_sample_shot_vector(self):
        """Test that getting the output shape of a tape that returns samples
        along with a device with a shot vector raises an error."""
        dev = qml.device("default.qubit", wires=3, shots=(1, 2, 3))

        with qml.tape.QuantumTape() as tape:
            qml.RY(0.3, wires=0)
            qml.RX(0.2, wires=0)
            qml.sample()

        with pytest.raises(
            MeasurementShapeError,
            match="returning samples along with a device with a shot vector",
        ):
            tape.shape(dev)


class TestNumericType:
    """Tests for determining the numeric type of the tape output."""

    @pytest.mark.parametrize(
        "ret", [qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0)), qml.probs(wires=[0])]
    )
    @pytest.mark.parametrize("shots", [None, 1, (1, 2, 3)])
    def test_float_measures(self, ret, shots):
        """Test that most measurements output floating point values and that
        the tape output domain correctly identifies this."""
        dev = qml.device("default.qubit", wires=3, shots=shots)

        @qml.qnode(dev)
        def circuit(a, b):
            qml.RY(a, wires=[0])
            qml.RZ(b, wires=[0])
            return qml.apply(ret)

        result = circuit(0.3, 0.2)

        # Double-check the domain of the QNode output
        assert np.issubdtype(result.dtype, float)
        assert circuit.qtape.numeric_type is float

    @pytest.mark.parametrize(
        "ret", [qml.state(), qml.density_matrix(wires=[0, 1]), qml.density_matrix(wires=[2, 0])]
    )
    def test_complex_state(self, ret):
        """Test that a tape with qml.state correctly determines that the output
        domain will be complex."""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(a, b):
            qml.RY(a, wires=[0])
            qml.RZ(b, wires=[0])
            return qml.apply(ret)

        result = circuit(0.3, 0.2)

        # Double-check the domain of the QNode output
        assert np.issubdtype(result.dtype, complex)
        assert circuit.qtape.numeric_type is complex

    @pytest.mark.parametrize("ret", [qml.sample(), qml.sample(qml.PauliZ(wires=0))])
    def test_sample_int_eigvals(self, ret):
        """Test that the tape can correctly determine the output domain for a
        sampling measurement with a Hermitian observable with integer
        eigenvalues."""
        dev = qml.device("default.qubit", wires=3, shots=5)

        arr = np.array(
            [
                1.32,
                2.312,
            ]
        )
        herm = np.outer(arr, arr)

        @qml.qnode(dev)
        def circuit():
            qml.RY(0.4, wires=[0])
            return qml.apply(ret)

        result = circuit()

        # Double-check the domain of the QNode output
        assert np.issubdtype(result.dtype, int)
        assert circuit.qtape.numeric_type is int

    # TODO: add cases for each interface once qml.Hermitian supports other
    # interfaces
    def test_sample_real_eigvals(self):
        """Test that the tape can correctly determine the output domain when
        sampling a Hermitian observable with real eigenvalues."""
        dev = qml.device("default.qubit", wires=3, shots=5)

        arr = np.array(
            [
                1.32,
                2.312,
            ]
        )
        herm = np.outer(arr, arr)

        @qml.qnode(dev)
        def circuit(a, b):
            qml.RY(0.4, wires=[0])
            return qml.sample(qml.Hermitian(herm, wires=0))

        result = circuit(0.3, 0.2)

        # Double-check the domain of the QNode output
        assert np.issubdtype(result.dtype, float)
        assert circuit.qtape.numeric_type is float

    @pytest.mark.autograd
    def test_sample_real_and_int_eigvals(self):
        """Test that the tape can correctly determine the output domain for
        multiple sampling measurements with a Hermitian observable with real
        eigenvalues and another one with integer eigenvalues."""
        dev = qml.device("default.qubit", wires=3, shots=5)

        arr = np.array(
            [
                1.32,
                2.312,
            ]
        )
        herm = np.outer(arr, arr)

        @qml.qnode(dev, interface="autograd")
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.sample(qml.Hermitian(herm, wires=0)), qml.sample(qml.PauliZ(1))

        result = circuit(0, 3)

        # Double-check the domain of the QNode output
        assert np.issubdtype(result.dtype, float)
        assert circuit.qtape.numeric_type is float

    def test_multi_type_measurements_numeric_type_error(self):
        """Test that querying the numeric type of a tape with several types of
        measurements raises an error."""
        a = 0.3
        b = 0.3

        with qml.tape.QuantumTape() as tape:
            qml.RY(a, wires=[0])
            qml.RZ(b, wires=[0])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0])

        with pytest.raises(
            ValueError,
            match="Getting the numeric type of a quantum script that contains multiple types of measurements is unsupported.",
        ):
            tape.numeric_type


class TestTapeDraw:
    """Test the tape draw method."""

    def test_default_kwargs(self):
        """Test tape draw with default keyword arguments."""

        with QuantumTape() as tape:
            qml.RX(1.23456, wires=0)
            qml.RY(2.3456, wires="a")
            qml.RZ(3.4567, wires=1.234)

        assert tape.draw() == qml.drawer.tape_text(tape)
        assert tape.draw(decimals=2) == qml.drawer.tape_text(tape, decimals=2)

    def test_show_matrices(self):
        """Test show_matrices keyword argument."""

        with QuantumTape() as tape:
            qml.QubitUnitary(qml.numpy.eye(2), wires=0)

        assert tape.draw() == qml.drawer.tape_text(tape)
        assert tape.draw(show_matrices=True) == qml.drawer.tape_text(tape, show_matrices=True)

    def test_max_length_keyword(self):
        """Test the max_length keyword argument."""

        with QuantumTape() as tape:
            for _ in range(50):
                qml.PauliX(0)

        assert tape.draw() == qml.drawer.tape_text(tape)
        assert tape.draw(max_length=20) == qml.drawer.tape_text(tape, max_length=20)
