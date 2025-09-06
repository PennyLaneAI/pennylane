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
# pylint: disable=protected-access,too-few-public-methods
import copy
from collections import defaultdict

import numpy as np
import pytest

import pennylane as qml
from pennylane import CircuitGraph
from pennylane.exceptions import PennyLaneDeprecationWarning, QuantumFunctionError
from pennylane.measurements import (
    ExpectationMP,
    MeasurementProcess,
    ProbabilityMP,
    counts,
    expval,
    probs,
    sample,
    var,
)
from pennylane.tape import QuantumScript, QuantumTape, expand_tape_state_prep


def TestOperationMonkeypatching():
    """Test that operations are monkeypatched only within the quantum tape"""
    with QuantumTape() as tape:
        op_ = qml.RX(0.432, wires=0)
        obs = qml.PauliX(wires="a")
        qml.expval(qml.PauliX(wires="a"))

    assert tape.operations == [op_]
    assert tape.observables == [obs]

    # now create an old QNode
    dev = qml.device("default.qubit", wires=[0, "a"])

    @qml.qnode(dev)
    def func(x):
        nonlocal op_
        op_ = qml.RX(x, wires=0)
        return qml.expval(qml.PauliX(wires="a"))

    # this should evaluate without error
    func(0.432)

    assert func.circuit.operations == [op_]


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

        assert len(tape.queue) == 6
        assert tape.operations == ops
        assert tape.observables == obs
        assert tape.batch_size is None

        assert tape.wires == qml.wires.Wires([0, "a", 4])

    def test_observable_processing(self, make_tape):
        """Test that observables are processed correctly"""
        tape, _, obs = make_tape

        # test that the internal tape.measurements list is created properly
        assert isinstance(tape.measurements[0], MeasurementProcess)
        assert isinstance(tape.measurements[0], qml.measurements.ExpectationMP)
        qml.assert_equal(tape.measurements[0].obs, obs[0])

        assert isinstance(tape.measurements[1], MeasurementProcess)
        assert isinstance(tape.measurements[1], qml.measurements.ProbabilityMP)

        # test the public observables property
        assert len(tape.observables) == 2
        assert tape.observables[0].name == "PauliX"
        assert isinstance(tape.observables[1], qml.measurements.ProbabilityMP)

        # test the public measurements property
        assert len(tape.measurements) == 2
        assert all(isinstance(m, MeasurementProcess) for m in tape.measurements)
        assert isinstance(tape.measurements[0], ExpectationMP)
        assert isinstance(tape.measurements[1], ProbabilityMP)

    def test_tensor_observables_matmul(self):
        """Test that tensor observables are correctly processed from the annotated
        queue. Here, we test multiple tensor observables constructed via matmul."""

        with QuantumTape() as tape:
            op_ = qml.RX(1.0, wires=0)
            t_obs1 = qml.PauliZ(0) @ qml.PauliX(1)
            t_obs2 = t_obs1 @ qml.PauliZ(3)
            qml.expval(t_obs2)

        assert tape.operations == [op_]
        assert tape.observables == [t_obs2]
        assert isinstance(tape.measurements[0], qml.measurements.ExpectationMP)
        assert tape.measurements[0].obs is t_obs2

    def test_tensor_observables_rmatmul(self):
        """Test that tensor observables are correctly processed from the annotated
        queue. Here, we test multiple tensor observables constructed via matmul
        with the observable occurring on the left hand side."""

        with QuantumTape() as tape:
            op_ = qml.RX(1.0, wires=0)
            t_obs1 = qml.PauliZ(1) @ qml.PauliX(0)
            t_obs2 = qml.Hadamard(2) @ t_obs1
            qml.expval(t_obs2)

        assert tape.operations == [op_]
        assert tape.observables == [t_obs2]
        assert isinstance(tape.measurements[0], qml.measurements.ExpectationMP)
        assert tape.measurements[0].obs is t_obs2

    def test_tensor_observables_tensor_matmul(self):
        """Test that tensor observables are correctly processed from the annotated
        queue". Here, wetest multiple tensor observables constructed via matmul
        between two tensor observables."""

        with QuantumTape() as tape:
            op_ = qml.RX(1.0, wires=0)
            t_obs1 = qml.PauliZ(0) @ qml.PauliX(1)
            t_obs2 = qml.PauliY(2) @ qml.PauliZ(3)
            t_obs = t_obs1 @ t_obs2
            qml.var(t_obs)

        assert tape.operations == [op_]
        assert tape.observables == [t_obs]
        assert isinstance(tape.measurements[0], qml.measurements.VarianceMP)
        assert tape.measurements[0].obs is t_obs

    def test_qubit_diagonalization(self, make_tape):
        """Test that qubit diagonalization works as expected"""
        tape, _, obs = make_tape

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

        assert len(tape.queue) == 1
        assert not tape.operations
        assert tape.measurements == [D]
        assert tape.observables == [C]
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

        assert len(tape.queue) == 4
        assert tape.operations == ops
        assert tape.observables == obs
        assert tape.batch_size is None

        assert not any(qml.equal(a, op) or qml.equal(b, op) for op in tape.operations)
        assert tape.wires == qml.wires.Wires([0, "a"])

    def test_state_preparation(self):
        """Test that state preparations are correctly processed"""
        params = [np.array([1, 0, 1, 0]) / np.sqrt(2), 1]

        with QuantumTape() as tape:
            A = qml.StatePrep(params[0], wires=[0, 1])
            B = qml.RX(params[1], wires=0)
            qml.expval(qml.PauliZ(wires=1))

        assert tape.operations == tape._ops == [A, B]
        assert tape.get_parameters() == params

    def test_state_preparation_queued_after_operation(self):
        """Test that no exception is raised if a state preparation comes
        after a quantum operation"""
        with QuantumTape() as tape:
            qml.PauliX(wires=0)
            qml.BasisState(np.array([0, 1]), wires=[0, 1])

        assert len(tape.operations) == 2
        qml.assert_equal(tape.operations[0], qml.PauliX(wires=0))
        qml.assert_equal(tape.operations[1], qml.BasisState(np.array([0, 1]), wires=[0, 1]))

    def test_measurement_before_operation(self):
        """Test that an exception is raised if a measurement occurs before an operation"""

        with pytest.raises(ValueError, match="must occur prior to measurements"):
            with QuantumTape():
                qml.expval(qml.PauliZ(wires=1))
                qml.RX(0.5, wires=0)
                qml.expval(qml.PauliZ(wires=1))

    def test_repr(self):
        """Test the string representation"""

        with QuantumTape() as tape:
            qml.RX(0.432, wires=0)

        s = repr(tape)
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

        with qml.queuing.AnnotatedQueue() as q:
            m_0 = qml.measure(0)
            qml.cond(m_0, f)(r)
            qml.apply(terminal_measurement)

        tape = qml.tape.QuantumScript.from_queue(q)
        target_wire = qml.wires.Wires(1)

        assert len(tape.circuit) == 5
        assert isinstance(tape.circuit[0], qml.measurements.MidMeasureMP)

        assert isinstance(tape.circuit[1], qml.ops.Conditional)
        assert isinstance(tape.circuit[1].base, qml.PauliX)
        assert tape.circuit[1].base.wires == target_wire

        assert isinstance(tape.circuit[2], qml.ops.Conditional)
        assert isinstance(tape.circuit[2].base, qml.RY)
        assert tape.circuit[2].base.wires == target_wire
        assert tape.circuit[2].base.data == (r,)

        assert isinstance(tape.circuit[3], qml.ops.Conditional)
        assert isinstance(tape.circuit[3].base, qml.PauliZ)
        assert tape.circuit[3].base.wires == target_wire

        assert tape.circuit[4] == terminal_measurement

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
        batch_size, when creating and when using `bind_new_parameters`."""

        # Test with tape construction
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(x, wires=0)
            qml.Rot(*rot, wires=1)
            qml.apply(qml.expval(qml.PauliZ(0)))
            qml.apply(qml.expval(qml.PauliX(1)))

        tape = qml.tape.QuantumScript.from_queue(q)
        assert tape.batch_size == exp_batch_size

        # Test with bind_new_parameters
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(0.2, wires=0)
            qml.Rot(1.0, 0.2, -0.3, wires=1)
            qml.apply(qml.expval(qml.PauliZ(0)))
            qml.apply(qml.expval(qml.PauliX(1)))

        tape = qml.tape.QuantumScript.from_queue(q)
        assert tape.batch_size is None

        tape = tape.bind_new_parameters([x] + rot, [0, 1, 2, 3])
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
        batch_size, when creating and when using `bind_new_parameters`."""

        tape = QuantumScript(
            [qml.RX(x, wires=0), qml.Rot(*rot, wires=1), qml.RX(y, wires=1)],
            [qml.expval(qml.PauliZ(0))],
        )
        with pytest.raises(
            ValueError, match="batch sizes of the quantum script operations do not match."
        ):
            _ = tape.batch_size

        tape = QuantumScript(
            [qml.RX(0.2, wires=0), qml.Rot(1.0, 0.2, -0.3, wires=1), qml.RX(0.2, wires=1)],
            [qml.expval(qml.PauliZ(0))],
        )
        tape = tape.bind_new_parameters([x] + rot + [y], [0, 1, 2, 3, 4])
        with pytest.raises(
            ValueError, match="batch sizes of the quantum script operations do not match."
        ):
            _ = tape.batch_size


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
            assert tape[idx] is exp_elem

        assert len(tape) == len(expected)

    def test_tape_as_list(self, make_tape):
        """Test that a tape can be converted to a list."""
        tape, ops, meas = make_tape
        tape = list(tape)

        expected = ops + meas
        for op_, exp_op in zip(tape, expected):
            assert op_ is exp_op

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
            for op_ in circuit:
                qml.apply(op_)

        # Check that the underlying circuit is as expected
        assert tape.circuit == circuit

        # Iterate over the tape by turning it into a list
        _ = list(tape)

        # Check that the underlying circuit is still as expected
        assert tape.circuit == circuit


class TestGraph:
    """Tests involving graph creation"""

    def test_graph_creation(self, mocker):
        """Test that the circuit graph is correctly created"""
        spy = mocker.spy(CircuitGraph, "__init__")

        with QuantumTape() as tape:
            op_ = qml.RX(1.0, wires=0)
            obs = qml.PauliZ(1)
            qml.expval(obs)

        # graph has not yet been created
        assert tape._graph is None
        spy.assert_not_called()

        # requesting the graph creates it
        g = tape.graph
        assert g.operations == [op_]
        assert g.observables == tape.measurements
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

        gate_types = defaultdict(int)
        expected_resources = qml.resource.Resources(num_wires=2, gate_types=gate_types)
        assert tape.specs["resources"] == expected_resources

        assert tape.specs["num_observables"] == 1
        assert tape.specs["num_trainable_params"] == 0

        assert len(tape.specs) == 4

    def test_specs_tape(self, make_tape):
        """Tests that regular tapes return correct specifications"""
        tape = make_tape

        specs = tape.specs

        assert len(specs) == 4

        gate_sizes = defaultdict(int, {1: 3, 2: 1})
        gate_types = defaultdict(int, {"RX": 2, "Rot": 1, "CNOT": 1})
        expected_resources = qml.resource.Resources(
            num_wires=3, num_gates=4, gate_types=gate_types, gate_sizes=gate_sizes, depth=3
        )
        assert specs["resources"] == expected_resources
        assert specs["num_observables"] == 2
        assert specs["num_trainable_params"] == 5

    def test_specs_add_to_tape(self, make_extendible_tape):
        """Test that tapes return correct specs after adding to them."""

        tape = make_extendible_tape
        specs1 = tape.specs

        assert len(specs1) == 4

        gate_sizes = defaultdict(int, {1: 3, 2: 1})
        gate_types = defaultdict(int, {"RX": 2, "Rot": 1, "CNOT": 1})

        expected_resoures = qml.resource.Resources(
            num_wires=3, num_gates=4, gate_types=gate_types, gate_sizes=gate_sizes, depth=3
        )
        assert specs1["resources"] == expected_resoures

        assert specs1["num_observables"] == 0
        assert specs1["num_trainable_params"] == 5

        with tape as tape:
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.1, wires=3)
            qml.expval(qml.PauliX(wires="a"))
            qml.probs(wires=[0, "a"])

        specs2 = tape.specs

        assert len(specs2) == 4

        gate_sizes = defaultdict(int, {1: 4, 2: 2})
        gate_types = defaultdict(int, {"RX": 2, "Rot": 1, "CNOT": 2, "RZ": 1})

        expected_resoures = qml.resource.Resources(
            num_wires=5, num_gates=6, gate_types=gate_types, gate_sizes=gate_sizes, depth=4
        )
        assert specs2["resources"] == expected_resoures

        assert specs2["num_observables"] == 2
        assert specs2["num_trainable_params"] == 6


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

        new_tape = tape.bind_new_parameters(new_params, [0, 1, 2, 3, 4])

        for pinfo, pval in zip(new_tape.par_info, new_params):
            assert pinfo["op"].data[pinfo["p_idx"]] == pval

        assert new_tape.get_parameters() == new_params
        assert tape.get_parameters() == params

    def test_setting_free_parameters(self, make_tape):
        """Test that free parameters are correctly modified after construction"""
        tape, params = make_tape
        new_params = [-0.654, 0.3]

        tape.trainable_params = [1, 3]
        new_tape = tape.bind_new_parameters(new_params, tape.trainable_params)

        count = 0
        for idx, pinfo in enumerate(new_tape.par_info):
            if idx in tape.trainable_params:
                assert pinfo["op"].data[pinfo["p_idx"]] == new_params[count]
                count += 1
            else:
                assert pinfo["op"].data[pinfo["p_idx"]] == params[idx]

        assert new_tape.get_parameters(trainable_only=False) == [
            params[0],
            new_params[0],
            params[2],
            new_params[1],
            params[4],
        ]
        assert tape.get_parameters(trainable_only=False) == params

    def test_setting_parameters_unordered(self, make_tape):
        """Test that an 'unordered' trainable_params set does not affect
        the setting of parameter values"""
        tape, params = make_tape
        new_params = [-0.654, 0.3]

        tape.trainable_params = [3, 1]
        new_tape = tape.bind_new_parameters(new_params, tape.trainable_params)

        assert new_tape.get_parameters(trainable_only=True) == new_params
        assert new_tape.get_parameters(trainable_only=False) == [
            params[0],
            new_params[0],
            params[2],
            new_params[1],
            params[4],
        ]
        assert tape.get_parameters(trainable_only=False) == params

    def test_setting_all_parameters(self, make_tape):
        """Test that all parameters are correctly modified after construction"""
        tape, params = make_tape
        new_params = [0.6543, -0.654, 0, 0.3, 0.6]

        tape.trainable_params = [1, 3]
        new_tape = tape.bind_new_parameters(new_params, [0, 1, 2, 3, 4])

        for pinfo, pval in zip(new_tape.par_info, new_params):
            assert pinfo["op"].data[pinfo["p_idx"]] == pval

        assert new_tape.get_parameters(trainable_only=False) == new_params
        assert tape.get_parameters(trainable_only=False) == params

    def test_setting_parameters_error(self, make_tape):
        """Test that exceptions are raised if incorrect parameters
        are attempted to be set"""
        tape, _ = make_tape

        with pytest.raises(ValueError, match="Number of provided parameters does not match"):
            tape.bind_new_parameters([0.54], [0, 1, 2, 3, 4])

        with pytest.raises(ValueError, match="Number of provided parameters does not match"):
            tape.bind_new_parameters([0.54, 0.54, 0.123], [0, 1])

    def test_array_parameter(self):
        """Test that array parameters integrate properly"""
        a = np.array([1, 1, 0, 0]) / np.sqrt(2)
        params = [a, 0.32, 0.76, 1.0]

        with QuantumTape() as tape:
            op_ = qml.StatePrep(params[0], wires=[0, 1])
            qml.Rot(params[1], params[2], params[3], wires=0)

        assert tape.num_params == len(params)
        assert tape.get_parameters() == params

        b = np.array([0.0, 1.0, 0.0, 0.0])
        new_params = [b, 0.543, 0.654, 0.123]
        new_tape = tape.bind_new_parameters(new_params, [0, 1, 2, 3])
        assert new_tape.get_parameters() == new_params
        assert tape.get_parameters() == params

        assert np.all(op_.data[0] == a)
        assert np.all(new_tape[0].data[0] == b)

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
        new_tape = tape.bind_new_parameters(new_params, [0, 1, 2, 3])
        assert new_tape.get_parameters() == new_params
        assert tape.get_parameters() == params

        assert np.all(obs.data[0] == H)
        assert np.all(new_tape[1].obs.data[0] == H2)


class TestInverseAdjoint:
    """Tests for tape inversion"""

    def test_adjoint(self):
        """Test that tape.adjoint is a copy of in-place inversion."""

        init_state = np.array([1, 1])
        p = [0.1, 0.2, 0.3, 0.4]

        with QuantumTape() as tape:
            qml.BasisState(init_state, wires=[0, "a"])
            qml.RX(p[0], wires=0)
            qml.adjoint(qml.Rot(*p[1:], wires=0))
            qml.CNOT(wires=[0, "a"])
            qml.probs(wires=0)
            qml.probs(wires="a")

        with QuantumTape() as tape2:
            adjoint_tape = tape.adjoint()

        assert tape2[0] is adjoint_tape

        assert id(adjoint_tape) != id(tape)
        assert isinstance(adjoint_tape, QuantumTape)


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
        assert new_tape.shots is tape.shots

        assert isinstance(new_tape.operations[0], qml.RZ)
        assert isinstance(new_tape.operations[1], qml.RY)
        assert isinstance(new_tape.operations[2], qml.RZ)

        # check that modifying the new tape does not affect the old tape

        new_tape.trainable_params = [0]

        assert tape.get_parameters() == [0.1, 0.2, 0.3]
        assert tape.trainable_params == [0, 1, 2]

    def test_decomposition_removing_parameters(self):
        """Test that decompositions which reduce the number of parameters
        on the tape retain tape consistency."""
        with QuantumTape() as tape:
            qml.BasisState(np.array([1]), wires=0)

        new_tape = tape.expand(depth=1)

        assert len(new_tape.operations) == 1
        assert new_tape.operations[0].name == "PauliX"
        assert new_tape.operations[0].wires.tolist() == [0]
        assert new_tape.num_params == 0
        assert new_tape.get_parameters() == []
        assert new_tape.shots is tape.shots

        assert isinstance(new_tape.operations[0], qml.PauliX)

    def test_decomposition_adding_parameters(self):
        """Test that decompositions which increase the number of parameters
        on the tape retain tape consistency."""
        with QuantumTape() as tape:
            qml.PauliX(wires=0)

        new_tape = tape.expand()

        assert len(new_tape.operations) == 2

        assert new_tape.operations[0].name == "RX"
        assert new_tape.operations[1].name == "GlobalPhase"

        assert new_tape.num_params == 2
        assert new_tape.get_parameters() == [np.pi, -np.pi / 2]
        assert new_tape.shots is tape.shots

    def test_stopping_criterion(self):
        """Test that gates specified in the stop_at
        argument are not expanded."""
        with QuantumTape() as tape:
            qml.U3(0, 1, 2, wires=0)
            qml.Rot(3, 4, 5, wires=0)
            qml.probs(wires=0)
            qml.probs(wires="a")

        new_tape = tape.expand(stop_at=lambda obj: getattr(obj, "name", None) in ["Rot"])
        assert len(new_tape.operations) == 4
        assert "Rot" in [i.name for i in new_tape.operations]
        assert "U3" not in [i.name for i in new_tape.operations]

    def test_depth_expansion(self):
        """Test expanding with depth=3"""
        with QuantumTape() as tape:
            # Will be decomposed into PauliX(0), PauliX(0)
            # Each PauliX will then be decomposed into PhaseShift, RX, PhaseShift.
            qml.BasisState(np.array([1, 1]), wires=[0, "a"])

            with QuantumTape():
                # will be decomposed into a RZ, RY, RZ
                qml.Rot(0.543, 0.1, 0.4, wires=0)

            qml.CNOT(wires=[0, "a"])
            qml.RY(0.2, wires="a")
            qml.probs(wires=0)
            qml.probs(wires="a")

        new_tape = tape.expand(depth=2)
        assert len(new_tape.operations) == 9

    @pytest.mark.parametrize("skip_first", (True, False))
    @pytest.mark.parametrize(
        "op, decomp",
        zip(
            [
                qml.BasisState([1, 0], wires=[0, 1]),
                qml.StatePrep([0, 1, 0, 0], wires=[0, 1]),
                qml.AmplitudeEmbedding([0, 1, 0, 0], wires=[0, 1]),
                qml.PauliZ(0),
            ],
            [
                qml.PauliX(0),
                qml.MottonenStatePreparation([0, 1, 0, 0], wires=[0, 1]),
                qml.MottonenStatePreparation([0, 1, 0, 0], wires=[0, 1]),
                qml.PauliZ(0),
            ],
        ),
    )
    def test_expansion_state_prep(self, skip_first, op, decomp):
        """Test that StatePrepBase operations are expanded correctly without
        expanding other operations in the tape.
        """
        ops = [
            op,
            qml.PauliZ(wires=0),
            qml.Rot(0.1, 0.2, 0.3, wires=0),
            qml.BasisState([0], wires=1),
            qml.StatePrep([0, 1], wires=0),
        ]
        tape = QuantumTape(ops=ops, measurements=[])
        new_tape = expand_tape_state_prep(tape, skip_first=skip_first)

        true_decomposition = []
        if skip_first:
            true_decomposition.append(op)
        else:
            true_decomposition.append(decomp)
        true_decomposition += [
            qml.PauliZ(wires=0),
            qml.Rot(0.1, 0.2, 0.3, wires=0),
            qml.MottonenStatePreparation([0, 1], wires=[0]),
        ]

        assert len(new_tape.operations) == len(true_decomposition)
        for tape_op, true_op in zip(new_tape.operations, true_decomposition):
            qml.assert_equal(tape_op, true_op)

    @pytest.mark.filterwarnings("ignore:The ``name`` property and keyword argument of")
    def test_stopping_criterion_with_depth(self):
        """Test that gates specified in the stop_at
        argument are not expanded."""
        with QuantumTape() as tape:
            # Will be decomposed into PauliX(0), PauliX(0)
            qml.BasisState(np.array([1, 1]), wires=[0, "a"])

            with QuantumTape():
                # will be decomposed into a RZ, RY, RZ
                qml.Rot(0.543, 0.1, 0.4, wires=0)

            qml.CNOT(wires=[0, "a"])
            qml.RY(0.2, wires="a")
            qml.probs(wires=0)
            qml.probs(wires="a")

        new_tape = tape.expand(
            depth=2, stop_at=lambda obj: getattr(obj, "name", None) in ["PauliX"]
        )
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
            qml.var(qml.Hermitian(np.array([[1, 2], [2, 4]]), wires=[1]))

        new_tape = tape.expand(expand_measurements=True)

        assert len(new_tape.operations) == 6

        expected = [
            qml.measurements.ProbabilityMP,
            qml.measurements.ExpectationMP,
            qml.measurements.VarianceMP,
        ]
        assert [isinstance(m, r) for m, r in zip(new_tape.measurements, expected)]

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

        with pytest.raises(QuantumFunctionError, match="Only observables that are qubit-wise"):
            tape.expand(expand_measurements=True)

    @pytest.mark.parametrize("ret", [expval, var, probs])
    @pytest.mark.parametrize("wires", [None, 0, [0]])
    def test_expand_tape_multiple_wires_non_commuting_no_obs_sampling(self, ret, wires):
        """Test if a QuantumFunctionError is raised during tape expansion if non-commuting
        observables (also involving computational basis sampling) are on the same wire"""
        with QuantumTape() as tape:
            qml.RX(0.3, wires=0)
            qml.RY(0.4, wires=1)
            ret(op=qml.PauliX(0))
            qml.sample(wires=wires)

        with pytest.raises(QuantumFunctionError, match="Only observables that are qubit-wise"):
            tape.expand(expand_measurements=True)

    @pytest.mark.parametrize("ret", [expval, var, probs])
    @pytest.mark.parametrize("wires", [None, 0, [0]])
    def test_expand_tape_multiple_wires_non_commuting_no_obs_counting(self, ret, wires):
        """Test if a QuantumFunctionError is raised during tape expansion if non-commuting
        observables (also involving computational basis sampling) are on the same wire"""
        with QuantumTape() as tape:
            qml.RX(0.3, wires=0)
            qml.RY(0.4, wires=1)
            ret(op=qml.PauliX(0))
            qml.counts(wires=wires)

        with pytest.raises(QuantumFunctionError, match="Only observables that are qubit-wise"):
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

        with pytest.raises(QuantumFunctionError, match=expected_error_msg):
            tape.expand(expand_measurements=True)

    def test_multiple_expand_no_change_original_tape(self):
        """Test that the original tape is not changed multiple time after maximal expansion."""
        with QuantumTape() as tape:
            qml.RX(0.1, wires=[0])
            qml.RY(0.2, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
            qml.expval(qml.PauliZ(0))

        expand_tape = tape.expand()
        circuit_after_first_expand = expand_tape.operations
        twice_expand_tape = tape.expand()
        circuit_after_second_expand = twice_expand_tape.operations
        for op1, op2 in zip(circuit_after_first_expand, circuit_after_second_expand):
            qml.assert_equal(op1, op2)

    def test_expand_does_not_affect_original_tape(self):
        """Test that expand_tape does not modify the inputted tape while creating a new one."""
        ops = [qml.RX(1.1, 0)]
        measurements = [qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(0))]
        tape = qml.tape.QuantumTape(ops, measurements)
        expanded = tape.expand()

        assert len(tape.operations) == 1
        qml.assert_equal(tape.operations[0], ops[0])
        assert len(tape.obs_sharing_wires) == 2
        for obs in tape.obs_sharing_wires:
            qml.assert_equal(obs, qml.X(0))
        qml.assert_equal(tape.measurements[0], qml.expval(qml.PauliX(0)))
        qml.assert_equal(tape.measurements[1], qml.expval(qml.PauliX(0)))
        assert tape.shots == qml.measurements.Shots(None)

        assert len(expanded.operations) == 2
        qml.assert_equal(expanded.operations[0], ops[0])
        qml.assert_equal(expanded.operations[1], qml.RY(-np.pi / 2, 0))  # new rotation
        assert len(expanded.obs_sharing_wires) == 2
        for obs in expanded.obs_sharing_wires:
            qml.assert_equal(obs, qml.Z(0))
        qml.assert_equal(expanded.measurements[0], qml.expval(qml.PauliZ(0)))
        qml.assert_equal(expanded.measurements[1], qml.expval(qml.PauliZ(0)))
        assert expanded.shots is tape.shots

    def test_expand_tape_does_not_check_mp_name_by_default(self, recwarn):
        """Test that calling expand_tape does not refer to MP.name"""

        def stop_at(obj):
            return obj.name in ["PauliX"]

        qs = qml.tape.QuantumScript(measurements=[qml.expval(qml.PauliZ(0))])
        qs.expand(stop_at=stop_at)
        assert len(recwarn) == 0


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

        assert tape.batch_size is None

        # test execution with no parameters
        res1 = dev.execute(tape)
        assert tape.get_parameters() == params

        # test setting parameters
        new_tape = tape.bind_new_parameters(params=[0.5, 0.6], indices=[0, 1])
        res3 = dev.execute(new_tape)
        assert not np.allclose(res1, res3, atol=tol, rtol=0)
        assert new_tape.get_parameters() == [0.5, 0.6]

    def test_no_output_execute(self):
        """Test that tapes with no measurement process return
        an empty list."""
        dev = qml.device("default.qubit", wires=2)
        params = [0.1, 0.2]

        with QuantumTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])

        res = dev.execute(tape)
        assert isinstance(res, tuple)

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

        res = dev.execute(tape)

        assert res.shape == ()

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

        res = dev.execute(tape)
        assert isinstance(res, tuple)
        assert len(res) == 2

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

        res = dev.execute(tape)
        assert isinstance(res, tuple)
        assert len(res) == 2

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

        res = dev.execute(tape)

        assert isinstance(res, tuple)
        assert len(res) == 2

        assert isinstance(res[0], np.float64)
        assert np.allclose(res[0], np.cos(x), atol=tol, rtol=0)

        assert isinstance(res[1], np.ndarray)
        final_state, _ = qml.devices.qubit.get_final_state(tape)
        assert np.allclose(res[1], np.abs(final_state.flatten()) ** 2, atol=tol, rtol=0)

    def test_single_mode_sample(self):
        """Test that there is only one array of values returned
        for a single wire qml.sample"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.sample(qml.PauliZ(0) @ qml.PauliX(1))

        tape._shots = qml.measurements.Shots(10)
        res = dev.execute(tape)
        assert res.shape == (10,)

    def test_multiple_samples(self):
        """Test that there is only one array of values returned
        for multiple samples"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.sample(qml.PauliZ(0))
            qml.sample(qml.PauliZ(1))

        tape._shots = qml.measurements.Shots(10)
        res = dev.execute(tape)
        assert isinstance(res, tuple)
        assert isinstance(res[0], np.ndarray)
        assert isinstance(res[1], np.ndarray)
        assert res[0].shape == (10,)
        assert res[1].shape == (10,)

    def test_samples_expval(self):
        """Test that multiple arrays of values are returned
        for combinations of samples and statistics"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.sample(qml.PauliZ(0))
            qml.expval(qml.PauliZ(1))

        tape._shots = qml.measurements.Shots(10)
        res = dev.execute(tape)
        assert isinstance(res, tuple)
        assert isinstance(res[0], np.ndarray)
        assert res[0].shape == (10,)
        assert isinstance(res[1], np.float64)
        assert res[1].shape == ()

    def test_decomposition(self, tol):
        """Test decomposition onto a device's supported gate set"""
        dev = qml.device("default.qubit", wires=1)
        from pennylane.devices.default_qubit import stopping_condition

        with QuantumTape() as tape:
            qml.U3(0.1, 0.2, 0.3, wires=[0])
            qml.expval(qml.PauliZ(0))

        def stop_fn(op):
            return isinstance(op, qml.measurements.MeasurementProcess) or stopping_condition(op)

        tape = tape.expand(stop_at=stop_fn)
        res = dev.execute(tape)
        assert np.allclose(res, np.cos(0.1), atol=tol, rtol=0)


class TestCVExecution:
    """Tests for CV tape execution"""

    def test_single_output_value(self):
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

        res = dev.batch_execute([tape])[0]
        assert res.shape == ()


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
        assert all(o1 is o2 for o1, o2 in zip(copied_tape.operations, tape.operations))
        assert all(o1 is o2 for o1, o2 in zip(copied_tape.observables, tape.observables))
        assert all(m1 is m2 for m1, m2 in zip(copied_tape.measurements, tape.measurements))

        # operation data is also a reference
        assert copied_tape.operations[0].wires is tape.operations[0].wires
        assert copied_tape.operations[0].data[0] is tape.operations[0].data[0]

        # check that all tape metadata is identical
        assert tape.get_parameters() == copied_tape.get_parameters()
        assert tape.wires == copied_tape.wires
        assert tape.data == copied_tape.data

    @pytest.mark.parametrize("copy_fn", [lambda tape: tape.copy(copy_operations=True), copy.copy])
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
        assert all(o1 is not o2 for o1, o2 in zip(copied_tape.operations, tape.operations))
        assert all(o1 is not o2 for o1, o2 in zip(copied_tape.observables, tape.observables))
        assert all(m1 is not m2 for m1, m2 in zip(copied_tape.measurements, tape.measurements))

        # however, the underlying operation data *is still shared*
        assert copied_tape.operations[0].wires is tape.operations[0].wires
        assert copied_tape.operations[0].data[0] is tape.operations[0].data[0]

        assert tape.get_parameters() == copied_tape.get_parameters()
        assert tape.wires == copied_tape.wires
        assert tape.data == copied_tape.data

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
        assert all(o1 is not o2 for o1, o2 in zip(copied_tape.operations, tape.operations))
        assert all(o1 is not o2 for o1, o2 in zip(copied_tape.observables, tape.observables))
        assert all(m1 is not m2 for m1, m2 in zip(copied_tape.measurements, tape.measurements))

        # The underlying operation data has also been copied
        assert copied_tape.operations[0].wires is not tape.operations[0].wires

        # however, the underlying operation *parameters* are still shared
        # to support PyTorch, which does not support deep copying of tensors
        assert copied_tape.operations[0].data[0] is tape.operations[0].data[0]


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
    return qml.execute([tape], dev, diff_method=qml.gradients.param_shift)


measures = [
    (qml.expval(qml.PauliZ(0)), ()),
    (qml.var(qml.PauliZ(0)), ()),
    (qml.probs(wires=[0]), (2,)),
    (qml.probs(wires=[0, 1]), (4,)),
    (qml.state(), (8,)),  # Assumes 3-qubit device
    (qml.density_matrix(wires=[0, 1]), (4, 4)),
    (
        qml.sample(qml.PauliZ(0)),
        None,
    ),  # Shape is None because the expected shape is in the test case
    (qml.sample(), None),  # Shape is None because the expected shape is in the test case
]

multi_measurements = [
    ([qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))], ((), ())),
    ([qml.probs(wires=[0]), qml.probs(wires=[1])], ((2,), (2,))),
    ([qml.probs(wires=[0]), qml.probs(wires=[1, 2])], ((2**1,), (2**2,))),
    ([qml.probs(wires=[0, 2]), qml.probs(wires=[1])], ((2**2,), (2**1,))),
    (
        [qml.probs(wires=[0]), qml.probs(wires=[1, 2]), qml.probs(wires=[0, 1, 2])],
        ((2**1,), (2**2,), (2**3,)),
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
        if shots is None and isinstance(measurement, qml.measurements.SampleMP):
            pytest.skip("Sample doesn't support analytic computations.")

        num_wires = 3
        dev = qml.device("default.qubit", wires=num_wires)

        a = np.array(0.1)
        b = np.array(0.2)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.apply(measurement)

        tape = qml.tape.QuantumScript.from_queue(q, shots=shots)
        shot_dim = shots if not isinstance(shots, tuple) else len(shots)
        if expected_shape is None:
            expected_shape = (shot_dim,)

        if isinstance(measurement, qml.measurements.SampleMP):
            if measurement.obs is not None:
                expected_shape = (shots,)

            else:
                expected_shape = (shots, num_wires)

        with pytest.warns(
            PennyLaneDeprecationWarning, match="``QuantumScript.shape`` is deprecated"
        ):
            assert tape.shape(dev) == expected_shape

    @pytest.mark.parametrize("measurement, _", measures)
    @pytest.mark.parametrize("shots", [None, 1, 10, (1, 2, 5, 3)])
    def test_output_shapes_single_qnode_check(self, measurement, _, shots):
        """Test that the output shape produced by the tape matches the output
        shape of a QNode for a single measurement."""
        if shots is None and isinstance(measurement, qml.measurements.SampleMP):
            pytest.skip("Sample doesn't support analytic computations.")

        if shots is not None and isinstance(measurement, qml.measurements.StateMP):
            pytest.skip("State and density matrix don't support finite shots and raise a warning.")

        # TODO: revisit when qml.sample without an observable has been updated
        # with shot vectors
        if (
            isinstance(shots, tuple)
            and isinstance(measurement, qml.measurements.SampleMP)
            and not measurement.obs
        ):
            pytest.skip("qml.sample with no observable is to be updated for shot vectors.")

        dev = qml.device("default.qubit", wires=3)

        a = np.array(0.1)
        b = np.array(0.2)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.apply(measurement)

        tape = qml.tape.QuantumScript.from_queue(q, shots=shots)
        program = dev.preprocess_transforms()
        res = qml.execute(
            [tape], dev, diff_method=qml.gradients.param_shift, transform_program=program
        )[0]

        if isinstance(res, tuple):
            res_shape = tuple(r.shape for r in res)
        else:
            res_shape = res.shape

        res_shape = res_shape if res_shape != tuple() else ()

        with pytest.warns(
            PennyLaneDeprecationWarning, match="``QuantumScript.shape`` is deprecated"
        ):
            assert tape.shape(dev) == res_shape

    @pytest.mark.autograd
    @pytest.mark.parametrize("measurements, expected", multi_measurements)
    @pytest.mark.parametrize("shots", [None, 1, 10])
    def test_multi_measure(self, measurements, expected, shots):
        """Test that the expected output shape is obtained when using multiple
        expectation value, variance and probability measurements."""
        dev = qml.device("default.qubit", wires=3)

        a = np.array(0.1)
        b = np.array(0.2)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            for m in measurements:
                qml.apply(m)

        tape = qml.tape.QuantumScript.from_queue(q, shots=shots)
        if isinstance(measurements[0], qml.measurements.SampleMP):
            expected[1] = shots
            expected = tuple(expected)

        with pytest.warns(
            PennyLaneDeprecationWarning, match="``QuantumScript.shape`` is deprecated"
        ):
            res = tape.shape(dev)
        assert res == expected

        execution_results = cost(tape, dev)
        results = execution_results[0]
        assert isinstance(results, tuple)
        assert len(expected) == len(measurements)
        assert res == expected

    @pytest.mark.autograd
    @pytest.mark.parametrize("measurements, expected", multi_measurements)
    def test_multi_measure_shot_vector(self, measurements, expected):
        """Test that the expected output shape is obtained when using multiple
        expectation value, variance and probability measurements with a shot
        vector."""
        if isinstance(measurements[0], qml.measurements.ProbabilityMP):
            num_wires = {len(m.wires) for m in measurements}
            if len(num_wires) > 1:
                pytest.skip(
                    "Multi-probs with varying number of varies when using a shot vector is to be updated in PennyLane."
                )

        shots = (1, 1, 5, 1)
        dev = qml.device("default.qubit", wires=3)

        a = np.array(0.1)
        b = np.array(0.2)

        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            for m in measurements:
                qml.apply(m)

        tape = qml.tape.QuantumScript.from_queue(q, shots=shots)
        # Modify expected to account for shot vector
        expected = tuple(expected for _ in shots)

        with pytest.warns(
            PennyLaneDeprecationWarning, match="``QuantumScript.shape`` is deprecated"
        ):
            res = tape.shape(dev)
        assert res == expected

    @pytest.mark.autograd
    @pytest.mark.parametrize("shots", [1, 10])
    def test_multi_measure_sample(self, shots):
        """Test that the expected output shape is obtained when using multiple
        qml.sample measurements."""
        dev = qml.device("default.qubit", wires=3)

        a = np.array(0.1)
        b = np.array(0.2)

        num_samples = 3
        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            for i in range(num_samples):
                qml.sample(qml.PauliZ(i))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shots)
        expected = tuple((shots,) for _ in range(num_samples))

        with pytest.warns(
            PennyLaneDeprecationWarning, match="``QuantumScript.shape`` is deprecated"
        ):
            res = tape.shape(dev)
        assert res == expected

    @pytest.mark.autograd
    def test_multi_measure_sample_shot_vector(self):
        """Test that the expected output shape is obtained when using multiple
        qml.sample measurements with a shot vector."""
        shots = (1, 1, 5, 1)
        dev = qml.device("default.qubit", wires=3)

        a = np.array(0.1)
        b = np.array(0.2)

        num_samples = 3
        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            for i in range(num_samples):
                qml.sample(qml.PauliZ(i))

        tape = qml.tape.QuantumScript.from_queue(q, shots=shots)
        expected = []
        for s in shots:
            expected.append(tuple((s,) for _ in range(num_samples)))

        expected = tuple(expected)

        with pytest.warns(
            PennyLaneDeprecationWarning, match="``QuantumScript.shape`` is deprecated"
        ):
            res = tape.shape(dev)

        for r, e in zip(res, expected):
            assert r == e

    @pytest.mark.autograd
    @pytest.mark.parametrize("measurement, _", measures)
    @pytest.mark.parametrize("shots", [None, 1, 10])
    def test_broadcasting_single(self, measurement, _, shots):
        """Test that the output shape produced by the tape matches the expected
        output shape for a single measurement and parameter broadcasting"""
        if shots is None and isinstance(measurement, qml.measurements.SampleMP):
            pytest.skip("Sample doesn't support analytic computations.")

        if isinstance(measurement, qml.measurements.StateMP) and measurement.wires is not None:
            pytest.skip("Density matrix does not support parameter broadcasting")

        num_wires = 3
        dev = qml.device("default.qubit", wires=num_wires)

        a = np.array([0.1, 0.2, 0.3])
        b = np.array([0.4, 0.5, 0.6])

        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            qml.apply(measurement)

        tape = qml.tape.QuantumScript.from_queue(q, shots=shots)
        program = dev.preprocess_transforms()
        expected = qml.execute([tape], dev, diff_method=None, transform_program=program)[0]

        with pytest.warns(
            PennyLaneDeprecationWarning, match="``QuantumScript.shape`` is deprecated"
        ):
            assert tape.shape(dev) == expected.shape

    @pytest.mark.autograd
    @pytest.mark.parametrize("measurement, expected", measures)
    @pytest.mark.parametrize("shots", [None, 1, 10])
    def test_broadcasting_multi(self, measurement, expected, shots):
        """Test that the output shape produced by the tape matches the expected
        output shape for multiple measurements and parameter broadcasting"""
        if shots is None and isinstance(measurement, qml.measurements.SampleMP):
            pytest.skip("Sample doesn't support analytic computations.")

        if isinstance(measurement, qml.measurements.StateMP):
            pytest.skip("State does not support multiple measurements")

        dev = qml.device("default.qubit", wires=3)

        a = np.array([0.1, 0.2, 0.3])
        b = np.array([0.4, 0.5, 0.6])

        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            for _ in range(2):
                qml.apply(measurement)

        tape = qml.tape.QuantumScript.from_queue(q, shots=shots)
        program = dev.preprocess_transforms()
        expected = qml.execute([tape], dev, diff_method=None, transform_program=program)[0]
        expected = tuple(i.shape for i in expected)

        with pytest.warns(
            PennyLaneDeprecationWarning, match="``QuantumScript.shape`` is deprecated"
        ):
            assert tape.shape(dev) == expected


class TestNumericType:
    """Tests for determining the numeric type of the tape output."""

    @pytest.mark.parametrize(
        "ret", [qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0)), qml.probs(wires=[0])]
    )
    @pytest.mark.parametrize("shots", [None, 1, (1, 2, 3)])
    def test_float_measures(self, ret, shots):
        """Test that most measurements output floating point values and that
        the tape output domain correctly identifies this."""
        dev = qml.device("default.qubit", wires=3)

        @qml.set_shots(shots)
        @qml.qnode(dev)
        def circuit(a, b):
            qml.RY(a, wires=[0])
            qml.RZ(b, wires=[0])
            return qml.apply(ret)

        result = circuit(0.3, 0.2)

        # Double-check the domain of the QNode output
        if isinstance(shots, tuple):
            assert np.issubdtype(result[0].dtype, float)
        else:
            assert np.issubdtype(result.dtype, float)

        tape = qml.workflow.construct_tape(circuit)(0.3, 0.2)

        with pytest.warns(
            PennyLaneDeprecationWarning, match="``QuantumScript.numeric_type`` is deprecated"
        ):
            assert tape.numeric_type is float

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

        tape = qml.workflow.construct_tape(circuit)(0.3, 0.2)

        with pytest.warns(
            PennyLaneDeprecationWarning, match="``QuantumScript.numeric_type`` is deprecated"
        ):
            assert tape.numeric_type is complex

    def test_sample_int(self):
        """Test that the tape can correctly determine the output domain for a
        sampling measurement with no observable."""
        dev = qml.device("default.qubit", wires=3)

        @qml.set_shots(5)
        @qml.qnode(dev)
        def circuit():
            qml.RY(0.4, wires=[0])
            return qml.sample()

        result = circuit()

        # Double-check the domain of the QNode output
        assert np.issubdtype(result.dtype, int)

        tape = qml.workflow.construct_tape(circuit)()

        with pytest.warns(
            PennyLaneDeprecationWarning, match="``QuantumScript.numeric_type`` is deprecated"
        ):
            assert tape.numeric_type is int

    # TODO: add cases for each interface once qml.Hermitian supports other
    # interfaces
    def test_sample_real_eigvals(self):
        """Test that the tape can correctly determine the output domain when
        sampling a Hermitian observable with real eigenvalues."""
        dev = qml.device("default.qubit", wires=3)

        arr = np.array(
            [
                1.32,
                2.312,
            ]
        )
        herm = np.outer(arr, arr)

        @qml.set_shots(5)
        @qml.qnode(dev)
        def circuit(a, b):
            # pylint: disable=unused-argument
            qml.RY(0.4, wires=[0])
            return qml.sample(qml.Hermitian(herm, wires=0))

        result = circuit(0.3, 0.2)

        # Double-check the domain of the QNode output
        assert np.issubdtype(result[0].dtype, float)

        tape = qml.workflow.construct_tape(circuit)(0.3, 0.2)

        with pytest.warns(
            PennyLaneDeprecationWarning, match="``QuantumScript.numeric_type`` is deprecated"
        ):
            assert tape.numeric_type is float

    @pytest.mark.autograd
    def test_sample_real_and_int_eigvals(self):
        """Test that the tape can correctly determine the output domain for
        multiple sampling measurements with a Hermitian observable with real
        eigenvalues and another sample with integer values."""
        dev = qml.device("default.qubit", wires=3)

        arr = np.array(
            [
                1.32,
                2.312,
            ]
        )
        herm = np.outer(arr, arr)

        @qml.set_shots(5)
        @qml.qnode(dev)
        def circuit(a, b):
            qml.RY(a, wires=0)
            qml.RX(b, wires=0)
            return qml.sample(qml.Hermitian(herm, wires=0)), qml.sample()

        result = circuit(0, 3)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].dtype == float
        assert result[1].dtype == int

        tape = qml.workflow.construct_tape(circuit)(0, 3)

        with pytest.warns(
            PennyLaneDeprecationWarning, match="``QuantumScript.numeric_type`` is deprecated"
        ):
            assert tape.numeric_type == (float, int)

    def test_multi_type_measurements_numeric_type_error(self):
        """Test that querying the numeric type of a tape with several types of
        measurements raises an error."""
        a = 0.3
        b = 0.3

        with qml.queuing.AnnotatedQueue() as q:
            qml.RY(a, wires=[0])
            qml.RZ(b, wires=[0])
            qml.expval(qml.PauliZ(0))
            qml.probs(wires=[0])

        tape = qml.tape.QuantumScript.from_queue(q)

        with pytest.warns(
            PennyLaneDeprecationWarning, match="``QuantumScript.numeric_type`` is deprecated"
        ):
            assert tape.numeric_type == (float, float)


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
