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
"""Unit tests for the QuantumTape"""
import copy

import numpy as np
import pytest

import pennylane as qml
from pennylane.tape import QuantumTape, TapeCircuitGraph
from pennylane.tape.measure import MeasurementProcess, expval, sample, var


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
        assert tape.interface is None

        assert tape.wires == qml.wires.Wires([0, "a", 4])
        assert tape._output_dim == len(obs[0].wires) + 2 ** len(obs[1].wires)

    def test_observable_processing(self, make_tape):
        """Test that observables are processed correctly"""
        tape, ops, obs = make_tape

        # test that the internal tape._measurements list is created properly
        assert isinstance(tape._measurements[0], MeasurementProcess)
        assert tape._measurements[0].return_type == qml.operation.Expectation
        assert tape._measurements[0].obs == obs[0]

        assert isinstance(tape._measurements[1], MeasurementProcess)
        assert tape._measurements[1].return_type == qml.operation.Probability

        # test the public observables property
        assert len(tape.observables) == 2
        assert tape.observables[0].name == "PauliX"
        assert tape.observables[1].return_type == qml.operation.Probability

        # test the public measurements property
        assert len(tape.measurements) == 2
        assert all(isinstance(m, MeasurementProcess) for m in tape.measurements)
        assert tape.observables[0].return_type == qml.operation.Expectation
        assert tape.observables[1].return_type == qml.operation.Probability

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
        assert tape.measurements[0].return_type is qml.operation.Expectation
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
        assert tape.measurements[0].return_type is qml.operation.Expectation
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
        assert tape.measurements[0].return_type is qml.operation.Expectation
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
        assert tape.measurements[0].return_type is qml.operation.Variance
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
        with pytest.raises(ValueError, match="must occur prior to any quantum"):
            with QuantumTape() as tape:
                B = qml.PauliX(wires=0)
                qml.BasisState(np.array([0, 1]), wires=[0, 1])

    def test_measurement_before_operation(self):
        """Test that an exception is raised if a measurement occurs before a operation"""

        with pytest.raises(ValueError, match="must occur prior to any measurements"):
            with QuantumTape() as tape:
                qml.expval(qml.PauliZ(wires=1))
                qml.RX(0.5, wires=0)
                qml.expval(qml.PauliZ(wires=1))

    def test_observable_with_no_measurement(self):
        """Test that an exception is raised if an observable is used without a measurement"""

        with pytest.raises(ValueError, match="does not have a measurement type specified"):
            with QuantumTape() as tape:
                qml.RX(0.5, wires=0)
                qml.Hermitian(np.array([[0, 1], [1, 0]]), wires=1)
                qml.expval(qml.PauliZ(wires=1))

        with pytest.raises(ValueError, match="does not have a measurement type specified"):
            with QuantumTape() as tape:
                qml.RX(0.5, wires=0)
                qml.PauliX(wires=0) @ qml.PauliY(wires=1)
                qml.expval(qml.PauliZ(wires=1))

    def test_sampling(self):
        """Test that the tape correctly marks itself as returning samples"""
        with QuantumTape() as tape:
            qml.expval(qml.PauliZ(wires=1))

        assert not tape.is_sampled

        with QuantumTape() as tape:
            qml.sample(qml.PauliZ(wires=0))

        assert tape.is_sampled


class TestGraph:
    """Tests involving graph creation"""

    def test_graph_creation(self, mocker):
        """Test that the circuit graph is correctly created"""
        spy = mocker.spy(TapeCircuitGraph, "__init__")

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

    def test_resources_empty_tape(self, make_empty_tape):
        """Test that empty tapes return empty resource counts."""
        tape = make_empty_tape

        assert tape.get_depth() == 0
        assert len(tape.get_resources()) == 0

    def test_resources_tape(self, make_tape):
        """Test that regular tapes return correct number of resources."""
        tape = make_tape

        assert tape.get_depth() == 3

        # Verify resource counts
        resources = tape.get_resources()
        assert len(resources) == 3
        assert resources["RX"] == 2
        assert resources["Rot"] == 1
        assert resources["CNOT"] == 1

    def test_resources_add_to_tape(self, make_extendible_tape):
        """Test that tapes return correct number of resources after adding to them."""
        tape = make_extendible_tape

        assert tape.get_depth() == 3

        resources = tape.get_resources()
        assert len(resources) == 3
        assert resources["RX"] == 2
        assert resources["Rot"] == 1
        assert resources["CNOT"] == 1

        with tape as tape:
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.1, wires=3)
            qml.expval(qml.PauliX(wires="a"))
            qml.probs(wires=[0, "a"])

        assert tape.get_depth() == 4

        resources = tape.get_resources()
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

    def test_parameter_processing(self, make_tape):
        """Test that parameters are correctly counted and processed"""
        tape, params = make_tape
        assert tape.num_params == len(params)
        assert tape.trainable_params == set(range(len(params)))
        assert tape.get_parameters() == params

    def test_set_trainable_params(self, make_tape):
        """Test that setting trainable parameters works as expected"""
        tape, params = make_tape
        trainable = {0, 2, 3}
        tape.trainable_params = trainable
        assert tape._trainable_params == trainable
        assert tape.num_params == 3
        assert tape.get_parameters() == [params[i] for i in tape.trainable_params]

        # add additional trainable parameters
        trainable = {1, 2, 3, 4}
        tape.trainable_params = trainable
        assert tape._trainable_params == trainable
        assert tape.num_params == 4
        assert tape.get_parameters() == [params[i] for i in tape.trainable_params]

    def test_changing_params(self, make_tape):
        """Test that changing trainable parameters works as expected"""
        tape, params = make_tape
        trainable = {0, 2, 3}
        tape.trainable_params = trainable
        assert tape._trainable_params == trainable
        assert tape.num_params == 3
        assert tape.get_parameters() == [params[i] for i in tape.trainable_params]
        assert tape.get_parameters(trainable_only=False) == params

    def test_set_trainable_params_error(self, make_tape):
        """Test that exceptions are raised if incorrect parameters
        are set as trainable"""
        tape, _ = make_tape

        with pytest.raises(ValueError, match="must be positive integers"):
            tape.trainable_params = {-1, 0}

        with pytest.raises(ValueError, match="must be positive integers"):
            tape.trainable_params = {0.5}

        with pytest.raises(ValueError, match="has at most 5 parameters"):
            tape.trainable_params = {0, 7}

    def test_setting_parameters(self, make_tape):
        """Test that parameters are correctly modified after construction"""
        tape, params = make_tape
        new_params = [0.6543, -0.654, 0, 0.3, 0.6]

        tape.set_parameters(new_params)

        for pinfo, pval in zip(tape._par_info.values(), new_params):
            assert pinfo["op"].data[pinfo["p_idx"]] == pval

        assert tape.get_parameters() == new_params

        new_params = [0.1, -0.2, 1, 5, 0]
        tape.data = new_params

        for pinfo, pval in zip(tape._par_info.values(), new_params):
            assert pinfo["op"].data[pinfo["p_idx"]] == pval

        assert tape.get_parameters() == new_params

    def test_setting_free_parameters(self, make_tape):
        """Test that free parameters are correctly modified after construction"""
        tape, params = make_tape
        new_params = [-0.654, 0.3]

        tape.trainable_params = {1, 3}
        tape.set_parameters(new_params)

        count = 0
        for idx, pinfo in tape._par_info.items():
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

        with monkeypatch.context() as m:
            m.setattr(tape, "_trainable_params", {3, 1})
            tape.set_parameters(new_params)

            assert tape.get_parameters(trainable_only=True) == [
                new_params[0],
                new_params[1],
            ]

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

        tape.trainable_params = {1, 3}
        tape.set_parameters(new_params, trainable_only=False)

        for pinfo, pval in zip(tape._par_info.values(), new_params):
            assert pinfo["op"].data[pinfo["p_idx"]] == pval

        assert tape.get_parameters(trainable_only=False) == new_params

    def test_setting_parameters_error(self, make_tape):
        """Test that exceptions are raised if incorrect parameters
        are attempted to be set"""
        tape, _ = make_tape

        with pytest.raises(ValueError, match="Number of provided parameters does not match"):
            tape.set_parameters([0.54])

        with pytest.raises(ValueError, match="Number of provided parameters does not match"):
            tape.trainable_params = {2, 3}
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


class TestInverse:
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

        tape.inv()

        # check that operation order is reversed
        assert tape.operations == [prep] + ops[::-1]

        # check that operations are inverted
        assert ops[0].inverse
        assert not ops[1].inverse
        assert ops[2].inverse

        # check that parameter order has reversed
        assert tape.get_parameters() == [init_state, p[1], p[2], p[3], p[0]]

    def test_parameter_transforms(self):
        """Test that inversion correctly changes trainable parameters"""
        init_state = np.array([1, 1])
        p = [0.1, 0.2, 0.3, 0.4]

        with QuantumTape() as tape:
            prep = qml.BasisState(init_state, wires=[0, "a"])
            ops = [qml.RX(p[0], wires=0), qml.Rot(*p[1:], wires=0).inv(), qml.CNOT(wires=[0, "a"])]
            m1 = qml.probs(wires=0)
            m2 = qml.probs(wires="a")

        tape.trainable_params = {1, 2}
        tape.inv()

        # check that operation order is reversed
        assert tape.trainable_params == {1, 4}
        assert tape.get_parameters() == [p[1], p[0]]

        # undo the inverse
        tape.inv()
        assert tape.trainable_params == {1, 2}
        assert tape.get_parameters() == [p[0], p[1]]
        assert tape._ops == ops


class TestExpand:
    """Tests for tape expansion"""

    def test_decomposition(self):
        """Test expanding a tape with operations that have decompositions"""
        with QuantumTape() as tape:
            qml.Rot(0.1, 0.2, 0.3, wires=0)

        new_tape = tape.expand()

        assert len(new_tape.operations) == 3
        assert new_tape.get_parameters() == [0.1, 0.2, 0.3]
        assert new_tape.trainable_params == {0, 1, 2}

        assert isinstance(new_tape.operations[0], qml.RZ)
        assert isinstance(new_tape.operations[1], qml.RY)
        assert isinstance(new_tape.operations[2], qml.RZ)

        # check that modifying the new tape does not affect the old tape

        new_tape.trainable_params = {0}
        new_tape.set_parameters([10])

        assert tape.get_parameters() == [0.1, 0.2, 0.3]
        assert tape.trainable_params == {0, 1, 2}

    def test_decomposition_removing_parameters(self):
        """Test that decompositions which reduce the number of parameters
        on the tape retain tape consistency."""
        with QuantumTape() as tape:
            qml.BasisState(np.array([1]), wires=0)

        new_tape = tape.expand()

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
        assert len(new_tape.operations) == 5

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

        new_tape = tape.expand(depth=2)
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

        assert len(new_tape.operations) == 6

        expected = [qml.operation.Probability, qml.operation.Expectation, qml.operation.Variance]
        assert [m.return_type is r for m, r in zip(new_tape.measurements, expected)]

        expected = [None, None, None]
        assert [m.obs is r for m, r in zip(new_tape.measurements, expected)]

        expected = [None, [1, -1, -1, 1], [0, 5]]
        assert [m.eigvals is r for m, r in zip(new_tape.measurements, expected)]

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

    @pytest.mark.parametrize("ret", [expval, var, sample])
    def test_expand_tape_multiple_wires_non_commuting(self, ret):
        """Test if a QuantumFunctionError is raised during tape expansion if non-commuting
        observables are on the same wire"""
        with QuantumTape() as tape:
            qml.RX(0.3, wires=0)
            qml.RY(0.4, wires=1)
            qml.expval(qml.PauliX(0))
            ret(qml.PauliZ(0))

        with pytest.raises(qml.QuantumFunctionError, match="Only observables that are qubit-wise"):
            tape.expand(expand_measurements=True)


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

        # test execution with no parameters
        res1 = tape.execute(dev)
        assert tape.get_parameters() == params

        # test execution with parameters
        res2 = tape.execute(dev, params=[0.5, 0.6])
        assert tape.get_parameters() == params

        # test setting parameters
        tape.set_parameters(params=[0.5, 0.6])
        res3 = tape.execute(dev)
        assert np.allclose(res2, res3, atol=tol, rtol=0)
        assert not np.allclose(res1, res2, atol=tol, rtol=0)
        assert tape.get_parameters() == [0.5, 0.6]

    def test_no_output_execute(self):
        """Test that tapes with no measurement process return
        an empty list."""
        dev = qml.device("default.qubit", wires=2)
        params = [0.1, 0.2]

        with QuantumTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])

        res = tape.execute(dev)
        assert res.size == 0
        assert np.all(res == np.array([]))

    def test_incorrect_output_dim_estimate(self):
        """Test that a quantum tape with an incorrect inferred output dimension
        corrects itself after evaluation."""
        dev = qml.device("default.qubit", wires=3)
        params = [1.0, 1.0, 1.0]

        with QuantumTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.RZ(params[2], wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=0)
            qml.probs(wires=[1])

        # estimate output dim should be correct
        assert tape.output_dim == sum([2, 2])

        # modify the output dim
        tape._output_dim = 2

        res = tape.execute(dev)
        assert tape.output_dim == sum([2, 2])

    def test_incorrect_ragged_output_dim_estimate(self):
        """Test that a quantum tape with an incorrect *ragged* output dimension
        estimate corrects itself after evaluation."""
        dev = qml.device("default.qubit", wires=3)
        params = [1.0, 1.0, 1.0]

        with QuantumTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.RZ(params[2], wires=[2])
            qml.CNOT(wires=[0, 1])
            qml.probs(wires=0)
            qml.probs(wires=[1, 2])

        # estimate output dim should be correct
        assert tape.output_dim == sum([2, 4])

        # modify the output dim
        tape._output_dim = 2

        res = tape.execute(dev)
        assert tape.output_dim == sum([2, 4])

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

        res = tape.execute(dev)
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

        res = tape.execute(dev)
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

        res = tape.execute(dev)
        assert res.shape == (2,)

        expected = [np.cos(x), np.cos(y) ** 2]
        assert np.allclose(res, expected, atol=tol, rtol=0)

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

        res = tape.execute(dev)

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

        res = tape.execute(dev)
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

        res = tape.execute(dev)
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

        res = tape.execute(dev)
        assert res[0].shape == (10,)
        assert isinstance(res[1], float)

    def test_decomposition(self, tol):
        """Test decomposition onto a device's supported gate set"""
        dev = qml.device("default.qubit", wires=1)

        with QuantumTape() as tape:
            qml.U3(0.1, 0.2, 0.3, wires=[0])
            qml.expval(qml.PauliZ(0))

        tape = tape.expand(stop_at=lambda obj: obj.name in dev.operations)
        res = tape.execute(dev)
        assert np.allclose(res, np.cos(0.1), atol=tol, rtol=0)

    def test_multiple_observables_same_wire(self):
        """Test if an error is raised when multiple observables are evaluated on the same wire
        without first running tape.expand()."""
        dev = qml.device("default.qubit", wires=2)

        with QuantumTape() as tape:
            qml.expval(qml.PauliX(0) @ qml.PauliZ(1))
            qml.expval(qml.PauliX(0))

        with pytest.raises(qml.QuantumFunctionError, match="Multiple observables are being"):
            tape.execute(dev)

        new_tape = tape.expand()
        new_tape.execute(dev)


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

        res = tape.execute(dev)
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

        res = tape.execute(dev)
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

        # The underlying operation data has also been copied
        assert copied_tape.operations[0].wires is not tape.operations[0].wires

        # however, the underlying operation *parameters* are still shared
        # to support PyTorch, which does not support deep copying of tensors
        assert copied_tape.operations[0].data[0] is tape.operations[0].data[0]

    def test_casting(self):
        """Test that copying and casting works as expected"""
        with QuantumTape() as tape:
            qml.BasisState(np.array([1, 0]), wires=[0, 1])
            qml.RY(0.5, wires=[1])
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(0) @ qml.PauliY(1))

        # copy and cast to a JacobianTape
        copied_tape = tape.copy(tape_cls=qml.tape.tapes.JacobianTape)

        # check that the copying worked
        assert copied_tape is not tape
        assert copied_tape.operations == tape.operations
        assert copied_tape.observables == tape.observables
        assert copied_tape.measurements == tape.measurements
        assert copied_tape.operations[0] is tape.operations[0]

        # check that the casting worked
        assert isinstance(copied_tape, qml.tape.tapes.JacobianTape)
        assert not isinstance(tape, qml.tape.tapes.JacobianTape)
