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
import pytest
import numpy as np

import pennylane as qml
from pennylane.beta.tapes import QuantumTape
from pennylane.beta.queuing import expval, var, sample, probs, MeasurementProcess


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
            expval(obs[0])
            obs += [probs(wires=[0, "a"])]

        return tape, ops, obs

    def test_qubit_queuing(self, make_tape):
        """Test that qubit quantum operations correctly queue"""
        tape, ops, obs = make_tape

        assert len(tape.queue) == 7
        assert tape.operations == ops
        assert tape.observables == obs
        assert tape.output_dim == 5

        assert tape.wires == qml.wires.Wires([0, "a", 4])
        assert tape._output_dim == len(obs[0].wires) + 2 ** len(obs[1].wires)

    def test_observable_processing(self, make_tape):
        """Test that observables are processed correctly"""
        tape, ops, obs = make_tape

        assert isinstance(tape._obs[0][0], MeasurementProcess)
        assert tape._obs[0][0].return_type == qml.operation.Expectation
        assert tape._obs[0][1] == obs[0]

        assert isinstance(tape._obs[1][0], MeasurementProcess)
        assert tape._obs[1][0].return_type == qml.operation.Probability

    def test_parameter_info(self, make_tape):
        """Test that parameter information is correctly extracted"""
        tape, ops, obs = make_tape
        assert tape._trainable_params == set(range(5))
        assert tape._par_info == {
            0: {"op": ops[0], "p_idx": 0, "grad_method": "F"},
            1: {"op": ops[1], "p_idx": 0, "grad_method": "F"},
            2: {"op": ops[1], "p_idx": 1, "grad_method": "F"},
            3: {"op": ops[1], "p_idx": 2, "grad_method": "F"},
            4: {"op": ops[3], "p_idx": 0, "grad_method": "0"},
        }

    def test_qubit_diagonalization(self, make_tape):
        """Test that qubit diagonalization works as expected"""
        tape, ops, obs = make_tape

        obs_rotations = [o.diagonalizing_gates() for o in obs]
        obs_rotations = [item for sublist in obs_rotations for item in sublist]

        for o1, o2 in zip(tape.diagonalizing_gates, obs_rotations):
            assert isinstance(o1, o2.__class__)
            assert o1.wires == o2.wires

    def test_tensor_construction(self):
        """Test that tensors are correctly queued"""
        with QuantumTape() as tape:
            A = qml.PauliX(wires=0)
            B = qml.PauliZ(wires=1)
            C = A @ B
            D = expval(C)

        assert len(tape.queue) == 4
        assert not tape.operations
        assert tape._obs == [(D, C)]
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
            expval(obs[0])
            obs += [probs(wires=[0, "a"])]

        assert len(tape.queue) == 5
        assert tape.operations == ops
        assert tape.observables == obs
        assert tape.output_dim == 5

        assert a not in tape.operations
        assert b not in tape.operations

        assert tape.wires == qml.wires.Wires([0, "a"])


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
            expval(qml.PauliX(wires="a"))
            probs(wires=[0, "a"])

        return tape, params

    def test_parameter_processing(self, make_tape):
        """Test that parameters are correctly counted and processed"""
        tape, params = make_tape
        assert tape.num_params == len(params)
        assert tape.trainable_params == set(range(len(params)))
        assert tape.get_parameters() == params

    def test_set_trainable_params(self, make_tape):
        """Test that changing trainable parameters works as expected"""
        tape, params = make_tape
        trainable = {0, 2, 3}
        tape.trainable_params = trainable
        assert tape._trainable_params == trainable
        assert tape.num_params == 3
        assert tape.get_parameters() == [params[i] for i in tape.trainable_params]
        assert tape.get_parameters(free_only=False) == params

    def test_set_trainable_params_error(self, make_tape):
        """Test that exceptions are raised if incorrect parameters
        are set as trainable"""
        tape, _ = make_tape

        with pytest.raises(ValueError, match="must be positive integers"):
            tape.trainable_params = {-1, 0}

        with pytest.raises(ValueError, match="must be positive integers"):
            tape.trainable_params = {0.5}

    def test_setting_parameters(self, make_tape):
        """Test that parameters are correctly modified after construction"""
        tape, params = make_tape
        new_params = [0.6543, -0.654, 0, 0.3, 0.6]

        tape.set_parameters(new_params)

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

        assert tape.get_parameters(free_only=False) == [
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
        tape.set_parameters(new_params, free_only=False)

        for pinfo, pval in zip(tape._par_info.values(), new_params):
            assert pinfo["op"].data[pinfo["p_idx"]] == pval

        assert tape.get_parameters(free_only=False) == new_params

    def test_setting_parameters_error(self, make_tape):
        """Test that exceptions are raised if incorrect parameters
        are attempted to be set"""
        tape, _ = make_tape

        with pytest.raises(ValueError, match="Number of provided parameters invalid"):
            tape.set_parameters([0.54])

        with pytest.raises(ValueError, match="Number of provided parameters invalid"):
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
            expval(qml.PauliZ(0) @ qml.PauliX(1))

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

    def test_incorrect_output_dim_estimate(self):
        """Test that a quantum tape with an incorrect output dimension
        estimate corrects itself after evaluation."""
        dev = qml.device("default.qubit", wires=3)
        params = [1.0, 1.0, 1.0]

        with QuantumTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.RZ(params[2], wires=[2])
            qml.CNOT(wires=[0, 1])
            probs(wires=0)
            probs(wires=[1])

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
            probs(wires=0)
            probs(wires=[1, 2])

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
            expval(qml.PauliZ(0) @ qml.PauliX(1))

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
            expval(qml.PauliZ(0))
            expval(qml.PauliX(1))

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
            expval(qml.PauliZ(0))
            var(qml.PauliX(1))

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
            expval(qml.PauliZ(0))
            probs(wires=[0, 1])

        assert tape.output_dim == 5

        res = tape.execute(dev)

        assert isinstance(res[0], float)
        assert np.allclose(res[0], np.cos(x), atol=tol, rtol=0)

        assert isinstance(res[1], np.ndarray)
        assert np.allclose(res[1], np.abs(dev.state) ** 2, atol=tol, rtol=0)

    def test_single_mode_sample(self):
        """Test that there is only one array of values returned
        for single mode samples"""
        dev = qml.device("default.qubit", wires=2, shots=10)
        x = 0.543
        y = -0.654

        with QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            sample(qml.PauliZ(0) @ qml.PauliX(1))

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
            sample(qml.PauliZ(0))
            sample(qml.PauliZ(1))

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
            sample(qml.PauliZ(0))
            expval(qml.PauliZ(1))

        res = tape.execute(dev)
        assert res[0].shape == (10,)
        assert isinstance(res[1], float)


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
            expval(qml.NumberOperator(0))

        assert tape.output_dim == 1

        res = tape.execute(dev)
        assert res.shape == (1,)

    def test_multiple_output_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple expval outputs"""
        dev = qml.device("default.gaussian", wires=2)
        x = 0.543
        y = -0.654

        with QuantumTape() as tape:
            qml.Displacement(x, 0, wires=[0])
            qml.Squeezing(y, 0, wires=[1])
            qml.Beamsplitter(np.pi / 4, 0, wires=[0, 1])
            expval(qml.PolyXP(np.diag([0, 1, 0]), wires=0))  # X^2
            var(qml.P(1))

        assert tape.output_dim == 2

        res = tape.execute(dev)
        assert res.shape == (2,)
