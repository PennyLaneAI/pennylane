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

    def test_tensor_process_queueion(self):
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

        with pytest.raises(ValueError, match="has at most 5 trainable parameters"):
            tape.trainable_params = {0, 7}

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


class TestGradMethod:
    """Tests for parameter gradient methods"""

    def test_non_differentiable(self):
        """Test that a non-differentiable parameter is
        correctly marked"""
        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with QuantumTape() as tape:
            qml.QubitStateVector(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            probs(wires=[0, 1])

        assert tape._grad_method(0) is None
        assert tape._grad_method(1) == "F"
        assert tape._grad_method(2) == "F"

        assert tape._par_info[0]["grad_method"] is None
        assert tape._par_info[1]["grad_method"] == "F"
        assert tape._par_info[2]["grad_method"] == "F"

    def test_independent(self):
        """Test that an independent variable is properly marked
        as having a zero gradient"""

        with QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            expval(qml.PauliY(0))

        assert tape._grad_method(0) == "F"
        assert tape._grad_method(1) == "0"

        assert tape._par_info[0]["grad_method"] == "F"
        assert tape._par_info[1]["grad_method"] == "0"

        # in non-graph mode, it is impossible to determine
        # if a parameter is independent or not
        tape._graph = None
        assert tape._grad_method(1, use_graph=False) == "F"


class TestJacobian:
    """Unit tests for the jacobian method"""

    def test_unknown_grad_method_error(self):
        """Test error raised if gradient method is unknown"""
        tape = QuantumTape()
        with pytest.raises(ValueError, match="Unknown gradient method"):
            tape.jacobian(None, method="unknown method")

    def test_non_differentiable_error(self):
        """Test error raised if attempting to differentiate with
        respect to a non-differentiable argument"""
        psi = np.array([1, 0, 1, 0]) / np.sqrt(2)

        with QuantumTape() as tape:
            qml.QubitStateVector(psi, wires=[0, 1])
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            qml.CNOT(wires=[0, 1])
            probs(wires=[0, 1])

        # by default all parameters are assumed to be trainable
        with pytest.raises(
            ValueError, match=r"Cannot differentiate with respect to parameter\(s\) {0}"
        ):
            tape.jacobian(None)

        # setting trainable parameters avoids this
        tape.trainable_params = {1, 2}
        dev = qml.device("default.qubit", wires=2)
        res = tape.jacobian(dev)
        assert res.shape == (4, 2)

    def test_analytic_method_with_unsupported_params(self):
        """Test that an exception is raised if method="A" but a parameter
        only support finite differences"""
        with QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            expval(qml.PauliY(0))

        dev = qml.device("default.qubit", wires=1)

        with pytest.raises(ValueError, match=r"analytic gradient method cannot be used"):
            tape.jacobian(dev, method="analytic")

    def test_analytic_method(self, mocker):
        """Test that calling the Jacobian with method=analytic correctly
        calls the analytic_pd method"""
        mock = mocker.patch("pennylane.beta.tapes.QuantumTape._grad_method")
        mock.return_value = "A"

        with QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            expval(qml.PauliY(0))

        dev = qml.device("default.qubit", wires=1)
        tape.analytic_pd = mocker.Mock()
        tape.analytic_pd.return_value = np.array([1.])

        tape.jacobian(dev, method="analytic")
        assert len(tape.analytic_pd.call_args_list) == 2

    def test_device_method(self, mocker):
        """Test that calling the Jacobian with method=device correctly
        calls the device_pd method"""
        with QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            expval(qml.PauliY(0))

        dev = qml.device("default.qubit", wires=1)
        dev.jacobian = mocker.Mock()

        tape.jacobian(dev, method="device")
        dev.jacobian.assert_called_once()

    def test_no_output_execute(self):
        """Test that tapes with no measurement process return
        an empty list."""
        dev = qml.device("default.qubit", wires=2)
        params = [0.1, 0.2]

        with QuantumTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])

        res = tape.jacobian(dev)
        assert res.size == 0

    def test_incorrect_output_dim_estimate(self):
        """Test that a quantum tape with an incorrect output dimension
        estimate raises an exception when computing the Jacobian."""
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

        with pytest.raises(ValueError, match=r"could not infer the correct output dimension"):
            # Note that we specify order=2 here. If we use first order differentiation,
            # the tape is able to correctly infer the correct output dimension
            # before the Jacobian is computed.
            tape.jacobian(dev, order=2)

    def test_incorrect_ragged_output_dim_estimate(self, mocker):
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
        with pytest.raises(ValueError, match=r"could not infer the correct output dimension"):
            res = tape.jacobian(dev, order=2)

    def test_independent_parameter(self, mocker):
        """Test that an independent parameter is skipped
        during the Jacobian computation."""
        numeric_spy = mocker.spy(QuantumTape, "numeric_pd")
        analytic_spy = mocker.spy(QuantumTape, "analytic_pd")

        with QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        res = tape.jacobian(dev)
        assert res.shape == (1, 2)

        # the numeric pd method is only called once
        assert len(numeric_spy.call_args_list) == 1

        # analytic pd should not be called at all
        assert len(analytic_spy.call_args_list) == 0

        # the numeric pd method is only called for parameter 0
        assert numeric_spy.call_args[0] == (tape, (0,), dev)

    def test_no_trainable_parameters(self, mocker):
        """Test that if the tape has no trainable parameters, no
        subroutines are called and the returned Jacobian is empty"""
        numeric_spy = mocker.spy(QuantumTape, "numeric_pd")
        analytic_spy = mocker.spy(QuantumTape, "analytic_pd")

        with QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[1])
            expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        tape.trainable_params = {}

        res = tape.jacobian(dev)
        assert res.size == 0
        assert np.all(res == np.array([[]]))

        numeric_spy.assert_not_called()
        analytic_spy.assert_not_called()

    def test_y0(self, mocker):
        """Test that if first order finite differences is used, then
        the tape is executed only once using the current parameter
        values."""
        execute_spy = mocker.spy(QuantumTape, "execute_device")
        numeric_spy = mocker.spy(QuantumTape, "numeric_pd")

        with QuantumTape() as tape:
            qml.RX(0.543, wires=[0])
            qml.RY(-0.654, wires=[0])
            expval(qml.PauliZ(0))

        dev = qml.device("default.qubit", wires=2)
        res = tape.jacobian(dev, order=1)

        # the execute device method is called once per parameter,
        # plus one global call
        assert len(execute_spy.call_args_list) == tape.num_params + 1
        assert "y0" in numeric_spy.call_args_list[0][1]
        assert "y0" in numeric_spy.call_args_list[1][1]

    def test_parameters(self, tol):
        """Test Jacobian computation works when parameters are both passed and not passed."""
        dev = qml.device("default.qubit", wires=2)
        params = [0.1, 0.2]

        with QuantumTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            expval(qml.PauliZ(0) @ qml.PauliX(1))

        # test Jacobian with no parameters
        res1 = tape.jacobian(dev)
        assert tape.get_parameters() == params

        # test Jacobian with parameters
        res2 = tape.jacobian(dev, params=[0.5, 0.6])
        assert tape.get_parameters() == params

        # test setting parameters
        tape.set_parameters(params=[0.5, 0.6])
        res3 = tape.jacobian(dev)
        assert np.allclose(res2, res3, atol=tol, rtol=0)
        assert not np.allclose(res1, res2, atol=tol, rtol=0)
        assert tape.get_parameters() == [0.5, 0.6]

    def test_numeric_pd_no_y0(self, mocker, tol):
        """Test that, if y0 is not passed when calling the numeric_pd method,
        y0 is calculated."""
        execute_spy = mocker.spy(QuantumTape, "execute_device")

        dev = qml.device("default.qubit", wires=2)
        params = [0.1, 0.2]

        with QuantumTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.CNOT(wires=[0, 1])
            expval(qml.PauliZ(0) @ qml.PauliX(1))

        # compute numeric gradient of parameter 0, without passing y0
        res1 = tape.numeric_pd(0, dev)
        assert len(execute_spy.call_args_list) == 2

        # compute y0 in advance
        y0 = tape.execute(dev)
        execute_spy.call_args_list = []
        res2 = tape.numeric_pd(0, dev, y0=y0)
        assert len(execute_spy.call_args_list) == 1
        assert np.allclose(res1, res2, atol=tol, rtol=0)

    def test_numeric_unknown_order(self):
        """Test that an exception is raised if the finite-difference
        order is not supported"""
        dev = qml.device("default.qubit", wires=2)
        params = [0.1, 0.2]

        with QuantumTape() as tape:
            qml.RX(1, wires=[0])
            qml.RY(1, wires=[1])
            qml.RZ(1, wires=[2])
            qml.CNOT(wires=[0, 1])
            expval(qml.operation.Tensor(qml.PauliZ(0) @ qml.PauliX(1), qml.PauliZ(2)))

        with pytest.raises(ValueError, match="Order must be 1 or 2"):
            tape.jacobian(dev, order=3)


class TestJacobianIntegration:
    """Integration tests for the Jacobian method"""

    def test_ragged_output(self):
        """Test that the Jacobian is correctly returned for a tape
        with ragged output"""
        dev = qml.device("default.qubit", wires=3)
        params = [1.0, 1.0, 1.0]

        with QuantumTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.RZ(params[2], wires=[2])
            qml.CNOT(wires=[0, 1])
            probs(wires=0)
            probs(wires=[1, 2])

        res = tape.jacobian(dev)
        assert res.shape == (6, 3)

    def test_ragged_output(self):
        """Test that the Jacobian is correctly returned for a tape
        with ragged output"""
        dev = qml.device("default.qubit", wires=3)
        params = [1.0, 1.0, 1.0]

        with QuantumTape() as tape:
            qml.RX(params[0], wires=[0])
            qml.RY(params[1], wires=[1])
            qml.RZ(params[2], wires=[2])
            qml.CNOT(wires=[0, 1])
            probs(wires=0)
            probs(wires=[1, 2])

        res = tape.jacobian(dev)
        assert res.shape == (6, 3)

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

        res = tape.jacobian(dev)
        assert res.shape == (1, 2)

        expected = np.array([[-np.sin(y) * np.sin(x), np.cos(y) * np.cos(x)]])
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

        res = tape.jacobian(dev)
        assert res.shape == (2, 2)

        expected = np.array([[-np.sin(x), 0], [0, np.cos(y)]])
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

        res = tape.jacobian(dev)
        assert res.shape == (2, 2)

        expected = np.array([[-np.sin(x), 0], [0, -2 * np.cos(y) * np.sin(y)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_prob_expectation_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with prob and expval outputs"""
        dev = qml.device("default.qubit", wires=2)
        x = 0.543
        y = -0.654

        with QuantumTape() as tape:
            qml.RX(x, wires=[0])
            qml.RY(y, wires=[1])
            qml.CNOT(wires=[0, 1])
            expval(qml.PauliZ(0))
            probs(wires=[0, 1])

        res = tape.jacobian(dev)
        assert res.shape == (5, 2)

        expected = (
            np.array(
                [
                    [-2 * np.sin(x), 0],
                    [
                        -(np.cos(y / 2) ** 2 * np.sin(x)),
                        -(np.cos(x / 2) ** 2 * np.sin(y)),
                    ],
                    [
                        -(np.sin(x) * np.sin(y / 2) ** 2),
                        (np.cos(x / 2) ** 2 * np.sin(y)),
                    ],
                    [
                        (np.sin(x) * np.sin(y / 2) ** 2),
                        (np.sin(x / 2) ** 2 * np.sin(y)),
                    ],
                    [
                        (np.cos(y / 2) ** 2 * np.sin(x)),
                        -(np.sin(x / 2) ** 2 * np.sin(y)),
                    ],
                ]
            )
            / 2
        )

        assert np.allclose(res, expected, atol=tol, rtol=0)


class TestJacobianCVIntegration:
    """Intgration tests for the Jacobian method and CV circuits"""

    def test_single_output_value(self, tol):
        """Tests correct Jacobian and output shape for a CV tape
        with a single output"""
        dev = qml.device("default.gaussian", wires=2)
        n = 0.543
        a = -0.654

        with QuantumTape() as tape:
            qml.ThermalState(n, wires=0)
            qml.Displacement(a, 0, wires=0)
            var(qml.NumberOperator(0))

        tape.trainable_params = {0, 1}
        res = tape.jacobian(dev)
        assert res.shape == (1, 2)

        expected = np.array([2 * a ** 2 + 2 * n + 1, 2 * a * (2 * n + 1)])
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_multiple_output_values(self, tol):
        """Tests correct output shape and evaluation for a tape
        with multiple outputs"""
        dev = qml.device("default.gaussian", wires=2)
        n = 0.543
        a = -0.654

        with QuantumTape() as tape:
            qml.ThermalState(n, wires=0)
            qml.Displacement(a, 0, wires=0)
            expval(qml.NumberOperator(0))
            var(qml.NumberOperator(0))

        tape.trainable_params = {0, 1}
        res = tape.jacobian(dev)
        assert res.shape == (2, 2)

        expected = np.array([[1, 2 * a], [2 * a ** 2 + 2 * n + 1, 2 * a * (2 * n + 1)]])
        assert np.allclose(res, expected, atol=tol, rtol=0)
