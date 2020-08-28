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

import pennylane as qml
from pennylane.beta.tapes import QuantumTape
from pennylane.beta.queuing import BetaTensor
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
            ops += [qml.CNOT(wires=[0, 'a'])]
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

        assert tape.wires == qml.wires.Wires([0, 'a', 4])
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
            0: {"op": ops[0], "p_idx": 0},
            1: {"op": ops[1], "p_idx": 0},
            2: {"op": ops[1], "p_idx": 1},
            3: {"op": ops[1], "p_idx": 2},
            4: {"op": ops[3], "p_idx": 0},
        }

    def test_qubit_diagonalization(self, make_tape):
        """Test that qubit diagonalization works as expected"""
        tape, ops, obs = make_tape

        obs_rotations = [o.diagonalizing_gates() for o in obs]
        obs_rotations = [item for sublist in obs_rotations for item in sublist]

        for o1, o2 in zip(tape.diagonalizing_gates, obs_rotations):
            assert isinstance(o1, o2.__class__)
            assert o1.wires == o2.wires


class TestParameters:
    """Tests for parameter processing, setting, and manipulation"""

    @pytest.fixture
    def make_tape(self):
        params = [0.432, 0.123, 0.546, 0.32, 0.76]

        with QuantumTape() as tape:
            qml.RX(params[0], wires=0)
            qml.Rot(*params[1:4], wires=0)
            qml.CNOT(wires=[0, 'a'])
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
            assert pinfo['op'].data[pinfo['p_idx']] == pval

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
                assert pinfo['op'].data[pinfo['p_idx']] == new_params[count]
                count += 1
            else:
                assert pinfo['op'].data[pinfo['p_idx']] == params[idx]

        assert tape.get_parameters(free_only=False) == [
            params[0], new_params[0], params[2], new_params[1], params[4]
        ]

    def test_setting_all_parameters(self, make_tape):
        """Test that all parameters are correctly modified after construction"""
        tape, params = make_tape
        new_params = [0.6543, -0.654, 0, 0.3, 0.6]

        tape.trainable_params = {1, 3}
        tape.set_parameters(new_params, free_only=False)

        for pinfo, pval in zip(tape._par_info.values(), new_params):
            assert pinfo['op'].data[pinfo['p_idx']] == pval

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
