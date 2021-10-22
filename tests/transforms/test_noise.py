# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Tests for the noise-adding transforms.
"""
import numpy as np
import pytest
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.operation import Expectation

from pennylane.transforms.noise import add_noise_to_tape


class TestAddNoiseToTape:
    """Tests for the add_noise_to_tape function"""

    with QuantumTape() as tape:
        qml.RX(0.9, wires=0)
        qml.RY(0.4, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RY(0.5, wires=0)
        qml.RX(0.6, wires=1)
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    def test_multiwire_noisy_op(self):
        """Tests if a ValueError is raised when multiqubit channels are requested"""
        with pytest.raises(ValueError, match="Adding noise to the tape is only"):
            add_noise_to_tape(self.tape, qml.QubitChannel, [])

    def test_invalid_position(self):
        """Test if a ValueError is raised when an invalid position is requested"""
        with pytest.raises(ValueError, match="Position must be either 'start', 'end', or 'all'"):
            add_noise_to_tape(self.tape, qml.AmplitudeDamping, 0.4, position="ABC")

    def test_not_noisy(self):
        """Test if a ValueError is raised when something that is not a noisy channel is fed to the
        noisy_op argument"""
        with pytest.raises(ValueError, match="The noisy_op argument must be a noisy operation"):
            add_noise_to_tape(self.tape, qml.PauliX, 0.4)

    def test_start(self):
        """Test if the expected tape is returned when the start position is requested"""
        tape = add_noise_to_tape(self.tape, qml.AmplitudeDamping, 0.4, position="start")

        with QuantumTape() as tape_exp:
            qml.AmplitudeDamping(0.4, wires=0)
            qml.AmplitudeDamping(0.4, wires=1)
            qml.RX(0.9, wires=0)
            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.RX(0.6, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(np.allclose(o1.parameters, o2.parameters) for o1, o2 in
                   zip(tape.operations, tape_exp.operations))
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == ["PauliZ", "PauliZ"]
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert tape.measurements[0].return_type is Expectation

    def test_all(self):
        """Test if the expected tape is returned when the all position is requested"""
        tape = add_noise_to_tape(self.tape, qml.PhaseDamping, 0.4, position="all")

        with QuantumTape() as tape_exp:
            qml.RX(0.9, wires=0)
            qml.PhaseDamping(0.4, wires=0)
            qml.RY(0.4, wires=1)
            qml.PhaseDamping(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.PhaseDamping(0.4, wires=0)
            qml.PhaseDamping(0.4, wires=1)
            qml.RY(0.5, wires=0)
            qml.PhaseDamping(0.4, wires=0)
            qml.RX(0.6, wires=1)
            qml.PhaseDamping(0.4, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(np.allclose(o1.parameters, o2.parameters) for o1, o2 in
                   zip(tape.operations, tape_exp.operations))
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == ["PauliZ", "PauliZ"]
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert tape.measurements[0].return_type is Expectation

    def test_end(self):
        """Test if the expected tape is returned when the end position is requested"""
        tape = add_noise_to_tape(self.tape, qml.GeneralizedAmplitudeDamping, [0.4, 0.5], position="end")

        with QuantumTape() as tape_exp:
            qml.RX(0.9, wires=0)
            qml.RY(0.4, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(0.5, wires=0)
            qml.RX(0.6, wires=1)
            qml.GeneralizedAmplitudeDamping(0.4, 0.5, wires=0)
            qml.GeneralizedAmplitudeDamping(0.4, 0.5, wires=1)
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        assert all(o1.name == o2.name for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(o1.wires == o2.wires for o1, o2 in zip(tape.operations, tape_exp.operations))
        assert all(np.allclose(o1.parameters, o2.parameters) for o1, o2 in
                   zip(tape.operations, tape_exp.operations))
        assert len(tape.measurements) == 1
        assert tape.observables[0].name == ["PauliZ", "PauliZ"]
        assert tape.observables[0].wires.tolist() == [0, 1]
        assert tape.measurements[0].return_type is Expectation
