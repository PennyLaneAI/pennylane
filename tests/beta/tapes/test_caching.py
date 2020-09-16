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
"""Tests for caching executions of the quantum tape and QNode."""
import numpy as np

import pennylane as qml
from pennylane.beta.queuing import expval
from pennylane.beta.tapes import QuantumTape
from pennylane.devices import DefaultQubit
from pennylane.utils import _hash_iterable


def get_tape(caching):
    """Creates a simple quantum tape"""
    with QuantumTape(caching=caching) as tape:
        qml.QubitUnitary(np.eye(2), wires=0)
        qml.RX(0.1, wires=0)
        qml.RX(0.2, wires=1)
        qml.CNOT(wires=[0, 1])
        expval(qml.PauliZ(wires=1))
    return tape


class TestTapeCaching:
    """Tests for caching when using quantum tape"""

    def test_set_and_get(self):
        """Test that the caching attribute can be set and accessed"""
        tape = QuantumTape()
        assert tape.caching == 0

        tape = QuantumTape(caching=10)
        assert tape.caching == 10

        tape.caching = 20
        assert tape.caching == 20

    def test_no_caching(self, mocker):
        """Test that no caching occurs when the caching attribute is equal to zero"""
        dev = qml.device("default.qubit", wires=2)
        tape = get_tape(0)

        spy = mocker.spy(DefaultQubit, "execute")
        tape.execute(device=dev)
        tape.execute(device=dev)

        assert len(spy.call_args_list) == 2
        assert len(tape._cache_execute) == 0

    def test_caching(self, mocker):
        """Test that caching occurs when the caching attribute is above zero"""
        dev = qml.device("default.qubit", wires=2)
        tape = get_tape(10)

        tape.execute(device=dev)
        spy = mocker.spy(DefaultQubit, "execute")
        tape.execute(device=dev)

        spy.assert_not_called()
        assert len(tape._cache_execute) == 1

    def test_add_to_cache_execute(self):
        """Test that the _cache_execute attribute is added to when the tape is executed"""
        dev = qml.device("default.qubit", wires=2)
        tape = get_tape(10)

        result = tape.execute(device=dev)
        cache_execute = tape._cache_execute
        params = tape.get_parameters()
        hashed = _hash_iterable(params)

        assert len(cache_execute) == 1
        assert hashed in cache_execute
        assert np.allclose(cache_execute[hashed], result)

    def test_something(self):
        dev = qml.device("default.qubit", wires=2)
        tape = get_tape(10)

        tape.trainable_parameters = {1}

