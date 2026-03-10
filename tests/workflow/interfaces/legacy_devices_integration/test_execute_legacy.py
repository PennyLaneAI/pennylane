# Copyright 2023 Xanadu Quantum Technologies Inc.

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
Interface independent tests for qml.execute
"""

import pytest
from default_qubit_legacy import DefaultQubitLegacy

import pennylane as qml
from pennylane.exceptions import QuantumFunctionError


def test_old_interface_no_device_jacobian_products():
    """Test that an error is always raised for the old device interface if device jacobian products are requested."""
    dev = DefaultQubitLegacy(wires=2)
    tape = qml.tape.QuantumScript([qml.RX(1.0, wires=0)], [qml.expval(qml.PauliZ(0))])
    with pytest.raises(QuantumFunctionError):
        qml.execute((tape,), dev, device_vjp=True)
