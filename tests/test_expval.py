# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.plugin.DefaultGaussian` device.
"""
# pylint: disable=protected-access,cell-var-from-loop
from pennylane import numpy as np
from scipy.linalg import block_diag

from defaults import pennylane as qml
from pennylane.ops import Identity
from pennylane.qnode import QuantumFunctionError
from pennylane.plugins import DefaultQubit

import pytest


def test_identity_raises_exception_if_outside_qnode():
    """expval: Tests that proper exceptions are raised if we try to call
    Idenity outside a QNode."""
    with pytest.raises(QuantumFunctionError, match="can only be used inside a qfunc"):
        Identity(wires=0)


# def test_identity_raises_exception_if_cannot_guess_device_type():
#     """expval: Tests that proper exceptions are raised if Identity fails to guess
#     whether on a device is CV or qubit."""
#     dev = qml.device("default.qubit", wires=1)
#     dev._expectation_map = {}

#     @qml.qnode(dev)
#     def circuit():
#         return qml.expval.Identity(wires=0)

#     with pytest.raises(
#         QuantumFunctionError,
#         match="Unable to determine whether this device supports CV or qubit",
#     ):
#         circuit()


def test_pass_positional_wires_to_expval(monkeypatch, capfd):
    """Tests whether the ability to pass wires as positional argument is retained"""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():
        return qml.expval.Identity(wires=0)

    with monkeypatch.context() as m:
        m.setattr(DefaultQubit, "pre_expval", lambda self: print(self.expval_queue))
        circuit()

    out, err = capfd.readouterr()
    assert "abc.Identity object" in out
