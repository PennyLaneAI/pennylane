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

import pennylane as qml
from pennylane.qnode import QuantumFunctionError
from pennylane.plugins import DefaultQubit

import pytest




def test_pass_positional_wires_to_observable(monkeypatch, capfd):
    """Tests whether the ability to pass wires as positional argument is retained"""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit():
        return qml.expval(qml.Identity(0))

    with monkeypatch.context() as m:
        m.setattr(DefaultQubit, "pre_measure", lambda self: print(self.obs_queue))
        circuit()

    out, err = capfd.readouterr()
    assert "pennylane.ops.Identity object" in out
