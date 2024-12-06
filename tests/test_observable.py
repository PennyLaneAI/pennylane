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
"""
Unit tests for the :mod:`pennylane.plugin.DefaultGaussian` device.
"""
# pylint: disable=protected-access,cell-var-from-loop


import pennylane as qml


def test_pass_positional_wires_to_observable():
    """Tests whether the ability to pass wires as positional argument is retained"""
    dev = qml.device("default.qubit", wires=1)

    obs = qml.Identity(0)

    @qml.qnode(dev)
    def circuit():
        return qml.expval(obs)

    tape = qml.workflow.construct_tape(circuit)()
    assert obs in tape.observables
