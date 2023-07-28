# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the the logging module"""
# pylint: disable=import-outside-toplevel, protected-access, no-member
import pytest
import logging

import pennylane as qml
import pennylane.logging as pl_logging


# pylint: disable=too-many-public-methods
@pytest.mark.logging
class TestLogging:
    """Tests for logging integration"""

    pl_logging.enable_logging()

    def test_basic_functionality_dq(self, caplog):
        "Test logging of QNode forward pass"
        dev = qml.device("default.qubit", wires=2)

        # Single log entry, QNode creation
        with caplog.at_level(logging.DEBUG):

            @qml.qnode(dev, diff_method=None)
            def circuit():
                qml.PauliX(wires=0)
                return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))

            assert len(caplog.records) == 1
            assert "Creating QNode" in caplog.text

        # Multiple log-record entries to be recorded mapping calling module to recorded message in given order:
        record_index_map = [
            (
                "pennylane.qnode",
                "Creating QNode(func=<function TestLogging.test_basic_functionality",
            ),
            (
                "pennylane.interfaces.execution",
                "Entry with args=(tapes=[<QuantumScript: wires=[0], params=0>], device=default.qubit, gradient_fn=None, interface=None",
            ),
            (
                "pennylane.interfaces.execution",
                "Entry with args=(fn=<function QubitDevice.batch_execute at ",
            ),
            (
                "pennylane._qubit_device",
                "Entry with args=(circuits=[<QuantumScript: wires=[0], params=0>]) called by",
            ),
            (
                "pennylane._qubit_device",
                "Entry with args=(circuit=<QuantumScript: wires=[0], params=0>, kwargs={}) called by",
            ),
        ]
        with caplog.at_level(logging.DEBUG):
            circuit()
        assert len(caplog.records) == 5
        for idx, r in enumerate(caplog.records):
            assert record_index_map[idx][0] in r.name
            assert record_index_map[idx][1] in r.getMessage()

    def test_basic_functionality_dq_backprop(self, caplog):
        "Test logging of QNode init and parameter-shift gradients"
        dev = qml.device("default.qubit", wires=2)
        params = qml.numpy.array(0.1234)

        # Single log entry, QNode creation
        with caplog.at_level(logging.DEBUG):

            @qml.qnode(dev, diff_method="backprop")
            def circuit(params):
                qml.RX(params, wires=0)
                return qml.expval(qml.PauliZ(0))

            assert len(caplog.records) == 1
            assert "Creating QNode" in caplog.text

        # Multiple log-record entries to be recorded mapping calling module to recorded message in given order:
        record_index_map = [
            (
                "pennylane.qnode",
                "Creating QNode(func=<function TestLogging.test_basic_functionality",
            ),
            (
                "pennylane.interfaces.execution",
                "Entry with args=(tapes=[<QuantumScript: wires=[0], params=1>], device=default.qubit.autograd, gradient_fn=backprop",
            ),
            (
                "pennylane.interfaces.execution",
                "Entry with args=(fn=<function QubitDevice.batch_execute at",
            ),
            (
                "pennylane._qubit_device",
                "Entry with args=(circuits=[<QuantumScript: wires=[0], params=1>]) called by",
            ),
            (
                "pennylane._qubit_device",
                "Entry with args=(circuit=<QuantumScript: wires=[0], params=1>, kwargs={}) called by",
            ),
        ]
        with caplog.at_level(logging.DEBUG):
            qml.grad(circuit)(params)
        assert len(caplog.records) == 5
        for idx, r in enumerate(caplog.records):
            assert record_index_map[idx][0] in r.name
            assert record_index_map[idx][1] in r.getMessage()

    def test_basic_functionality_dq_ps(self, caplog):
        "Test logging of QNode init and parameter-shift gradients"
        dev = qml.device("default.qubit", wires=2)
        params = qml.numpy.array(0.1234)

        # Single log entry, QNode creation
        with caplog.at_level(logging.DEBUG):

            @qml.qnode(dev, diff_method="parameter-shift")
            def circuit(params):
                qml.RX(params, wires=0)
                return qml.expval(qml.PauliZ(0))

            assert len(caplog.records) == 1
            assert "Creating QNode" in caplog.text

        # Multiple log-record entries to be recorded mapping calling module to recorded message in given order:
        record_index_map = [
            (
                "pennylane.qnode",
                "Creating QNode(func=<function TestLogging.test_basic_functionality_dq_ps",
            ),
            (
                "pennylane.interfaces.execution",
                "Entry with args=(tapes=[<QuantumScript: wires=[0], params=1>], device=default.qubit, gradient_fn=<pennylane.gradients.gradient_transform.gradient_transform object at",
            ),
            (
                "pennylane.interfaces.execution",
                "Entry with args=(fn=<function QubitDevice.batch_execute at ",
            ),
            (
                "pennylane.interfaces.autograd",
                "Entry with args=(parameters=([tensor(0.1234, requires_grad=True)],), tapes=[<QuantumScript: wires=[0], params=1>], device=default.qubit, execute_fn=<function cache_execute",
            ),
            (
                "pennylane._qubit_device",
                "Entry with args=(circuits=[<QuantumScript: wires=[0], params=1>]) called by",
            ),
            (
                "pennylane._qubit_device",
                "Entry with args=(circuit=<QuantumScript: wires=[0], params=1>, kwargs={}) called by",
            ),
            (
                "pennylane.interfaces.autograd",
                "Entry with args=(ans=([array(0.99239588)], []), parameters=([tensor(0.1234, requires_grad=True)],), tapes=[<QuantumScript: wires=[0], params=1>], device=default.qubit, execute_fn=<function cache_execute",
            ),
            ("pennylane.interfaces.autograd", "Entry with args=(dy=([array(1.)], [])) called by"),
            (
                "pennylane._qubit_device",
                "Entry with args=(circuits=[<QuantumScript: wires=[0], params=1>, <QuantumScript: wires=[0], params=1>]) called by",
            ),
            (
                "pennylane._qubit_device",
                "Entry with args=(circuit=<QuantumScript: wires=[0], params=1>, kwargs={}) called by",
            ),
            (
                "pennylane._qubit_device",
                "Entry with args=(circuit=<QuantumScript: wires=[0], params=1>, kwargs={}) called by",
            ),
            (
                "pennylane.interfaces.autograd",
                "Entry with args=(jacs=[array(-0.12308706)], dy=[array(1.)], multi_measurements=[False], shots=None) called by",
            ),
        ]
        with caplog.at_level(logging.DEBUG):
            qml.grad(circuit)(params)
        assert len(caplog.records) == 12
        for idx, r in enumerate(caplog.records):
            assert record_index_map[idx][0] in r.name
            assert record_index_map[idx][1] in r.getMessage()
