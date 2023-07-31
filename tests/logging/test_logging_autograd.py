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
import logging
import pytest
import re

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
            {
                "log_origin": "pennylane.qnode",
                "log_body": [
                    "Creating QNode(func=<function TestLogging.test_basic_functionality",
                ],
            },
            {
                "log_origin": "pennylane.interfaces.execution",
                "log_body": [
                    "device=<DefaultQubit device (wires=2, shots=None) at",
                    "gradient_fn=None, interface=None, grad_on_execution=best, gradient_kwargs={}, cache=True, cachesize=10000, max_diff=1, override_shots=False, expand_fn=device, max_expansion=10, device_batch_transform=True",
                ],
            },
            {
                "log_origin": "pennylane.interfaces.execution",
                "log_body": [
                    "Entry with args=(fn=<function QubitDevice.batch_execute at ",
                ],
            },
            {
                "log_origin": "pennylane._qubit_device",
                "log_body": [
                    "Entry with args=(circuits=[<QuantumScript: wires=[0], params=0>]) called by",
                ],
            },
            {
                "log_origin": "pennylane._qubit_device",
                "log_body": [
                    "Entry with args=(circuit=<QuantumScript: wires=[0], params=0>, kwargs={}) called by",
                ],
            },
        ]
        with caplog.at_level(logging.DEBUG):
            circuit()
        assert len(caplog.records) == 5
        for idx, r in enumerate(caplog.records):
            assert record_index_map[idx]["log_origin"] in r.name
            for msg in record_index_map[idx]["log_body"]:
                assert msg in r.getMessage()

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
            {
                "log_origin": "pennylane.qnode",
                "log_body": [
                    "Creating QNode(func=<function TestLogging.test_basic_functionality_dq_backprop",
                    "device=<DefaultQubit device (wires=2, shots=None)",
                    "interface=auto, diff_method=backprop, expansion_strategy=gradient, max_expansion=10, grad_on_execution=best, mode=None, cache=True, cachesize=10000, max_diff=1, gradient_kwargs={}",
                ],
            },
            {
                "log_origin": "pennylane.interfaces.execution",
                "log_body": [
                    "Entry with args=(tapes=[<QuantumScript: wires=[0], params=1>]",
                    "device=<DefaultQubitAutograd device (wires=2, shots=None) at",
                    "gradient_fn=backprop, interface=autograd, grad_on_execution=best, gradient_kwargs={}, cache=True, cachesize=10000, max_diff=1, override_shots=False, expand_fn=device, max_expansion=10, device_batch_transform=True) called by",
                ],
            },
            {
                "log_origin": "pennylane.interfaces.execution",
                "log_body": [
                    "Entry with args=(fn=<function QubitDevice.batch_execute at",
                ],
            },
            {
                "log_origin": "pennylane._qubit_device",
                "log_body": [
                    "Entry with args=(circuits=[<QuantumScript: wires=[0], params=1>]) called by",
                ],
            },
            {
                "log_origin": "pennylane._qubit_device",
                "log_body": [
                    "Entry with args=(circuit=<QuantumScript: wires=[0], params=1>, kwargs={}) called by",
                ],
            },
        ]
        with caplog.at_level(logging.DEBUG):
            qml.grad(circuit)(params)
        assert len(caplog.records) == 5
        for idx, r in enumerate(caplog.records):
            assert record_index_map[idx]["log_origin"] in r.name
            for msg in record_index_map[idx]["log_body"]:
                assert msg in r.getMessage()

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
            {
                "log_origin": "pennylane.qnode",
                "log_body": [
                    "Creating QNode(func=<function TestLogging.test_basic_functionality_dq_ps",
                ],
            },
            {
                "log_origin": "pennylane.interfaces.execution",
                "log_body": [
                    "Entry with args=(tapes=[<QuantumScript: wires=[0], params=1>]",
                    "device=<DefaultQubit device (wires=2, shots=None)",
                    "gradient_fn=<pennylane.gradients.gradient_transform.gradient_transform object",
                    "interface=autograd, grad_on_execution=best, gradient_kwargs={}, cache=True, cachesize=10000, max_diff=1, override_shots=False, expand_fn=device, max_expansion=10, device_batch_transform=True) called by",
                ],
            },
            {
                "log_origin": "pennylane.interfaces.execution",
                "log_body": [
                    "Entry with args=(fn=<function QubitDevice.batch_execute at ",
                    "pass_kwargs=False, return_tuple=True, expand_fn=<function _preprocess_expand_fn.<locals>.device_expansion_function at",
                ],
            },
            {
                "log_origin": "pennylane.interfaces.autograd",
                "log_body": [
                    "Entry with args=(parameters=([tensor(0.1234, requires_grad=True)],), tapes=[<QuantumScript: wires=[0], params=1>]",
                    "device=<DefaultQubit device (wires=2, shots=None) at",
                    "execute_fn=<function cache_execute.<locals>.fn at",
                    "gradient_fn=<pennylane.gradients.gradient_transform.gradient_transform object at",
                    "gradient_kwargs={}, _n=1, max_diff=1) called by",
                ],
            },
            {
                "log_origin": "pennylane._qubit_device",
                "log_body": [
                    "Entry with args=(circuits=[<QuantumScript: wires=[0], params=1>]) called by",
                ],
            },
            {
                "log_origin": "pennylane._qubit_device",
                "log_body": [
                    "Entry with args=(circuit=<QuantumScript: wires=[0], params=1>, kwargs={}) called by",
                ],
            },
            {
                "log_origin": "pennylane.interfaces.autograd",
                "log_body": [
                    "Entry with args=(ans=([array(0.99239588)], []), parameters=([tensor(0.1234, requires_grad=True)],), tapes=[<QuantumScript: wires=[0], params=1>],",
                    "device=<DefaultQubit device (wires=2, shots=None) at",
                    "execute_fn=<function cache_execute.<locals>.fn at",
                    "gradient_fn=<pennylane.gradients.gradient_transform.gradient_transform object",
                    "gradient_kwargs={}, _n=1, max_diff=1) called by",
                ],
            },
            {
                "log_origin": "pennylane.interfaces.autograd",
                "log_body": [
                    "Entry with args=(dy=([array(1.)], [])) called by",
                ],
            },
            {
                "log_origin": "pennylane._qubit_device",
                "log_body": [
                    "Entry with args=(circuits=[<QuantumScript: wires=[0], params=1>, <QuantumScript: wires=[0], params=1>]) called by",
                ],
            },
            {
                "log_origin": "pennylane._qubit_device",
                "log_body": [
                    "Entry with args=(circuit=<QuantumScript: wires=[0], params=1>, kwargs={}) called by",
                ],
            },
            {
                "log_origin": "pennylane._qubit_device",
                "log_body": [
                    "Entry with args=(circuit=<QuantumScript: wires=[0], params=1>, kwargs={}) called by",
                ],
            },
            {
                "log_origin": "pennylane.interfaces.autograd",
                "log_body": [
                    "Entry with args=(jacs=[array(-0.12308706)], dy=[array(1.)], multi_measurements=[False], shots=None) called by",
                ],
            },
        ]
        with caplog.at_level(logging.DEBUG):
            qml.grad(circuit)(params)
        assert len(caplog.records) == 12
        for idx, r in enumerate(caplog.records):
            assert record_index_map[idx]["log_origin"] in r.name
            if idx > 2:
                continue
            for msg in record_index_map[idx]["log_body"]:
                assert msg in r.getMessage()
