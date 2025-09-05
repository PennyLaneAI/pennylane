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

import pennylane as qml

_grad_log_map = {
    "adjoint": "diff_method=adjoint, interface=autograd, grad_on_execution=best, gradient_kwargs={}",
    "backprop": "diff_method=backprop, interface=autograd, grad_on_execution=best, gradient_kwargs={}",
    "parameter-shift": "diff_method=<transform: param_shift>",
}


@pytest.mark.logging
class TestLogging:
    """Tests for logging integration"""

    def test_qd_dev_creation(self, caplog):
        "Test logging of device creation"

        with caplog.at_level(logging.DEBUG):
            qml.device("default.qubit", wires=2)

        assert len(caplog.records) == 1
        assert "Calling <__init__(self=<default.qubit device" in caplog.text

    def test_qd_qnode_creation(self, caplog):
        "Test logging of QNode creation"

        dev = qml.device("default.qubit", wires=2)

        # Single log entry, QNode creation
        with caplog.at_level(logging.DEBUG):

            @qml.qnode(dev, diff_method=None)
            def circuit():  # pylint: disable=unused-variable
                qml.PauliX(wires=0)
                return qml.expval(qml.PauliZ(0)), qml.var(qml.PauliZ(0))

        assert len(caplog.records) == 1
        assert "Creating QNode" in caplog.text

    def test_dq_qnode_execution(self, caplog):
        "Test logging of QNode forward pass"

        dev = qml.device("default.qubit", wires=2)

        with caplog.at_level(logging.DEBUG):
            params = qml.numpy.array(0.1234)

            @qml.qnode(dev, diff_method=None)
            def circuit(params):
                qml.RX(params, wires=0)
                return qml.expval(qml.PauliZ(0))

            circuit(params)
        assert len(caplog.records) == 11
        log_records_expected = [
            (
                "pennylane.workflow.qnode",
                ["Creating QNode(func=<function TestLogging.test_dq_qnode_execution"],
            ),
            (
                "pennylane.workflow.qnode",
                ["Calling <construct(self=<QNode: device='<default.qubit device"],
            ),
            (
                "pennylane.workflow.execution",
                [
                    "device=<default.qubit device (wires=2)",
                    "diff_method=None, interface=auto",
                ],
            ),
            (
                "pennylane.workflow.resolution",
                ["Calling <_resolve_diff_method("],
            ),
            (
                "pennylane.devices.default_qubit",
                ["Calling <setup_execution_config(self=<default.qubit device (wires=2)"],
            ),
            (
                "pennylane.devices.default_qubit",
                ["Calling <preprocess_transforms(self=<default.qubit device (wires=2)"],
            ),
        ]

        for expected, actual in zip(log_records_expected, caplog.records[:5]):
            assert expected[0] in actual.name
            for msg in expected[1]:
                assert msg in actual.getMessage()

    @pytest.mark.parametrize(
        "diff_method,num_records", [("parameter-shift", 23), ("backprop", 13), ("adjoint", 17)]
    )
    def test_dq_qnode_execution_grad(self, caplog, diff_method, num_records):
        "Test logging of QNode with parametrized gradients"

        dev = qml.device("default.qubit", wires=2)
        params = qml.numpy.array(0.1234)

        # Single log entry, QNode creation
        with caplog.at_level(logging.DEBUG):

            @qml.qnode(dev, diff_method=diff_method)
            def circuit(params):
                qml.RX(params, wires=0)
                return qml.expval(qml.PauliZ(0))

            qml.grad(circuit)(params)

        assert len(caplog.records) == num_records

        log_records_expected = [
            (
                "pennylane.workflow.qnode",
                [
                    "Creating QNode(func=<function TestLogging.test_dq_qnode_execution_grad",
                    "device=<default.qubit device (wires=2)",
                    f"interface=Interface.AUTO, diff_method={diff_method}, grad_on_execution=best,",
                ],
            ),
            (
                "pennylane.workflow.execution",
                [
                    "Entry with args=(tapes=(<QuantumScript: wires=[0], params=1>,)",
                    _grad_log_map[diff_method],
                ],
            ),
        ]

        for expected, actual in zip(log_records_expected, caplog.records[:1]):
            assert expected[0] in actual.name
            for exp_msg in expected[1]:
                assert exp_msg in actual.getMessage()

    def test_execution_debugging_qutrit_mixed(self, caplog):
        """Test logging of QNode forward pass from default qutrit mixed."""

        with caplog.at_level(logging.DEBUG):
            dev = qml.device("default.qutrit.mixed", wires=2)
            params = qml.numpy.array(0.1234)

            @qml.qnode(dev, diff_method=None)
            def circuit(params):
                qml.TRX(params, wires=0)
                return qml.expval(qml.GellMann(0, 3))

            circuit(params)

        assert len(caplog.records) == 9

        log_records_expected = [
            (
                "pennylane.devices.default_qutrit_mixed",
                ["Calling <__init__(self=<default.qutrit.mixed device (wires=2)"],
            ),
            (
                "pennylane.workflow.qnode",
                ["Creating QNode(func=<function TestLogging.test_execution_debugging_qutrit_mixed"],
            ),
            (
                "pennylane.workflow.qnode",
                ["Calling <construct(self=<QNode: device='<default.qutrit.mixed device (wires=2)"],
            ),
            (
                "pennylane.workflow.execution",
                [
                    "device=<default.qutrit.mixed device (wires=2)",
                    "diff_method=None, interface=Interface.AUTOGRAD",
                ],
            ),
        ]

        for expected, actual in zip(log_records_expected, caplog.records[:2]):
            assert expected[0] in actual.name
            assert all(msg in actual.getMessage() for msg in expected[1])
