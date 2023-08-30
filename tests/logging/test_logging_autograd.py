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
import pennylane.logging as pl_logging


_grad_log_map = {
    "adjoint": "gradient_fn=device, interface=autograd, grad_on_execution=best, gradient_kwargs={'use_device_state': True, 'method': 'adjoint_jacobian'}",
    "backprop": "gradient_fn=backprop, interface=autograd, grad_on_execution=best, gradient_kwargs={}",
    "parameter-shift": "gradient_fn=<gradient_transform: param_shift>",
}


def enable_and_configure_logging():
    pl_logging.enable_logging()

    pl_logger = logging.root.manager.loggerDict["pennylane"]
    plqn_logger = logging.root.manager.loggerDict["pennylane.qnode"]

    # Ensure logs messages are propagated for pytest capture
    pl_logger.propagate = True
    plqn_logger.propagate = True


@pytest.mark.logging
class TestLogging:
    """Tests for logging integration"""

    def test_qd_qnode_creation(self, caplog):
        "Test logging of QNode creation"

        enable_and_configure_logging()

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

        enable_and_configure_logging()

        dev = qml.device("default.qubit", wires=2)

        with caplog.at_level(logging.DEBUG):
            dev = qml.device("default.qubit", wires=2)
            params = qml.numpy.array(0.1234)

            @qml.qnode(dev, diff_method=None)
            def circuit(params):
                qml.RX(params, wires=0)
                return qml.expval(qml.PauliZ(0))

            circuit(params)

        assert len(caplog.records) == 5

        log_records_expected = [
            (
                "pennylane.qnode",
                ["Creating QNode(func=<function TestLogging.test_dq_qnode_execution"],
            ),
            (
                "pennylane.interfaces.execution",
                [
                    "device=<DefaultQubit device (wires=2, shots=None)",
                    "gradient_fn=None, interface=None",
                ],
            ),
        ]

        for idx, r in enumerate(caplog.records[0:2]):
            assert log_records_expected[idx][0] in r.name
            for msg in log_records_expected[idx][1]:
                assert msg in r.getMessage()

    @pytest.mark.parametrize(
        "diff_method", [("parameter-shift", 12), ("backprop", 5), ("adjoint", 9)]
    )
    def test_dq_qnode_execution_grad(self, caplog, diff_method):
        "Test logging of QNode with parameterised gradients"

        enable_and_configure_logging()

        dev = qml.device("default.qubit", wires=2)
        params = qml.numpy.array(0.1234)

        # Single log entry, QNode creation
        with caplog.at_level(logging.DEBUG):
            dev = qml.device("default.qubit", wires=2)
            params = qml.numpy.array(0.1234)

            @qml.qnode(dev, diff_method=diff_method[0])
            def circuit(params):
                qml.RX(params, wires=0)
                return qml.expval(qml.PauliZ(0))

            qml.grad(circuit)(params)

        assert len(caplog.records) == diff_method[1]

        log_records_expected = [
            (
                "pennylane.qnode",
                [
                    "Creating QNode(func=<function TestLogging.test_dq_qnode_execution_grad",
                    "device=<DefaultQubit device (wires=2, shots=None)",
                    f"interface=auto, diff_method={diff_method[0]}, expansion_strategy=gradient, max_expansion=10, grad_on_execution=best,",
                ],
            ),
            (
                "pennylane.interfaces.execution",
                [
                    "Entry with args=(tapes=(<QuantumScript: wires=[0], params=1>,)",
                    _grad_log_map[diff_method[0]],
                ],
            ),
        ]

        for idx, r in enumerate(caplog.records[0:2]):
            assert log_records_expected[idx][0] in r.name
            for msg in log_records_expected[idx][1]:
                assert msg in r.getMessage()
