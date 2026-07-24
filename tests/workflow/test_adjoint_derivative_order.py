# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for higher-order derivatives requested with the adjoint method."""

from dataclasses import replace

import pytest

import pennylane as qp
from pennylane.devices import DefaultQubit, Device, ExecutionConfig
from pennylane.exceptions import QuantumFunctionError
from pennylane.workflow import _resolve_diff_method


# pylint: disable=unused-argument
class DeviceResolvingBestToAdjoint(Device):
    """Test device that advertises support before resolving ``best`` to ``adjoint``."""

    def execute(self, circuits, execution_config=None):
        return 0

    def supports_derivatives(self, execution_config=None, circuit=None):
        return True

    def setup_execution_config(self, config=None, circuit=None):
        return replace(config, gradient_method="adjoint")


@pytest.mark.parametrize("support_method", ["supports_derivatives", "supports_jvp", "supports_vjp"])
def test_default_qubit_does_not_advertise_higher_order_adjoint(support_method):
    """DefaultQubit should accurately report that adjoint supports only first derivatives."""
    config = ExecutionConfig(gradient_method="adjoint", derivative_order=2)

    assert getattr(DefaultQubit(), support_method)(config) is False


def test_explicit_higher_order_adjoint_is_rejected():
    """An explicit request for higher-order adjoint derivatives should fail informatively."""
    config = ExecutionConfig(gradient_method="adjoint", derivative_order=2)

    with pytest.raises(QuantumFunctionError, match="adjoint.*higher-order derivatives"):
        _resolve_diff_method(config, DefaultQubit())


def test_higher_order_adjoint_is_rejected_after_device_resolution():
    """A plugin must not bypass the guard by resolving ``best`` to ``adjoint``."""
    config = ExecutionConfig(gradient_method="best", derivative_order=2)

    with pytest.raises(QuantumFunctionError, match="adjoint.*higher-order derivatives"):
        _resolve_diff_method(config, DeviceResolvingBestToAdjoint())


def test_first_order_adjoint_remains_supported():
    """The guard should not affect supported first-order adjoint derivatives."""
    config = ExecutionConfig(gradient_method="adjoint", derivative_order=1)

    resolved_config = _resolve_diff_method(config, DefaultQubit())

    assert resolved_config.gradient_method == "adjoint"


@pytest.mark.torch
def test_torch_classical_preprocessing_fails_fast():
    """Adjoint must not return an incomplete Torch Hessian through classical preprocessing."""
    torch = pytest.importorskip("torch")
    dev = DefaultQubit(wires=1)

    @qp.qnode(dev, interface="torch", diff_method="adjoint", max_diff=2)
    def circuit(angle):
        qp.RY(angle, wires=0)
        return qp.expval(qp.PauliZ(0))

    x = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    angle = torch.tanh(1.7 * x + 0.2)

    with pytest.raises(QuantumFunctionError, match="adjoint.*higher-order derivatives"):
        circuit(angle)
