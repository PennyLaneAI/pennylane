# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fixtures and functions for QNode interface tests."""

import warnings
import itertools

import pytest
import pennylane as qml
from param_shift_dev import ParamShiftDerivativesDevice


@pytest.fixture(autouse=True)
def suppress_tape_property_deprecation_warning():
    warnings.filterwarnings(
        "ignore", "The tape/qtape property is deprecated", category=qml.PennyLaneDeprecationWarning
    )


def get_device(device_name, wires, seed=None):
    if device_name == "param_shift.qubit":
        return ParamShiftDerivativesDevice(seed=seed)
    if device_name == "lightning.qubit":
        return qml.device("lightning.qubit", wires=wires)
    return qml.device(device_name, seed=seed)


_dev_names = ("default.qubit", "param_shift.qubit", "lightning.qubit", "reference.qubit")
_diff_methods = ("backprop", "adjoint", "finite-diff", "parameter-shift", "hadamard", "spsa")
_grad_on_exec = (True, False)
_device_vjp = (True, False)


def generate_test_matrix(
    parametrize_devices=True,
    parametrize_diff_methods=True,
    parametrize_grad_on_execution=True,
    parametrize_device_vjp=True,
    xfail_conditions=None,
    skip_conditions=None,
):
    """Generates the test matrix for different combinations."""

    xfail_conditions = xfail_conditions or []
    skip_conditions = skip_conditions or []

    def _first_reason(conditions, params):
        """Returns the first condition that is met."""
        for condition in conditions:
            reason = condition(*params)
            if reason:
                return reason
        return None

    def _test_matrix_iter():
        """Yields tuples of (device_name, diff_method, grad_on_execution, device_vjp)."""

        dev_names = _dev_names if parametrize_devices else (None,)
        diff_methods = _diff_methods if parametrize_diff_methods else (None,)
        grad_on_execs = _grad_on_exec if parametrize_grad_on_execution else (None,)
        device_vjp = _device_vjp if parametrize_device_vjp else (None,)

        combinations = itertools.product(dev_names, diff_methods, grad_on_execs, device_vjp)
        for device_name, diff_method, grad_on_execution, device_vjp in combinations:

            if diff_method == "adjoint" and device_name not in ("default.qubit", "lightning.qubit"):
                continue  # adjoint diff is only supported on DQ and LQ

            if device_name == "param_shift.qubit" and diff_method != "parameter-shift":
                continue  # param_shift.qubit is not intended to be used with anything else

            if diff_method not in ("adjoint", "backprop") and grad_on_execution:
                continue  # Gradient transforms cannot be used with grad_on_execution=True

            params = device_name, diff_method, grad_on_execution, device_vjp
            filtered_params = tuple(p for p in params if p is not None)
            if xfail_reason := _first_reason(xfail_conditions, params):
                yield pytest.param(*filtered_params, marks=pytest.mark.xfail(reason=xfail_reason))
            elif skip_reason := _first_reason(skip_conditions, params):
                yield pytest.param(*filtered_params, marks=pytest.mark.skip(reason=skip_reason))
            else:
                yield filtered_params if len(filtered_params) > 1 else filtered_params[0]

    return [_params for _params in _test_matrix_iter()]
