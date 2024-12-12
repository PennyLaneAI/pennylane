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
from typing import Callable

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


_device_names = ("default.qubit", "param_shift.qubit", "lightning.qubit", "reference.qubit")
_diff_methods = ("backprop", "adjoint", "finite-diff", "parameter-shift", "hadamard", "spsa")
_grad_on_execution = (True, False)
_device_vjp = (True, False)


def generate_test_matrix(xfail_condition: Callable, skip_condition: Callable):
    """Generates the test matrix for different combinations."""

    def _test_matrix_iter():
        """Yields tuples of (device_name, diff_method, grad_on_execution, device_vjp)."""

        all_combinations = itertools.product(_device_names, _diff_methods, _grad_on_execution)
        for device_name, diff_method, grad_on_execution, device_vjp in all_combinations:

            if diff_method == "adjoint" and device_name not in ("default.qubit", "lightning.qubit"):
                continue  # adjoint diff is only supported on DQ and LQ

            if device_name == "param_shift.qubit" and diff_method != "parameter-shift":
                continue  # param_shift.qubit is not intended to be used with anything else

            xfail_reason = xfail_condition(device_name, diff_method, grad_on_execution, device_vjp)
            skip_reason = skip_condition(device_name, diff_method, grad_on_execution, device_vjp)

            if xfail_reason:
                yield pytest.param(
                    device_name,
                    diff_method,
                    grad_on_execution,
                    device_vjp,
                    marks=pytest.mark.xfail(reason=xfail_reason),
                )

            elif skip_reason:
                yield pytest.param(
                    device_name,
                    diff_method,
                    grad_on_execution,
                    device_vjp,
                    marks=pytest.mark.skip(reason=skip_reason),
                )

            else:
                yield device_name, diff_method, grad_on_execution, device_vjp

    return [_params for _params in _test_matrix_iter()]
