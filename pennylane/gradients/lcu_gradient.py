# Copyright 2023 Xanadu Quantum Technologies Inc.

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
This module contains functions for computing the parameter-shift gradient
of a qubit-based quantum tape.
"""
import pennylane as qml
from pennylane.measurements import MutualInfoMP, StateMP, VarianceMP, VnEntropyMP

from .finite_difference import _all_zero_grad_new, _no_trainable_grad_new

from .gradient_transform import (
    choose_grad_methods,
    grad_method_validation,
    gradient_analysis,
    gradient_transform,
)

from pennylane.transforms.metric_tensor import _get_aux_wire
@gradient_transform
def lcu_grad(
    tape,
    argnum=None,
    shots=None,
    aux_wire=None,
    device_wires=None,
):
    r"""Transform a QNode to compute the finite-difference gradient of all gate
    parameters with respect to its inputs. This function is adapted to the new return system.

    Args:
        tape (pennylane.QNode or .QuantumTape): quantum tape or QNode to differentiate
        argnum (int or list[int] or None): Trainable parameter indices to differentiate
            with respect to. If not provided, the derivatives with respect to all
            trainable parameters are returned.

    Returns:
        function or tuple[list[QuantumTape], function]

    **Example**

    """
    if argnum is None and not tape.trainable_params:
        return _no_trainable_grad_new(tape, shots)

    gradient_analysis(tape, grad_fn=lcu_grad)
    method = "analytic"
    diff_methods = grad_method_validation(method, tape)

    if all(g == "0" for g in diff_methods):
        return _all_zero_grad_new(tape, shots)

    gradient_tapes = []

    method_map = choose_grad_methods(diff_methods, argnum)
    argnum = [i for i, dm in method_map.items() if dm == "A"]
    print(method_map)

    # Get default for aux_wire
    aux_wire = _get_aux_wire(aux_wire, tape, device_wires)

    if any(isinstance(m, VarianceMP) for m in tape.measurements):
        g_tapes, processing_fn = var_lcu(tape, argnum, aux_wire)
    else:
        g_tapes, processing_fn = expval_lcu(tape, argnum, aux_wire)

    return gradient_tapes, processing_fn


def expval_lcu(tape, argnum, aux_wire):
    print("1", argnum, tape.trainable_params)
    argnum = argnum or tape.trainable_params
    print("2", argnum, tape.trainable_params)
    g_tapes = []

    def processing_fn(results):
        return results

    return g_tapes, processing_fn


def var_lcu(tape, argnum, aux_wire):
    argnum = argnum or tape.trainable_params
    print("ici", argnum)
    g_tapes = []

    def processing_fn(results):
        return results

    return g_tapes, processing_fn
