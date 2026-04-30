# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
This module contains experimental features for
resource estimation.

.. warning::

    This module is experimental. Frequent changes will occur,
    with no guarantees of stability or backwards compatibility.


Resource Estimation
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.estimator_beta

.. autosummary::
    :toctree: api

    ~estimate
    ~LabsResourceConfig

Qubit Tracking Functionality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.estimator_beta

.. autosummary::
    :toctree: api

    ~Allocate
    ~Deallocate
    ~estimate_wires_from_circuit
    ~estimate_wires_from_resources
    ~MarkClean
    ~MarkQubits

State Preparation
~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.estimator_beta.templates

.. autosummary::
    :toctree: api

    ~LabsMottonenStatePreparation
    ~LabsCosineWindow
    ~LabsSumOfSlatersPrep

Alternate Decompositions
~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.estimator_beta

.. autosummary::
    :toctree: api

    ~selectpaulirot_controlled_resource_decomp
    ~paulirot_controlled_resource_decomp
    ~ch_resource_decomp
    ~ch_toffoli_based_resource_decomp
    ~hadamard_controlled_resource_decomp
    ~hadamard_toffoli_based_controlled_decomp

Templates
~~~~~~~~~

.. currentmodule:: pennylane.labs.estimator_beta.templates

.. autosummary::
    :toctree: api

    ~OutOfPlaceIntegerComparator
    ~RegisterEquality

"""

import numpy as np

import pennylane as qp
from pennylane.estimator import *

from pennylane.estimator.ops.op_math.symbolic import apply_adj, apply_controlled
from pennylane.estimator.resource_mapping import _map_to_resource_op

from .estimate import estimate
from .wires_manager import (
    Allocate,
    Deallocate,
    MarkClean,
    MarkQubits,
    estimate_wires_from_circuit,
    estimate_wires_from_resources,
)
from .resource_config import LabsResourceConfig
from .templates import LabsCosineWindow as CosineWindow
from .templates import LabsMottonenStatePreparation as MottonenStatePreparation
from .templates import LabsSumOfSlatersPrep as SumOfSlatersPrep

from .templates import (
    OutOfPlaceIntegerComparator,
    RegisterEquality,
    selectpaulirot_controlled_resource_decomp,
)
from .ops import (
    ch_resource_decomp,
    ch_toffoli_based_resource_decomp,
    hadamard_controlled_resource_decomp,
    hadamard_toffoli_based_controlled_decomp,
    paulirot_controlled_resource_decomp,
)


@apply_controlled.register
def _(action: Allocate | Deallocate, num_ctrl_wires, num_zero_ctrl):
    return action


@apply_adj.register
def _(action: Allocate):
    return Deallocate(allocated_register=action)


@apply_adj.register
def _(action: Deallocate):
    if action.allocated_register is not None:
        return action.allocated_register

    return Allocate(action.num_wires, state=action.state, restored=action.restored)


@_map_to_resource_op.register
def _(op: qp.CosineWindow):
    return CosineWindow(num_wires=len(op.wires), wires=op.wires)


@_map_to_resource_op.register
def _(op: qp.MottonenStatePreparation):
    return MottonenStatePreparation(num_wires=len(op.wires), wires=op.wires)


@_map_to_resource_op.register
def _(op: qp.SumOfSlatersPrep):
    from pennylane.templates.state_preparations.sum_of_slaters import (  # pylint: disable=import-outside-toplevel
        select_sos_rows,
    )

    indices = op.hyperparameters["indices"]
    n = len(op.wires)
    v_bits = qp.math.int_to_binary(np.array(indices), n).T
    selector_ids, _ = select_sos_rows(v_bits)
    return SumOfSlatersPrep(
        num_coeffs=len(indices), num_wires=len(op.wires), num_bits=len(selector_ids), wires=op.wires
    )
