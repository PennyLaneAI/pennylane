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
    ~LabsAdder
    ~LabsOutAdder
    ~LabsMultiplier
    ~ClassicalOutMultiplier
    ~LabsModExp
    ~LabsPhaseAdder

"""

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

from .templates import LabsAdder as Adder
from .templates import LabsModExp as ModExp
from .templates import LabsMultiplier as Multiplier
from .templates import LabsOutAdder as OutAdder
from .templates import LabsPhaseAdder as PhaseAdder

from .templates import (
    OutOfPlaceIntegerComparator,
    RegisterEquality,
    selectpaulirot_controlled_resource_decomp,
    ClassicalOutMultiplier,
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
def _(op: qp.Adder):
    mod = op.hyperparameters["mod"]
    x_wires = op.hyperparameters["x_wires"]
    return Adder(
        len(x_wires),
        mod,
        wires=x_wires,
    )


@_map_to_resource_op.register
def _(op: qp.OutAdder):
    mod = op.hyperparameters["mod"]
    x_wires = op.hyperparameters["x_wires"]
    y_wires = op.hyperparameters["y_wires"]
    output_wires = op.hyperparameters["output_wires"]

    return OutAdder(
        len(x_wires),
        len(y_wires),
        len(output_wires),
        mod=mod,
        wires=x_wires + y_wires + output_wires,
    )


@_map_to_resource_op.register
def _(op: qp.Multiplier):
    mod = op.hyperparameters["mod"]
    x_wires = op.hyperparameters["x_wires"]
    return Multiplier(
        len(x_wires),
        mod=mod,
        wires=x_wires,
    )


@_map_to_resource_op.register
def _(op: qp.ModExp):
    mod = op.hyperparameters["mod"]
    x_wires = op.hyperparameters["x_wires"]
    output_wires = op.hyperparameters["output_wires"]
    return ModExp(
        len(x_wires),
        len(output_wires),
        mod=mod,
        wires=x_wires + output_wires,
    )


@_map_to_resource_op.register
def _(op: qp.PhaseAdder):
    mod = op.hyperparameters["mod"]
    x_wires = op.hyperparameters["x_wires"]

    if mod != 2 ** (len(x_wires)):  # An extra wire was prepended
        x_wires = x_wires[1:]

    return PhaseAdder(
        len(x_wires),
        mod=mod,
        wires=x_wires,
    )
