# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains special logic of decomposing symbolic operations."""

from __future__ import annotations

import functools

import pennylane as qml

from .controlled_decomposition import base_to_custom_ctrl_op
from .decomposition_rule import DecompositionRule, register_resources
from .resources import adjoint_resource_rep, resource_rep


class AdjointDecomp(DecompositionRule):  # pylint: disable=too-few-public-methods
    """The adjoint version of a decomposition rule."""

    def __init__(self, base_decomposition: DecompositionRule):
        self._base_decomposition = base_decomposition
        super().__init__(self._get_impl(), self._get_resource_fn())

    def _get_impl(self):
        """The implementation of the adjoint of a gate."""

        def _impl(*_, base, **__):
            qml.adjoint(self._base_decomposition._impl)(  # pylint: disable=protected-access
                *base.parameters, base.wires, **base.hyperparameters
            )

        return _impl

    def _get_resource_fn(self):
        """The resource function of the adjoint of a gate."""

        def _resource_fn(base_class, base_params):  # pylint: disable=unused-argument
            base_resources = self._base_decomposition.compute_resources(**base_params)
            return {
                adjoint_resource_rep(decomp_op.op_type, decomp_op.params): count
                for decomp_op, count in base_resources.gate_counts.items()
            }

        return _resource_fn


def _same_type_adjoint_resource(base_class, base_params):
    """Resources of the adjoint of a gate whose adjoint is an instance of its own type."""
    # This assumes that the adjoint of the gate has the same resources as the gate itself.
    return {resource_rep(base_class, **base_params): 1}


@register_resources(_same_type_adjoint_resource)
def has_adjoint_decomp(*_, base, **__):
    """Decompose the adjoint of a gate whose adjoint is an instance of its own type."""
    base.adjoint()


def _adjoint_adjoint_resource(*_, base_params, **__):
    """Resources of the adjoint of the adjoint of a gate."""
    base_class, base_params = base_params["base_class"], base_params["base_params"]
    return {resource_rep(base_class, **base_params): 1}


@register_resources(_adjoint_adjoint_resource)
def adjoint_adjoint_decomp(*_, wires, base):  # pylint: disable=unused-argument
    """Decompose the adjoint of the adjoint of a gate."""
    base.base._unflatten(*base.base._flatten())  # pylint: disable=protected-access


def _adjoint_controlled_resource(base_class, base_params):
    """Resources of the adjoint of a controlled gate whose base has adjoint."""

    num_control_wires = base_params["num_control_wires"]
    controlled_base_class = base_params["base_class"]

    # Handle controlled-X gates, the adjoint is just themselves
    if controlled_base_class is qml.X:
        if num_control_wires == 1:
            return {resource_rep(qml.CNOT): 1}
        if num_control_wires == 2:
            return {resource_rep(qml.Toffoli): 1}
        return {
            resource_rep(
                qml.MultiControlledX,
                num_control_wires=num_control_wires,
                num_zero_control_values=base_params["num_zero_control_values"],
                num_work_wires=base_params["num_work_wires"],
            ): 1
        }

    # Handle custom controlled gates. The adjoint of a general controlled operator that
    # is equivalent to a custom controlled operator should just be the custom controlled
    # operator given that its base has_adjoint.
    if (controlled_base_class, num_control_wires) in base_to_custom_ctrl_op():
        custom_op_type = base_to_custom_ctrl_op()[(controlled_base_class, num_control_wires)]
        # All gates in base_to_custom_ctrl_op do not have resource params.
        return {resource_rep(custom_op_type): 1}

    # Handle the general case, here we assume that the adjoint of a controlled gate
    # whose base has an adjoint that is of its own type, should have the same resource
    # rep as the controlled gate itself. For example, Adjoint(Controlled(O)) should
    # have the same resources as Controlled(O) if the adjoint of O is another O.
    return {resource_rep(base_class, **base_params): 1}


@register_resources(_adjoint_controlled_resource)
def adjoint_controlled_decomp(*_, base, **__):
    """Decompose the adjoint of a controlled gate whose base has adjoint.

    Precondition:
    - isinstance(base, qml.ops.Controlled) and base.base.has_adjoint

    """
    qml.ctrl(
        base.base.adjoint(),
        control=base.control_wires,
        control_values=base.control_values,
        work_wires=base.work_wires,
    )


@functools.lru_cache()
def has_adjoint_ops():
    """A set of operators whose adjoint is an instance of its own type."""
    return {
        # identity
        qml.Identity,
        qml.GlobalPhase,
        # non-parametric gates
        qml.H,
        qml.X,
        qml.Y,
        qml.Z,
        qml.SWAP,
        qml.ECR,
        # single-qubit parametric gates
        qml.Rot,
        qml.U1,
        qml.U2,
        qml.U3,
        qml.RX,
        qml.RY,
        qml.RZ,
        qml.PhaseShift,
        # multi-qubit parametric gates
        qml.MultiRZ,
        qml.PauliRot,
        qml.PCPhase,
        qml.IsingXX,
        qml.IsingYY,
        qml.IsingZZ,
        qml.IsingXY,
        qml.PSWAP,
        qml.CPhaseShift00,
        qml.CPhaseShift01,
        qml.CPhaseShift10,
        # matrix gates
        qml.QubitUnitary,
        qml.DiagonalQubitUnitary,
        qml.BlockEncode,
        qml.SpecialUnitary,
        # custom controlled ops
        qml.CH,
        qml.CY,
        qml.CZ,
        qml.CNOT,
        qml.CSWAP,
        qml.CCZ,
        qml.Toffoli,
        qml.MultiControlledX,
        qml.CRX,
        qml.CRY,
        qml.CRZ,
        qml.CRot,
        qml.ControlledPhaseShift,
        # arithmetic ops
        qml.QubitSum,
        qml.IntegerComparator,
        # qchem ops
        qml.SingleExcitation,
        qml.SingleExcitationMinus,
        qml.SingleExcitationPlus,
        qml.DoubleExcitation,
        qml.DoubleExcitationPlus,
        qml.DoubleExcitationMinus,
        qml.OrbitalRotation,
        qml.FermionicSWAP,
        # templates
        qml.CommutingEvolution,
    }
