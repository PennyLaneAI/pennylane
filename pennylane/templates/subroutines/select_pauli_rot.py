# Copyright 2018-2025 Xanadu Quantum Technologies Inc.

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
Contains the SelectPauliRot template.
"""

from pennylane import capture, math
from pennylane.core.operator import Operator2, abstractify
from pennylane.decomposition import (
    add_decomps,
    change_op_basis_resource_rep,
    register_resources,
    resource_rep,
)
from pennylane.ops import CNOT, RZ, Hadamard, S, adjoint, change_op_basis, prod
from pennylane.ops.op_math import Prod
from pennylane.ops.op_math.adjoint2 import _adjoint_abstract
from pennylane.templates.state_preparations.mottonen import _apply_uniform_rotation_dagger
from pennylane.typing import Float, Wire
from pennylane.wires import Wires


class SelectPauliRot(Operator2):
    r"""Applies individual single-qubit Pauli rotations depending on the state of
    designated control qubits.

    This operator, also called a **multiplexed rotation** or **uniformly controlled rotation**,
    applies a sequence of multi-controlled rotations about the same axis to a single target qubit.
    The rotation angles are selected based on the state of the control qubits.
    Its definition is given by:

    .. math::

       \sum_i | i \rangle \langle i | \otimes R_P(\alpha_i),

    where :math:`| i \rangle` refers to the computational basis state of the control register,
    the :math:`\{\alpha_i\}` are the rotation angles, and :math:`R_P` denotes a Pauli rotation
    about the Pauli operator :math:`P` applied to the target qubit.

    .. figure:: ../../../doc/_static/templates/subroutines/select_pauli_rot.png
                    :align: center
                    :width: 70%
                    :target: javascript:void(0);

    For more details, see `Möttönen and Vartiainen (2005), Fig 7a <https://arxiv.org/abs/quant-ph/0504100>`_.

    Args:
        angles (tensor_like): The rotation angles to be applied. The length of the angles array must
            be :math:`2^n`, where :math:`n` is the number of ``control_wires``.
        control_wires (Sequence[int]): The control qubits used to select the rotation.
        target_wire (Sequence[int]): The wire where the rotations are applied.
        rot_axis (str): The axis around which the rotation is performed.
            It can take the value ``X``, ``Y`` or ``Z``. Default is ``Z``.

    Raises:
        ValueError: If the length of the angles array is not :math:`2^n`, where :math:`n` is the number
            of ``control_wires``.
        ValueError: If ``rot_axis`` has a value different from ``X``, ``Y`` or ``Z``.
        ValueError: If the number of the target wires is not one.

    .. seealso:: :class:`~.Select`.

    **Example**

    .. code-block:: python

        angles = np.array([1.0, 2.0, 3.0, 4.0])

        wires = qp.registers({"control": 2, "target": 1})
        dev = qp.device("default.qubit", wires=3)

        @qp.qnode(dev)
        def circuit():
            qp.SelectPauliRot(
                angles,
                control_wires=wires["control"],
                target_wire=wires["target"],
                rot_axis="Y",
            )
            return qp.state()

    >>> print(circuit()) # doctest: +SKIP
    [0.8776+0.j 0.4794+0.j 0.    +0.j 0.    +0.j 0.    +0.j 0.    +0.j
     0.    +0.j 0.    +0.j]
    """

    dynamic_argnames = ("angles",)
    wire_argnames = ("control_wires", "target_wire")
    compilable_argnames = ("rot_axis",)

    arg_specs = {"angles": Float[-1], "control_wires": Wire[-1], "target_wire": Wire}

    grad_method = None
    ndim_params = (1,)

    def __init__(
        self, angles, control_wires, target_wire, rot_axis="Z"
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments

        if math.shape(angles)[-1] != 2 ** len(control_wires):
            raise ValueError("Number of angles must be 2^(len(control_wires))")

        if rot_axis not in ["X", "Y", "Z"]:
            raise ValueError("'rot_axis' can only take the values 'X', 'Y' and 'Z'.")

        if len(Wires(target_wire)) != 1:
            raise ValueError("Only one target wire can be specified")

        super().__init__(
            angles, control_wires=control_wires, target_wire=target_wire, rot_axis=rot_axis
        )

    def __abstract_init__(self, angles, control_wires, target_wire, rot_axis):
        if math.shape(angles)[-1] != 2 ** len(control_wires):
            raise ValueError("Number of angles must be 2^(len(control_wires))")
        if rot_axis not in ["X", "Y", "Z"]:
            raise ValueError("'rot_axis' can only take the values 'X', 'Y' and 'Z'.")
        if (
            not isinstance(target_wire, int)
            and target_wire.shape != (1,)
            and target_wire.shape != ()
        ):
            raise ValueError("Only one target wire can be specified")
        return super().__abstract_init__(angles, control_wires, target_wire, rot_axis)


# pylint: disable=unused-arguments
def _select_pauli_rot_resource(angles, control_wires, target_wire, rot_axis):

    num_wires = len(control_wires) + 1

    prod_res = {
        abstractify(RZ): 2 ** (num_wires - 1),
        abstractify(CNOT): 2 ** (num_wires - 1) if num_wires > 1 else 0,
    }
    if rot_axis == "Z":
        return prod_res

    if rot_axis == "X":
        return {
            change_op_basis_resource_rep(
                Hadamard, resource_rep(Prod, resources=prod_res), Hadamard
            ): 1,
        }

    prod_rep1 = resource_rep(Prod, resources={abstractify(Hadamard): 1, _adjoint_abstract(S): 1})
    prod_rep2 = resource_rep(Prod, resources={abstractify(S): 1, abstractify(Hadamard): 1})

    return {
        change_op_basis_resource_rep(
            prod_rep1, resource_rep(Prod, resources=prod_res), prod_rep2
        ): 1,
    }


# Not exact resources because rotations might be skipped based on angles
@register_resources(_select_pauli_rot_resource, exact=False)
def decompose_select_pauli_rot(angles, control_wires, target_wire, rot_axis):
    r"""Decomposes the SelectPauliRot"""

    wires = Wires(control_wires) + Wires(target_wire)

    if capture.enabled():
        wires = math.array(wires, like="jax")

    match rot_axis:
        case "X":
            change_op_basis(
                Hadamard(wires[-1]),
                prod(_apply_uniform_rotation_dagger)(RZ, angles, wires[-2::-1], wires[-1]),
                Hadamard(wires[-1]),
            )
        case "Y":
            change_op_basis(
                Hadamard(wires[-1]) @ adjoint(S(wires[-1])),
                prod(_apply_uniform_rotation_dagger)(RZ, angles, wires[-2::-1], wires[-1]),
                S(wires[-1]) @ Hadamard(wires[-1]),
            )
        case "Z":
            _apply_uniform_rotation_dagger(RZ, angles, wires[-2::-1], wires[-1])


add_decomps(SelectPauliRot, decompose_select_pauli_rot)
