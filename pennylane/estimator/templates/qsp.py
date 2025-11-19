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
"""
Contains templates for Quantum Signal Processing (QSP) based subroutines.
"""
from pennylane.estimator.ops.op_math.symbolic import Adjoint, Controlled
from pennylane.estimator.ops.qubit.parametric_ops_single_qubit import Rot
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    _dequeue,
    resource_rep,
)
from pennylane.wires import Wires, WiresLike


class GQSP(ResourceOperator):
    r"""Resource class for the Generalized Quantum Signal Processing (GQSP) algorithm.
    As described in theorem 3 of `Generalized Quantum Signal Processing (2024)
    <https://arxiv.org/pdf/2308.01501>`_.

    Args:
        signal_operator (:class:`~.pennylane.estimator.resource_operator.ResourceOperator`): The
            signal operator which encodes the target hamiltonian.
        poly_deg (int): the maximum positive degree :math:`d` of the polynomial transformation
        neg_poly_deg (int): the maximum negative degree :math:`k` of the polynomial transformation
        is_controlled (bool): Is ``True`` if the provided ``signal_operator`` is already block-encoded
            via a signle qubit control.
        rot_precision (float, None): The precision with which to apply the general SU(2) rotation gates.
        wires (Sequence[int], None): The wires the operation acts on. This includes both the wires of the
            signal operator and the control wire required for block-encoding.

    Resources:
        The resources are obtained as described in theorem 3 of `Generalized Quantum Signal Processing
        (2024) <https://arxiv.org/pdf/2308.01501>`_.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    """

    resource_keys = {"cmpr_signal_op", "poly_deg", "neg_poly_deg", "is_controlled", "rot_precision"}

    def __init__(
        self,
        signal_operator: ResourceOperator,
        poly_deg: int,
        neg_poly_deg: int = 0,
        is_controlled: bool = True,
        rot_precision: float | None = None,
        wires: WiresLike = None,
    ):
        _dequeue(signal_operator)  # remove operator
        self.queue()

        self.poly_deg = poly_deg
        self.neg_poly_deg = neg_poly_deg
        self.is_controlled = is_controlled
        self.rot_precision = rot_precision
        self.cmpr_signal_op = signal_operator.resource_rep_from_op()

        self.num_wires = signal_operator.num_wires
        if not is_controlled:
            self.num_wires += 1  # add control wire

        if wires:
            self.wires = Wires(wires)
            if base_wires := signal_operator.wires:
                self.wires = Wires.all_wires([self.wires, base_wires])
            if len(self.wires) != self.num_wires:
                raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}.")
        else:
            self.wires = None

    @property
    def resource_params(self):
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * cmpr_signal_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                  The compressed representation of signal operator which encodes the target hamiltonian.
                * poly_deg (int): the maximum positive degree :math:`d` of the polynomial transformation
                * neg_poly_deg (int): the maximum negative degree :math:`k` of the polynomial
                  transformation
                * is_controlled (bool): Is ``True`` if the provided ``signal_operator`` is already
                  block-encoded via a signle qubit control.
                * rot_precision (float, None): The precision with which to apply the general SU(2)
                  rotation gates.
        """

        return {
            "cmpr_signal_op": self.cmpr_signal_op,
            "poly_deg": self.poly_deg,
            "neg_poly_deg": self.neg_poly_deg,
            "is_controlled": self.is_controlled,
            "rot_precision": self.rot_precision,
        }

    @classmethod
    def resource_rep(
        cls,
        cmpr_signal_op: CompressedResourceOp,
        poly_deg: int,
        neg_poly_deg: int,
        is_controlled: bool,
        rot_precision: float | None,
    ):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            cmpr_signal_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                The compressed representation of signal operator which encodes the target hamiltonian.
            poly_deg (int): the maximum positive degree :math:`d` of the polynomial transformation
            neg_poly_deg (int): the maximum negative degree :math:`k` of the polynomial transformation
            is_controlled (bool): Is ``True`` if the provided ``signal_operator`` is already
                block-encoded via a signle qubit control.
            rot_precision (float, None): The precision with which to apply the general SU(2)
                rotation gates.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_wires = cmpr_signal_op.num_wires
        if not is_controlled:
            num_wires += 1  # add control wire

        params = {
            "cmpr_signal_op": cmpr_signal_op,
            "poly_deg": poly_deg,
            "neg_poly_deg": neg_poly_deg,
            "is_controlled": is_controlled,
            "rot_precision": rot_precision,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        cmpr_signal_op: CompressedResourceOp,
        poly_deg: int,
        neg_poly_deg: int,
        is_controlled: bool,
        rot_precision: float | None,
    ):
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            cmpr_signal_op (:class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`):
                The compressed representation of signal operator which encodes the target hamiltonian.
            poly_deg (int): the maximum positive degree :math:`d` of the polynomial transformation
            neg_poly_deg (int): the maximum negative degree :math:`k` of the polynomial transformation
            is_controlled (bool): Is ``True`` if the provided ``signal_operator`` is already block-encoded
                via a signle qubit control.
            rot_precision (float, None): The precision with which to apply the general SU(2) rotation gates.
            wires (Sequence[int], None): The wires the operation acts on. This includes both the wires of the
                signal operator and the control wire required for block-encoding.

        Resources:
            The resources are obtained as described in theorem 3 of `Generalized Quantum Signal Processing
            (2024) <https://arxiv.org/pdf/2308.01501>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        rot = Rot.resource_rep(precision=rot_precision)
        adj_cmpr_signal_op = Adjoint.resource_rep(cmpr_signal_op)

        if not is_controlled:
            cmpr_signal_op = Controlled.resource_rep(
                base_cmpr_op=cmpr_signal_op,
                num_ctrl_wires=1,
                num_zero_ctrl=1,
            )

            adj_cmpr_signal_op = Controlled.resource_rep(
                base_cmpr_op=cmpr_signal_op,
                num_ctrl_wires=1,
                num_zero_ctrl=0,
            )

        if neg_poly_deg == 0:
            return [GateCount(rot, poly_deg + 1), GateCount(cmpr_signal_op, poly_deg)]

        return [
            GateCount(rot, poly_deg + neg_poly_deg + 1),
            GateCount(cmpr_signal_op, poly_deg),
            GateCount(adj_cmpr_signal_op, neg_poly_deg),
        ]
