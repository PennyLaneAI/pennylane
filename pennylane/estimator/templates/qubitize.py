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
r"""Resource operators for PennyLane subroutine templates."""
import math

import numpy as np

from pennylane.estimator.compact_hamiltonian import THCHamiltonian
from pennylane.estimator.ops.op_math.controlled_ops import MultiControlledX, Toffoli
from pennylane.estimator.ops.op_math.symbolic import Adjoint, Controlled
from pennylane.estimator.ops.qubit.non_parametric_ops import X
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    _dequeue,
    resource_rep,
)
from pennylane.estimator.templates.select import SelectTHC
from pennylane.estimator.templates.stateprep import PrepTHC
from pennylane.estimator.wires_manager import Allocate, Deallocate
from pennylane.wires import Wires, WiresLike

# pylint: disable=signature-differs, arguments-differ, too-many-arguments


class QubitizeTHC(ResourceOperator):
    r"""Resource class for qubitization of tensor hypercontracted Hamiltonian.

    .. note::

            This decomposition assumes that an appropriately sized phase gradient state is available.
            Users should ensure that the cost of constructing this state has been accounted for.
            See also :class:`~.pennylane.estimator.templates.PhaseGradient`.

    Args:
        thc_ham (:class:`~.pennylane.estimator.compact_hamiltonian.THCHamiltonian`): A tensor hypercontracted
            Hamiltonian for which the walk operator is being created.
        prep_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator` | None): An optional
            resource operator, corresponding to the prepare routine. If :code:`None`, the
            default :class:`~.pennylane.estimator.templates.stateprep.PrepTHC` will be used.
        select_op (:class:`~.pennylane.estimator.resource_operator.ResourceOperator` | None): An optional
            resource operator, corresponding to the select routine. If :code:`None`, the
            default :class:`~.pennylane.estimator.templates.select.SelectTHC` will be used.
        coeff_precision (int | None): The number of bits used to represent the precision for loading
            the coefficients of Hamiltonian.
        rotation_precision (int | None): The number of bits used to represent the precision for loading
            the rotation angles for :code:`select_op`.
        wires (WiresLike | None): the wires on which the operator acts

    Resources:
        The resources are calculated based on `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> thc_ham = qre.THCHamiltonian(num_orbitals=20, tensor_rank=40)
    >>> prep = qre.PrepTHC(thc_ham, coeff_precision=20, select_swap_depth=2)
    >>> res = qre.estimate(qre.QubitizeTHC(thc_ham, prep_op=prep))
    >>> print(res)
    --- Resources: ---
     Total wires: 400
       algorithmic wires: 102
       allocated wires: 298
         zero state: 298
         any state: 0
     Total gates : 5.617E+4
       'Toffoli': 3.501E+3,
       'CNOT': 4.031E+4,
       'X': 2.231E+3,
       'Z': 41,
       'S': 80,
       'Hadamard': 1.001E+4

    .. details::
        :title: Usage Details

        **Precision Precedence**

        The :code:`coeff_precision` and :code:`rotation_precision` arguments are used to determine
        the number of bits for loading the coefficients and the rotation angles, respectively.
        The final value is determined by the following precedence:

        * If provided, the values from :code:`coeff_precision` and :code:`rotation_precision` are used.
        * If :code:`coeff_precision` or :code:`rotation_precision` are not provided or are set to `None`,
          the precisions from :code:`prep_op` and :code:`select_op` take precedence.
        * If both of the above are not specified, the default value of ``15`` bits is used.

    """

    resource_keys = {"thc_ham", "prep_op", "select_op", "coeff_precision", "rotation_precision"}

    def __init__(
        self,
        thc_ham: THCHamiltonian,
        prep_op: ResourceOperator | None = None,
        select_op: ResourceOperator | None = None,
        coeff_precision: int | None = None,
        rotation_precision: int | None = None,
        wires: WiresLike | None = None,
    ):
        if not isinstance(thc_ham, THCHamiltonian):
            raise TypeError(
                f"Unsupported Hamiltonian representation for QubitizeTHC."
                f"This method works with thc Hamiltonian, {type(thc_ham)} provided"
            )

        self.thc_ham = thc_ham
        self.coeff_precision = coeff_precision
        self.rotation_precision = rotation_precision

        num_orb = thc_ham.num_orbitals
        tensor_rank = thc_ham.tensor_rank
        num_coeff = num_orb + tensor_rank * (tensor_rank + 1) / 2  # N+M(M+1)/2
        coeff_register = int(math.ceil(math.log2(num_coeff)))

        if coeff_precision is None:
            coeff_precision = prep_op.coeff_precision if prep_op else 15
        self.coeff_precision = coeff_precision

        if rotation_precision is None:
            rotation_precision = select_op.rotation_precision if select_op else 15
        self.rotation_precision = rotation_precision

        if prep_op is None:
            prep_op = PrepTHC(
                thc_ham,
                coeff_precision=coeff_precision,
            )
        _dequeue(prep_op)
        self.prep_op = prep_op.resource_rep_from_op()

        if select_op is None:
            select_op = SelectTHC(
                thc_ham,
                rotation_precision=rotation_precision,
            )
        _dequeue(select_op)
        self.select_op = select_op.resource_rep_from_op()

        # Algorithmic wires for the walk operator, based on section III D in arXiv:2011.03494.
        # The auxiliary wires are excluded and accounted for by the included templates: QROM, SemiAdder, SelectTHC.
        # The total algorithmic qubits are thus given by: N + 2*n_M + ceil(log(d)) + \aleph + 6 + m
        # where \aleph is coeff_precision, m = 2n_M + \aleph + 2, N = 2*num_orb,
        # d = num_orb + tensor_rank(tensor_rank+1)/2, and n_M = log_2(tensor_rank+1).
        self.num_wires = (
            num_orb * 2
            + 4 * int(np.ceil(math.log2(tensor_rank + 1)))
            + coeff_register
            + 8
            + coeff_precision
        )
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * thc_ham (:class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`): A tensor hypercontracted
                  Hamiltonian for which the walk operator is being created.
                * prep_op (:class:`~pennylane.estimator.resource_operator.CompressedResourceOp` | None): An optional compressed
                  resource operator, corresponding to the prepare routine. If :code:`None`, the
                  default :class:`~.pennylane.estimator.templates.PrepTHC` will be used.
                * select_op (:class:`~pennylane.estimator.resource_operator.CompressedResourceOp` | None): An optional compressed
                  resource operator, corresponding to the select routine. If :code:`None`, the
                  default :class:`~.pennylane.estimator.templates.SelectTHC` will be used.
                * coeff_precision (int | None): The number of bits used to represent the precision for loading
                  the coefficients of Hamiltonian.
                * rotation_precision (int | None): The number of bits used to represent the precision for loading
                  the rotation angles.
        """
        return {
            "thc_ham": self.thc_ham,
            "prep_op": self.prep_op,
            "select_op": self.select_op,
            "coeff_precision": self.coeff_precision,
            "rotation_precision": self.rotation_precision,
        }

    @classmethod
    def resource_rep(
        cls,
        thc_ham: THCHamiltonian,
        prep_op: CompressedResourceOp | None = None,
        select_op: CompressedResourceOp | None = None,
        coeff_precision: int | None = None,
        rotation_precision: int | None = None,
    ) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            thc_ham (:class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`): A tensor hypercontracted
                Hamiltonian for which the walk operator is being created.
            prep_op (:class:`~pennylane.estimator.resource_operator.CompressedResourceOp` | None): An optional compressed
                resource operator, corresponding to the prepare routine. If :code:`None`, the
                default :class:`~.pennylane.estimator.tempaltes.PrepTHC` will be used.
            select_op (:class:`~pennylane.estimator.resource_operator.CompressedResourceOp` | None): An optional compressed
                resource operator, corresponding to the select routine. If :code:`None`, the
                default :class:`~.pennylane.estimator.templates.SelectTHC` will be used.
            coeff_precision (int | None): The number of bits used to represent the precision for loading
                the coefficients of Hamiltonian.
            rotation_precision (int | None): The number of bits used to represent the precision for loading
                the rotation angles.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        if not isinstance(thc_ham, THCHamiltonian):
            raise TypeError(
                f"Unsupported Hamiltonian representation for QubitizeTHC."
                f"This method works with thc Hamiltonian, {type(thc_ham)} provided"
            )

        num_orb = thc_ham.num_orbitals
        tensor_rank = thc_ham.tensor_rank
        num_coeff = num_orb + tensor_rank * (tensor_rank + 1) / 2  # N+M(M+1)/2
        coeff_register = int(math.ceil(math.log2(num_coeff)))

        if coeff_precision is None:
            coeff_precision = prep_op.params["coeff_precision"] if prep_op else 15
        if rotation_precision is None:
            rotation_precision = select_op.params["rotation_precision"] if select_op else 15

        num_wires = (
            num_orb * 2
            + 4 * int(np.ceil(math.log2(tensor_rank + 1)))
            + coeff_register
            + 8
            + coeff_precision
        )

        params = {
            "thc_ham": thc_ham,
            "prep_op": prep_op,
            "select_op": select_op,
            "coeff_precision": coeff_precision,
            "rotation_precision": rotation_precision,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        thc_ham: THCHamiltonian,
        prep_op: CompressedResourceOp | None = None,
        select_op: CompressedResourceOp | None = None,
        coeff_precision: int | None = None,
        rotation_precision: int | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        .. note::

            This decomposition assumes that an appropriately sized phase gradient state is available.
            Users should ensure that the cost of constructing this state has been accounted for.
            See also :class:`~.pennylane.estimator.templates.PhaseGradient`.

        Args:
            thc_ham (:class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`): a tensor hypercontracted
                Hamiltonian for which the walk operator is being created
            prep_op (:class:`~pennylane.estimator.resource_operator.CompressedResourceOp` | None): An optional compressed
                resource operator, corresponding to the prepare routine. If :code:`None`, the
                default :class:`~.pennylane.estimator.templates.PrepTHC` will be used.
            select_op (:class:`~pennylane.estimator.resource_operator.CompressedResourceOp` | None): An optional compressed
                resource operator, corresponding to the select routine. If :code:`None`, the
                default :class:`~.pennylane.estimator.templates.SelectTHC` will be used.
            coeff_precision (int | None): The number of bits used to represent the precision for loading
                the coefficients of Hamiltonian.
            rotation_precision (int | None): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        gate_list = []

        tensor_rank = thc_ham.tensor_rank
        m_register = int(np.ceil(np.log2(tensor_rank)))

        select_kwargs = {
            "thc_ham": thc_ham,
            "select_swap_depth": select_op.params["select_swap_depth"] if select_op else None,
            "num_batches": select_op.params["num_batches"] if select_op else 1,
        }
        if rotation_precision:
            select_kwargs["rotation_precision"] = rotation_precision

        if rotation_precision or select_op is None:
            # Select cost from Figure 5 in arXiv:2011.03494
            select_op = resource_rep(SelectTHC, select_kwargs)

        gate_list.append(GateCount(select_op))

        prep_kwargs = {
            "thc_ham": thc_ham,
            "select_swap_depth": prep_op.params["select_swap_depth"] if prep_op else None,
        }
        if coeff_precision:
            prep_kwargs["coeff_precision"] = coeff_precision

        if coeff_precision or prep_op is None:
            # Prep cost from Figure 3 and 4 in arXiv:2011.03494
            prep_op = resource_rep(PrepTHC, prep_kwargs)

        gate_list.append(GateCount(prep_op))
        gate_list.append(GateCount(resource_rep(Adjoint, {"base_cmpr_op": prep_op})))

        # reflection cost from Eq. 44 in arXiv:2011.03494
        coeff_precision = prep_op.params["coeff_precision"]

        toffoli = resource_rep(Toffoli)
        gate_list.append(GateCount(toffoli, 2 * m_register + coeff_precision + 4))

        return gate_list

    @classmethod
    def controlled_resource_decomp(
        cls, num_ctrl_wires: int, num_zero_ctrl: int, target_resource_params: dict
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for the controlled version of the operator.

        .. note::

            This decomposition assumes that an appropriately sized phase gradient state is available.
            Users should ensure that the cost of constructing this state has been accounted for.
            See also :class:`~.pennylane.estimator.templates.PhaseGradient`.

        Args:
            num_ctrl_wires (int): the number of wires the operation is controlled on
            num_zero_ctrl (int): the number of control wires, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): A dictionary containing the resource params of the target operator.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        gate_list = []
        thc_ham = target_resource_params["thc_ham"]
        prep_op = target_resource_params["prep_op"]
        select_op = target_resource_params["select_op"]

        coeff_precision = target_resource_params.get("coeff_precision")
        rotation_precision = target_resource_params.get("rotation_precision")

        tensor_rank = thc_ham.tensor_rank
        m_register = int(np.ceil(np.log2(tensor_rank)))

        if num_ctrl_wires > 1:
            mcx = resource_rep(
                MultiControlledX,
                {
                    "num_ctrl_wires": num_ctrl_wires,
                    "num_zero_ctrl": num_zero_ctrl,
                },
            )
            gate_list.append(Allocate(1))
            gate_list.append(GateCount(mcx, 2))

        select_kwargs = {
            "thc_ham": thc_ham,
            "select_swap_depth": select_op.params["select_swap_depth"] if select_op else None,
            "num_batches": select_op.params["num_batches"] if select_op else 1,
        }
        if rotation_precision:
            select_kwargs["rotation_precision"] = rotation_precision

        if rotation_precision or select_op is None:
            # Controlled Select cost from Fig 5 in arXiv:2011.03494
            select_op = resource_rep(SelectTHC, select_kwargs)
        gate_list.append(
            GateCount(
                resource_rep(
                    Controlled,
                    {"base_cmpr_op": select_op, "num_ctrl_wires": 1, "num_zero_ctrl": 0},
                )
            )
        )

        prep_kwargs = {
            "thc_ham": thc_ham,
            "select_swap_depth": prep_op.params["select_swap_depth"] if prep_op else None,
        }
        if coeff_precision:
            prep_kwargs["coeff_precision"] = coeff_precision

        if coeff_precision or prep_op is None:
            # Prep cost from Fig 3 and 4 in arXiv:2011.03494
            prep_op = resource_rep(PrepTHC, prep_kwargs)

        gate_list.append(GateCount(prep_op))
        gate_list.append(GateCount(resource_rep(Adjoint, {"base_cmpr_op": prep_op})))

        # reflection cost from Eq. 44 in arXiv:2011.03494
        coeff_precision = prep_op.params["coeff_precision"]

        toffoli = resource_rep(Toffoli)
        gate_list.append(GateCount(toffoli, 2 * m_register + coeff_precision + 4))

        if num_ctrl_wires > 1:
            gate_list.append(Deallocate(1))
        elif num_zero_ctrl > 0:
            gate_list.append(GateCount(resource_rep(X), 2 * num_zero_ctrl))

        return gate_list
