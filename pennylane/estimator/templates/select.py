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
r"""Resource operators for select templates."""

import math

import numpy as np

import pennylane.estimator as qre
from pennylane.estimator import Allocate, Deallocate
from pennylane.estimator.compact_hamiltonian import THCHamiltonian
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.wires import Wires, WiresLike

# pylint: disable= signature-differs, arguments-differ


class SelectTHC(ResourceOperator):
    r"""Resource class for creating the custom Select operator for tensor hypercontracted (THC) Hamiltonian.

    .. note::

            This decomposition assumes that an appropriately sized phase gradient state is available.
            Users should ensure that the cost of constructing this state has been accounted for.
            See also :class:`~.pennylane.estimator.templates.PhaseGradient`.

    Args:
        thc_ham (:class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`): A tensor hypercontracted
            Hamiltonian on which the select operator is being applied.
        batched_rotations (int | None): The maximum number of rotation angles to load simultaneously
            into temporary quantum registers for processing in the Givens rotation circuits.
            The default value of ``None`` loads all angles at once, where the batch size is equal to
            the number of orbitals minus one.
        rotation_precision (int): The number of bits used to represent the precision for loading
            the rotation angles for basis rotation. The default value is set to ``15`` bits.
        select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.subroutines.QROM`
            used to trade-off extra wires for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.
        wires (WiresLike | None): the wires on which the operator acts

    Resources:
        The resources are calculated based on Figure 5 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> thc_ham =  qre.THCHamiltonian(num_orbitals=20, tensor_rank=40)
    >>> res = qre.estimate(qre.SelectTHC(thc_ham, rotation_precision=15))
    >>> print(res)
    --- Resources: ---
     Total wires: 371
        algorithmic wires: 58
        allocated wires: 313
             zero state: 313
             any state: 0
     Total gates : 1.959E+4
      'Toffoli': 2.219E+3,
      'CNOT': 1.058E+4,
      'X': 268,
      'Z': 41,
      'S': 80,
      'Hadamard': 6.406E+3

    Let's also see how the resources change when batched rotations are used:

    >>> res = qre.estimate(qre.SelectTHC(thc_ham, batched_rotations=10, rotation_precision=15))
    >>> print(res)
    --- Resources: ---
     Total wires: 227
       algorithmic wires: 58
       allocated wires: 169
         zero state: 169
         any state: 0
     Total gates : 2.534E+4
       'Toffoli': 2.633E+3,
       'CNOT': 1.440E+4,
       'X': 804.0,
       'Z': 41,
       'S': 80,
       'Hadamard': 7.378E+3

    We can see that by using batched rotations, the number of allocated wires decreases
    significantly, at the cost of an increased number of Toffoli gates.

    """

    resource_keys = {"thc_ham", "batched_rotations", "rotation_precision", "select_swap_depth"}

    def __init__(
        self,
        thc_ham: THCHamiltonian,
        batched_rotations: int | None = None,
        rotation_precision: int = 15,
        select_swap_depth: int | None = None,
        wires: WiresLike | None = None,
    ):

        if not isinstance(thc_ham, THCHamiltonian):
            raise TypeError(
                f"Unsupported Hamiltonian representation for SelectTHC."
                f"This method works with thc Hamiltonian, {type(thc_ham)} provided"
            )

        if not isinstance(rotation_precision, int):
            raise TypeError(
                f"`rotation_precision` must be an integer, but type {type(rotation_precision)} was provided."
            )

        if batched_rotations is not None and (
            batched_rotations <= 0 or batched_rotations > thc_ham.num_orbitals - 1
        ):
            raise ValueError(
                f"`batched_rotations` must be a positive integer less than the number of orbitals {thc_ham.num_orbitals}, but got {batched_rotations}."
            )

        self.thc_ham = thc_ham
        self.batched_rotations = batched_rotations
        self.rotation_precision = rotation_precision
        self.select_swap_depth = select_swap_depth
        num_orb = thc_ham.num_orbitals
        tensor_rank = thc_ham.tensor_rank

        # 6 auxiliary wires account for:
        # - 2 spin registers
        # - 1 for rotation on auxiliary qubit
        # - 1 flag for success of inequality
        # - 1 flag for one-body vs two-body rotation
        # - 1 to control swap of \mu and \nu registers.
        # 2*n_M wires are for \mu and \nu registers, where n_M = log_2(tensor_rank+1)
        # coeff_precision wires for the keep register
        # num_orb*2 for state register
        self.num_wires = num_orb * 2 + 2 * int(np.ceil(math.log2(tensor_rank + 1))) + 6

        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")

        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * thc_ham (:class:`~.pennylane.estimator.compact_hamiltonian.THCHamiltonian`): a tensor hypercontracted
                  Hamiltonian on which the select operator is being applied
                * batched_rotations (int | None): The maximum number of rotation angles to load simultaneously
                  into temporary quantum registers for processing in the Givens rotation circuits.
                  The default value of :code:`None` loads all angles at once, where the batch size is equal to
                  the number of orbitals minus one.
                * rotation_precision (int): The number of bits used to represent the precision for loading
                  the rotation angles for basis rotation. The default value is set to ``15`` bits.
                * select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.QROM`
                  used to trade-off extra wires for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.
        """
        return {
            "thc_ham": self.thc_ham,
            "batched_rotations": self.batched_rotations,
            "rotation_precision": self.rotation_precision,
            "select_swap_depth": self.select_swap_depth,
        }

    @classmethod
    def resource_rep(
        cls,
        thc_ham: THCHamiltonian,
        batched_rotations: int | None = None,
        rotation_precision: int = 15,
        select_swap_depth: int | None = None,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            thc_ham (:class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`): A tensor hypercontracted
                Hamiltonian on which the select operator is being applied.
            batched_rotations (int | None): The maximum number of rotation angles to load simultaneously
                into temporary quantum registers for processing in the Givens rotation circuits.
                The default value of :code:`None` loads all angles at once, where the batch size is equal to
                the number of orbitals minus one.
            rotation_precision (int): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation. The default value is set to ``15`` bits.
            select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.QROM`
                used to trade-off extra wires for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """

        if not isinstance(thc_ham, THCHamiltonian):
            raise TypeError(
                f"Unsupported Hamiltonian representation for SelectTHC."
                f"This method works with thc Hamiltonian, {type(thc_ham)} provided"
            )

        if not isinstance(rotation_precision, int):
            raise TypeError(
                f"`rotation_precision` must be an integer, but type {type(rotation_precision)} was provided."
            )

        if batched_rotations is not None and (
            batched_rotations <= 0 or batched_rotations > thc_ham.num_orbitals - 1
        ):
            raise ValueError(
                f"`batched_rotations` must be a positive integer less than the number of orbitals {thc_ham.num_orbitals}, but got {batched_rotations}."
            )

        num_orb = thc_ham.num_orbitals
        tensor_rank = thc_ham.tensor_rank

        # 6 auxiliary wires account for:
        # - 2 spin registers
        # - 1 for rotation on auxiliary qubit
        # - 1 flag for success of inequality
        # - 1 flag for one-body vs two-body rotation
        # - 1 to control swap of \mu and \nu registers.
        # 2*n_M wires are for \mu and \nu registers, where n_M = log_2(tensor_rank+1)
        # coeff_precision wires for the keep register
        # num_orb*2 for state register
        num_wires = num_orb * 2 + 2 * int(np.ceil(math.log2(tensor_rank + 1))) + 6
        params = {
            "thc_ham": thc_ham,
            "batched_rotations": batched_rotations,
            "rotation_precision": rotation_precision,
            "select_swap_depth": select_swap_depth,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        thc_ham: THCHamiltonian,
        batched_rotations: int | None = None,
        rotation_precision: int = 15,
        select_swap_depth: int | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        .. note::

            This decomposition assumes that an appropriately sized phase gradient state is available.
            Users should ensure that the cost of constructing this state has been accounted for.
            See also :class:`~.pennylane.estimator.templates.PhaseGradient`.

        Args:
            thc_ham (:class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`): A tensor hypercontracted
                Hamiltonian on which the select operator is being applied.
            batched_rotations (int | None): The maximum number of rotation angles to load simultaneously
                into temporary quantum registers for processing in the Givens rotation circuits.
                The default value of :code:`None` loads all angles at once, where the batch size is equal to
                the number of orbitals minus one.
            rotation_precision (int): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation. The default value is set to ``15`` bits.
            select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.QROM`
                used to trade-off extra wires for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.

        Resources:
            The resources are calculated based on Figure 5 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_.
            The resources are modified to remove the control from the Select operation.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        num_orb = thc_ham.num_orbitals
        tensor_rank = thc_ham.tensor_rank

        gate_list = []
        # Total select cost from Eq. 43 in arXiv:2011.03494

        # 4 swaps on state registers controlled on spin qubits
        cswap = resource_rep(qre.CSWAP)
        gate_list.append(GateCount(cswap, 4 * num_orb))

        if batched_rotations is None:
            batched_rotations = num_orb - 1

        restore_qrom = True
        if batched_rotations == num_orb - 1:
            restore_qrom = False

        num_givens_blocks = int(np.ceil((num_orb - 1) / batched_rotations))

        # Data output for rotations
        gate_list.append(Allocate(rotation_precision * batched_rotations))
        print(tensor_rank+num_orb, rotation_precision, batched_rotations, restore_qrom, select_swap_depth, num_givens_blocks)

        # QROM to load rotation angles for 2-body integrals
        qrom_twobody = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": tensor_rank + num_orb,
                "size_bitstring": rotation_precision * batched_rotations,
                "restored": restore_qrom,
                "select_swap_depth": select_swap_depth,
            },
        )
        gate_list.append(GateCount(qrom_twobody, num_givens_blocks))

        # Cost for rotations by adding the rotations into the phase gradient state
        semiadder = resource_rep(
            qre.Controlled,
            {
                "base_cmpr_op": resource_rep(
                    qre.SemiAdder,
                    {"max_register_size": rotation_precision - 1},
                ),
                "num_ctrl_wires": 1,
                "num_zero_ctrl": 0,
            },
        )
        gate_list.append(GateCount(semiadder, num_orb-1))

        # Adjoint of QROM for 2-body integrals Eq. 34 in arXiv:2011.03494
        gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": qrom_twobody})))
        # Adjoint of semiadder for 2-body integrals
        gate_list.append(
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": semiadder}), num_orb - 1)
        )

        # QROM to load rotation angles for one body integrals
        qrom_onebody = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": tensor_rank,
                "size_bitstring": rotation_precision * batched_rotations,
                "restored": restore_qrom,
                "select_swap_depth": select_swap_depth,
            },
        )
        gate_list.append(GateCount(qrom_onebody, num_givens_blocks))

        # Cost for rotations by adding the rotations into the phase gradient state
        gate_list.append(GateCount(semiadder, num_orb - 1))

        # Clifford cost for rotations
        h = resource_rep(qre.Hadamard)
        s = resource_rep(qre.S)
        s_dagg = resource_rep(qre.Adjoint, {"base_cmpr_op": s})

        gate_list.append(GateCount(h, 4 * (num_orb)))
        gate_list.append(GateCount(s, 2 * num_orb))
        gate_list.append(GateCount(s_dagg, 2 * num_orb))

        # Adjoint of QROM for one body integrals Eq. 35 in arXiv:2011.03494
        gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": qrom_onebody})))

        # Adjoint of semiadder for one body integrals
        gate_list.append(
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": semiadder}), num_orb - 1)
        )

        # Z gate in the center of rotations
        gate_list.append(qre.GateCount(resource_rep(qre.Z)))

        cz = resource_rep(qre.CZ)
        gate_list.append(qre.GateCount(cz, 1))

        # 1 cswap between the spin registers
        gate_list.append(qre.GateCount(cswap, 1))
        gate_list.append(Deallocate(rotation_precision * batched_rotations))

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
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            The resources are calculated based on Figure 5 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        thc_ham = target_resource_params["thc_ham"]
        rotation_precision = target_resource_params["rotation_precision"]
        select_swap_depth = target_resource_params["select_swap_depth"]
        batched_rotations = target_resource_params["batched_rotations"]

        num_orb = thc_ham.num_orbitals
        tensor_rank = thc_ham.tensor_rank

        gate_list = []

        if num_ctrl_wires > 1:
            mcx = resource_rep(
                qre.MultiControlledX,
                {
                    "num_ctrl_wires": num_ctrl_wires,
                    "num_zero_ctrl": num_zero_ctrl,
                },
            )
            gate_list.append(Allocate(1))
            gate_list.append(GateCount(mcx, 2))

        # 4 swaps on state registers controlled on spin qubits
        cswap = resource_rep(qre.CSWAP)
        gate_list.append(GateCount(cswap, 4 * num_orb))

        if batched_rotations is None:
            batched_rotations = num_orb - 1

        restore_qrom = True
        if batched_rotations == num_orb - 1:
            restore_qrom = False

        num_givens_blocks = int(np.ceil((num_orb - 1) / batched_rotations))

        # Data output for rotations
        gate_list.append(Allocate(rotation_precision * batched_rotations))

        # QROM for loading rotation angles for 2-body integrals
        qrom_twobody = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": tensor_rank + num_orb,
                "size_bitstring": rotation_precision * batched_rotations,
                "restored": restore_qrom,
                "select_swap_depth": select_swap_depth,
            },
        )
        gate_list.append(GateCount(qrom_twobody, num_givens_blocks))

        # Cost for rotations by adding the rotations into the phase gradient state
        semiadder = resource_rep(
            qre.Controlled,
            {
                "base_cmpr_op": resource_rep(
                    qre.SemiAdder,
                    {"max_register_size": rotation_precision - 1},
                ),
                "num_ctrl_wires": 1,
                "num_zero_ctrl": 0,
            },
        )
        gate_list.append(GateCount(semiadder, num_orb - 1))

        # Adjoint of QROM for 2-body integrals Eq. 34 in arXiv:2011.03494
        gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": qrom_twobody})))
        # Adjoint of semiadder for 2-body integrals
        gate_list.append(
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": semiadder}), num_orb - 1)
        )

        # QROM for loading rotation angles for one body integrals
        qrom_onebody = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": tensor_rank,
                "size_bitstring": rotation_precision * batched_rotations,
                "restored": restore_qrom,
                "select_swap_depth": select_swap_depth,
            },
        )
        gate_list.append(GateCount(qrom_onebody, num_givens_blocks))

        # Cost for rotations by adding the rotations into the phase gradient state
        gate_list.append(GateCount(semiadder, num_orb - 1))

        # Clifford cost for rotations
        h = resource_rep(qre.Hadamard)
        s = resource_rep(qre.S)
        s_dagg = resource_rep(qre.Adjoint, {"base_cmpr_op": s})

        gate_list.append(GateCount(h, 4 * (num_orb)))
        gate_list.append(GateCount(s, 2 * num_orb))
        gate_list.append(GateCount(s_dagg, 2 * num_orb))

        # Adjoint of QROM for one body integrals Eq. 35 in arXiv:2011.03494
        gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": qrom_onebody})))
        # Adjoint of semiadder for one body integrals
        gate_list.append(
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": semiadder}), num_orb - 1)
        )

        # Z gate in the center of rotations
        cz = resource_rep(qre.CZ)
        gate_list.append(qre.GateCount(cz, 1))

        ccz = resource_rep(
            qre.Controlled,
            {
                "base_cmpr_op": qre.Z.resource_rep(),
                "num_ctrl_wires": 2,
                "num_zero_ctrl": 1,
            },
        )
        gate_list.append(qre.GateCount(ccz, 1))

        # 1 cswap between the spin registers
        gate_list.append(qre.GateCount(cswap, 1))

        gate_list.append(Deallocate(rotation_precision * batched_rotations))

        if num_ctrl_wires > 1:
            gate_list.append(Deallocate(1))
        elif num_zero_ctrl > 0:
            gate_list.append(GateCount(resource_rep(qre.X), 2 * num_zero_ctrl))

        return gate_list
