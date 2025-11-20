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
from pennylane.estimator.compact_hamiltonian import PauliHamiltonian, THCHamiltonian
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
        rotation_precision (int | None): The number of bits used to represent the precision for loading
            the rotation angles for basis rotation. If :code:`None` is provided, the default value from the
            :class:`~.pennylane.estimator.resource_config.ResourceConfig` is used.
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

    """

    resource_keys = {"thc_ham", "rotation_precision", "select_swap_depth"}

    def __init__(
        self,
        thc_ham: THCHamiltonian,
        rotation_precision: int | None = None,
        select_swap_depth: int | None = None,
        wires: WiresLike | None = None,
    ):

        if not isinstance(thc_ham, THCHamiltonian):
            raise TypeError(
                f"Unsupported Hamiltonian representation for SelectTHC."
                f"This method works with thc Hamiltonian, {type(thc_ham)} provided"
            )

        if not (isinstance(rotation_precision, int) or rotation_precision is None):
            raise TypeError(
                f"`rotation_precision` must be an integer, but type {type(rotation_precision)} was provided."
            )

        self.thc_ham = thc_ham
        self.rotation_precision = rotation_precision
        self.select_swap_depth = select_swap_depth
        num_orb = thc_ham.num_orbitals
        tensor_rank = thc_ham.tensor_rank

        # num_orb*2 for state register, 2*log(M) for \mu and \nu registers
        # 6 extras are for 2 spin registers, 1 for rotation on ancilla, 1 flag for success of inequality,
        # 1 flag for one-body vs two-body and 1 to control swap of \mu and \nu registers.
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
                * rotation_precision (int | None): The number of bits used to represent the precision for loading
                  the rotation angles for basis rotation. If :code:`None` is provided, the default value from the
                  :class:`~.pennylane.estimator.resource_config.ResourceConfig` is used.
                * select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.QROM`
                  used to trade-off extra wires for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.
        """
        return {
            "thc_ham": self.thc_ham,
            "rotation_precision": self.rotation_precision,
            "select_swap_depth": self.select_swap_depth,
        }

    @classmethod
    def resource_rep(
        cls,
        thc_ham: THCHamiltonian,
        rotation_precision: int | None = None,
        select_swap_depth: int | None = None,
    ) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            thc_ham (:class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`): A tensor hypercontracted
                Hamiltonian on which the select operator is being applied.
            rotation_precision (int | None): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation. If :code:`None` is provided, the default value from the
                :class:`~.pennylane.estimator.resource_config.ResourceConfig` is used.
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

        if not isinstance(rotation_precision, int) and rotation_precision is not None:
            raise TypeError(
                f"`rotation_precision` must be an integer, but type {type(rotation_precision)} was provided."
            )
        num_orb = thc_ham.num_orbitals
        tensor_rank = thc_ham.tensor_rank

        # num_orb*2 for state register, 2*log(M) for \mu and \nu registers
        # 6 extras are for 2 spin registers, 1 for rotation on ancilla, 1 flag for success of inequality,
        # 1 flag for one-body vs two-body and 1 to control swap of \mu and \nu registers.
        num_wires = num_orb * 2 + 2 * int(np.ceil(math.log2(tensor_rank + 1))) + 6
        params = {
            "thc_ham": thc_ham,
            "rotation_precision": rotation_precision,
            "select_swap_depth": select_swap_depth,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        thc_ham,
        rotation_precision: int | None = None,
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
            rotation_precision (int | None): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation. If :code:`None` is provided, the default value from the
                :class:`~.pennylane.estimator.resource_config.ResourceConfig` is used.
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

        # Data output for rotations
        gate_list.append(Allocate(rotation_precision * (num_orb - 1)))

        # QROM to load rotation angles for 2-body integrals
        qrom_twobody = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": tensor_rank + num_orb,
                "size_bitstring": rotation_precision,
                "restored": False,
                "select_swap_depth": select_swap_depth,
            },
        )
        gate_list.append(GateCount(qrom_twobody))

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

        # QROM to load rotation angles for one body integrals
        qrom_onebody = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": tensor_rank,
                "size_bitstring": rotation_precision,
                "restored": False,
                "select_swap_depth": select_swap_depth,
            },
        )
        gate_list.append(GateCount(qrom_onebody))

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
        gate_list.append(Deallocate(rotation_precision * (num_orb - 1)))

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

        # Data output for rotations
        gate_list.append(Allocate(rotation_precision * (num_orb - 1)))

        # QROM for loading rotation angles for 2-body integrals
        qrom_twobody = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": tensor_rank + num_orb,
                "size_bitstring": rotation_precision,
                "restored": False,
                "select_swap_depth": select_swap_depth,
            },
        )
        gate_list.append(GateCount(qrom_twobody))

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
                "size_bitstring": rotation_precision,
                "restored": False,
                "select_swap_depth": select_swap_depth,
            },
        )
        gate_list.append(GateCount(qrom_onebody))

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

        gate_list.append(Deallocate(rotation_precision * (num_orb - 1)))

        if num_ctrl_wires > 1:
            gate_list.append(Deallocate(1))
        elif num_zero_ctrl > 0:
            gate_list.append(GateCount(resource_rep(qre.X), 2 * num_zero_ctrl))

        return gate_list


class SelectPauli(ResourceOperator):
    r"""Resource class for the Select subroutine, when working with a hamiltonian expressed as a linear
    combination of unitaries (LCU) where each unitary is a Pauli word.

    Args:
        pauli_ham (:class:`~pennylane.estimator.compact_hamiltonian.PauliHamiltonian`): A hamiltonian
            expressed as a linear combination of Pauli words, over which we will apply Select.
        wires (Sequence[int], None): The wires the operation acts on.

    Resources:
        The resources are based on the analysis in `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_ section III.A,
        'Unary Iteration and Indexed Operations'. See Figures 4, 6, and 7.

    .. seealso:: :class:`~.pennylane.Select`, :class:`~.pennylane.estimator.subroutines.Select`

    **Example**

    The resources for this operation are computed below. Note each of the 25 Pauli strings is
    treated as an independant term, and thus the Select operation has :math:`L = 25` target operators
    to apply (see figure 7 of `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_).

    >>> import pennylane.estimator as qre
    >>> pauli_terms = {"XX":10, "ZZ":10, "Y":5}
    >>> pauli_ham = qre.PauliHamiltonian(num_qubits=5, pauli_dist=pauli_terms)
    >>> pauli_ham
    PauliHamiltonian(num_qubits=5, num_pauli_words=25, max_weight=2)
    >>> res = qre.estimate(qre.SelectPauli(pauli_ham))
    >>> print(res)
    --- Resources: ---
     Total wires: 14
       algorithmic wires: 10
       allocated wires: 4
         zero state: 4
         any state: 0
     Total gates : 292
       'Toffoli': 24,
       'CNOT': 93,
       'X': 48,
       'Z': 5,
       'S': 10,
       'Hadamard': 112

    """

    resource_keys = {"pauli_ham"}

    def __init__(self, pauli_ham: PauliHamiltonian, wires: WiresLike = None) -> None:
        self.queue()
        self.pauli_ham = pauli_ham

        num_select_ops = pauli_ham.num_pauli_words
        num_ctrl_wires = math.ceil(math.log2(num_select_ops))

        num_wires = pauli_ham.num_qubits + num_ctrl_wires

        if wires:
            self.wires = Wires(wires)
            if len(self.wires) != num_wires:
                raise ValueError(
                    f"Expected {num_wires} wires ({num_ctrl_wires} control + {pauli_ham.num_qubits} target), got {len(self.wires)}."
                )
            self.num_wires = num_wires
        else:
            self.wires = None
            self.num_wires = num_wires

    @classmethod
    def resource_decomp(cls, pauli_ham: PauliHamiltonian):  # pylint: disable=unused-argument
        r"""The resources for a select implementation taking advantage of the unary iterator trick.

        Args:
            pauli_ham (:class:`~pennylane.estimator.compact_hamiltonian.PauliHamiltonian`): A hamiltonian
            expressed as a linear combination of Pauli words, over which we will apply Select.

        Resources:
            The resources are based on the analysis in `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_ section III.A,
            'Unary Iteration and Indexed Operations'. See Figures 4, 6, and 7.

            Note: This implementation assumes we have access to :math:`n - 1` additional work qubits,
            where :math:`n = \left\lceil log_{2}(N) \right\rceil` and :math:`N` is the number of batches of unitaries
            to select.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_types = []
        x = qre.X.resource_rep()
        cnot = qre.CNOT.resource_rep()
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})

        num_ops = pauli_ham.num_pauli_words
        work_qubits = math.ceil(math.log2(num_ops)) - 1

        gate_types.append(Allocate(work_qubits))

        cz = qre.CZ.resource_rep()
        cy = qre.CY.resource_rep()
        op_counts = [0, 0, 0]  # cx, cy, cz counts

        if (pauli_dist := pauli_ham.pauli_dist) is not None:
            for pw, freq in pauli_dist.items():
                x_count, y_count, z_count = (0, 0, 0)
                for pauli_op in pw:
                    if pauli_op == "X":
                        x_count += 1
                    elif pauli_op == "Y":
                        y_count += 1
                    else:
                        z_count += 1

                op_counts[0] += x_count * freq
                op_counts[1] += y_count * freq
                op_counts[2] += z_count * freq
        else:
            op_counts[0] = pauli_ham.max_weight * (pauli_ham.num_pauli_words // 3)
            op_counts[1] = pauli_ham.max_weight * (pauli_ham.num_pauli_words // 3)
            op_counts[2] = pauli_ham.max_weight * (
                (pauli_ham.num_pauli_words // 3) + (pauli_ham.num_pauli_words % 3)
            )

        gate_types.append(GateCount(cnot, op_counts[0]))
        gate_types.append(GateCount(cy, op_counts[1]))
        gate_types.append(GateCount(cz, op_counts[2]))

        gate_types.append(GateCount(x, 2 * (num_ops - 1)))  # conjugate 0 controlled toffolis
        gate_types.append(GateCount(cnot, num_ops - 1))
        gate_types.append(GateCount(l_elbow, num_ops - 1))
        gate_types.append(GateCount(r_elbow, num_ops - 1))

        gate_types.append(Deallocate(work_qubits))
        return gate_types

    @classmethod
    def adjoint_resource_decomp(cls, target_resource_params=None):
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            Because each target operation is self-adjoint, the resources of the adjoint operation results
            are same as the original operation (up to some re-ordering of the application of the gates).

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(**target_resource_params))]

    @classmethod
    def controlled_resource_decomp(cls, num_ctrl_wires, num_zero_ctrl, target_resource_params=None):
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the
                operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are
                controlled when in the :math:`|0\rangle` state
            target_resource_params (dict | None): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are based on the analysis in `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_ section III.A,
            'Unary Iteration and Indexed Operations'. See Figures 4, 6, and 7. This presents the cost of
            a single qubit controlled Select operator. In the case of multiple control wires, we use one
            additional auxiliary qubit and two multi-controlled X gates.

            Note: This implementation assumes we have access to :math:`n` additional work qubits,
            where :math:`n = \left\lceil log_{2}(N) \right\rceil` and :math:`N` is the number of batches of unitaries
            to select.

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_types = []
        pauli_ham = target_resource_params["pauli_ham"]

        x = qre.X.resource_rep()
        cnot = qre.CNOT.resource_rep()
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})

        if num_ctrl_wires > 1:
            mcx = qre.MultiControlledX.resource_rep(num_ctrl_wires, num_zero_ctrl)
            gate_types.append(Allocate(1))
            gate_types.append(GateCount(mcx))

        else:
            if num_zero_ctrl == 1:
                gate_types.append(GateCount(x, 2))

        # Cost Single-Controlled Select (Unary Iterator)
        num_ops = pauli_ham.num_pauli_words
        work_qubits = math.ceil(math.log2(num_ops)) - 1

        gate_types.append(Allocate(work_qubits))

        cz = qre.CZ.resource_rep()
        cy = qre.CY.resource_rep()
        op_counts = [0, 0, 0]  # cx, cy, cz counts

        if (pauli_dist := pauli_ham.pauli_dist) is not None:
            for pw, freq in pauli_dist.items():
                x_count, y_count, z_count = (0, 0, 0)
                for pauli_op in pw:
                    if pauli_op == "X":
                        x_count += 1
                    elif pauli_op == "Y":
                        y_count += 1
                    else:
                        z_count += 1

                op_counts[0] += x_count * freq
                op_counts[1] += y_count * freq
                op_counts[2] += z_count * freq
        else:
            op_counts[0] = pauli_ham.max_weight * (pauli_ham.num_pauli_words // 3)
            op_counts[1] = pauli_ham.max_weight * (pauli_ham.num_pauli_words // 3)
            op_counts[2] = pauli_ham.max_weight * (
                (pauli_ham.num_pauli_words // 3) + (pauli_ham.num_pauli_words % 3)
            )

        gate_types.append(GateCount(cnot, op_counts[0]))
        gate_types.append(GateCount(cy, op_counts[1]))
        gate_types.append(GateCount(cz, op_counts[2]))

        gate_types.append(GateCount(x, 2 * (num_ops - 1)))  # conjugate 0 controlled toffolis
        gate_types.append(GateCount(cnot, num_ops - 1))
        gate_types.append(GateCount(l_elbow, num_ops - 1))
        gate_types.append(GateCount(r_elbow, num_ops - 1))

        gate_types.append(Deallocate(work_qubits))

        # Clean up controls:
        if num_ctrl_wires > 1:
            gate_types.append(GateCount(mcx))
            gate_types.append(Deallocate(1))

        return gate_types

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * pauli_ham (:class:`~pennylane.estimator.compact_hamiltonian.PauliHamiltonian`): A
                  hamiltonian expressed as a linear combination of Pauli words, over which we will apply
                  Select.

        """
        return {"pauli_ham": self.pauli_ham}

    @classmethod
    def resource_rep(cls, pauli_ham: PauliHamiltonian) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            pauli_ham (:class:`~pennylane.estimator.compact_hamiltonian.PauliHamiltonian`): A hamiltonian
            expressed as a linear combination of Pauli words, over which we will apply Select.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        num_select_ops = pauli_ham.num_pauli_words
        num_ctrl_wires = math.ceil(math.log2(num_select_ops))

        num_wires = pauli_ham.num_qubits + num_ctrl_wires
        params = {"pauli_ham": pauli_ham}
        return CompressedResourceOp(cls, num_wires, params)
