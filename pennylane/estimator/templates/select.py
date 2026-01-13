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
from collections import defaultdict

import numpy as np

import pennylane.estimator as qre
from pennylane.estimator import Allocate, Deallocate
from pennylane.estimator.compact_hamiltonian import PauliHamiltonian, THCHamiltonian
from pennylane.estimator.resource_operator import (
    GateCount,
    ResourceOperator,
    resource_rep,
)
from pennylane.wires import Wires, WiresLike

# pylint: disable=arguments-differ,super-init-not-called, signature-differs


class SelectTHC(ResourceOperator):
    r"""Resource class for creating the custom ``Select`` operator for tensor hypercontracted (THC)
    Hamiltonian.

    .. note::

            This decomposition assumes that an appropriately sized phase gradient state is available.
            Users should ensure that the cost of constructing this state has been accounted for.
            See also :class:`~.pennylane.estimator.templates.subroutines.PhaseGradient`.

    Args:
        thc_ham (:class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`): A tensor hypercontracted
            Hamiltonian on which this ``Select`` operator is being applied.
        num_batches (int): The number of batches for loading Givens rotation angles
            into temporary quantum registers. Must be less than the number of orbitals in ``thc_ham``.
            The default value of ``1`` loads all angles in one batch.
        rotation_precision (int): The number of bits used to represent the precision for loading
            the rotation angles for basis rotation. The default value is set to ``15`` bits.
        select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.subroutines.QROM`
            used to trade off extra wires for reduced circuit depth. Defaults to :code:`None`, in which
            case, the ``select_swap_depth`` is set to the optimal depth that minimizes the total
            ``T``-gate count.
        wires (WiresLike | None): the wires on which the operator acts

    Raises:
        TypeError: If ``thc_ham`` is not a :class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`.
        TypeError: If ``rotation_precision`` is not a positive integer.
        ValueError: If ``num_batches`` is not a positive integer or is greater than or equal
            to the number of orbitals in ``thc_ham``.
        ValueError: If the number of provided ``wires`` does not match the calculated
            ``num_wires`` required for the operation.

    Resources:
        The resources are calculated based on Figure 5 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_ and
        Figure 4 in `arXiv:2501.06165 <https://arxiv.org/abs/2501.06165>`_.

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> thc_ham = qre.THCHamiltonian(num_orbitals=20, tensor_rank=40)
    >>> res = qre.estimate(qre.SelectTHC(thc_ham, rotation_precision=15))
    >>> print(res)
    --- Resources: ---
     Total wires: 356
       algorithmic wires: 58
       allocated wires: 298
         zero state: 298
         any state: 0
     Total gates : 3.336E+4
       'Toffoli': 2.249E+3,
       'CNOT': 2.344E+4,
       'X': 392,
       'Z': 41,
       'S': 80,
       'Hadamard': 7.160E+3

    Let's also see how the resources change when more batches are used for the rotations:

    >>> res = qre.estimate(qre.SelectTHC(thc_ham, num_batches=2, rotation_precision=15))
    >>> print(res)
    --- Resources: ---
     Total wires: 221
       algorithmic wires: 58
       allocated wires: 163
         zero state: 163
         any state: 0
     Total gates : 3.461E+4
       'Toffoli': 2.345E+3,
       'CNOT': 2.438E+4,
       'X': 582,
       'Z': 45,
       'S': 80,
       'Hadamard': 7.178E+3

    We can see that by batching the rotations, the number of allocated wires decreases
    at the cost of an increased number of Toffoli gates.

    """

    resource_keys = {"thc_ham", "num_batches", "rotation_precision", "select_swap_depth"}

    def __init__(  # pylint: disable=too-many-arguments
        self,
        thc_ham: THCHamiltonian,
        num_batches: int = 1,
        rotation_precision: int = 15,
        select_swap_depth: int | None = None,
        wires: WiresLike | None = None,
    ):

        if not isinstance(thc_ham, THCHamiltonian):
            raise TypeError(
                f"Unsupported Hamiltonian representation for SelectTHC."
                f"This method works with thc Hamiltonian, {type(thc_ham)} provided"
            )

        if not isinstance(rotation_precision, int) or rotation_precision <= 0:
            raise TypeError(
                f"`rotation_precision` must be a positive integer, but got {rotation_precision}."
            )

        if not isinstance(num_batches, int) or num_batches not in range(1, thc_ham.num_orbitals):
            raise ValueError(
                f"`num_batches` must be a positive integer less than the number of orbitals ({thc_ham.num_orbitals}), but got {num_batches}."
            )

        self.thc_ham = thc_ham
        self.num_batches = num_batches
        self.rotation_precision = rotation_precision
        self.select_swap_depth = select_swap_depth
        num_orb = thc_ham.num_orbitals
        tensor_rank = thc_ham.tensor_rank

        # Based on section III D in arXiv:2011.03494
        # Algorithmic wires for the walk operator, auxiliary wires are accounted for by the QROM
        # and SemiAdder operators.
        # 2*n_M wires are for \mu and \nu registers, where n_M = log_2(tensor_rank+1)
        # num_orb*2 for state register and 6 are flags.
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
                  Hamiltonian on which this ``Select`` operator is being applied
                * num_batches (int): The number of batches for loading Givens rotation angles
                  into temporary quantum registers. Must be less than the number of orbitals in ``thc_ham``.
                  The default value of ``1`` loads all angles in one batch.
                * rotation_precision (int): The number of bits used to represent the precision for loading
                  the rotation angles for basis rotation. The default value is set to ``15`` bits.
                * select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.subroutines.QROM`
                  used to trade off extra wires for reduced circuit depth. Defaults to :code:`None`, in which
                  case, the ``select_swap_depth`` is set to the optimal depth that minimizes the total
                  ``T``-gate count.
        """
        return {
            "thc_ham": self.thc_ham,
            "num_batches": self.num_batches,
            "rotation_precision": self.rotation_precision,
            "select_swap_depth": self.select_swap_depth,
        }

    @classmethod
    def resource_rep(
        cls,
        thc_ham: THCHamiltonian,
        num_batches: int = 1,
        rotation_precision: int = 15,
        select_swap_depth: int | None = None,
    ) -> ResourceOperator:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            thc_ham (:class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`): A tensor hypercontracted
                Hamiltonian on which this ``Select`` operator is being applied.
            num_batches (int): The number of batches for loading Givens rotation angles
                into temporary quantum registers. Must be less than the number of orbitals in ``thc_ham``.
                The default value of ``1`` loads all angles in one batch.
            rotation_precision (int): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation. The default value is set to ``15`` bits.
            select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.subroutines.QROM`
                used to trade off extra wires for reduced circuit depth. Defaults to :code:`None`, in which
                case, the ``select_swap_depth`` is set to the optimal depth that minimizes the total
                ``T``-gate count.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.ResourceOperator`: the operator in a compressed representation
        """

        if not isinstance(thc_ham, THCHamiltonian):
            raise TypeError(
                f"Unsupported Hamiltonian representation for SelectTHC."
                f"This method works with thc Hamiltonian, {type(thc_ham)} provided"
            )

        if not isinstance(rotation_precision, int) or rotation_precision <= 0:
            raise TypeError(
                f"`rotation_precision` must be a positive integer, but got {rotation_precision}."
            )

        if not isinstance(num_batches, int) or num_batches not in range(1, thc_ham.num_orbitals):
            raise ValueError(
                f"`num_batches` must be a positive integer less than the number of orbitals ({thc_ham.num_orbitals}), but got {num_batches}."
            )

        num_orb = thc_ham.num_orbitals
        tensor_rank = thc_ham.tensor_rank

        num_wires = num_orb * 2 + 2 * int(np.ceil(math.log2(tensor_rank + 1))) + 6
        params = {
            "thc_ham": thc_ham,
            "num_batches": num_batches,
            "rotation_precision": rotation_precision,
            "select_swap_depth": select_swap_depth,
        }
        return ResourceOperator(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        thc_ham: THCHamiltonian,
        num_batches: int = 1,
        rotation_precision: int = 15,
        select_swap_depth: int | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        .. note::

            This decomposition assumes that an appropriately sized phase gradient state is available.
            Users should ensure that the cost of constructing this state has been accounted for.
            See also :class:`~.pennylane.estimator.templates.subroutines.PhaseGradient`.

        Args:
            thc_ham (:class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`): A tensor hypercontracted
                Hamiltonian on which this ``Select`` operator is being applied.
            num_batches (int): The number of batches for loading Givens rotation angles
                into temporary quantum registers. Must be less than the number of orbitals in ``thc_ham``.
                The default value of ``1`` loads all angles in one batch.
            rotation_precision (int): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation. The default value is set to ``15`` bits.
            select_swap_depth (int | None): A parameter of :class:`~.pennylane.estimator.templates.subroutines.QROM`
                used to trade off extra wires for reduced circuit depth. Defaults to :code:`None`, in which
                case, the ``select_swap_depth`` is set to the optimal depth that minimizes the total
                ``T``-gate count.

        Resources:
            The resources are calculated based on Figure 5 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_ and
            Figure 4 in `arXiv:2501.06165 <https://arxiv.org/abs/2501.06165>`_.

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

        restore_qrom = num_batches != 1

        batched_rotations = int(np.ceil((num_orb - 1) / num_batches))

        # Data output for rotations
        gate_list.append(Allocate(rotation_precision * batched_rotations))

        # QROM to load rotation angles for both 1-body and 2-body integrals
        qrom_full = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": tensor_rank + num_orb,
                "size_bitstring": rotation_precision * batched_rotations,
                "restored": restore_qrom,
                "select_swap_depth": select_swap_depth,
            },
        )
        gate_list.append(GateCount(qrom_full, num_batches))

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

        # Adjoint of QROM for loading both 1-body and 2-body integrals Eq. 34 in arXiv:2011.03494
        gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": qrom_full})))
        # Adjoint of semiadder for 1-body and 2-body integrals
        gate_list.append(
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": semiadder}), num_orb - 1)
        )

        # QROM to load rotation angles for two body integrals
        qrom_twobody = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": tensor_rank,
                "size_bitstring": rotation_precision * batched_rotations,
                "restored": restore_qrom,
                "select_swap_depth": select_swap_depth,
            },
        )
        gate_list.append(GateCount(qrom_twobody, num_batches))

        # Cost for rotations by adding the rotations into the phase gradient state
        gate_list.append(GateCount(semiadder, num_orb - 1))

        # Clifford cost for rotations
        h = resource_rep(qre.Hadamard)
        s = resource_rep(qre.S)
        s_dagg = resource_rep(qre.Adjoint, {"base_cmpr_op": s})

        gate_list.append(GateCount(h, 4 * (num_orb)))
        gate_list.append(GateCount(s, 2 * num_orb))
        gate_list.append(GateCount(s_dagg, 2 * num_orb))

        # Adjoint of QROM for two body integrals Eq. 35 in arXiv:2011.03494
        gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": qrom_twobody})))

        # Adjoint of semiadder for two body integrals
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
            See also :class:`~.pennylane.estimator.templates.subroutines.PhaseGradient`.

        Args:
            num_ctrl_wires (int): the number of wires the operation is controlled on
            num_zero_ctrl (int): the number of control wires, that are controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): A dictionary containing the resource parameters of the target operator.

        Resources:
            The resources are calculated based on Figure 5 in `arXiv:2011.03494 <https://arxiv.org/abs/2011.03494>`_.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """

        thc_ham = target_resource_params["thc_ham"]
        rotation_precision = target_resource_params["rotation_precision"]
        select_swap_depth = target_resource_params["select_swap_depth"]
        num_batches = target_resource_params["num_batches"]

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

        restore_qrom = num_batches != 1

        batched_rotations = int(np.ceil((num_orb - 1) / num_batches))

        # Data output for rotations
        gate_list.append(Allocate(rotation_precision * batched_rotations))

        # QROM for loading rotation angles for 1-body and 2-body integrals
        qrom_full = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": tensor_rank + num_orb,
                "size_bitstring": rotation_precision * batched_rotations,
                "restored": restore_qrom,
                "select_swap_depth": select_swap_depth,
            },
        )
        gate_list.append(GateCount(qrom_full, num_batches))

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

        # Adjoint of QROM for 1-body and 2-body integrals Eq. 34 in arXiv:2011.03494
        gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": qrom_full})))
        # Adjoint of semiadder for 1-body and 2-body integrals
        gate_list.append(
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": semiadder}), num_orb - 1)
        )

        # QROM for loading rotation angles for two body integrals
        qrom_twobody = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": tensor_rank,
                "size_bitstring": rotation_precision * batched_rotations,
                "restored": restore_qrom,
                "select_swap_depth": select_swap_depth,
            },
        )
        gate_list.append(GateCount(qrom_twobody, num_batches))

        # Cost for rotations by adding the rotations into the phase gradient state
        gate_list.append(GateCount(semiadder, num_orb - 1))

        # Clifford cost for rotations
        h = resource_rep(qre.Hadamard)
        s = resource_rep(qre.S)
        s_dagg = resource_rep(qre.Adjoint, {"base_cmpr_op": s})

        gate_list.append(GateCount(h, 4 * (num_orb)))
        gate_list.append(GateCount(s, 2 * num_orb))
        gate_list.append(GateCount(s_dagg, 2 * num_orb))

        # Adjoint of QROM for two body integrals Eq. 35 in arXiv:2011.03494
        gate_list.append(GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": qrom_twobody})))
        # Adjoint of semiadder for two body integrals
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


class SelectPauli(ResourceOperator):
    r"""Resource class for the ``Select`` opreation used with a Hamiltonian expressed as a linear
    combination of unitaries (LCU) where each unitary is a Pauli word.

    Args:
        pauli_ham (:class:`~pennylane.estimator.compact_hamiltonian.PauliHamiltonian`): A Hamiltonian
            expressed as a linear combination of Pauli words, over which ``Select`` is applied.
        wires (WiresLike | None): the wires the operation acts on

    Resources:
        The resources are based on the analysis in `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_, Section III.A,
        'Unary Iteration and Indexed Operations', and Figures 4, 6, and 7.

        Note: This implementation assumes we have access to :math:`n - 1` additional auxiliary qubits,
        where :math:`n = \left\lceil log_{2}(N) \right\rceil` and :math:`N` is the number of batches of unitaries
        to select.

    Raises:
        TypeError: If the input ``pauli_ham`` isn't an instance of
            :class:`~pennylane.estimator.compact_hamiltonian.PauliHamiltonian`.
        ValueError: if the wires provided don't match the number of wires expected by the operator

    .. seealso:: :class:`~.pennylane.Select`

    **Example**

    The resources for this operation are computed using:

    >>> import pennylane.estimator as qre
    >>> pauli_ham = qre.PauliHamiltonian(num_qubits=4, pauli_terms={"XY": 1, "Z": 2})
    >>> select_pauli = qre.SelectPauli(pauli_ham)
    >>> print(qre.estimate(select_pauli))
    --- Resources: ---
     Total wires: 7
       algorithmic wires: 6
       allocated wires: 1
         zero state: 1
         any state: 0
     Total gates : 27
       'Toffoli': 2,
       'CNOT': 8,
       'X': 4,
       'Z': 1,
       'S': 2,
       'Hadamard': 10
    """

    resource_keys = {"pauli_ham"}

    def __init__(self, pauli_ham: PauliHamiltonian, wires: WiresLike = None) -> None:
        self.queue()

        if not isinstance(pauli_ham, PauliHamiltonian):
            raise TypeError(
                f"'pauli_ham' must be an instance of PauliHamiltonian, got {type(pauli_ham)}"
            )
        self.pauli_ham = pauli_ham

        num_ctrl_wires = math.ceil(math.log2(pauli_ham.num_terms))
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
    def resource_decomp(
        cls, pauli_ham: PauliHamiltonian
    ) -> list[GateCount]:  # pylint: disable=unused-argument
        r"""The resources for a select implementation taking advantage of the unary iterator trick.

        Args:
            pauli_ham (:class:`~pennylane.estimator.compact_hamiltonian.PauliHamiltonian`): A Hamiltonian
                expressed as a linear combination of Pauli words, over which ``Select`` is applied.

        Resources:
            The resources are based on the analysis in `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_, Section III.A,
            'Unary Iteration and Indexed Operations', and Figures 4, 6, and 7.

            Note: This implementation assumes we have access to :math:`n - 1` additional auxiliary qubits,
            where :math:`n = \left\lceil log_{2}(N) \right\rceil` and :math:`N` is the number of batches of unitaries
            to select.

        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        gate_types = []
        x = qre.X.resource_rep()
        cnot = qre.CNOT.resource_rep()
        l_elbow = resource_rep(qre.TemporaryAND)
        r_elbow = resource_rep(qre.Adjoint, {"base_cmpr_op": l_elbow})

        pauli_terms = pauli_ham.pauli_terms
        if not isinstance(pauli_terms, dict):
            commuting_groups = pauli_terms
            pauli_terms = defaultdict(int)
            for group in commuting_groups:
                for pw, freq in group.items():
                    pauli_terms[pw] += freq

        num_ops = pauli_ham.num_terms
        work_qubits = math.ceil(math.log2(num_ops)) - 1

        gate_types.append(Allocate(work_qubits))

        cz = qre.CZ.resource_rep()
        cy = qre.CY.resource_rep()
        op_counts = [0, 0, 0]  # cx, cy, cz counts

        for pw, freq in pauli_terms.items():
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
    def adjoint_resource_decomp(cls, target_resource_params: dict) -> list[GateCount]:
        r"""Returns a list representing the resources for the adjoint of the operator.

        Args:
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            Because each target operation is self-adjoint, the resources of the adjoint operation are
            the same as the original operation (up to some re-ordering of the application of the gates).

        Returns:
            list[:class:`~.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        return [GateCount(cls.resource_rep(**target_resource_params))]

    @classmethod
    def controlled_resource_decomp(
        cls, num_ctrl_wires: int, num_zero_ctrl: int, target_resource_params: dict
    ) -> list[GateCount]:
        r"""Returns a list representing the resources for a controlled version of the operator.

        Args:
            num_ctrl_wires (int): the number of qubits the
                operation is controlled on
            num_zero_ctrl (int): the number of control qubits, that are
                controlled when in the :math:`|0\rangle` state
            target_resource_params (dict): A dictionary containing the resource parameters
                of the target operator.

        Resources:
            The resources are based on the analysis in `Babbush et al. (2018) <https://arxiv.org/pdf/1805.03662>`_, Section III.A,
            'Unary Iteration and Indexed Operations'. See Figures 4, 6, and 7. This presents the cost of
            a single qubit controlled ``Select`` operator. In the case of multiple control wires, we use one
            additional auxiliary qubit and two multi-controlled ``X`` gates.

            Note: This implementation assumes we have access to :math:`n` additional auxiliary qubits,
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
        pauli_terms = pauli_ham.pauli_terms
        if not isinstance(pauli_terms, dict):
            commuting_groups = pauli_terms
            pauli_terms = defaultdict(int)
            for group in commuting_groups:
                for pw, freq in group.items():
                    pauli_terms[pw] += freq

        num_ops = pauli_ham.num_terms
        work_qubits = math.ceil(math.log2(num_ops))

        gate_types.append(Allocate(work_qubits))

        cz = qre.CZ.resource_rep()
        cy = qre.CY.resource_rep()
        op_counts = [0, 0, 0]  # cx, cy, cz counts

        for pw, freq in pauli_terms.items():
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
                  Hamiltonian expressed as a linear combination of Pauli words, over which ``Select``
                  is applied.

        """
        return {"pauli_ham": self.pauli_ham}

    @classmethod
    def resource_rep(cls, pauli_ham: PauliHamiltonian) -> ResourceOperator:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute the resources.

        Args:
            pauli_ham (:class:`~pennylane.estimator.compact_hamiltonian.PauliHamiltonian`): A Hamiltonian
                expressed as a linear combination of Pauli words, over which ``Select`` is applied.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.ResourceOperator`: the operator in a compressed representation
        """
        num_ctrl_wires = math.ceil(math.log2(pauli_ham.num_terms))
        num_wires = pauli_ham.num_qubits + num_ctrl_wires
        params = {"pauli_ham": pauli_ham}
        return ResourceOperator(cls, num_wires, params)
