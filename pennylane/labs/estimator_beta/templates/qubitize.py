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

import numpy as np


import pennylane.labs.estimator_beta as qre
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)

from pennylane.labs.estimator_beta import Allocate, Deallocate
from pennylane.labs.estimator_beta.compact_hamiltonian import DFHamiltonian
from pennylane.wires import Wires, WiresLike

# pylint: disable=signature-differs, arguments-differ, too-many-arguments


class QubitizeDF(ResourceOperator):
    r"""Resource class for qubitization of tensor hypercontracted Hamiltonian.

    .. note::

            This decomposition assumes that an appropriately sized phase gradient state is available.
            Users should ensure that the cost of constructing this state has been accounted for.
            See also :class:`~.pennylane.estimator.templates.PhaseGradient`.

    Args:
        df_ham (:class:`~.pennylane.estimator.compact_hamiltonian.DFHamiltonian`): A tensor hypercontracted
            Hamiltonian for which the walk operator is being created.
        amplitude_amplification_precision (int | None): The number of bits used to represent the precision for single
            qubit rotation in amplitude amplification in outer and inner prep.
        coeff_precision (int | None): The number of bits used to represent the precision for loading
            the coefficients of Hamiltonian.
        rotation_precision (int | None): The number of bits used to represent the precision for loading
            the rotation angles for :code:`select_op`.
        select_swap_depths (int | None): A parameter of :class:`~.pennylane.estimator.templates.subroutines.QROM`
            used to trade-off extra wires for reduced circuit depth. Defaults to :code:`None`, which internally determines the optimal depth.
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
     Total wires: 381
        algorithmic wires: 68
        allocated wires: 313
             zero state: 313
             any state: 0
     Total gates : 5.628E+4
      'Toffoli': 3.504E+3,
      'CNOT': 4.138E+4,
      'X': 2.071E+3,
      'Z': 41,
      'S': 80,
      'Hadamard': 9.213E+3

    .. details::
        :title: Usage Details

        **Precision Precedence**

        The :code:`coeff_precision` and :code:`rotation_precision` arguments are used to determine
        the number of bits for loading the coefficients and the rotation angles, respectively.
        The final value is determined by the following precedence:

        * If provided, the precisions from :code:`prep_op` and :code:`select_op` take precedence.
        * If :code:`prep_op`, and :code:`select_op` are not provided or have the precision value set to `None`,
          the values for :code:`coeff_precision`, and :code:`rotation_precision` arguments are used.
        * If both of the above are not specified, the value set in
          :class:`~.pennylane.estimator.resource_config.ResourceConfig` is used.

    """

    resource_keys = {
        "df_ham",
#        "num_batches",
        "amplitude_amplification_precision",
        "coeff_precision",
        "rotation_precision",
        "select_swap_depths",
    }

    def __init__(
        self,
        df_ham: DFHamiltonian,
#        num_batches: int = 1,
        amplitude_amplification_precision: int | None = None,
        coeff_precision: int | None = None,
        rotation_precision: int | None = None,
        select_swap_depths: int | None = None,
        wires: WiresLike | None = None,
    ):
        if not isinstance(df_ham, DFHamiltonian):
            raise TypeError(
                f"Unsupported Hamiltonian representation for QubitizeDF."
                f"This method works with double factorized Hamiltonian, {type(df_ham)} provided"
            )

        self.df_ham = df_ham
        self.amplitude_amplification_precision = amplitude_amplification_precision
        self.coeff_precision = coeff_precision
        self.rotation_precision = rotation_precision
        self.select_swap_depths = select_swap_depths

        num_orb = df_ham.num_orbitals
        xi = df_ham.num_orbitals
        L = df_ham.num_fragments
        Lxi = df_ham.num_eigenvectors
        nlxi = int(np.ceil(np.log2(Lxi + num_orb)))

        nxi = int(np.ceil(np.log2(xi)))
        nl = int(np.ceil(np.log2(L + 1)))

        # Based on section Eq. C40 in arXiv:2011.03494
        self.num_wires = (
            num_orb * 2
            + 2 * nl
            + 3 * nxi
            + amplitude_amplification_precision
            + 4 * coeff_precision
            + rotation_precision
            + nlxi
            + num_orb * rotation_precision
            + 9
        )
        if wires is not None and len(Wires(wires)) != self.num_wires:
            raise ValueError(f"Expected {self.num_wires} wires, got {len(Wires(wires))}")
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * df_ham (:class:`~pennylane.labs.estimator_beta.compact_hamiltonian.DFHamiltonian`): A double factorized
                  Hamiltonian for which the walk operator is being created.
                * coeff_precision (int | None): The number of bits used to represent the precision for loading
                  the coefficients of Hamiltonian.
                * rotation_precision (int | None): The number of bits used to represent the precision for loading
                  the rotation angles.
                * select_swap_depths (int | None): A parameter of :class:`~.pennylane.estimator.templates.subroutines.QROM`
                  used to trade-off extra wires for reduced circuit depth. Defaults to :code:`None`,
                  which internally determines the optimal depth.
        """
        return {
            "df_ham": self.df_ham,
            "amplitude_amplification_precision": self.amplitude_amplification_precision,
            "coeff_precision": self.coeff_precision,
            "rotation_precision": self.rotation_precision,
            "select_swap_depths": self.select_swap_depths,
        }

    @classmethod
    def resource_rep(
        cls,
        df_ham: DFHamiltonian,
        amplitude_amplification_precision: int | None = None,
        coeff_precision: int | None = None,
        rotation_precision: int | None = None,
        select_swap_depths: int | None = None,
    ) -> CompressedResourceOp:
        """Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            df_ham (:class:`~pennylane.labs.estimator_beta.compact_hamiltonian.DFHamiltonian`): A double factorized
                Hamiltonian for which the walk operator is being created.
            amplitude_amplification_precision (int | None): The number of bits used to represent the precision for single
                qubit rotation in amplitude amplification in outer and inner prep.
            coeff_precision (int | None): The number of bits used to represent the precision for loading
                the coefficients of Hamiltonian.
            rotation_precision (int | None): The number of bits used to represent the precision for loading
                the rotation angles.
            select_swap_depths (int | None): A parameter of :class:`~.pennylane.estimator.templates.subroutines.QROM`
                used to trade-off extra wires for reduced circuit depth. Defaults to :code:`None`, which
                internally determines the optimal depth.

        Returns:
            :class:`~.pennylane.estimator.resource_operator.CompressedResourceOp`: the operator in a compressed representation
        """
        if not isinstance(df_ham, DFHamiltonian):
            raise TypeError(
                f"Unsupported Hamiltonian representation for QubitizeDF."
                f"This method works with thc Hamiltonian, {type(df_ham)} provided"
            )

        num_orb = df_ham.num_orbitals
        xi = df_ham.num_orbitals
        L = df_ham.num_fragments
        Lxi = df_ham.num_eigenvectors
        nlxi = int(np.ceil(np.log2(Lxi + num_orb)))

        nxi = int(np.ceil(np.log2(xi)))
        nl = int(np.ceil(np.log2(L + 1)))
        print("nl: ", nl, "nxi: ", nxi, "nlxi: ", int(np.ceil(np.log2(Lxi + num_orb))))

        # Numbers have been adjusted to remove the auxilliary wires accounted for by different templates
        num_wires = (
            num_orb * 2
            + 2 * nl
            + 3 * nxi
            + amplitude_amplification_precision
            + 4 * coeff_precision
            + rotation_precision
            + nlxi
            + num_orb * rotation_precision
            + 9
        )
        params = {
            "df_ham": df_ham,
            "amplitude_amplification_precision": amplitude_amplification_precision,
            "coeff_precision": coeff_precision,
            "rotation_precision": rotation_precision,
            "select_swap_depths": select_swap_depths,
        }
        return CompressedResourceOp(cls, num_wires, params)

    @classmethod
    def resource_decomp(
        cls,
        df_ham: DFHamiltonian,
        amplitude_amplification_precision: int | None = None,
        coeff_precision: int | None = None,
        rotation_precision: int | None = None,
        select_swap_depths: int | None = None,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object represents a quantum gate
        and the number of times it occurs in the decomposition.

        .. note::

            This decomposition assumes that an appropriately sized phase gradient state is available.
            Users should ensure that the cost of constructing this state has been accounted for.
            See also :class:`~.pennylane.estimator.templates.PhaseGradient`.

        Args:
            df_ham (:class:`~pennylane.estimator.compact_hamiltonian.THCHamiltonian`): a double factorized
                Hamiltonian for which the walk operator is being created
            amplitude_amplification_precision (int | None): The number of bits used to represent the precision for single
                qubit rotation in amplitude amplification in outer and inner prep.
            coeff_precision (int | None): The number of bits used to represent the precision for loading
                the coefficients of Hamiltonian.
            rotation_precision (int | None): The number of bits used to represent the precision for loading
                the rotation angles for basis rotation.
            select_swap_depths (int | None): A parameter of :class:`~.pennylane.estimator.templates.subroutines.QROM`
                used to trade-off extra wires for reduced circuit depth. Defaults to :code:`None`,
                which internally determines the optimal depth.


        Returns:
            list[:class:`~.pennylane.estimator.resource_operator.GateCount`]: A list of ``GateCount`` objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.

        """
        gate_list = []
        num_orbitals = df_ham.num_orbitals
        xi = df_ham.num_orbitals
        Lxi = df_ham.num_eigenvectors
        L = df_ham.num_fragments
        num_coeff = Lxi + num_orbitals

        nl_register = int(np.ceil(np.log2(L + 1)))
        nxi = int(np.ceil(np.log2(xi)))
        coeff_register = int(np.ceil(np.log2(num_coeff)))

        # Fig 16 in arXiv:2011.03494
        # Step1 : Prep
        # 1a, Prepare equal superposition over L+1 basis states, step 1a

        eta = (L & -L).bit_length() - 1

        # paper assumes cost of inequality to be equal to a adder's cost
        comparator = resource_rep(qre.SemiAdder, {"max_register_size": nl_register - eta})
        gate_list.append(GateCount(comparator, 2))

        # Rotate an ancilla to obtain amplitude for sucess
        # and invert the rotation
        toffoli = resource_rep(qre.Toffoli)
        gate_list.append(Allocate(amplitude_amplification_precision))
        gate_list.append(GateCount(toffoli, 4 * (amplitude_amplification_precision - 3)))
        gate_list.append(Deallocate(amplitude_amplification_precision))

        # Reflection on \lceil log(L+1) \rceil - eta - 1 qubits
        gate_list.append(GateCount(toffoli, 2 * (nl_register - eta - 1)))

        # Inequality test again
        gate_list.append(GateCount(comparator, 2))

        # step 1b
        qrom_prep1 = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": L + 1,
                "size_bitstring": nl_register + coeff_precision,
                "borrow_qubits": False,
                "select_swap_depth": select_swap_depths,
            },
        )
        gate_list.append(GateCount(qrom_prep1, 1))

        # step 1c
        ineq = resource_rep(qre.SemiAdder, {"max_register_size": coeff_precision + 1})
        gate_list.append(GateCount(ineq, 2))

        # step 1d
        cswap = resource_rep(qre.CSWAP)
        gate_list.append(GateCount(cswap, 2 * nl_register + 2))

        # Step2 : Output data from the l register
        qrom_output = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": L + 1,
                "size_bitstring": nxi + coeff_register + amplitude_amplification_precision + 1,
                "borrow_qubits": False,
                "select_swap_depth": select_swap_depths,
            },
        )
        gate_list.append(GateCount(qrom_output, 1))

        # Step3 : Prepare the state on p register controlled on l register
        # step 3a:
        # copy the nxi register: i
        gate_list.append(GateCount(toffoli, 4 * nxi - 4))

        # controlled Hadamards - catalytic decomposition: ii
        gate_list.append(GateCount(toffoli, 4 * nxi))

        # inequality test on xi register: iii
        xi_comparator = resource_rep(qre.SemiAdder, {"max_register_size": nxi + 1})
        gate_list.append(GateCount(xi_comparator, 4))

        # rotate and invert the rotation of ancilla: iv and vi
        gate_list.append(Allocate(amplitude_amplification_precision))
        gate_list.append(GateCount(toffoli, 8 * amplitude_amplification_precision - 16))
        gate_list.append(Deallocate(amplitude_amplification_precision))

        # Reflection on the result of inequality: v
        cz = resource_rep(qre.CZ)
        gate_list.append(GateCount(cz, 4))

        # Controlled Hadamards: vii and ix
        gate_list.append(GateCount(toffoli, 8 * nxi))

        # Reflect about the zero state: viii
        gate_list.append(GateCount(toffoli, 4 * nxi - 4))

        # inequality test again: x
        gate_list.append(GateCount(xi_comparator, 4))

        # step3b: Add the offset to the second register
        adder = resource_rep(qre.SemiAdder, {"max_register_size": coeff_register})
        gate_list.append(GateCount(adder, 4))

        # step3c: QROM to output alt and keep values
        qrom_prep2 = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": num_coeff,
                "size_bitstring": nxi + coeff_precision + 2,
                "borrow_qubits": False,
                "select_swap_depth": select_swap_depths,
            },
        )
        gate_list.append(GateCount(qrom_prep2, 1))

        # step3d: Inequality test and controlled swaps
        gate_list.append(GateCount(ineq, 4))
        gate_list.append(GateCount(cswap, 4 * nxi))

        # Step 4: Apply number operators via rotations
        # step4a: Add offset to the second register
        gate_list.append(GateCount(adder, 2))

        # step4b: QROM for the rotation angles
        # For 2-body
        qrom_rot_twobody = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": num_coeff,
                "size_bitstring": num_orbitals * rotation_precision,
                "borrow_qubits": False,
                "select_swap_depth": select_swap_depths,
            },
        )
        gate_list.append(GateCount(qrom_rot_twobody, 1))

        # step4c: controlled swaps controlled on spin qubit
        gate_list.append(GateCount(cswap, 2 * num_orbitals))

        # step4d: Controlled rotations based on semiadder
        rotation_adder = resource_rep(qre.SemiAdder, {"max_register_size": rotation_precision - 1})
        gate_list.append(GateCount(rotation_adder, 4 * num_orbitals))

        # step4e: Z1 controlled on success of prep of l and p registers
        ccz = resource_rep(qre.CCZ)
        gate_list.append(GateCount(ccz, 2))

        # step4f: reverse the controlled rotations and cswaps
        gate_list.append(
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": rotation_adder}), 4 * num_orbitals)
        )
        gate_list.append(GateCount(cswap, 2 * num_orbitals))

        # step4g: Reverse the qrom
        qrom_rot_twobody_adj = resource_rep(qre.Adjoint, {"base_cmpr_op": qrom_rot_twobody})
        gate_list.append(GateCount(qrom_rot_twobody_adj, 1))

        # step4h: Reverse the addition
        gate_list.append(GateCount(adder, 2))

        # Step 5: Invert the state prep cost, same as step: 3, with a different QROM cost
        # Appropriately changed step 3 numbers except for QROM
        qrom_prep2_adj = resource_rep(qre.Adjoint, {"base_cmpr_op": qrom_prep2})
        gate_list.append(GateCount(qrom_prep2_adj, 1))

        # Step 6: Reflection cost
        gate_list.append(GateCount(toffoli, nxi + coeff_precision + 2))

        # Step 7: Repeat steps 2-5 for one-electron integrals
        # Appropriately doubled the resources, adding QROMs here
        qrom_prep2_onebody = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": Lxi,
                "size_bitstring": nxi + coeff_precision + 2,
                "borrow_qubits": False,
                "select_swap_depth": select_swap_depths,
            },
        )
        gate_list.append(GateCount(qrom_prep2_onebody))

        qrom_rot_onebody = resource_rep(
            qre.QROM,
            {
                "num_bitstrings": Lxi,
                "size_bitstring": num_orbitals * rotation_precision,
                "borrow_qubits": False,
                "select_swap_depth": select_swap_depths,
            },
        )
        gate_list.append(GateCount(qrom_rot_onebody, 1))

        qrom_rot_onebody_adj = resource_rep(qre.Adjoint, {"base_cmpr_op": qrom_rot_onebody})
        gate_list.append(GateCount(qrom_rot_onebody_adj, 1))

        qrom_prep2_onebody_adj = resource_rep(qre.Adjoint, {"base_cmpr_op": qrom_prep2_onebody})
        gate_list.append(GateCount(qrom_prep2_onebody_adj, 1))

        # Step 8: Invert the QROM in step:2 and 1
        qrom_output_adj = resource_rep(qre.Adjoint, {"base_cmpr_op": qrom_output})
        gate_list.append(GateCount(qrom_output_adj, 1))

        qrom_prep1_adj = resource_rep(qre.Adjoint, {"base_cmpr_op": qrom_prep1})
        gate_list.append(GateCount(qrom_prep1_adj, 1))

        # And the preparation in step:1 : Adjusted the numbers in step 1 to be doubled.

        # Step 9: Reflection needed for walk operator
        gate_list.append(GateCount(toffoli, nl_register + nxi + 2 * coeff_precision + 1))

        return gate_list
