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
r"""Resource operators for PennyLane state preparation templates."""
import math
from collections import defaultdict
from typing import Dict

import pennylane as qml
from pennylane.labs import resource_estimation as re
from pennylane.labs.resource_estimation.qubit_manager import AllocWires, FreeWires
from pennylane.labs.resource_estimation.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourceOperator,
    resource_rep,
)

# pylint: disable=arguments-differ, protected-access, non-parent-init-called, too-many-arguments,


class ResourceMPSPrep(ResourceOperator):
    r"""Resource class for the MPSPrep template.

    The resource operation for preparing an initial state from a matrix product state (MPS) 
    representation.

    Args:
        num_mps_matrices (int): the number of matrices in the MPS representation
        max_bond_dim (int): the bond dimension of the MPS representation
        precision (Union[None, float], optional): the precision used when loading the MPS matricies
        wires (Sequence[int], optional): the wires the operation acts on

    Resources:
        The resources for MPSPrep are according to the decomposition, which uses the generic 
        :class:`~.labs.resource_estimation.ResourceQubitUnitary`. The decomposition is based on 
        the routine described in `Fomichev et al. (2024) <https://arxiv.org/pdf/2310.18410>`_.

    .. seealso:: :class:`~.MPSPrep`

    **Example**

    The resources for this operation are computed using:

    >>> mps = plre.ResourceMPSPrep(num_mps_matrices=10, max_bond_dim=5)
    >>> print(plre.estimate_resources(mps, gate_set={"CNOT", "RZ", "RY"}))
    --- Resources: ---
     Total qubits: 13
     Total gates : 1.654E+3
     Qubit breakdown:
      clean qubits: 3, dirty qubits: 0, algorithmic qubits: 10
     Gate breakdown:
      {'RZ': 728, 'CNOT': 774, 'RY': 152}
    """
    resource_keys = {"num_mps_matrices", "max_bond_dim", "precision"}

    def __init__(self, num_mps_matrices, max_bond_dim, precision=None, wires=None):
        self.num_wires = num_mps_matrices

        self.precision = precision
        self.max_bond_dim = max_bond_dim
        self.num_mps_matrices = num_mps_matrices
        super().__init__(wires=wires)

    @property
    def resource_params(self) -> Dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_mps_matrices (int): the number of matrices in the MPS representation
                * max_bond_dim (int): the bond dimension of the MPS representation
                * precision (Union[None, float], optional): The precision used when loading the 
                  MPS matricies.
        """
        return {"num_mps_matrices": self.num_mps_matrices, "max_bond_dim": self.max_bond_dim, "precision": self.precision}

    @classmethod
    def resource_rep(cls, num_mps_matrices, max_bond_dim, precision=None) -> CompressedResourceOp:
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_mps_matrices (int): the number of matrices in the MPS representation
            max_bond_dim (int): the bond dimension of the MPS representation
            precision (Union[None, float], optional): The precision used when loading the MPS matricies.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {"num_mps_matrices": num_mps_matrices, "max_bond_dim": max_bond_dim, "precision": precision}
        return CompressedResourceOp(cls, params)

    @classmethod
    def default_resource_decomp(
        cls,
        num_mps_matrices,
        max_bond_dim,
        precision = None,
        **kwargs,
    ) -> list[GateCount]:
        r"""Returns a list representing the resources of the operator. Each object in the list
        represents a gate and the number of times it occurs in the circuit.

        Args:
            num_mps_matrices (int): the number of matrices in the MPS representation
            max_bond_dim (int): the bond dimension of the MPS representation
            precision (Union[None, float], optional): The precision used when loading
                the MPS matricies.

        Resources:
            The resources for MPSPrep are according to the decomposition, which uses the generic 
            :class:`~.labs.resource_estimation.ResourceQubitUnitary`. The decomposition is based on 
            the routine described in `Fomichev et al. (2024) <https://arxiv.org/pdf/2310.18410>`_.
        
        Returns:
            list[GateCount]: A list of GateCount objects, where each object
            represents a specific quantum gate and the number of times it appears
            in the decomposition.
        """
        precision = precision or kwargs["config"]["precision_mps_prep"]
        num_work_wires = math.ceil(math.log2(max_bond_dim))
        log2_chi = min(num_work_wires, math.ceil(num_mps_matrices / 2))

        gate_lst = [AllocWires(num_work_wires)]

        for index in range(1, num_mps_matrices + 1):
            qubit_unitary_wires = min(index + 1, log2_chi + 1, (num_mps_matrices - index) + 2)
            qubit_unitary = re.ResourceQubitUnitary.resource_rep(num_wires=qubit_unitary_wires, precision=precision)
            gate_lst.append(GateCount(qubit_unitary))

        gate_lst.append(FreeWires(num_work_wires))
        return gate_lst

    @classmethod
    def tracking_name(cls, num_mps_matrices, max_bond_dim, precision) -> str:
        return f"MPSPrep({num_mps_matrices}, {max_bond_dim}, {precision})"


class ResourceQROMStatePreparation(ResourceOperator):
    r"""Resource class for the QROMStatePreparation template.

    This operation implements the state preparation method described
    in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_.

    Args:
        state_vector (tensor_like): The state vector of length :math:`2^n` to be prepared on :math:`n` wires.
        wires (Sequence[int]): The wires on which to prepare the state.
        precision_wires (Sequence[int]): The wires allocated for storing the binary representations of the
            rotation angles utilized in the template.
        work_wires (Sequence[int], optional):  The work wires used for the QROM operations. Defaults to ``None``.

    Resource Parameters:
        * num_state_qubits (int): number of qubits required to represent the state-vector
        * num_precision_wires (int): number of qubits that specify the precision of the rotation angles
        * num_work_wires (int): additional qubits which optimize the implementation
        * positive_and_real (bool): flag that the coefficients of the statevector are all real and positive.

    Resources:
        The resources for QROMStatePreparation are according to the decomposition as described
        in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_.

    .. seealso:: :class:`~.QROMStatePreparation`

    **Example**

    The resources for this operation are computed using:

    >>> re.ResourceQROMStatePreparation.resources(
    ...     num_state_qubits=5,
    ...     num_precision_wires=3,
    ...     num_work_wires=3,
    ...     positive_and_real=True,
    ... )
    defaultdict(<class 'int'>, {QROM: 1, Adjoint(QROM): 1,
    QROM: 1, Adjoint(QROM): 1, QROM: 1, Adjoint(QROM): 1,
    QROM: 1, Adjoint(QROM): 1, QROM: 1, Adjoint(QROM): 1, CRY: 15})
    """

    resource_keys = {"num_state_qubits", "precision", "positive_and_real"}

    def __init__(self, num_state_qubits, precision=None, positive_and_real=False, wires=None):
        # Overriding the default init method to allow for CompactState as an input.
        self.num_wires = num_state_qubits
        self.precision = precision
        self.positive_and_real = positive_and_real
        super().__init__(wires=wires)

    @classmethod
    def default_resource_decomp(
        cls,
        num_state_qubits,
        precision,
        positive_and_real,
        **kwargs,
    ):
        r"""Returns a dictionary representing the resources of the operator. The
        keys are the operators and the associated values are the counts.

        Args:
            num_state_qubits (int): number of qubits required to represent the state-vector
            num_precision_wires (int): number of qubits that specify the precision of the rotation angles
            num_work_wires (int): additional qubits which optimize the implementation
            num_phase_gradient_wires (int): number of qubits where the phase gradient state is stored. Must be equal
                to ``num_precision_wires``
            positive_and_real (bool): flag that the coefficients of the statevector are all real and positive.

        Resources:
            The resources for QROMStatePreparation are according to the decomposition as described
            in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_.
        """
        gate_counts = []
        precision = precision or kwargs["config"]["precision_qrom_state_prep"]
        num_precision_wires = abs(math.floor(math.log2(precision)))

        gate_counts.append(AllocWires(num_precision_wires))

        for j in range(1, num_state_qubits):
            num_bitstrings = 2**j
            num_bit_flips = max(2 ** (j - 1), 1)

            gate_counts.append(
                GateCount(
                    re.ResourceQROM.resource_rep(
                        num_bitstrings,
                        num_precision_wires,
                        num_bit_flips,
                        clean=False,
                    )
                )
            )

            gate_counts.append(
                GateCount(
                    re.ResourceAdjoint.resource_rep(
                        re.resource_rep(
                            re.ResourceQROM,
                            {
                                "num_bitstrings": num_bitstrings,
                                "num_bit_flips": num_bit_flips,
                                "size_bitstring": num_precision_wires,
                                "clean": False,
                            },
                        ),
                    )
                )
            )

        t = re.ResourceT.resource_rep()
        h = re.ResourceHadamard.resource_rep()

        # SemiAdder T-cost estimation. Deduce based in image 1 and non-simetrics cnots: https://arxiv.org/pdf/1709.06648
        # TODO: Update once we have qml.SemiAdder
        gate_counts.append(
            GateCount(
                t,
                (2 * (2 * (num_precision_wires - 1)) + 4 * (2 * num_precision_wires - 1))
                * num_state_qubits,
            )
        )

        gate_counts.append(GateCount(h, 2 * num_state_qubits))

        if not positive_and_real:
            gate_counts.append(
                GateCount(
                    re.ResourceQROM.resource_rep(
                        2**num_state_qubits,
                        num_precision_wires,
                        2 ** (num_state_qubits - 1),
                        clean=False,
                    )
                )
            )

            # SemiAdder T-cost estimation. Deduce based in image 1 and non-simetrics cnots: https://arxiv.org/pdf/1709.06648
            # TODO: Update once we have qml.SemiAdder
            gate_counts.append(
                GateCount(
                    t,
                    2 * (2 * (num_precision_wires - 1)) + 4 * (2 * num_precision_wires - 1),
                )
            )

            gate_counts.append(
                GateCount(
                    re.ResourceAdjoint.resource_rep(
                        re.resource_rep(
                            re.ResourceQROM,
                            {
                                "num_bitstrings": num_bitstrings,
                                "num_bit_flips": num_bit_flips,
                                "size_bitstring": num_precision_wires,
                                "clean": False,
                            },
                        ),
                    )
                )
            )

        gate_counts.append(FreeWires(num_precision_wires))
        return gate_counts

    # @staticmethod
    # def optimized_ww_decomp(
    #     num_state_qubits,
    #     precision,
    #     positive_and_real,
    #     **kwargs,
    # ):
    #     r"""Returns a dictionary representing the resources of the operator. The
    #     keys are the operators and the associated values are the counts.

    #     Args:
    #         num_state_qubits (int): number of qubits required to represent the state-vector
    #         num_precision_wires (int): number of qubits that specify the precision of the rotation angles
    #         num_work_wires (int): additional qubits which optimize the implementation
    #         num_phase_gradient_wires (int): number of qubits where the phase gradient state is stored. Must be equal
    #             to ``num_precision_wires``
    #         positive_and_real (bool): flag that the coefficients of the statevector are all real and positive.

    #     Resources:
    #         The resources for QROMStatePreparation are according to the decomposition as described
    #         in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_.
    #     """
    #     gate_counts = []
    #     precision = precision or kwargs["config"]["precision_qrom_state_prep"]
    #     num_precision_wires = abs(math.floor(math.log2(precision)))

    #     gate_counts.append(AddQubits(num_precision_wires))
    #     available_wires = clean_qubits() - num_precision_wires

    #     for j in range(1, num_state_qubits):
    #         num_bitstrings = 2**j
    #         num_bit_flips = max(2 ** (j - 1), 1)

    #         W_opt = re.ResourceQROM._t_optimized_select_swap_width(num_bitstrings, num_precision_wires)
    #         l = math.ceil(math.log2(math.ceil(num_bitstrings / W_opt)))
    #         l_new = l
    #         if tight_qubit_budget() and available_wires < ((W_opt - 1) * num_precision_wires + (l - 1)):
    #             for p in range(0, int(math.log2(W_opt))+ 1):
    #             # for W_opt_new in range(1, W_opt):
    #                 W_opt_new = 2**p
    #                 l_new = math.ceil(math.log2(math.ceil(num_bitstrings / W_opt_new)))
    #                 if available_wires < ((W_opt_new - 1) * num_precision_wires + (l_new - 1)):
    #                     break
    #                 W_opt = W_opt_new

    #         gate_counts.append(
    #             GateCount(
    #                 re.ResourceQROM.resource_rep(
    #                     num_bitstrings,
    #                     num_bit_flips,
    #                     num_precision_wires,
    #                     select_swap_depth=W_opt,
    #                     clean=False,
    #                 )
    #             )
    #         )

    #         gate_counts.append(
    #             GateCount(
    #                 re.ResourceAdjoint.resource_rep(
    #                     base_class=re.ResourceQROM,
    #                     base_params={
    #                         "num_bitstrings": num_bitstrings,
    #                         "num_bit_flips": num_bit_flips,
    #                         "size_bitstring": num_precision_wires,
    #                         "select_swap_depth": W_opt,
    #                         "clean": False,
    #                     },
    #                 )
    #             )
    #         )

    #     t = re.ResourceT.resource_rep()
    #     h = re.ResourceHadamard.resource_rep()

    #     # SemiAdder T-cost estimation. Deduce based in image 1 and non-simetrics cnots: https://arxiv.org/pdf/1709.06648
    #     # TODO: Update once we have qml.SemiAdder
    #     gate_counts.append(
    #         GateCount(
    #             t,
    #             (2 * (2 * (num_precision_wires - 1)) + 4 * (2 * num_precision_wires - 1)) * num_state_qubits,
    #         )
    #     )

    #     gate_counts.append(GateCount(h, 2 * num_state_qubits))

    #     if not positive_and_real:
    #         num_bitstrings = 2**num_state_qubits
    #         W_opt = re.ResourceQROM._t_optimized_select_swap_width(num_bitstrings, num_precision_wires)
    #         l = math.ceil(math.log2(math.ceil(num_bitstrings / W_opt)))

    #         if tight_qubit_budget() and available_wires < ((W_opt - 1) * num_precision_wires + (l - 1)):
    #             for p in range(0, int(math.log2(W_opt)) + 1):
    #             # for W_opt_new in range(1, W_opt):
    #                 W_opt_new = 2**p
    #                 l_new = math.ceil(math.log2(math.ceil(num_bitstrings / W_opt_new)))
    #                 if available_wires < ((W_opt_new - 1) * num_precision_wires + (l_new - 1)):
    #                     break
    #                 W_opt = W_opt_new

    #         gate_counts.append(
    #             GateCount(
    #                 re.ResourceQROM.resource_rep(
    #                     2**num_state_qubits,
    #                     2 ** (num_state_qubits - 1),
    #                     num_precision_wires,
    #                     select_swap_depth=W_opt,
    #                     clean=False,
    #                 )
    #             )
    #         )

    #         # SemiAdder T-cost estimation. Deduce based in image 1 and non-simetrics cnots: https://arxiv.org/pdf/1709.06648
    #         # TODO: Update once we have qml.SemiAdder
    #         gate_counts.append(
    #             GateCount(
    #                 t,
    #                 2 * (2 * (num_precision_wires - 1)) + 4 * (2 * num_precision_wires - 1),
    #             )
    #         )

    #         gate_counts.append(
    #             GateCount(
    #                 re.ResourceAdjoint.resource_rep(
    #                     base_class=re.ResourceQROM,
    #                     base_params={
    #                         "num_bitstrings": 2**num_state_qubits,
    #                         "num_bit_flips": 2 ** (num_state_qubits - 1),
    #                         "size_bitstring": num_precision_wires,
    #                         "select_swap_depth": W_opt,
    #                         "clean": False,
    #                     },
    #                 )
    #             )
    #         )

    #     gate_counts.append(FreeWires(num_precision_wires))
    #     return gate_counts

    # @staticmethod
    # def zero_swap_w_decomp(
    #     num_state_qubits,
    #     precision,
    #     positive_and_real,
    #     **kwargs,
    # ):
    #     r"""Returns a dictionary representing the resources of the operator. The
    #     keys are the operators and the associated values are the counts.

    #     Args:
    #         num_state_qubits (int): number of qubits required to represent the state-vector
    #         num_precision_wires (int): number of qubits that specify the precision of the rotation angles
    #         num_work_wires (int): additional qubits which optimize the implementation
    #         num_phase_gradient_wires (int): number of qubits where the phase gradient state is stored. Must be equal
    #             to ``num_precision_wires``
    #         positive_and_real (bool): flag that the coefficients of the statevector are all real and positive.

    #     Resources:
    #         The resources for QROMStatePreparation are according to the decomposition as described
    #         in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_.
    #     """
    #     gate_counts = []
    #     precision = precision or kwargs["config"]["precision_qrom_state_prep"]
    #     num_precision_wires = abs(math.floor(math.log2(precision)))

    #     gate_counts.append(AddQubits(num_precision_wires))

    #     for j in range(1, num_state_qubits):
    #         num_bitstrings = 2**j
    #         num_bit_flips = max(2 ** (j - 1), 1)

    #         gate_counts.append(
    #             GateCount(
    #                 re.ResourceQROM.resource_rep(
    #                     num_bitstrings,
    #                     num_bit_flips,
    #                     num_precision_wires,
    #                     select_swap_depth=1,
    #                     clean=False,
    #                 )
    #             )
    #         )

    #         gate_counts.append(
    #             GateCount(
    #                 re.ResourceAdjoint.resource_rep(
    #                     base_class=re.ResourceQROM,
    #                     base_params={
    #                         "num_bitstrings": num_bitstrings,
    #                         "num_bit_flips": num_bit_flips,
    #                         "size_bitstring": num_precision_wires,
    #                         "select_swap_depth": 1,
    #                         "clean": False,
    #                     },
    #                 )
    #             )
    #         )

    #     t = re.ResourceT.resource_rep()
    #     h = re.ResourceHadamard.resource_rep()

    #     # SemiAdder T-cost estimation. Deduce based in image 1 and non-simetrics cnots: https://arxiv.org/pdf/1709.06648
    #     # TODO: Update once we have qml.SemiAdder
    #     gate_counts.append(
    #         GateCount(
    #             t,
    #             (2 * (2 * (num_precision_wires - 1)) + 4 * (2 * num_precision_wires - 1)) * num_state_qubits,
    #         )
    #     )

    #     gate_counts.append(GateCount(h, 2 * num_state_qubits))

    #     if not positive_and_real:
    #         gate_counts.append(
    #             GateCount(
    #                 re.ResourceQROM.resource_rep(
    #                     2**num_state_qubits,
    #                     2 ** (num_state_qubits - 1),
    #                     num_precision_wires,
    #                     select_swap_depth=1,
    #                     clean=False,
    #                 )
    #             )
    #         )

    #         # SemiAdder T-cost estimation. Deduce based in image 1 and non-simetrics cnots: https://arxiv.org/pdf/1709.06648
    #         # TODO: Update once we have qml.SemiAdder
    #         gate_counts.append(
    #             GateCount(
    #                 t,
    #                 2 * (2 * (num_precision_wires - 1)) + 4 * (2 * num_precision_wires - 1),
    #             )
    #         )

    #         gate_counts.append(
    #             GateCount(
    #                 re.ResourceAdjoint.resource_rep(
    #                     base_class=re.ResourceQROM,
    #                     base_params={
    #                         "num_bitstrings": 2**num_state_qubits,
    #                         "num_bit_flips": 2 ** (num_state_qubits - 1),
    #                         "size_bitstring": num_precision_wires,
    #                         "select_swap_depth": 1,
    #                         "clean": False,
    #                     },
    #                 )
    #             )
    #         )

    #     gate_counts.append(FreeWires(num_precision_wires))
    #     return gate_counts

    @property
    def resource_params(self) -> dict:
        r"""Returns a dictionary containing the minimal information needed to compute the resources.

        Returns:
            dict: A dictionary containing the resource parameters:
                * num_state_qubits (int): number of qubits required to represent the state-vector
                * num_precision_wires (int): number of qubits that specify the precision of the rotation angles
                * num_work_wires (int): additional qubits which optimize the implementation
                * positive_and_real (bool): flag that the coefficients of the statevector are all real and positive.
        """

        return {
            "num_state_qubits": self.num_wires,
            "precision": self.precision,
            "positive_and_real": self.positive_and_real,
        }

    @classmethod
    def resource_rep(cls, num_state_qubits, precision=None, positive_and_real=False):
        r"""Returns a compressed representation containing only the parameters of
        the Operator that are needed to compute a resource estimation.

        Args:
            num_state_qubits (int): number of qubits required to represent the state-vector
            num_precision_wires (int): number of qubits that specify the precision of the rotation angles
            num_work_wires (int): additional qubits which optimize the implementation
            positive_and_real (bool): flag that the coefficients of the statevector are all real and positive.

        Returns:
            CompressedResourceOp: the operator in a compressed representation
        """
        params = {
            "num_state_qubits": num_state_qubits,
            "precision": precision,
            "positive_and_real": positive_and_real,
        }
        return CompressedResourceOp(cls, params)
