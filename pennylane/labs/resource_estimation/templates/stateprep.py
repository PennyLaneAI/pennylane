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
import itertools

from typing import Dict
from collections import defaultdict

import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import Operation
from pennylane.labs import resource_estimation as re
from pennylane.labs.resource_estimation import CompressedResourceOp, ResourceOperator

# pylint: disable=arguments-differ, protected-access

def sum_by_prefix(vector, prefix):
    r"""Calculates the sum of elements in a vector whose index, when represented in binary, starts with a given prefix.

    Args:
        vector (TensorLike): A 1D vector of numerical values.
        prefix (str): A string representing the binary prefix to match.

    Returns:
        (float) The sum of the elements in the vector whose index binary representation starts with the given prefix.


    Example:
        >>> vector = [1, 3, 5, 2, 1, 3, 2, 2]
        >>> prefix = "10"
        >>> sum_by_prefix(vector, prefix)
        1 + 3 = 4  # Elements at indices 4 and 5 (binary 100 and 101) are summed.

        >>> prefix = "01"
        >>> sum_by_prefix(vector, prefix)
        5 + 2 = 7 # Elements at indices 2 and 3 (binary 010 and 011) are summed.

        >>> prefix = "1"
        >>> sum_by_prefix(vector, prefix)
        1 + 3 + 2 + 2 = 8 # Elements at indices 4, 5, 6, and 7 (binary 100, 101, 110 and 111) are summed.
    """

    n = len(vector).bit_length() - 1
    sum_result = 0
    for i, value in enumerate(vector):
        bitstring = qml.math.binary_repr(i, n)
        if bitstring.startswith(prefix):
            sum_result += value
    return sum_result

def get_basis_state_list(n_wires, add_zero = False):
    r"""Generates a list of binary strings representing basis states.

    Args:
        n_wires (int): The number of wires in the system.
        add_zero (bool, optional): Whether to append a '0' to each binary string. Defaults to False.

    Returns:
        list[str]: A list of binary strings representing the basis states.
        Each string has length `n_wires` (or `n_wires + 1` if `add_zero` is True).

    Example:
        >>> get_basis_state_list(2)
        ['00', '01', '10', '11']

        >>> get_basis_state_list(3, add_zero=True)
        ['0000', '0010', '0100', '0110', '1000', '1010', '1100', '1110']
    """

    if add_zero:
      return [''.join(map(str, bits)) + '0' for bits in itertools.product([0, 1], repeat=n_wires)]
    else:
      return [''.join(map(str, bits)) for bits in itertools.product([0, 1], repeat=n_wires)]

def func_to_binary(n_precision, x , func):
  r"""Converts a value within the range [0, 1) to its binary representation with a specified precision.

    This function applies a given transformation function (`func`) to the input value `x` and then converts
    the result to a binary string. The transformation function should map values from the interval [0, 1) to
    the interval [0, 1).

    Args:
        n_precision (int): The number of bits to use for the binary representation.
        x (float): The value to convert to binary. Must be in the range [0, 1).
        func (callable): A function that transforms the input value.

    Returns:
        str: The binary representation of the transformed value, with the specified precision.

    Example:
        >>> func_to_binary(3, 0.25, lambda x: np.sqrt(x))
        '100'

        Expected value as `\sqrt{0.25} = 0.5`, and it's binary representation is `0.100`.

    """

  return bin(int(2**(n_precision) + 2**(n_precision)*func(x)))[-n_precision:]


class ResourceStatePrep(qml.StatePrep, ResourceOperator):
    """Resource class for StatePrep.

    Resources:
        Uses the resources as defined in the ResourceMottonenStatePreperation template.
    """

    @staticmethod
    def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        return {re.ResourceMottonenStatePreparation.resource_rep(num_wires): 1}

    def resource_params(self) -> dict:
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, num_wires) -> CompressedResourceOp:
        params = {"num_wires": num_wires}
        return CompressedResourceOp(cls, params)

    @classmethod
    def tracking_name(cls, num_wires) -> str:
        return f"StatePrep({num_wires})"


class ResourceMottonenStatePreparation(qml.MottonenStatePreparation, ResourceOperator):
    """Resource class for the MottonenStatePreparation template.
    
    Using the resources as described in https://arxiv.org/pdf/quant-ph/0407010. 
    """

    @staticmethod
    def _resource_decomp(num_wires, **kwargs) -> Dict[CompressedResourceOp, int]:
        gate_types = {}
        rz = re.ResourceRZ.resource_rep()
        cnot = re.ResourceCNOT.resource_rep()

        r_count = 2 ** (num_wires + 2) - 5
        cnot_count = 2 ** (num_wires + 2) - 4 * num_wires - 4

        if r_count:
            gate_types[rz] = r_count

        if cnot_count:
            gate_types[cnot] = cnot_count
        return gate_types

    def resource_params(self) -> dict:
        return {"num_wires": len(self.wires)}

    @classmethod
    def resource_rep(cls, num_wires) -> CompressedResourceOp:
        params = {"num_wires": num_wires}
        return CompressedResourceOp(cls, params)

    @classmethod
    def tracking_name(cls, num_wires) -> str:
        return f"MottonenStatePrep({num_wires})"


class ResourceSuperposition(qml.Superposition, ResourceOperator):
    """Resource class for the Superposition template."""

    @staticmethod
    def _resource_decomp(num_stateprep_wires, num_basis_states, size_basis_state, **kwargs) -> Dict[CompressedResourceOp, int]:
        r"""The resources are computed following the PennyLane decomposition of
        the class. This class was designed based on the method described in 
        https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.5.040339. 

        We use the following (somewhat naive) assumptions to approximate the
        resources: 
        
        -  The MottonenStatePreparation routine is assumed for the state prep
        component. 
        -  The permutation block requires 2 multi-controlled X gates and a 
        series of CNOT gates. On average we will be controlling on and flipping 
        half the number of bits in :code:`size_basis`. (i.e for any given basis
        state, half will be ones and half will be zeros). 
        -  If the number of basis states provided spans the set of all basis states,
        then we don't need to permute. In general, there is a probability associated
        with not needing to permute wires if the basis states happen to match, we 
        estimate this quantity aswell. 
        """
        gate_types = {}
        msp = re.ResourceMottonenStatePreparation.resource_rep(num_stateprep_wires)
        gate_types[msp] = 1

        cnot = re.ResourceCNOT.resource_rep()
        num_zero_ctrls = size_basis_state // 2
        multi_x = re.ResourceMultiControlledX.resource_rep(
            num_ctrl_wires=size_basis_state, num_ctrl_values=num_zero_ctrls, num_work_wires=0,
        )

        basis_size = 2 ** size_basis_state
        prob_matching_basis_states = num_basis_states / basis_size
        num_permutes =  round(num_basis_states * (1 - prob_matching_basis_states))
        
        if num_permutes:
            gate_types[cnot] = num_permutes * (size_basis_state // 2)  # average number of bits to flip
            gate_types[multi_x] = 2 * num_permutes  # for compute and uncompute

        return gate_types
    
    def resource_params(self) -> Dict:
        bases = self.hyperparameters["bases"]
        num_basis_states = len(bases)
        size_basis_state = len(bases[0])  # assuming they are all the same size
        num_stateprep_wires = math.ceil(math.log2(len(self.coeffs)))

        return {
            "num_stateprep_wires": num_stateprep_wires, 
            "num_basis_states": num_basis_states,
            "size_basis_state": size_basis_state,
        }
    
    @classmethod
    def resource_rep(cls, num_stateprep_wires, num_basis_states, size_basis_state) -> CompressedResourceOp:
        params = {
            "num_stateprep_wires": num_stateprep_wires, 
            "num_basis_states": num_basis_states,
            "size_basis_state": size_basis_state,
        }
        return CompressedResourceOp(cls, params)


class ResourceQROMStatePreparation(Operation, re.ResourceOperator):
    r"""Prepares a quantum state using a Quantum Read-Only Memory (QROM) based approach.

        This operation decomposes the state preparation into a sequence of QROM operations and controlled rotations.

        Args:
            state_vector (TensorLike): The state vector to prepare.
            wires (Sequence[int]): The wires on which to prepare the state.
            precision_wires (Sequence[int]): The wires used for storing the binary representations of the
                amplitudes and phases.
            work_wires (Sequence[int], optional):  The wires used as work wires for the QROM operations. Defaults to ``None``.

        **Example**

        .. code-block::

            dev = qml.device("default.qubit", wires=6)
            state_vector = np.array([1,0,0,0]) / 2.0
            wires = [0, 1]
            precision_wires = [2, 3, 4]
            work_wires = [5]

            @qml.qnode(dev)
            def circuit():
                qml.QROMStatePreparation(state_vector, wires, precision_wires, work_wires)
                return qml.state()

            print(circuit())

        .. details::
            :title: Usage Details

            This operation implements the state preparation method described in
            `arXiv:quant-ph/0208112 <https://arxiv.org/abs/quant-ph/0208112>`.  It uses a QROM to store
            the binary representations of the amplitudes and phases of the target state, and then uses
            controlled rotations to apply these values to the target qubits.

            The input `state_vector` must have a length that is a power of 2. The number of `wires`
            must be at least :math:`\log_2(\text{len}(state\_vector))`. The number of `precision_wires` determines the
            precision with which the amplitudes and phases are encoded.

            The `work_wires` are used as auxiliary qubits in the QROM operation. They should be distinct
            from the `wires` and `precision_wires`.

            The decomposition involves encoding the probabilities and phases of the state vector using
            QROMs and then applying controlled rotations based on the values stored in the `precision_wires`.
            The decomposition applies CRY rotations for amplitude encoding and controlled GlobalPhase rotations for the phase encoding.

            The user must ensure that the number of precision wires is enough to store the values. The relation between the number of precision wires `n_p` and the precision `p` is given by :math:`p = 2^{-n_p}`.
    """

    def __init__(self, state_vector, wires, precision_wires, work_wires = None, id = None):

        self.state_vector = state_vector
        self.hyperparameters["input_wires"] = qml.wires.Wires(wires)
        self.hyperparameters["precision_wires"] = qml.wires.Wires(precision_wires)
        self.hyperparameters["work_wires"] = qml.wires.Wires(() if work_wires is None else work_wires)

        all_wires = self.hyperparameters["input_wires"] + self.hyperparameters["precision_wires"] + self.hyperparameters["work_wires"]

        super().__init__(state_vector, wires= all_wires, id=id)

    def decomposition(self):  # pylint: disable=arguments-differ
        filtered_hyperparameters = {
            key: value for key, value in self.hyperparameters.items() if key != "input_wires"
        }
        return self.compute_decomposition(
            self.parameters[0], wires=self.hyperparameters["input_wires"], **filtered_hyperparameters
        )

    @staticmethod
    def compute_decomposition(state_vector, wires, precision_wires, work_wires):  # pylint: disable=arguments-differ
        """
        Computes the decomposition operations for the given state vector.

        Args:
            state_vector (array-like): The state vector.
            wires (list): List of control wires.
            precision_wires (list): List of precision wires.
            work_wires (list or None): List of work wires (can be None).

        Returns:
            list: List of decomposition operations.
        """
        # Square each component of the state vector
        probs = qml.math.abs(state_vector) ** 2
        phases = qml.math.angle(state_vector) % (2 * np.pi)

        decomp_ops = []
        num_iterations = int(qml.math.log2(len(probs)))
        for i in range(num_iterations):

            # Calculation of the numerator and denominator of the function f (Eq.5 [arXiv:quant-ph/0208112])
            prefixes = get_basis_state_list(n_wires = i)
            probs_denominator = qml.math.array([sum_by_prefix(probs, prefix=p) for p in prefixes])

            prefixes_with_zero = get_basis_state_list(n_wires = i, add_zero=True)
            probs_numerator = qml.math.array([sum_by_prefix(probs, prefix=p) for p in prefixes_with_zero])

            eps = 1e-8  # Small constant to avoid division by zero

            # Compute the binary representations of the angles θi
            func = lambda x: 2 * qml.math.arccos(qml.math.sqrt(x))/np.pi
            thetas_binary = [
                func_to_binary(len(precision_wires), probs_numerator[j] / (probs_denominator[j] + eps), func)
                for j in range(len(probs_numerator))
            ]

            # Apply the QROM operation to encode the thetas binary representation
            decomp_ops.append(
                qml.QROM(
                    bitstrings=thetas_binary,
                    target_wires=precision_wires,
                    control_wires=wires[: i],
                    work_wires=work_wires,
                    clean=False,
                )
            )

            # Turn binary representation into proper rotation
            for ind, wire in enumerate(precision_wires):
                rotation_angle = 2 ** (- ind - 1)
                decomp_ops.append(qml.CRY(np.pi * rotation_angle, wires=[wire, wires[i]]))

            # Clean wires used to store the theta values
            decomp_ops.append(
                qml.adjoint(qml.QROM)(
                    bitstrings=thetas_binary,
                    target_wires=precision_wires,
                    control_wires=wires[: i],
                    work_wires=work_wires,
                    clean=False,
                )
            )

        # Compute the binary representations of the phases
        func = lambda x: (x) / (2 * np.pi)
        thetas_binary = [
            func_to_binary(len(precision_wires), phase, func)
            for phase in phases
        ]

        # Apply the QROM operation to encode the thetas binary representation
        decomp_ops.append(
            qml.QROM(
                bitstrings=thetas_binary,
                target_wires=precision_wires,
                control_wires=wires,
                work_wires=work_wires,
                clean=False,
            )
        )

        # Turn binary representation into proper rotation
        for ind, wire in enumerate(precision_wires):
            rotation_angle = 2 ** (- ind - 1)
            decomp_ops.append(qml.ctrl(qml.GlobalPhase((2 * np.pi)*(-rotation_angle), wires=wires[0]), control = wire))

        # Clean wires used to store the theta values
        decomp_ops.append(
            qml.adjoint(qml.QROM)(
                bitstrings=thetas_binary,
                target_wires=precision_wires,
                control_wires=wires,
                work_wires=work_wires,
                clean=False,
            )
        )

        return decomp_ops

    @staticmethod
    def _resource_decomp(num_state_qubits, num_precision_wires, num_work_wires, positive_and_real, **kwargs):
        """The resources associated with a single QROMPrep"""
        gate_types = defaultdict(int)

        for j in range(num_state_qubits):
            num_bitstrings = 2**j
            num_bit_flips = 2**(j-1)
            num_control_wires = j
            
            gate_types[
                re.ResourceQROM.resource_rep(
                    num_bitstrings, 
                    num_bit_flips,
                    num_control_wires,
                    num_work_wires,
                    num_precision_wires,
                    clean=False,
                )
            ] += 1

            gate_types[
                re.ResourceAdjoint.resource_rep(
                    base_class=re.ResourceQROM, 
                    base_params={
                        "num_bitstrings": num_bitstrings,
                        "num_bit_flips": num_bit_flips,
                        "num_control_wires": num_control_wires,
                        "num_work_wires": num_work_wires,
                        "size_bitstring": num_precision_wires,
                        "clean": False,
                    }
                )
            ] += 1

        c_ry = re.ResourceCRY.resource_rep()
        gate_types[c_ry] = num_precision_wires * num_state_qubits

        c_gp = re.ResourceControlled.resource_rep(re.ResourceGlobalPhase, {}, 1, 0, 0)

        if not positive_and_real:
            gate_types[
                re.ResourceQROM.resource_rep(
                    2**num_state_qubits,
                    2**(num_state_qubits - 1),
                    num_state_qubits,
                    num_work_wires,
                    num_precision_wires,
                    clean=False,
                )
            ] += 1

            gate_types[c_gp] = num_precision_wires
            
            gate_types[
                re.ResourceAdjoint.resource_rep(
                    base_class=re.ResourceQROM, 
                    base_params={
                        "num_bitstrings": 2**num_state_qubits,
                        "num_bit_flips": 2**(num_state_qubits - 1),
                        "num_control_wires": num_state_qubits,
                        "num_work_wires": num_work_wires,
                        "size_bitstring": num_precision_wires,
                        "clean": False,
                    }
                )
            ] += 1

        return gate_types

    def resource_params(self) -> dict:
        """The key parameters required to expand the resources of QROMPrep."""
        state_vector = self.state_vector
        positive_and_real = True
        
        for c in state_vector:
            if c.imag != 0 or c.real < 0:
                positive_and_real = False
                break
        
        num_state_qubits = int(math.log2(len(self.state_vector)))
        num_precision_wires = len(self.hyperparameters["precision_wires"])
        num_work_wires = len(self.hyperparameters["work_wires"])
        
        return {
            "num_state_qubits": num_state_qubits,
            "num_precision_wires": num_precision_wires,
            "num_work_wires": num_work_wires,
            "positive_and_real": positive_and_real,
        }

    @classmethod
    def resource_rep(cls, num_state_qubits, num_precision_wires, num_work_wires, positive_and_real):
        params = {
            "num_state_qubits": num_state_qubits,
            "num_precision_wires": num_precision_wires,
            "num_work_wires": num_work_wires,
            "positive_and_real": positive_and_real,
        }
        return re.CompressedResourceOp(cls, params)
