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
r"""
Contains the QROMStatePreparation template.
"""

import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from pennylane.wires import Wires

import itertools


def _sum_by_prefix(vector, prefix):
    """Calculates the sum of elements in a vector whose index, when represented in binary, starts with a given prefix.

    Args:
        vector (TensorLike): A 1D vector of numerical values.
        prefix (str): A string representing the binary prefix to match.

    Returns:
        (float) The sum of the elements in the vector whose index binary representation starts with the given prefix.


    Example:
        >>> vector = [1, 3, 5, 2, 1, 3, 2, 2]
        >>> prefix = "10"
        >>> _sum_by_prefix(vector, prefix)
        1 + 3 = 4  # Elements at indices 4 and 5 (binary 100 and 101) are summed.

        >>> prefix = "01"
        >>> _sum_by_prefix(vector, prefix)
        5 + 2 = 7 # Elements at indices 2 and 3 (binary 010 and 011) are summed.

        >>> prefix = "1"
        >>> _sum_by_prefix(vector, prefix)
        1 + 3 + 2 + 2 = 8 # Elements at indices 4, 5, 6, and 7 (binary 100, 101, 110 and 111) are summed.
    """

    n = len(vector).bit_length() - 1
    sum_result = 0
    for i, value in enumerate(vector):
        bitstring = qml.math.binary_repr(i, n)
        if bitstring.startswith(prefix):
            sum_result += value
    return sum_result


def _get_basis_state_list(n_wires, add_zero=False):
    """Generates a list of binary strings representing basis states.

    Args:
        n_wires (int): The number of wires in the system.
        add_zero (bool, optional): Whether to append a '0' to each binary string. Defaults to False.

    Returns:
        list[str]: A list of binary strings representing the basis states.
        Each string has length `n_wires` (or `n_wires + 1` if `add_zero` is True).

    Example:
        >>> _get_basis_state_list(2)
        ['00', '01', '10', '11']

        >>> _get_basis_state_list(3, add_zero=True)
        ['0000', '0010', '0100', '0110', '1000', '1010', '1100', '1110']
    """

    if add_zero:
        return ["".join(map(str, bits)) + "0" for bits in itertools.product([0, 1], repeat=n_wires)]
    else:
        return ["".join(map(str, bits)) for bits in itertools.product([0, 1], repeat=n_wires)]


def _func_to_binary(n_precision, x, func):
    """Converts a value within the range [0, 1) to its binary representation with a specified precision.

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
        >>> _func_to_binary(3, 0.25, lambda x: np.sqrt(x))
        '100'

        Expected value as `\sqrt{0.25} = 0.5`, and it's binary representation is `0.100`.

    """

    binary_rep = bin(int(2 ** (n_precision + 1) + 2 ** (n_precision) * func(x)))
    if binary_rep[-n_precision - 1] == "1":
        return "1" * n_precision

    return binary_rep[-n_precision:]


class QROMStatePreparation(Operation):
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
        state_vector = np.array([1/2,-1/2,1/2,1/2])
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

        This operation implements the state preparation method described
        in `arXiv:quant-ph/0208112 <https://arxiv.org/abs/quant-ph/0208112>`. It uses a QROM to store
        the binary representations of the amplitudes and phases of the target state, and then uses
        controlled rotations to apply these values to the target qubits.

        The input `state_vector` must have a length that is a power of 2. The number of ``wires``
        must be :math:`\log_2(\text{len}(state\_vector))`. The number of ``precision_wires`` determines the
        precision with which the amplitudes and phases are encoded.

        The ``work_wires`` are used as auxiliary qubits in the QROM operation.

        The decomposition involves encoding the probabilities and phases of the state vector using
        QROMs and then applying controlled rotations based on the values stored in the `precision_wires`.
        The decomposition applies CRY rotations for amplitude encoding and controlled GlobalPhase
        rotations for the phase encoding.
    """

    def __init__(self, state_vector, wires, precision_wires, work_wires=None, id=None):

        n_amplitudes = len(state_vector)
        if n_amplitudes != 2 ** len(Wires(wires)):
            raise ValueError(
                f"State vectors must be of length {2 ** len(wires)}; vector has length {n_amplitudes}."
            )

        if not qml.math.is_abstract(state_vector[0]):
            norm = qml.math.sum(qml.math.abs(state_vector) ** 2)
            if not qml.math.allclose(norm, 1.0, atol=1e-3):
                raise ValueError(
                    f"State vectors have to be of norm 1.0, vector has squared norm {norm}"
                )

        self.state_vector = state_vector
        self.hyperparameters["input_wires"] = qml.wires.Wires(wires)
        self.hyperparameters["precision_wires"] = qml.wires.Wires(precision_wires)
        self.hyperparameters["work_wires"] = qml.wires.Wires(
            () if work_wires is None else work_wires
        )

        all_wires = (
            self.hyperparameters["input_wires"]
            + self.hyperparameters["precision_wires"]
            + self.hyperparameters["work_wires"]
        )

        super().__init__(state_vector, wires=all_wires, id=id)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    def _flatten(self):
        hyperparameters = (
            ("wires", self.hyperparameters["input_wires"]),
            ("precision_wires", self.hyperparameters["precision_wires"]),
            ("work_wires", self.hyperparameters["work_wires"]),
        )
        return (self.state_vector,), hyperparameters

    @classmethod
    def _unflatten(cls, data, metadata):
        hyperparams_dict = dict(metadata)
        return cls(data[0], **hyperparams_dict)

    def map_wires(self, wire_map):
        new_wires = Wires(
            [wire_map.get(wire, wire) for wire in self.hyperparameters["input_wires"]]
        )
        new_precision_wires = Wires(
            [wire_map.get(wire, wire) for wire in self.hyperparameters["precision_wires"]]
        )

        new_work_wires = Wires(
            [wire_map.get(wire, wire) for wire in self.hyperparameters["work_wires"]]
        )
        return QROMStatePreparation(
            self.state_vector, new_wires, new_precision_wires, new_work_wires
        )

    def decomposition(self):  # pylint: disable=arguments-differ
        filtered_hyperparameters = {
            key: value for key, value in self.hyperparameters.items() if key != "input_wires"
        }
        return self.compute_decomposition(
            self.parameters[0],
            wires=self.hyperparameters["input_wires"],
            **filtered_hyperparameters,
        )

    @staticmethod
    def compute_decomposition(
        state_vector, wires, precision_wires, work_wires
    ):  # pylint: disable=arguments-differ
        """
        Computes the decomposition operations for the given state vector.

        Args:

            state_vector (TensorLike): The state vector to prepare.
            wires (Sequence[int]): The wires on which to prepare the state.
            precision_wires (Sequence[int]): The wires used for storing the binary representations of the
                amplitudes and phases.
            work_wires (Sequence[int], optional):  The wires used as work wires for the QROM operations. Defaults to ``None``.

        Returns:
            list: List of decomposition operations.
        """

        probs = qml.math.abs(state_vector) ** 2
        phases = qml.math.angle(state_vector) % (2 * np.pi)

        decomp_ops = []
        num_iterations = int(qml.math.log2(len(probs)))
        for i in range(num_iterations):

            # Calculation of the numerator and denominator of the function f(x) (Eq.5 [arXiv:quant-ph/0208112])
            prefixes = _get_basis_state_list(n_wires=i)
            probs_denominator = [_sum_by_prefix(probs, prefix=p) for p in prefixes]

            prefixes_with_zero = _get_basis_state_list(n_wires=i, add_zero=True)
            probs_numerator = [_sum_by_prefix(probs, prefix=p) for p in prefixes_with_zero]

            eps = 1e-8  # Small constant to avoid division by zero

            # Compute the binary representations of the angles θi
            func = lambda x: 2 * qml.math.arccos(qml.math.sqrt(x)) / np.pi
            thetas_binary = [
                _func_to_binary(
                    len(precision_wires), probs_numerator[j] / (probs_denominator[j] + eps), func
                )
                for j in range(len(probs_numerator))
            ]
            # Apply the QROM operation to encode the thetas binary representation
            decomp_ops.append(
                qml.QROM(
                    bitstrings=thetas_binary,
                    target_wires=precision_wires,
                    control_wires=wires[:i],
                    work_wires=work_wires,
                    clean=False,
                )
            )

            # Turn binary representation into proper rotation
            for ind, wire in enumerate(precision_wires):
                rotation_angle = 2 ** (-ind - 1)
                decomp_ops.append(qml.CRY(np.pi * rotation_angle, wires=[wire, wires[i]]))

            # Clean wires used to store the theta values
            decomp_ops.append(
                qml.adjoint(qml.QROM)(
                    bitstrings=thetas_binary,
                    target_wires=precision_wires,
                    control_wires=wires[:i],
                    work_wires=work_wires,
                    clean=False,
                )
            )

        if not qml.math.allclose(phases, 0.0):
            # Compute the binary representations of the phases
            func = lambda x: (x) / (2 * np.pi)
            thetas_binary = [_func_to_binary(len(precision_wires), phase, func) for phase in phases]

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

            for ind, wire in enumerate(precision_wires):
                rotation_angle = 2 ** (-ind - 1)
                decomp_ops.append(
                    qml.ctrl(
                        qml.GlobalPhase((2 * np.pi) * (-rotation_angle), wires=wires[0]),
                        control=wire,
                    )
                )

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
