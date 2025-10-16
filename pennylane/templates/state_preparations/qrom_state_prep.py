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
r"""Contains the QROMStatePreparation template."""

import numpy as np

import pennylane as qml
from pennylane.operation import Operation
from pennylane.wires import Wires


def _float_to_binary(val, num_bits):
    r"""Converts a value within the range [0, 1) to its binary representation with a specified precision.

    Args:
        val (float): The value to convert to binary. Must be in the range [0, 1).
        num_bits (int): the number of bits to use for the binary representation

    Returns:
        str: The binary representation of the value, with the specified precision.

    **Example**

        >>> _float_to_binary(0.5, 3)
        '100'

        Expected value as the binary representation of `0.5` is `0.100`.
    """

    binary_rep = bin(int(2 ** (num_bits + 1) + 2 ** (num_bits) * val))
    if binary_rep[-num_bits - 1] == "1":
        return "1" * num_bits

    return binary_rep[-num_bits:]


class QROMStatePreparation(Operation):
    r"""Prepares a quantum state using Quantum Read-Only Memory (QROM).

    This operation implements the state preparation method described
    in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_.

    Args:
        state_vector (tensor_like): The state vector of length :math:`2^n` to be prepared on :math:`n` wires.
        wires (Sequence[int]): The wires on which to prepare the state.
        precision_wires (Sequence[int]): The wires allocated for storing the binary representations of the
            rotation angles utilized in the template.
        work_wires (Sequence[int], optional):  The work wires used for the QROM operations. Defaults to ``None``.

    Raises:
        ValueError: If the length of the input state vector array is not :math:`2^n` where :math:`n` is an integer, or if
            its norm is not equal to one.

    **Example**

    .. code-block:: python

        import numpy as np

        probs_vector = np.array([0.5, 0., 0.25, 0.25])

        dev = qml.device("default.qubit", wires = 6)

        wires = qml.registers({"work_wires": 1, "prec_wires": 3, "state_wires": 2})

        @qml.qnode(dev)
        def circuit():
            qml.QROMStatePreparation(
                np.sqrt(probs_vector), wires["state_wires"], wires["prec_wires"], wires["work_wires"]
            )
            return qml.probs(wires["state_wires"])

    .. code-block:: pycon

        >>> circuit()
        array([0.5 , 0.  , 0.25, 0.25])

    .. seealso:: :class:`~.QROM`

    .. details::
        :title: Usage Details

        The ``precision_wires`` are used as the target wires in the underlying QROM operations.
        The number of ``precision_wires`` determines the precision with which the rotation angles of the
        template are encoded. This means that the binary representation of the angle is truncated up to
        the :math:`m`-th digit, where :math:`m` is the number of precision wires given. See  Eq. 5 in
        `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_ for more details.
        The ``work_wires`` correspond to auxiliary qubits that can be specified in :class:`~.QROM` to
        reduce the overall resource requirements on the implementation.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self, state_vector, wires, precision_wires, work_wires=None, id=None
    ):  # pylint: disable=too-many-arguments

        n_amplitudes = qml.math.shape(state_vector)[0]
        if n_amplitudes != 2 ** len(Wires(wires)):
            raise ValueError(
                f"State vectors must be of length {2 ** len(wires)}; vector has length {n_amplitudes}."
            )

        norm = qml.math.linalg.norm(state_vector)
        if not qml.math.allclose(norm, 1.0, atol=1e-3):
            raise ValueError(
                f"Input state vectors must have a norm 1.0, the vector has squared norm {norm}"
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

    @staticmethod
    def compute_decomposition(
        state_vector, wires, input_wires, precision_wires, work_wires
    ):  # pylint: disable=arguments-differ
        r"""
        Computes the decomposition operations for the given state vector.

        Args:

            state_vector (tensor_like): The state vector to prepare.
            wires (Sequence[int]): The wires which the operator acts on.
            input_wires (Sequence[int]): The wires on which to prepare the state.
            precision_wires (Sequence[int]): The wires allocated for storing the binary representations of the
                rotation angles utilized in the template.
            work_wires (Sequence[int]):  The wires used as work wires for the QROM operations. Defaults to ``None``.

        Returns:
            list: List of decomposition operations.
        """

        probs = qml.math.abs(state_vector) ** 2
        phases = qml.math.angle(state_vector) % (2 * np.pi)
        eps = 1e-15  # Small constant to avoid division by zero

        decomp_ops = []
        num_iterations = int(qml.math.log2(qml.math.shape(probs)[0]))
        rotation_angles = [2 ** (-ind - 1) for ind in range(len(precision_wires))]

        for i in range(num_iterations):

            probs_aux = qml.math.reshape(probs, [1, -1])

            # Calculation of the numerator and denominator of the function f(x) (Eq.5 [arXiv:quant-ph/0208112])
            for itx in range(i + 1):
                probs_denominator = qml.math.sum(probs_aux, axis=1)
                probs_aux = qml.math.reshape(probs_aux, [int(2 ** (itx + 1)), -1])
                probs_numerator = qml.math.sum(probs_aux, axis=1)[::2]

            # Compute the binary representations of the angles θi
            thetas_binary = [
                _float_to_binary(
                    2
                    * qml.math.arccos(
                        qml.math.sqrt(probs_numerator[j] / (probs_denominator[j] + eps))
                    )
                    / np.pi,
                    len(precision_wires),
                )
                for j in range(qml.math.shape(probs_numerator)[0])
            ]
            # Apply the QROM operation to encode the thetas binary representation
            decomp_ops.append(
                qml.QROM(
                    bitstrings=thetas_binary,
                    target_wires=precision_wires,
                    control_wires=input_wires[:i],
                    work_wires=work_wires,
                    clean=False,
                )
            )

            # Turn binary representation into proper rotation
            for ind, wire in enumerate(precision_wires):
                decomp_ops.append(qml.CRY(np.pi * rotation_angles[ind], wires=[wire, wires[i]]))

            # Clean wires used to store the theta values
            decomp_ops.append(
                qml.adjoint(qml.QROM)(
                    bitstrings=thetas_binary,
                    target_wires=precision_wires,
                    control_wires=input_wires[:i],
                    work_wires=work_wires,
                    clean=False,
                )
            )

        if not qml.math.allclose(phases, 0.0):
            # Compute the binary representations of the phases

            thetas_binary = [
                _float_to_binary(phase / (2 * np.pi), len(precision_wires)) for phase in phases
            ]

            # Apply the QROM operation to encode the thetas binary representation
            decomp_ops.append(
                qml.QROM(
                    bitstrings=thetas_binary,
                    target_wires=precision_wires,
                    control_wires=input_wires,
                    work_wires=work_wires,
                    clean=False,
                )
            )

            for ind, wire in enumerate(precision_wires):
                decomp_ops.append(
                    qml.ctrl(
                        qml.GlobalPhase(
                            (2 * np.pi) * (-rotation_angles[ind]), wires=input_wires[0]
                        ),
                        control=wire,
                    )
                )

            decomp_ops.append(
                qml.adjoint(qml.QROM)(
                    bitstrings=thetas_binary,
                    target_wires=precision_wires,
                    control_wires=input_wires,
                    work_wires=work_wires,
                    clean=False,
                )
            )

        return decomp_ops
