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

import itertools

import numpy as np

import pennylane as qml
from pennylane.operation import Operation
from pennylane.wires import Wires


def _x_to_binary(n_precision, x):
    r"""Converts a value within the range [0, 1) to its binary representation with a specified precision.

    Args:
        n_precision (int): The number of bits to use for the binary representation.
        x (float): The value to convert to binary. Must be in the range [0, 1).

    Returns:
        str: The binary representation of the value, with the specified precision.

    Example:
        >>> _x_to_binary(3, 0.5)
        '100'

        Expected value as the binary representation of `0.5` is `0.100`.
    """

    binary_rep = bin(int(2 ** (n_precision + 1) + 2 ** (n_precision) * x))
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
        in `arXiv:quant-ph/0208112 <https://arxiv.org/abs/quant-ph/0208112>`_. It uses a QROM to store
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

    def __init__(
        self, state_vector, wires, precision_wires, work_wires=None, id=None
    ):  # pylint: disable=too-many-arguments

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
        r"""
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

            probs_aux = probs.reshape(1, -1)

            # Calculation of the numerator and denominator of the function f(x) (Eq.5 [arXiv:quant-ph/0208112])
            for itx in range(i + 1):
                probs_denominator = probs_aux.sum(axis=1)
                probs_aux = probs_aux.reshape(int(2 ** (itx + 1)), -1)
                probs_numerator = probs_aux.sum(axis=1)[::2]

            eps = 1e-8  # Small constant to avoid division by zero

            # Compute the binary representations of the angles θi
            func = lambda x: 2 * qml.math.arccos(qml.math.sqrt(x)) / np.pi
            thetas_binary = [
                _x_to_binary(
                    len(precision_wires), func(probs_numerator[j] / (probs_denominator[j] + eps))
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
            thetas_binary = [_x_to_binary(len(precision_wires), func(phase)) for phase in phases]

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
