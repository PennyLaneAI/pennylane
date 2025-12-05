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
r"""Contains the MultiplexerStatePreparation template."""

import numpy as np

import pennylane as qml
from pennylane import math
from pennylane.operation import Operation
from pennylane.wires import Wires


class MultiplexerStatePreparation(Operation):
    r"""Prepares a quantum state using multiplexed rotations.

    This operation implements the state preparation method described
    in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_.

    Args:
        state_vector (tensor_like): The state vector of length :math:`2^n` to be prepared on :math:`n` wires.
        wires (Sequence[int]): The wires on which to prepare the state.

    Raises:
        ValueError: If the length of the input state vector array is not :math:`2^n` where :math:`n` is an integer, or if
            its norm is not equal to one.

    **Example**

    .. code-block:: python

        import numpy as np

        probs_vector = np.array([0.5, 0., 0.25, 0.25])

        dev = qml.device("default.qubit", wires = 2)

        wires = [0,1]

        @qml.qnode(dev)
        def circuit():
            qml.MultiplexerStatePreparation(
                np.sqrt(probs_vector), wires
            )
            return qml.probs(wires)

    .. code-block:: pycon

        >>> np.round(circuit(), 2)
        array([0.5 , 0.  , 0.25, 0.25])

    .. seealso:: :class:`~.SelectPauliRot`

    """

    resource_keys = {"num_wires"}

    # pylint: disable=too-many-positional-arguments
    def __init__(self, state_vector, wires, id=None):  # pylint: disable=too-many-arguments

        n_amplitudes = math.shape(state_vector)[0]
        if n_amplitudes != 2 ** len(Wires(wires)):
            raise ValueError(
                f"State vectors must be of length {2 ** len(wires)}; vector has length {n_amplitudes}."
            )

        norm = math.linalg.norm(state_vector)
        if not math.allclose(norm, 1.0, atol=1e-3):
            raise ValueError(
                f"Input state vectors must have a norm 1.0, the vector has squared norm {norm}"
            )

        self.state_vector = state_vector
        wires = Wires(wires)
        super().__init__(state_vector, wires=wires, id=id)

    @classmethod
    def _primitive_bind_call(cls, *args, **kwargs):
        return cls._primitive.bind(*args, **kwargs)

    @property
    def resource_params(self) -> dict:
        return {
            "num_wires": len(self.wires),
        }

    @staticmethod
    def compute_decomposition(state_vector, wires):  # pylint: disable=arguments-differ
        with qml.queuing.AnnotatedQueue() as q:
            _multiplexer_state_prep_decomposition(state_vector, wires)

        if qml.queuing.QueuingManager.recording():
            for op in q.queue:
                qml.apply(op)

        return q.queue


def _multiplexer_state_prep_decomposition_resources(num_wires) -> dict:
    resources = dict.fromkeys(
        [
            qml.resource_rep(qml.SelectPauliRot, num_wires=i + 1, rot_axis="Y")
            for i in range(num_wires)
        ],
        1,
    )

    resources[qml.resource_rep(qml.DiagonalQubitUnitary, num_wires=num_wires)] = 1

    return resources


@qml.register_resources(_multiplexer_state_prep_decomposition_resources, exact=False)
def _multiplexer_state_prep_decomposition(state_vector, wires):  # pylint: disable=arguments-differ
    r"""
    Computes the decomposition operations for the given state vector.

    Args:
        state_vector (tensor_like): The state vector to prepare.
        wires (Sequence[int]): The wires which the operator acts on.

    Returns:
        list: List of decomposition operations.
    """

    probs = math.abs(state_vector) ** 2
    phases = math.angle(state_vector) % (2 * np.pi)
    eps = 1e-15  # Small constant to avoid division by zero

    num_iterations = int(math.log2(math.shape(probs)[0]))

    shapes = [[int(2 ** (i + 1)), -1] for i in range(num_iterations)]
    for i in range(num_iterations):

        probs_aux = math.reshape(probs, [1, -1])

        # Calculation of the numerator and denominator of the function f(x) (Eq.5 [arXiv:quant-ph/0208112])
        for itx in range(i + 1):
            probs_denominator = math.sum(probs_aux, axis=1)
            probs_aux = math.reshape(probs_aux, shapes[itx])
            probs_numerator = math.sum(probs_aux, axis=1)[::2]

        # Compute the angles θi
        thetas = [
            2 * math.arccos(math.sqrt(probs_numerator[j] / (probs_denominator[j] + eps)))
            for j in range(math.shape(probs_numerator)[0])
        ]
        # Apply the SelectPauliRot operation to apply the theta rotations
        qml.SelectPauliRot(thetas, target_wire=wires[i], control_wires=wires[:i], rot_axis="Y")

    if not qml.math.is_abstract(phases):
        if not math.allclose(phases, 0.0):
            # Compute the phases
            thetas = [1j * phase for phase in phases]

            # Apply the DiagonalQubitUnitary operation to encode the phases
            qml.DiagonalQubitUnitary(math.exp(thetas), wires=wires)


qml.add_decomps(MultiplexerStatePreparation, _multiplexer_state_prep_decomposition)
