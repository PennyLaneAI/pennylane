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
from pennylane import math, queuing
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.operation import Operation
from pennylane.wires import Wires


class MultiplexerStatePreparation(Operation):
    r"""Prepares a quantum state using multiplexed rotations.

    This operation implements the state preparation method described
    in `arXiv:0208112 <https://arxiv.org/abs/quant-ph/0208112>`_.

    Args:
        state_vector (tensor_like): The state vector of length :math:`2^n` to be prepared on
            :math:`n` wires.
        wires (Sequence[int]): The wires on which to prepare the state.

    Raises:
        ValueError: If the length of the input state vector array is not :math:`2^n`, where
            :math:`n` is the number, or if the norm of the input state is not unity.

    **Example**

    .. code-block:: python

        probs_vector = np.array([0.5, 0., 0.25, 0.25])

        dev = qml.device("default.qubit", wires = 2)

        wires = [0, 1]

        @qml.qnode(dev)
        def circuit():
            qml.MultiplexerStatePreparation(np.sqrt(probs_vector), wires)
            return qml.probs(wires)

    .. code-block:: pycon

        >>> np.round(circuit(), 2)
        array([0.5 , 0.  , 0.25, 0.25])

    .. seealso::

        :class:`~.SelectPauliRot` for a description of the main building blocks used to
        implement this operation.

    """

    resource_keys = {"num_wires"}

    # pylint: disable=too-many-positional-arguments, too-many-arguments
    def __init__(self, state_vector, wires, id=None):

        wires = Wires(wires)
        n_amplitudes = math.shape(state_vector)[0]
        if n_amplitudes != 2 ** len(wires):
            raise ValueError(
                f"State vector must be of length {2 ** len(wires)}; got length {n_amplitudes}."
            )

        if not math.is_abstract(state_vector):
            norm = math.linalg.norm(state_vector)
            if not math.allclose(norm, 1.0, atol=1e-3):
                raise ValueError(
                    f"State vector must have norm 1.0; the input state vector has norm {norm}"
                )

        self.state_vector = state_vector
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
        with queuing.AnnotatedQueue() as q:
            _multiplexer_state_prep_decomposition(state_vector, wires)

        if queuing.QueuingManager.recording():
            for op in q.queue:
                qml.apply(op)

        return q.queue


def _multiplexer_state_prep_decomposition_resources(num_wires) -> dict:
    r"""Computes the resources of MultiplexerStatePreparation."""
    resources = dict.fromkeys(
        [resource_rep(qml.SelectPauliRot, num_wires=i + 1, rot_axis="Y") for i in range(num_wires)],
        1,
    )

    resources[resource_rep(qml.DiagonalQubitUnitary, num_wires=num_wires)] = 1

    return resources


@register_resources(_multiplexer_state_prep_decomposition_resources, exact=False)
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

    num_iterations = int(math.log2(math.shape(probs)[0]))

    shapes = []
    for i in range(num_iterations):
        shapes.append([int(2 ** (i + 1)), -1])
        probs_aux = math.reshape(probs, [1, -1])

        # From Eq. 5 of arXiv:quant-ph/0208112.
        for itx in range(i + 1):
            probs_denominator = math.sum(probs_aux, axis=1)
            probs_aux = math.reshape(probs_aux, shapes[itx])
            probs_numerator = math.sum(probs_aux, axis=1)[::2]

        # arcos(x) = arctan2(sqrt(1-x^2), x)
        thetas = 2 * math.arctan2(
            math.sqrt(probs_denominator - probs_numerator),
            math.sqrt(probs_numerator),
        )

        qml.SelectPauliRot(thetas, target_wire=wires[i], control_wires=wires[:i], rot_axis="Y")

    if not math.is_abstract(phases):
        if not math.allclose(phases, 0.0):
            qml.DiagonalQubitUnitary(math.exp(1j * phases), wires=wires)

    else:
        qml.DiagonalQubitUnitary(math.exp(1j * phases), wires=wires)


add_decomps(MultiplexerStatePreparation, _multiplexer_state_prep_decomposition)
