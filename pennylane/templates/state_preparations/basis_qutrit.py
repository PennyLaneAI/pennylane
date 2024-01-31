# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Contains the BasisStatePreparation template.
"""

import pennylane as qml
from pennylane.operation import Operation, AnyWires


class QutritBasisStatePreparation(Operation):
    r"""
    Prepares a basis state on the given wires using a sequence of TShift gates.

    .. warning::

        ``basis_state`` influences the circuit architecture and is therefore incompatible with
        gradient computations.

    Args:
        basis_state (array): Input array of shape ``(n,)``, where n is the number of wires
            the state preparation acts on.
        wires (Iterable): wires that the template acts on

    **Example**

    .. code-block:: python

        dev = qml.device("default.qutrit", wires=4)

        @qml.qnode(dev)
        def circuit(basis_state, obs):
            qml.QutritBasisStatePreparation(basis_state, wires=range(4))
            return [qml.expval(qml.THermitian(obs, wires=i)) for i in range(4)]

        basis_state = [0, 1, 1, 0]
        obs = np.array([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)

    >>> print(circuit(basis_state, obs))
    [array(0.70710678), array(-0.70710678), array(-0.70710678), array(0.70710678)]
    """

    num_params = 1
    num_wires = AnyWires
    grad_method = None

    def __init__(self, basis_state, wires, id=None):
        basis_state = qml.math.stack(basis_state)

        # check if the `basis_state` param is batched
        batched = len(qml.math.shape(basis_state)) > 1

        state_batch = basis_state if batched else [basis_state]

        for i, state in enumerate(state_batch):
            shape = qml.math.shape(state)

            if len(shape) != 1:
                raise ValueError(
                    f"Basis states must be one-dimensional; state {i} has shape {shape}."
                )

            n_bits = shape[0]
            if n_bits != len(wires):
                raise ValueError(
                    f"Basis states must be of length {len(wires)}; state {i} has length {n_bits}."
                )

            if any(bit not in [0, 1, 2] for bit in state):
                raise ValueError(
                    f"Basis states must only consist of 0s, 1s, and 2s; state {i} is {state}"
                )

        # TODO: basis_state should be a hyperparameter, not a trainable parameter.
        # However, this breaks a test that ensures compatibility with batch_transform.
        # The transform should be rewritten to support hyperparameters as well.
        super().__init__(basis_state, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(basis_state, wires):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.


         .. seealso:: :meth:`~.BasisState.decomposition`.

        Args:
            basis_state (array): Input array of shape ``(len(wires),)``
            wires (Any or Iterable[Any]): wires that the operator acts on

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> qml.QutritBasisStatePreparation.compute_decomposition(basis_state=[1, 2], wires=["a", "b"])
        [Tshift(wires=['a']),
        Tshift(wires=['b']),
        TShift(wires=['b'])]
        """

        op_list = []
        for wire, state in zip(wires, basis_state):
            for _ in range(0, state):
                op_list.append(qml.TShift(wire))
        return op_list
