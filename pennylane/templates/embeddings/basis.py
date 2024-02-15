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
Contains the BasisEmbedding template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.wires import Wires


class BasisEmbedding(Operation):
    r"""Encodes :math:`n` binary features into a basis state of :math:`n` qubits.

    For example, for ``features=np.array([0, 1, 0])`` or ``features=2`` (binary 10), the
    quantum system will be prepared in state :math:`|010 \rangle`.

    .. warning::

        ``BasisEmbedding`` calls a circuit whose architecture depends on the binary features.
        The ``features`` argument is therefore not differentiable when using the template, and
        gradients with respect to the argument cannot be computed by PennyLane.

    Args:
        features (tensor_like): binary input of shape ``(len(wires), )``
        wires (Any or Iterable[Any]): wires that the template acts on

    Example:

        Basis embedding encodes the binary feature vector into a basis state.

        .. code-block:: python

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def circuit(feature_vector):
                qml.BasisEmbedding(features=feature_vector, wires=range(3))
                return qml.state()

            X = [1,1,1]

        The resulting circuit is:

        >>> print(qml.draw(circuit, expansion_strategy="device")(X))
        0: ──X─┤  State
        1: ──X─┤  State
        2: ──X─┤  State

        And, the output state is:

        >>> print(circuit(X))
            [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 1.+0.j]

        Thus, ``[1,1,1]`` is mapped to :math:`|111 \rangle`.

    """

    num_wires = AnyWires
    grad_method = None

    def _flatten(self):
        basis_state = self.hyperparameters["basis_state"]
        basis_state = tuple(basis_state) if isinstance(basis_state, list) else basis_state
        return tuple(), (self.wires, basis_state)

    @classmethod
    def _unflatten(cls, _, metadata) -> "BasisEmbedding":
        return cls(features=metadata[1], wires=metadata[0])

    def __init__(self, features, wires, id=None):
        if isinstance(features, list):
            features = qml.math.stack(features)

        tracing = qml.math.is_abstract(features)

        if qml.math.shape(features) == ():
            if not tracing and features >= 2 ** len(wires):
                raise ValueError(
                    f"Features must be of length {len(wires)}, got features={features} which is >= {2 ** len(wires)}"
                )
            bin = 2 ** np.arange(len(wires))[::-1]
            features = qml.math.where((features & bin) > 0, 1, 0)

        wires = Wires(wires)
        shape = qml.math.shape(features)

        if len(shape) != 1:
            raise ValueError(f"Features must be one-dimensional; got shape {shape}.")

        n_features = shape[0]
        if n_features != len(wires):
            raise ValueError(
                f"Features must be of length {len(wires)}; got length {n_features} (features={features})."
            )

        if not tracing:
            features = list(qml.math.toarray(features))
            if not set(features).issubset({0, 1}):
                raise ValueError(f"Basis state must only consist of 0s and 1s; got {features}")

        self._hyperparameters = {"basis_state": features}

        super().__init__(wires=wires, id=id)

    @property
    def num_params(self):
        return 0

    @staticmethod
    def compute_decomposition(wires, basis_state):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.BasisEmbedding.decomposition`.

        Args:
            features (tensor-like): binary input of shape ``(len(wires), )``
            wires (Any or Iterable[Any]): wires that the operator acts on

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> features = torch.tensor([1, 0, 1])
        >>> qml.BasisEmbedding.compute_decomposition(features, wires=["a", "b", "c"])
        [X('a'),
         X('c')]
        """
        if not qml.math.is_abstract(basis_state):
            ops_list = []
            for wire, bit in zip(wires, basis_state):
                if bit == 1:
                    ops_list.append(qml.X(wire))
            return ops_list

        ops_list = []
        for wire, state in zip(wires, basis_state):
            ops_list.append(qml.PhaseShift(state * np.pi / 2, wire))
            ops_list.append(qml.RX(state * np.pi, wire))
            ops_list.append(qml.PhaseShift(state * np.pi / 2, wire))

        return ops_list
