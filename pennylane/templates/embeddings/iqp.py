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
Contains the IQPEmbedding template.
"""
# pylint: disable=too-many-arguments
import copy
from itertools import combinations

from pennylane import capture, math
from pennylane.control_flow import for_loop, while_loop
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.operation import Operation
from pennylane.ops import RZ, H, MultiRZ
from pennylane.wires import Wires

has_jax = True
try:
    from jax import numpy as jnp
except ModuleNotFoundError:  # pragma: no cover
    has_jax = False  # pragma: no cover


class IQPEmbedding(Operation):
    r"""
    Encodes :math:`n` features into :math:`n` qubits using diagonal gates of an IQP circuit.

    The embedding has been proposed by `Havlicek et al. (2018) <https://arxiv.org/abs/1804.11326>`_.

    The basic IQP circuit can be repeated by specifying ``n_repeats``. Repetitions can make the
    embedding "richer" through interference.

    .. warning::

        ``IQPEmbedding`` calls a circuit that involves non-trivial classical processing of the
        features. The ``features`` argument is therefore **not differentiable** when using the template, and
        gradients with respect to the features cannot be computed by PennyLane.

    An IQP circuit is a quantum circuit of a block of Hadamards, followed by a block of gates that are
    diagonal in the computational basis. Here, the diagonal gates are single-qubit ``RZ`` rotations, applied to each
    qubit and encoding the :math:`n` features, followed by two-qubit ZZ entanglers,
    :math:`e^{-i x_i x_j \sigma_z \otimes \sigma_z}`. The entangler applied to wires ``(wires[i], wires[j])``
    encodes the product of features ``features[i]*features[j]``. The pattern in which the entanglers are
    applied is either the default, or a custom pattern:

    * If ``pattern`` is not specified, the default pattern will be used, in which the entangling gates connect all
      pairs of neighbours:

      |

      .. figure:: ../../_static/templates/embeddings/iqp.png
          :align: center
          :width: 50%
          :target: javascript:void(0);

      |

    * Else, ``pattern`` is a list of wire pairs ``[[a, b], [c, d],...]``, applying the entangler
      on wires ``[a, b]``, ``[c, d]``, etc. For example, ``pattern = [[0, 1], [1, 2]]`` produces
      the following entangler pattern:

      |

      .. figure:: ../../_static/templates/embeddings/iqp_custom.png
          :align: center
          :width: 50%
          :target: javascript:void(0);

      |

      Since diagonal gates commute, the order of the entanglers does not change the result.

    Args:
        features (tensor_like): tensor of features to encode
        wires (Any or Iterable[Any]): wires that the template acts on
        n_repeats (int): number of times the basic embedding is repeated
        pattern (list[int]): specifies the wires and features of the entanglers

    Raises:
        ValueError: if inputs do not have the correct format

    .. details::
        :title: Usage Details

        A typical usage example of the template is the following:

        .. code-block:: python

            import pennylane as qp

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def circuit(features):
                qml.IQPEmbedding(features, wires=range(3))
                return [qml.expval(qml.Z(w)) for w in range(3)]

            circuit([1., 2., 3.])

        **Repeating the embedding**

        The embedding can be repeated by specifying the ``n_repeats`` argument:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit(features):
                qml.IQPEmbedding(features, wires=range(3), n_repeats=4)
                return [qml.expval(qml.Z(w)) for w in range(3)]

            circuit([1., 2., 3.])

        Every repetition uses exactly the same quantum circuit.

        **Using a custom entangler pattern**

        A custom entangler pattern can be used by specifying the ``pattern`` argument. A pattern has to be
        a nested list of dimension ``(K, 2)``, where ``K`` is the number of entanglers to apply.

        .. code-block:: python

            pattern = [[1, 2], [0, 2], [1, 0]]

            @qml.qnode(dev)
            def circuit(features):
                qml.IQPEmbedding(features, wires=range(3), pattern=pattern)
                return [qml.expval(qml.Z(w)) for w in range(3)]

            circuit([1., 2., 3.])

        Since diagonal gates commute, the order of the wire pairs has no effect on the result.

        .. code-block:: python

            from pennylane import numpy as np

            pattern1 = [[1, 2], [0, 2], [1, 0]]
            pattern2 = [[1, 0], [0, 2], [1, 2]]  # a reshuffling of pattern1

            @qml.qnode(dev)
            def circuit(features, pattern):
                qml.IQPEmbedding(features, wires=range(3), pattern=pattern, n_repeats=3)
                return [qml.expval(qml.Z(w)) for w in range(3)]

            res1 = circuit([1., 2., 3.], pattern=pattern1)
            res2 = circuit([1., 2., 3.], pattern=pattern2)

            assert np.allclose(res1, res2)

        **Non-consecutive wires**

        In principle, the user can also pass a non-consecutive wire list to the template.
        For single qubit gates, the i'th feature is applied to the i'th wire index (which may not be the i'th wire).
        For the entanglers, the product of i'th and j'th features is applied to the wire indices at the i'th and j'th
        position in ``wires``.

        For example, for ``wires=[2, 0, 1]`` the ``RZ`` block applies the first feature to wire 2,
        the second feature to wire 0, and the third feature to wire 1.

        Likewise, using the default pattern, the entangler block applies the product of the first and second
        feature to the wire pair ``[2, 0]``, the product of the second and third feature to ``[2, 1]``, and so
        forth.

    """

    grad_method = None

    resource_keys = {"pattern_size", "n_repeats", "num_wires"}

    def __init__(self, features, wires, n_repeats=1, pattern=None, id=None):
        shape = math.shape(features)

        if len(shape) not in {1, 2}:
            raise ValueError(
                "Features must be a one-dimensional tensor, or two-dimensional "
                f"when broadcasting; got shape {shape}."
            )

        n_features = shape[-1]
        if n_features != len(wires):
            raise ValueError(f"Features must be of length {len(wires)}; got length {n_features}.")

        if pattern is None:
            # default is an all-to-all pattern
            pattern = tuple(combinations(wires, 2))
        self._hyperparameters = {"pattern": pattern, "n_repeats": n_repeats}

        super().__init__(features, wires=wires, id=id)

    @property
    def resource_params(self) -> dict:
        return {
            "pattern_size": len(self.hyperparameters["pattern"]),
            "n_repeats": self.hyperparameters["n_repeats"],
            "num_wires": len(self.wires),
        }

    def map_wires(self, wire_map):
        # pylint: disable=protected-access
        new_op = copy.deepcopy(self)
        new_op._wires = Wires([wire_map.get(wire, wire) for wire in self.wires])
        new_op._hyperparameters["pattern"] = [
            [wire_map.get(w, w) for w in wires] for wires in new_op._hyperparameters["pattern"]
        ]
        return new_op

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (1,)

    @staticmethod
    def compute_decomposition(
        features, wires, n_repeats, pattern
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.



        .. seealso:: :meth:`~.IQPEmbedding.decomposition`.

        Args:
            features (tensor_like): tensor of features to encode
            wires (Any or Iterable[Any]): wires that the template acts on

        Returns:
            list[.Operator]: decomposition of the operator

        **Example**

        >>> features = torch.tensor([1., 2., 3.])
        >>> pattern = [(0, 1), (0, 2), (1, 2)]
        >>> qml.IQPEmbedding.compute_decomposition(features, wires=[0, 1, 2], n_repeats=2, pattern=pattern)
        [H(0), RZ(tensor(1.), wires=[0]),
         H(1), RZ(tensor(2.), wires=[1]),
         H(2), RZ(tensor(3.), wires=[2]),
         MultiRZ(tensor(2.), wires=[0, 1]), MultiRZ(tensor(3.), wires=[0, 2]), MultiRZ(tensor(6.), wires=[1, 2]),
         H(0), RZ(tensor(1.), wires=[0]),
         H(1), RZ(tensor(2.), wires=[1]),
         H(2), RZ(tensor(3.), wires=[2]),
         MultiRZ(tensor(2.), wires=[0, 1]), MultiRZ(tensor(3.), wires=[0, 2]), MultiRZ(tensor(6.), wires=[1, 2])]
        """
        wires = Wires(wires)
        op_list = []
        if math.ndim(features) > 1:
            # If broadcasting is used, we want to iterate over the wires axis of the features,
            # not over the broadcasting dimension. The latter is passed on to the rotations.
            features = math.T(features)

        for _ in range(n_repeats):
            for i in range(len(wires)):  # pylint: disable=consider-using-enumerate
                op_list.append(H(wires=wires[i]))
                op_list.append(RZ(features[i], wires=wires[i]))

            for wire_pair in pattern:
                # get the position of the wire indices in the array
                idx1, idx2 = wires.indices(wire_pair)
                # apply product of two features as entangler
                op_list.append(MultiRZ(features[idx1] * features[idx2], wires=wire_pair))

        return op_list


def _iqp_embedding_resources(pattern_size, n_repeats, num_wires):
    return {
        resource_rep(RZ): n_repeats * num_wires,
        resource_rep(H): n_repeats * num_wires,
        resource_rep(MultiRZ, num_wires=2): pattern_size * n_repeats,
    }


@register_resources(_iqp_embedding_resources)
def _iqp_embedding_decomposition(features, wires, n_repeats, pattern):

    if has_jax and capture.enabled():
        wires, pattern, features = jnp.array(wires), jnp.array(pattern), jnp.array(features)

    if math.ndim(features) > 1:
        features = math.T(features)

    @for_loop(n_repeats)
    def outer_loop(_):

        @for_loop(len(wires))
        def single_qubit_loop(i):
            H(wires=wires[i])
            RZ(features[i], wires=wires[i])

        single_qubit_loop()  # pylint: disable=no-value-for-parameter

        @for_loop(len(pattern))
        def pattern_loop(j):

            @while_loop(lambda curr: wires[curr] != pattern[j][0])
            def search_loop_1(curr):
                curr += 1
                return curr

            idx1 = search_loop_1(0)

            @while_loop(lambda curr: wires[curr] != pattern[j][1])
            def search_loop_2(curr):
                curr += 1
                return curr

            idx2 = search_loop_2(0)

            MultiRZ(features[idx1] * features[idx2], wires=pattern[j])

        pattern_loop()  # pylint: disable=no-value-for-parameter

    outer_loop()  # pylint: disable=no-value-for-parameter


add_decomps(IQPEmbedding, _iqp_embedding_decomposition)
