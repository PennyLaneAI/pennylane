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

from itertools import combinations

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import RZ, H, MultiRZ
from pennylane.typing import Float, Wire
from pennylane.wires import WiresLike


class IQPEmbedding(Operator2):
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

    * Else, ``pattern`` is a list of index pairs ``[[a, b], [c, d], ...]`` into ``wires``, applying
      the entangler on ``(wires[a], wires[b])``, ``(wires[c], wires[d])``, etc. For example,
      ``pattern = [[0, 1], [1, 2]]`` produces the following entangler pattern:

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
        pattern (list[list[int]]): pairs of indices into ``wires`` that specify the entanglers.
            ``None`` (default) applies an entangler to every pair of wires.

    Raises:
        ValueError: if inputs do not have the correct format

    .. details::
        :title: Usage Details

        A typical usage example of the template is the following:

        .. code-block:: python

            import pennylane as qp

            dev = qp.device('default.qubit', wires=3)

            @qp.qnode(dev)
            def circuit(features):
                qp.IQPEmbedding(features, wires=range(3))
                return [qp.expval(qp.Z(w)) for w in range(3)]

            circuit([1., 2., 3.])

        **Repeating the embedding**

        The embedding can be repeated by specifying the ``n_repeats`` argument:

        .. code-block:: python

            @qp.qnode(dev)
            def circuit(features):
                qp.IQPEmbedding(features, wires=range(3), n_repeats=4)
                return [qp.expval(qp.Z(w)) for w in range(3)]

            circuit([1., 2., 3.])

        Every repetition uses exactly the same quantum circuit.

        **Using a custom entangler pattern**

        A custom entangler pattern can be used by specifying the ``pattern`` argument. A pattern has to be
        a nested list of dimension ``(K, 2)``, where ``K`` is the number of entanglers to apply.
        Each pair contains indices into ``wires``, not wire labels. For example, with
        ``wires=["z", "a", "k"]``, ``pattern=[[0, 2]]`` entangles ``"z"`` and ``"k"``.

        .. code-block:: python

            pattern = [[1, 2], [0, 2], [1, 0]]

            @qp.qnode(dev)
            def circuit(features):
                qp.IQPEmbedding(features, wires=range(3), pattern=pattern)
                return [qp.expval(qp.Z(w)) for w in range(3)]

            circuit([1., 2., 3.])

        Since diagonal gates commute, the order of the wire pairs has no effect on the result.

        .. code-block:: python

            from pennylane import numpy as np

            pattern1 = [[1, 2], [0, 2], [1, 0]]
            pattern2 = [[1, 0], [0, 2], [1, 2]]  # a reshuffling of pattern1

            @qp.qnode(dev)
            def circuit(features, pattern):
                qp.IQPEmbedding(features, wires=range(3), pattern=pattern, n_repeats=3)
                return [qp.expval(qp.Z(w)) for w in range(3)]

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
        feature to the wire pair ``[2, 0]``, the product of the first and third feature to ``[2, 1]``,
        and the product of the second and third feature to ``[0, 1]``.

    """

    dynamic_argnames = ("features",)
    compilable_argnames = ("n_repeats", "pattern")
    arg_specs = {"features": Float[-1], "wires": Wire[-1]}

    ndim_params = (1,)

    def __init__(self, features, wires, n_repeats=1, pattern=None):
        if isinstance(features, (list, tuple)):
            features = math.stack(features)

        shape = math.shape(features)

        if len(shape) not in {1, 2}:
            raise ValueError(
                "Features must be a one-dimensional tensor, or two-dimensional "
                f"when broadcasting; got shape {shape}."
            )

        n_features = shape[-1]
        if n_features != len(wires):
            raise ValueError(f"Features must be of length {len(wires)}; got length {n_features}.")

        # ``pattern`` is compilable, so store hashable nested tuples of indices into ``wires``.
        if pattern is None:
            pattern = tuple(combinations(range(len(wires)), 2))
        else:
            pattern = tuple(tuple(pair) for pair in pattern)

        super().__init__(features, wires, n_repeats, pattern)


# pylint: disable=unused-argument
def _iqp_embedding_resources(features, wires, n_repeats, pattern):
    return {
        RZ: n_repeats * len(wires),
        H: n_repeats * len(wires),
        MultiRZ(Float, Wire[2]): len(pattern) * n_repeats,
    }


@register_resources(_iqp_embedding_resources, exact=False)
def _iqp_embedding_decomposition(features, wires: WiresLike, n_repeats, pattern):

    if capture.enabled() or compiler.active():
        wires, pattern, features = (
            math.array(wires, like="jax"),
            math.array(pattern, like="jax"),
            math.array(features, like="jax"),
        )

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
            idx1, idx2 = pattern[j][0], pattern[j][1]
            MultiRZ(features[idx1] * features[idx2], wires=[wires[idx1], wires[idx2]])

        pattern_loop()  # pylint: disable=no-value-for-parameter

    outer_loop()  # pylint: disable=no-value-for-parameter


add_decomps(IQPEmbedding, _iqp_embedding_decomposition)
