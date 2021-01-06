# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Contains the ``IQPEmbedding`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from collections.abc import Iterable
from itertools import combinations

import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.ops import RZ, MultiRZ, Hadamard
from pennylane.templates import broadcast
from pennylane.templates.utils import (
    check_shape,
    check_type,
    get_shape,
)
from pennylane.wires import Wires


def _preprocess(features, wires, pattern, n_repeats):
    """Validate and pre-process inputs as follows:

    * Check that the features tensor is one-dimensional.
    * Check that the first dimension of the features tensor
      has length :math:`n`, where :math:`n` is the number of qubits.
    * If pattern is None, create default pattern. Else cast wires in user-provided pattern to Wires objects.

    Args:
        features (tensor_like): input features to pre-process
        wires (Wires): wires that template acts on
        pattern (list[int]): specifies the wires and features of the entanglers
        n_repeats (int): number of times the basic embedding is repeated

    Returns:
        list[Wires]: preprocessed pattern
    """

    if qml.tape_mode_active():

        shape = qml.math.shape(features)

        if len(shape) != 1:
            raise ValueError(f"Features must be a one-dimensional tensor; got shape {shape}.")

        n_features = shape[0]
        if n_features != len(wires):
            raise ValueError(f"Features must be of length {len(wires)}; got length {n_features}.")

        if pattern is None:
            # default is an all-to-all pattern
            pattern = [Wires(wire_pair) for wire_pair in combinations(wires, 2)]
        else:
            # convert wire pairs to Wires object
            pattern = [Wires(wire_pair) for wire_pair in pattern]

    else:
        expected_shape = (len(wires),)
        check_shape(
            features,
            expected_shape,
            msg="Features must be of shape {}; got {}"
            "".format(expected_shape, get_shape(features)),
        )

        check_type(
            n_repeats,
            [int],
            msg="'n_repeats' must be an integer; got type {}".format(type(n_repeats)),
        )

        if pattern is None:
            # default is an all-to-all pattern
            pattern = [Wires(wire_pair) for wire_pair in combinations(wires, 2)]
        else:
            # do some checks
            check_type(
                pattern,
                [Iterable, type(None)],
                msg="'pattern' must be a list of pairs of wires; got {}".format(pattern),
            )
            shape = get_shape(pattern)
            if len(shape) != 2 or shape[1] != 2:
                raise ValueError(
                    "'pattern' must be a list of pairs of wires; got {}".format(pattern)
                )

            # convert wire pairs to Wires object
            pattern = [Wires(wire_pair) for wire_pair in pattern]

    return pattern


@template
def IQPEmbedding(features, wires, n_repeats=1, pattern=None):
    r"""
    Encodes :math:`n` features into :math:`n` qubits using diagonal gates of an IQP circuit.

    The embedding has been proposed by `Havlicek et al. (2018) <https://arxiv.org/pdf/1804.11326.pdf>`_.

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
        features (tensor_like): array of features to encode
        wires (Iterable or Wires): Wires that the template acts on. Accepts an iterable of numbers or strings, or
            a Wires object.
        n_repeats (int): number of times the basic embedding is repeated
        pattern (list[int]): specifies the wires and features of the entanglers

    Raises:
        ValueError: if inputs do not have the correct format

    .. UsageDetails::

        A typical usage example of the template is the following:

        .. code-block:: python

            import pennylane as qml
            from pennylane.templates import IQPEmbedding

            dev = qml.device('default.qubit', wires=3)

            @qml.qnode(dev)
            def circuit(features=None):
                IQPEmbedding(features=features, wires=range(3))
                return [qml.expval(qml.PauliZ(w)) for w in range(3)]

            circuit(features=[1., 2., 3.])

        **Do not pass features as a positional argument to the qnode**

        The ``features`` argument cannot be passed to the quantum node
        as a positional argument. This is due to the fact that the embedding performs non-trivial calculations
        on the features. As a consequence, the following code **will produce an error**:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit(features):
               IQPEmbedding(features=features, wires=range(3), n_repeats=2)
               return [qml.expval(qml.PauliZ(w)) for w in range(3)]

            circuit([1., 2., 3.])

        >>> ValueError: 'features' cannot be differentiable

        **Repeating the embedding**

        The embedding can be repeated by specifying the ``n_repeats`` argument:

        .. code-block:: python

            @qml.qnode(dev)
            def circuit(features=None):
                IQPEmbedding(features=features, wires=range(3), n_repeats=4)
                return [qml.expval(qml.PauliZ(w)) for w in range(3)]

            circuit(features=[1., 2., 3.])

        Every repetition uses exactly the same quantum circuit.

        **Using a custom entangler pattern**

        A custom entangler pattern can be used by specifying the ``pattern`` argument. A pattern has to be
        a nested list of dimension ``(K, 2)``, where ``K`` is the number of entanglers to apply.

        .. code-block:: python

            pattern = [[1, 2], [0, 2], [1, 0]]

            @qml.qnode(dev)
            def circuit(features=None):
                IQPEmbedding(features=features, wires=range(3), pattern=pattern)
                return [qml.expval(qml.PauliZ(w)) for w in range(3)]

            circuit(features=[1., 2., 3.])

        Since diagonal gates commute, the order of the wire pairs has no effect on the result.

        .. code-block:: python

            from pennylane import numpy as np

            pattern1 = [[1, 2], [0, 2], [1, 0]]
            pattern2 = [[1, 0], [0, 2], [1, 2]]  # a reshuffling of pattern1

            @qml.qnode(dev)
            def circuit(features=None, pattern=None):
                IQPEmbedding(features=features, wires=range(3), pattern=pattern, n_repeats=3)
                return [qml.expval(qml.PauliZ(w)) for w in range(3)]

            res1 = circuit(features=[1., 2., 3.], pattern=pattern1)
            res2 = circuit(features=[1., 2., 3.], pattern=pattern2)

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

    wires = Wires(wires)
    pattern = _preprocess(features, wires, pattern, n_repeats)

    for i in range(n_repeats):

        # first block of Hadamards
        broadcast(unitary=Hadamard, pattern="single", wires=wires)
        # encode features into block of RZ rotations
        broadcast(unitary=RZ, pattern="single", wires=wires, parameters=features)

        # create new features for entangling block
        products = []
        for wire_pair in pattern:
            # get the position of the wire indices in the array
            idx1, idx2 = wires.indices(wire_pair)
            # create products of parameters
            products.append(features[idx1] * features[idx2])

        broadcast(unitary=MultiRZ, pattern=pattern, wires=wires, parameters=products)
