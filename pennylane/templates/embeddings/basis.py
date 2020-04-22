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
Contains the ``BasisEmbedding`` template.
"""
# pylint: disable-msg=too-many-branches,too-many-arguments,protected-access
from collections import Iterable

from pennylane.templates.decorator import template
from pennylane.templates.utils import check_shape, check_wires, get_shape, check_type
import pennylane as qml


@template
def BasisEmbedding(features, wires):
    r"""Encodes :math:`n` binary features into a basis state of :math:`n` qubits.

    For example, for ``features=np.array([0, 1, 0])``, the quantum system will be
    prepared in state :math:`|010 \rangle`.

    .. warning::

        ``BasisEmbedding`` calls a circuit whose architecture depends on the binary features.
        The ``features`` argument is therefore not differentiable when using the template, and
        gradients with respect to the argument cannot be computed by PennyLane.

    Args:
        features (array): binary input array of shape ``(n, )``
        wires (Sequence[int] or int): qubit indices that the template acts on

    Raises:
        ValueError: if inputs do not have the correct format
    """

    #############
    # Input checks

    wires = check_wires(wires)

    check_type(
        features, [Iterable], msg="'features' must be iterable; got type {}".format(type(features))
    )

    expected_shape = (len(wires),)
    check_shape(
        features,
        expected_shape,
        msg="'features' must be of shape {}; got {}" "".format(expected_shape, get_shape(features)),
    )

    if any([b not in [0, 1] for b in features]):
        raise ValueError("'basis_state' must only consist of 0s and 1s; got {}".format(features))

    ###############

    for wire, bit in zip(wires, features):
        if bit == 1:
            qml.PauliX(wire)
