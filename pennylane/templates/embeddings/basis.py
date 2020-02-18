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
import numpy as np

from pennylane.templates.decorator import template
from pennylane.ops import BasisState
from pennylane.templates.utils import _check_shape, _check_wires, _get_shape


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

    wires = _check_wires(wires)

    expected_shape = (len(wires),)
    _check_shape(
        features,
        expected_shape,
        msg="'features' must be of shape {}; got {}"
        "".format(expected_shape, _get_shape(features)),
    )

    if any([b not in [0, 1] for b in features]):
        raise ValueError("'basis_state' must only consist of 0s and 1s; got {}".format(features))

    ###############

    features = np.array(features)
    BasisState(features, wires=wires)
