# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A function to compute the center of a Lie algebra"""
from itertools import combinations

import numpy as np

from pennylane.pauli import PauliSentence


def center(g, pauli=False):
    r"""
    A function to compute the center of a Lie algebra.

    Given a Lie algebra :math:`\mathfrak{g} = \{h_1,.., h_d\}`, the center :math:`\mathfrak{\xi}(\mathfrak{g})`
    is given by all elements in :math:`\mathfrak{g}` that commute with `all` other elements in :math:`\mathfrak{g}`,

    .. math:: \mathfrak{\xi}(\mathfrak{g}) := \{h \in \mathfrak{g} | [h, h_i]=0 \forall h_i \in \mathfrak{g} \}

    Args:
        ops (List[Union[Operator, PauliSentence]]): List of operators for which to find the center.
        pauli (bool): Indicates whether it is assumed that :class:`~PauliSentence` instances are input and returned.
            This can help with performance to avoid unnecessary conversions from :class:`~PauliSentence` to :class:`~Operator`
            or :class:`~PauliWord` and vice versa. Note that the input in that case also has to be a list of :class:`~PauliSentence` instances.
            Default is ``False``.

    Returns:
        List[Union[Operator, PauliSentence]]: Center of g


    """

    if not pauli:
        g = [o.pauli_rep for o in g]

    d = len(g)
    commutators = np.zeros((d, d), dtype=int)
    for (j, op1), (k, op2) in combinations(enumerate(g), r=2):
        res = op1.commutator(op2)
        res.simplify()
        if res != PauliSentence({}):
            commutators[j, k] = 1  # dummy value to indicate operators dont commute
            commutators[k, j] = 1

    mask = np.all(commutators == 0, axis=0)
    center = list(np.array(g)[mask])

    if not pauli:
        center = [op.operation() for op in center]
    return center
