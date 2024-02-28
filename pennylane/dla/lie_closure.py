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
"""A function to compute the Lie closure of a set of operators"""
# pylint: disable=too-many-arguments
from functools import reduce

from copy import copy
import numpy as np

import pennylane as qml

from pennylane.operation import Operator
from pennylane.pauli import PauliSentence, PauliWord


class VSpace:
    """
    Class representing the linearly independent basis of a vector space.

    The main purpose of this class is to store and process ``M``, which
    is a dictionary-of-keys (DOK) style sparse representation of the set of basis vectors. You can
    think of it as the numpy-equivalent of a PauliSentence: each ``PauliWord`` (key of ``PauliSentence``)
    represents one row of ``M`` with the coefficient (value of ``PauliSentence``).
    For example the set of 3 linearly independent generators ``X(0) + X(1), X(0) + X(2), X(0) + 0.5 * Y(0)``
    can be represented as

    .. code-block::python3
        [
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0.5]
        ]

    where each column represents one sentence, and each row represents the coefficient of the respective word in the sentence.
    To make sense of this representation one additionally needs to keep track of the mapping between keys and rows. In this case we have

    .. code-block::python3
        pw_to_idx = {
            X(0) : 0,
            X(1) : 1,
            X(2) : 2,
            Y(0) : 3
        }

    where we have set the numbering based on appearance in the list of generators. This mapping is in general not unique.

    Args:
        generators [Iterable[Union[PauliWord, PauliSentence, Operator]]]: Operators that span the vector space.

    **Example**

    Take the linearly dependent set of operators and span the VSpace.

    .. code-block::python3
        ops = [
            PauliSentence({
                PauliWord({0:"X", 1:"X"}) : 1.,
                PauliWord({0:"Y", 1:"Y"}) : 1.
            }),
            PauliSentence({
                PauliWord({0:"X", 1:"X"}) : 1.,
            }),
            PauliSentence({
                PauliWord({0:"Y", 1:"Y"}) : 1.,
            }),
        ]

        vspace = VSpace(ops)

    It automatically detects that the third operator is linearly dependent on the former two, so it does not add the third operator to the basis.

    >>> vspace.basis
    [1.0 * X(0) @ X(1)
     + 1.0 * Y(0) @ Y(1),
     1.0 * X(0) @ X(1)]

    We can also retrospectively add operators.

    >>> vspace.add(PauliWord({0:"X"}))
    [1.0 * X(0) @ X(1)
     + 1.0 * Y(0) @ Y(1),
     1.0 * X(0) @ X(1),
     1.0 * X(0)]

    Again, checks of linear independence are always performed. So in the following example no operator is added.

    >>> vspace.add(PauliWord({0:"Y", 1:"Y"}))
    [1.0 * X(0) @ X(1)
     + 1.0 * Y(0) @ Y(1),
     1.0 * X(0) @ X(1),
     1.0 * X(0)]
    """

    def __init__(self, generators):

        # Get all Pauli words that are present in at least one Pauli sentence
        all_pws = list(reduce(set.__or__, [set(ps.keys()) for ps in generators]))
        num_pw = len(all_pws)
        # Create a dictionary mapping from PauliWord to row index
        self._pw_to_idx = {pw: i for i, pw in enumerate(all_pws)}

        # Initialize VSpace properties trivially
        self._basis = []
        rank = 0

        self._M = np.zeros((num_pw, rank), dtype=float)
        self._rank = rank
        self._num_pw = num_pw

        # Add all generators that are linearly independent
        self.add(generators)

    @property
    def basis(self):
        """List of basis operators of VSpace"""
        return self._basis

    def __len__(self):
        return len(self.basis)

    def add(self, other):
        """Adding a list of PauliSentences if they are linearly independent"""
        if isinstance(other, (PauliWord, PauliSentence, Operator)):
            other = [other]

        other = [
            qml.pauli.pauli_sentence(op) if not isinstance(op, PauliSentence) else op
            for op in other
        ]

        for ps in other:
            # TODO: Potential speed-up by computing the maximal linear independent set for all current basis vectors + other, essentially algorithm1 in https://arxiv.org/abs/1012.5256
            self._M, self._pw_to_idx, self._rank, self._num_pw, is_independent = (
                self._add_if_independent(self._M, ps, self._pw_to_idx, self._rank, self._num_pw)
            )
            if is_independent:
                self._basis.append(ps)
        return self._basis

    @staticmethod
    def _add_if_independent(M, pauli_sentence, pw_to_idx, rank, num_pw, tol=1e-15):
        r"""
        Checks if ``pauli_sentence`` is linearly independent.

        This is done in the following way: ``M`` (see description in class) is extended by ``pauli_sentence``.
        If the added operator has a PauliWord (key) that is new to ``pw_to_idx``, then we have to add a new row
        and already know that it has to be linearly independent.
        If it contains the same PauliWords, we need to compute the new rank and compare it with the old rank.
        If the rank is the same, the operator is linearly dependent and not added. Else, the rank is incrased by 1
        and the extended M becomes our new M.

        Args:
            M (ndarray): coefficient matrix for current LIS
            pauli_sentence (`~.PauliSentence`): Pauli sentence for which to add a column if independent
            pw_to_idx (dict): map from ``PauliWord`` to row index in ``M``
            rank (int): current rank of ``M``, equal to its number of columns
            num_pw (int): current number of ``PauliWord``\ s, equal to the number of rows in ``M``
        Returns:
            ndarray: updated coefficient matrix for the LIS
            dict: updated map from ``PauliWord`` to row index in ``M``. Includes new ``PauliWord`` keys
                from the input ``pauli_sentence`` if it was linearly independent
            int: updated rank/number of columns of ``M``
            int: updated number of ``PauliWord``\ s/number of rows of ``M``
            bool: whether ``pauli_sentence`` was linearly independent and its column was added to ``M``
        """
        new_pws = [pw for pw in pauli_sentence.keys() if pw not in pw_to_idx]
        new_num_pw = num_pw + len(new_pws)

        if new_num_pw < rank + 1:
            # Can't span rank+1 independent vectors in fewer than rank+1 dimensions
            # The input PauliSentence must have been linearly dependent
            return M, pw_to_idx, rank, num_pw, False

        M = np.pad(M, ((0, new_num_pw - num_pw), (0, 1)))

        # If there are new PauliWords (i.e. new basis vectors), the candidate vector must be linearly independent
        if new_num_pw > num_pw:
            new_pw_to_idx = copy(pw_to_idx)
            for i, pw in enumerate(new_pws, start=num_pw):
                new_pw_to_idx[pw] = i
            # Add new PauliSentence entries to matrix
            for pw, value in pauli_sentence.items():
                M[new_pw_to_idx[pw], rank] = value

            return M, new_pw_to_idx, rank + 1, new_num_pw, True

        # Add new PauliSentence entries to matrix
        for pw, value in pauli_sentence.items():
            M[pw_to_idx[pw], rank] = value

        # Check if new vector is linearly dependent on the current basis
        v = M[:, -1].copy()  # remove copy to normalize M
        v /= np.linalg.norm(v)
        A = M[:, :-1]
        v = v - A @ qml.math.linalg.inv(qml.math.conj(A.T) @ A) @ qml.math.conj(A).T @ v

        if np.linalg.norm(v) > tol:
            return M, pw_to_idx, rank + 1, new_num_pw, True

        return M[:num_pw, :rank], pw_to_idx, rank, num_pw, False

    def __repr__(self):
        return str(self.basis)

    def __eq__(self, other):
        """
        Two VSpaces are equivalent when they span the same dimensional space.
        This is checked here by having matching PauliWord keys in the sparse DOK representation and having the same rank.
        """
        if not self._num_pw == other._num_pw:
            return False
        if not set(self._pw_to_idx.keys()) == set(other._pw_to_idx.keys()):
            return False

        rank1 = np.linalg.matrix_rank(self._M)
        rank2 = np.linalg.matrix_rank(other._M)
        rank3 = np.linalg.matrix_rank(np.concatenate([self._M, other._M], axis=1))

        return rank1 == rank2 and rank2 == rank3
