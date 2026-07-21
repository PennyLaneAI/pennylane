# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Fermionic algebra primitives for the GAN Hamiltonian.

This module provides the building blocks for the electronic (fermionic) part of
the GAN Hamiltonian: individual creation and annihilation operators
(:class:`FermiOp`), ordered products of them (:class:`FermiWord`), and linear
combinations of such products (:class:`FermiSentence`).

The operators act on two distinct single-particle spaces --- molecular
(``"mol"``) and metallic (``"met"``) modes --- and obey the canonical fermionic
anticommutation relations,

.. math::

    \{a_i, a_j^\dagger\} = \delta_{ij}, \qquad
    \{a_i, a_j\} = \{a_i^\dagger, a_j^\dagger\} = 0,

where the Kronecker delta requires both the mode index and the space (molecular
vs. metallic) to match. :meth:`FermiWord.normal_order` uses these relations to
rewrite an arbitrary operator product into a canonical normal-ordered form.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class FermiType(Enum):
    """Whether a fermionic operator is a creation (``"+"``) or annihilation (``"-"``) operator."""

    CREATION = "+"
    ANNIHILATION = "-"


class FermiSpace(Enum):
    """The single-particle space a fermionic operator acts on: molecular or metallic."""

    MOLECULAR = "mol"
    METALLIC = "met"


@dataclass(frozen=True)
class FermiOp:
    """A single fermionic creation or annihilation operator.

    An operator is fully specified by its type (creation or annihilation), the
    space it acts on (molecular or metallic), and its integer mode index. The
    dataclass is frozen so that operators are hashable and can be used as keys
    and inside :class:`FermiWord`.

    Args:
        op_type (FermiType): whether this is a creation or annihilation operator.
        space (FermiSpace): the molecular or metallic space the operator acts on.
        mode (int): the mode index within that space.
    """

    op_type: FermiType
    space: FermiSpace
    mode: int

    @staticmethod
    def creation_mol(mode):
        """Return a molecular creation operator on the given mode.

        Args:
            mode (int): the molecular mode index.

        Returns:
            FermiOp: a creation operator on the molecular space.
        """
        return FermiOp(op_type=FermiType.CREATION, space=FermiSpace.MOLECULAR, mode=mode)

    @staticmethod
    def annihilation_mol(mode):
        """Return a molecular annihilation operator on the given mode.

        Args:
            mode (int): the molecular mode index.

        Returns:
            FermiOp: an annihilation operator on the molecular space.
        """
        return FermiOp(op_type=FermiType.ANNIHILATION, space=FermiSpace.MOLECULAR, mode=mode)

    @staticmethod
    def creation_met(mode):
        """Return a metallic creation operator on the given mode.

        Args:
            mode (int): the metallic mode index.

        Returns:
            FermiOp: a creation operator on the metallic space.
        """
        return FermiOp(op_type=FermiType.CREATION, space=FermiSpace.METALLIC, mode=mode)

    @staticmethod
    def annihilation_met(mode):
        """Return a metallic annihilation operator on the given mode.

        Args:
            mode (int): the metallic mode index.

        Returns:
            FermiOp: an annihilation operator on the metallic space.
        """
        return FermiOp(op_type=FermiType.ANNIHILATION, space=FermiSpace.METALLIC, mode=mode)

    def __repr__(self):
        symbol = "+" if self.op_type == FermiType.CREATION else "-"

        if self.space == FermiSpace.METALLIC:
            space = "met"
        else:
            space = "mol"

        return f"({symbol}, {space}:{self.mode})"


class FermiWord:
    """An ordered product of fermionic operators.

    A ``FermiWord`` represents a single monomial in the fermionic operators,
    i.e. an ordered product :math:`o_0 o_1 \\cdots o_{n-1}` of :class:`FermiOp`
    factors. The empty product is the identity (see :meth:`identity`). Words are
    hashable (by their ordered operators) so they can serve as dictionary keys,
    e.g. as the fermionic part of a :class:`~.GanFragment` term.

    Args:
        ops (Sequence[FermiOp]): the ordered operators making up the word.
    """

    def __init__(self, ops: Sequence[FermiOp]):
        self.ops = list(ops)

    def normal_order(self) -> FermiSentence:
        r"""Rewrite the word in canonical normal order.

        Repeatedly applies the fermionic anticommutation relations to bring the
        operators into a fixed canonical order, returning the result as a
        :class:`FermiSentence` (a normal-ordered word may expand into several
        words because of the :math:`\{a_i, a_i^\dagger\} = 1` contractions).

        The ordering pushes annihilation operators to the right of creation
        operators; an annihilation/creation pair on the *same* mode and space
        produces a contraction term (the word with that pair removed) plus a
        sign flip, while a mismatched pair simply anticommutes with a sign flip.
        Same-type operators are ordered by mode (and molecular before metallic),
        each swap contributing a factor of :math:`-1`. Words containing a
        repeated adjacent operator vanish and are dropped.

        Returns:
            FermiSentence: the normal-ordered form as a linear combination of
            words.
        """

        unordered_words = defaultdict(float, {deepcopy(self): 1})
        sentence = defaultdict(float)

        while unordered_words:
            word, coeff = unordered_words.popitem()
            n_ops = len(word.ops)

            for i in range(n_ops):
                cur = i
                for j in reversed(range(i)):
                    l_type = word[j].op_type
                    r_type = word[cur].op_type
                    l_space = word[j].space
                    r_space = word[cur].space
                    l_mode = word[j].mode
                    r_mode = word[cur].mode

                    ## {a_i, c_j} = 0
                    if l_type == FermiType.ANNIHILATION and r_type == FermiType.CREATION:
                        if l_mode != r_mode or l_space != r_space:
                            word[cur], word[j] = word[j], word[cur]
                            coeff *= -1
                            cur -= 1
                            continue

                        ## {a_i, c_i} = 1
                        word[cur], word[j] = word[j], word[cur]
                        new_word = FermiWord(word[:j] + word[cur + 1 :])
                        unordered_words[new_word] += coeff
                        coeff *= -1
                        cur -= 1
                        continue

                    ## {a_i, a_j} = {c_i, c_j} = 0
                    if l_type == r_type and l_space == r_space and l_mode >= r_mode:
                        word[cur], word[j] = word[j], word[cur]
                        coeff *= -1
                        cur -= 1
                        continue

                    ## {a_i, a_j} = {c_i, c_j} = 0
                    if (
                        l_type == r_type
                        and l_space == FermiSpace.MOLECULAR
                        and r_space == FermiSpace.METALLIC
                    ):
                        word[cur], word[j] = word[j], word[cur]
                        coeff *= -1
                        cur -= 1
                        continue

                    break

            word = FermiWord(word)

            if not word.is_zero():
                sentence[word] += coeff

            sentence = {word: coeff for word, coeff in sentence.items() if not np.isclose(coeff, 0)}

        return FermiSentence(sentence)

    def is_zero(self):
        """Whether the word is identically the zero operator.

        A fermionic word vanishes when it contains two identical adjacent
        operators (e.g. :math:`a_i^\\dagger a_i^\\dagger = 0`).

        Returns:
            bool: ``True`` if any two adjacent operators are equal.
        """
        return any(self.ops[i] == self.ops[i + 1] for i in range(len(self.ops) - 1))

    def __eq__(self, other):
        """Whether two words have identical ordered operators."""
        return self.ops == other.ops

    def __getitem__(self, i):
        """Return the operator (or slice of operators) at index ``i``."""
        return self.ops[i]

    def __setitem__(self, i, val):
        """Set the operator at index ``i``.

        Args:
            i (int): the position to overwrite.
            val (FermiOp): the replacement operator.

        Raises:
            TypeError: if ``val`` is not a :class:`FermiOp`.
        """
        if not isinstance(val, FermiOp):
            raise TypeError(f"FermiWords must contain FermiOps, got {type(val)}.")

        self.ops[i] = val

    def __hash__(self):
        """Return a hash based on the ordered operators."""
        return hash(tuple(self.ops))

    def __add__(self, other: FermiWord | float) -> FermiSentence:
        """Add two words, or a word and a scalar.

        Adding two words returns their sum as a :class:`FermiSentence`. Adding a
        ``float`` adds that multiple of the identity word.

        Args:
            other (FermiWord | float): the word or scalar to add.

        Returns:
            FermiSentence: the resulting linear combination.

        Raises:
            TypeError: if ``other`` is neither a ``FermiWord`` nor a ``float``.
        """

        if isinstance(other, FermiWord):
            sentence = defaultdict(float)
            sentence[self] += 1
            sentence[other] += 1
            return FermiSentence(sentence)

        if isinstance(other, float):
            identity = FermiWord([])
            return FermiSentence({self: 1, identity: other})

        raise TypeError(f"Cannot add FermiWord with {type(other)}.")

    def __mul__(self, scalar: float) -> FermiSentence:
        """Scale the word by a scalar, returning a single-term sentence.

        Args:
            scalar (float): the scalar multiplier.

        Returns:
            FermiSentence: a sentence mapping this word to ``scalar``.
        """
        return FermiSentence({self: scalar})

    def __matmul__(self, other) -> FermiWord:
        """Concatenate two words into their operator product.

        Args:
            other (FermiWord): the word to multiply on the right.

        Returns:
            FermiWord: the word whose operators are this word's followed by
            ``other``'s.

        Raises:
            TypeError: if ``other`` is not a ``FermiWord``.
        """
        if not isinstance(other, FermiWord):
            raise TypeError(f"Cannot multiply FermiWord with type {type(other)}.")

        return FermiWord(self.ops + other.ops)

    def __str__(self):
        return str(self.ops)

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def identity():
        """Return the identity word (the empty operator product).

        Returns:
            FermiWord: a word with no operators.
        """
        return FermiWord([])


class FermiSentence:
    """A linear combination of :class:`FermiWord` objects.

    Stores a mapping from each word to its (scalar) coefficient, representing a
    general fermionic operator as a sum of operator products.

    Args:
        words (dict[FermiWord, float]): the word-to-coefficient mapping.
    """

    def __init__(self, words: dict[FermiWord, float]):
        self.words = words

    def __add__(self, other: FermiWord | FermiSentence | float) -> FermiSentence:
        """Add a word, sentence, or scalar to this sentence.

        Coefficients of shared words are summed; a ``float`` is added as a
        multiple of the identity word.

        Args:
            other (FermiWord | FermiSentence | float): the term(s) to add.

        Returns:
            FermiSentence: the combined linear combination.

        Raises:
            TypeError: if ``other`` is not a ``FermiWord``, ``FermiSentence``,
                or ``float``.
        """
        d = defaultdict(float)

        for key, value in self.words.items():
            d[key] += value

        if isinstance(other, FermiWord):
            d[other] += 1
            return FermiSentence(d)

        if isinstance(other, FermiSentence):
            for key, value in other.words.items():
                d[key] += value

            return FermiSentence(d)

        if isinstance(other, float):
            d[FermiWord([])] += other
            return FermiSentence(d)

        raise TypeError(f"Cannot add FermiSentence with {type(other)}.")

    def __mul__(self, scalar: float) -> FermiSentence:
        """Scale every word's coefficient by ``scalar``.

        Args:
            scalar (float): the scalar multiplier.

        Returns:
            FermiSentence: the scaled linear combination.
        """
        d = defaultdict(float)
        for key, value in self.words.items():
            d[key] += scalar * value

        return FermiSentence(d)

    def __matmul__(self, other: FermiWord | FermiSentence):
        """Multiply this sentence by a word or another sentence.

        Multiplying by a :class:`FermiWord` right-concatenates it onto every
        word. Multiplying by a :class:`FermiSentence` distributes over all
        word pairs, concatenating operators and multiplying coefficients.

        Args:
            other (FermiWord | FermiSentence): the right operand.

        Returns:
            FermiSentence: the product as a linear combination of words.
        Raises:
            TypeError: if ``other`` is neither a ``FermiWord`` nor a
                ``FermiSentence``.
        """
        if isinstance(other, FermiWord):
            return FermiSentence({key @ other: value for key, value in self.words.items()})

        if isinstance(other, FermiSentence):
            d = defaultdict(float)

            for l_key, l_value in self.words.items():
                for r_key, r_value in other.words.items():
                    d[l_key @ r_key] += l_value * r_value

            return FermiSentence(d)

        raise TypeError(f"Cannot matmul FermiSentence with {type(other)}.")

    def __eq__(self, other):
        """Whether two sentences have the same words and coefficients."""
        return self.words == other.words

    def __repr__(self):
        return " + ".join(f"{coeff}*{word}" for word, coeff in self.words.items())
