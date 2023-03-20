# Copyright 2018-2023 Xanadu Quantum Technologies Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains the Shots class to hold shot-related information."""

from collections import namedtuple
from collections.abc import Sequence as SequenceType
from functools import singledispatchmethod
from typing import Sequence, Tuple, Union
import numpy as np

ShotCopies = namedtuple("ShotCopies", ["shots", "copies"])
"""A namedtuple that represents a shot quantity being repeated some number of times."""


def valid_int(s):
    """Returns True if s is an non-negative integer."""
    return isinstance(s, int) and s > 0


def valid_tuple(s):
    """Returns True if s is a tuple of the form (shots, copies)."""
    return isinstance(s, tuple) and len(s) == 2 and valid_int(s[0]) and valid_int(s[1])


class Shots:
    """A data class that stores shot information."""

    total_shots: int = None
    """The total number of shots to be executed."""

    shot_vector: Tuple[ShotCopies] = None
    """The list of ShotCopies to be executed. Each element is of the form (shots, copies)."""

    _SHOT_ERROR = ValueError(
        "Shots must be a single non-negative integer, a tuple pair of the form (shots, counts), or a sequence of these types."
    )

    _frozen = False

    def __new__(cls, shots):
        return shots if isinstance(shots, cls) else object.__new__(cls)

    def __setattr__(self, name, value):
        if self._frozen:
            raise AttributeError(
                "Shots is an immutable class. Consider creating a new instance if you need different shot values."
            )
        return super().__setattr__(name, value)

    @singledispatchmethod
    def __init__(self, shots):
        if isinstance(shots, self.__class__):
            return  # self already _is_ shots as defined by __new__
        raise self._SHOT_ERROR

    @__init__.register
    def __int_init__(self, shots: int):
        if shots < 1:
            raise self._SHOT_ERROR
        self.total_shots = shots
        self.shot_vector = (ShotCopies(shots, 1),)
        self._frozen = True

    @__init__.register
    def __None_init__(self, shots: type(None)):  # pylint:disable=unused-argument
        self.total_shots = None
        self.shot_vector = ()
        self._frozen = True

    @__init__.register
    def __tuple_init__(self, shots: tuple):
        if len(shots) != 2:
            raise self._SHOT_ERROR
        shots, copies = shots
        self.total_shots = shots * copies
        self.shot_vector = (ShotCopies(shots, copies),)
        self._frozen = True

    @__init__.register
    def __list_init__(self, shots: SequenceType):  # pylint:disable=inconsistent-return-statements
        if not all(valid_int(s) or valid_tuple(s) for s in shots):
            raise self._SHOT_ERROR
        if all(valid_tuple(s) for s in shots):
            return self.__all_tuple_init__(shots)
        if any(valid_tuple(s) for s in shots):
            return self.__mixed_init__(shots)

        if len(set(shots)) == 1:
            # All shots are identical, only require a single shot tuple
            self.shot_vector = (ShotCopies(shots=shots[0], copies=len(shots)),)
            self.total_shots = shots[0] * len(shots)
            return
        # Iterate through the shots, and group consecutive identical shots
        split_at_repeated = np.split(shots, np.diff(shots).nonzero()[0] + 1)
        self.shot_vector = tuple(ShotCopies(shots=i[0], copies=len(i)) for i in split_at_repeated)
        self.total_shots = int(np.sum(np.prod(self.shot_vector, axis=1)))
        self._frozen = True

    def __all_tuple_init__(self, shots: Sequence[Tuple]):
        res = []
        total_shots = 0
        current_shots, current_count = shots[0]
        for s in shots[1:]:
            if s[0] == current_shots:
                current_count += s[1]
            else:
                res.append(ShotCopies(current_shots, current_count))
                total_shots += current_shots * current_count
                current_shots, current_count = s
        self.shot_vector = tuple(res + [ShotCopies(current_shots, current_count)])
        self.total_shots = total_shots + current_shots * current_count
        self._frozen = True

    def __mixed_init__(self, shots: Sequence[Union[int, Tuple[int, int]]]):
        return self.__all_tuple_init__([s if isinstance(s, tuple) else (s, 1) for s in shots])

    @property
    def has_partitioned_shots(self):
        """Checks if the device was instructed to perform executions with partitioned shots.

        Returns:
            bool: whether or not shots are partitioned
        """
        return len(self.shot_vector) > 1 or self.shot_vector[0].copies > 1
