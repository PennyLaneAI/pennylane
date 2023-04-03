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
# pylint:disable=inconsistent-return-statements
from collections import namedtuple
from typing import Sequence, Tuple

ShotCopies = namedtuple("ShotCopies", ["shots", "copies"])
"""A namedtuple that represents a shot quantity being repeated some number of times.

For example, ``ShotCopies(shots=10, copies=2)`` indicates two executions with 10 shots each for 20 shots total.
"""


def valid_int(s):
    """Returns True if s is a positive integer."""
    return isinstance(s, int) and s > 0


def valid_tuple(s):
    """Returns True if s is a tuple of the form (shots, copies)."""
    return isinstance(s, tuple) and len(s) == 2 and valid_int(s[0]) and valid_int(s[1])


class Shots:
    """
    A data class that stores shot information.

    Args:
        shots (Union[None, int, Tuple[int, int], Sequence[int, Tuple[int, int]]]): Raw shot information

    Defining shots enables users to specify circuit executions, and the Shots class standardizes
    the internal representation of shots. There are three ways to specify shot values:

    * A positive integer
    * A tuple consisting of a pair of positive integers of the form ``(shots, copies)``
    * A list of multiple shot values matching one or both of the above value types

    The tuple-pair of the form ``(shots, copies)`` is represented internally by a namedtuple called
    ``ShotCopies``. The first value is the number of shots to execute, and the second value is the
    number of times to repeat a circuit with that number of shots.

    The ``Shots`` class exposes two properties:

    * ``total_shots``, the total number of shots to be executed
    * ``shot_vector``, the list of ``ShotCopies`` to be executed

    Instances of this class are static. If an instance is passed to the constructor, that same
    instance is returned. If an instance is constructed with a ``None`` value, ``total_shots``
    will be ``None``, and this will suggest analytic execution.
    """

    total_shots: int = None
    """The total number of shots to be executed."""

    shot_vector: Tuple[ShotCopies] = None
    """The tuple of ShotCopies to be executed. Each element is of the form (shots, copies)."""

    _SHOT_ERROR = ValueError(
        "Shots must be a single positive integer, a tuple pair of the form (shots, copies), or a sequence of these types."
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

    def __init__(self, shots):
        if shots is None:
            self.total_shots = None
            self.shot_vector = ()
        elif isinstance(shots, int):
            if shots < 1:
                raise self._SHOT_ERROR
            self.total_shots = shots
            self.shot_vector = (ShotCopies(shots, 1),)
        elif isinstance(shots, Sequence):
            if not all(valid_int(s) or valid_tuple(s) for s in shots):
                raise self._SHOT_ERROR
            self.__all_tuple_init__([s if isinstance(s, tuple) else (s, 1) for s in shots])
        elif isinstance(shots, self.__class__):
            return  # self already _is_ shots as defined by __new__
        else:
            raise self._SHOT_ERROR

        self._frozen = True

    def __all_tuple_init__(self, shots: Sequence[Tuple]):
        res = []
        total_shots = 0
        current_shots, current_copies = shots[0]
        for s in shots[1:]:
            if s[0] == current_shots:
                current_copies += s[1]
            else:
                res.append(ShotCopies(current_shots, current_copies))
                total_shots += current_shots * current_copies
                current_shots, current_copies = s
        self.shot_vector = tuple(res + [ShotCopies(current_shots, current_copies)])
        self.total_shots = total_shots + current_shots * current_copies

    @property
    def has_partitioned_shots(self):
        """Checks if the device was instructed to perform executions with partitioned shots.

        Returns:
            bool: whether or not shots are partitioned
        """
        return len(self.shot_vector) > 1 or self.shot_vector[0].copies > 1
