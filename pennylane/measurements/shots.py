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
        shots (Union[None, int, Sequence[int, Tuple[int, int]]]): Raw shot information

    Defining shots enables users to specify circuit executions, and the Shots class standardizes
    the internal representation of shots. There are three ways to specify shot values:

    * The value ``None``
    * A positive integer
    * A sequence consisting of either positive integers or a tuple-pair of positive integers of the form ``(shots, copies)``

    The tuple-pair of the form ``(shots, copies)`` is represented internally by a namedtuple called
    :class:`~ShotCopies`. The first value is the number of shots to execute, and the second value is the
    number of times to repeat a circuit with that number of shots.

    The ``Shots`` class exposes two properties:

    * ``total_shots``, the total number of shots to be executed
    * ``shot_vector``, the tuple of :class:`~ShotCopies` to be executed

    Instances of this class are static. If an instance is passed to the constructor, that same
    instance is returned. If an instance is constructed with a ``None`` value, ``total_shots``
    will be ``None``.  This indicates analytic execution.

    **Examples**

    Example constructing a Shots instance with ``None``:

    >>> shots = Shots(None)
    >>> shots.total_shots, shots.shot_vector
    (None, ())

    Example constructing a Shots instance with an int:

    >>> shots = Shots(100)
    >>> shots.total_shots, shots.shot_vector
    (100, (ShotCopies(shots=100, copies=1),))

    Example constructing a Shots instance with another instance:

    >>> shots = Shots(100)
    >>> Shots(shots) is shots
    True

    Example constructing a Shots instance with a sequence of ints:

    >>> shots = Shots([100, 200])
    >>> shots.total_shots, shots.shot_vector
    (300, (ShotCopies(shots=100, copies=1), ShotCopies(shots=200, copies=1)))

    Example constructing a Shots instance with a sequence of tuple-pairs:

    >>> shots = Shots(((100, 3), (200, 4),))
    >>> shots.total_shots, shots.shot_vector
    (1100, (ShotCopies(shots=100, copies=3), ShotCopies(shots=200, copies=4)))

    Example constructing a Shots instance with a sequence of both ints and tuple-pairs.
    Note that the first stand-alone ``100`` gets absorbed into the subsequent tuple because the
    shot value matches:

    >>> shots = Shots((10, 100, (100, 3), (200, 4),))
    >>> shots.total_shots, shots.shot_vector
    (1210, (ShotCopies(shots=10, copies=1), ShotCopies(shots=100, copies=4), ShotCopies(shots=200, copies=4)))

    One should also note that specifying a single tuple of length 2 is considered two different
    shot values, and *not* a tuple-pair representing shots and copies to avoid special behaviour
    depending on the iterable type:

    >>> shots = Shots((100, 2))
    >>> shots.total_shots, shots.shot_vector
    (102, (ShotCopies(shots=100, copies=1), ShotCopies(shots=2, copies=1)))

    >>> shots = Shots(((100, 2),))
    >>> shots.total_shots, shots.shot_vector
    (200, (ShotCopies(shots=100, copies=2),))
    """

    total_shots: int = None
    """The total number of shots to be executed."""

    shot_vector: Tuple[ShotCopies] = None
    """The tuple of :class:`~ShotCopies` to be executed. Each element is of the form ``(shots, copies)``."""

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
        """
        Evaluates to True if this instance represents either multiple shot
        quantities, or the same shot quantity repeated multiple times.

        Returns:
            bool: whether or not shots are partitioned
        """
        if self.total_shots is None:
            return False
        return len(self.shot_vector) > 1 or self.shot_vector[0].copies > 1
