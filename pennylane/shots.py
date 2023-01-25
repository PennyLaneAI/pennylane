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
"""This module contains the ShotAPI class to hold shot-related information."""

from collections import namedtuple
from collections.abc import Sequence
from functools import singledispatchmethod
import numpy as np

ShotTuple = namedtuple("ShotTuple", ["shots", "copies"])


class ShotAPI:
    """A data class that stores shot information needed in order to execute tapes."""

    # pylint:disable=too-few-public-methods

    total_shots, shot_vector, shot_list = 0, [], []

    @singledispatchmethod
    def __init__(self, shots):
        if shots is not None:
            raise ValueError(
                "Shots must be a single non-negative integer or a sequence of non-negative integers."
            )

    @__init__.register
    def __int_init__(self, shots: int):
        if shots < 1:
            raise ValueError(f"The specified number of shots needs to be at least 1. Got {shots}.")
        self.total_shots = shots
        self.shot_list = [shots]
        self.shot_vector = [ShotTuple(shots, 1)]

    @__init__.register
    def __list_init__(self, shots: Sequence):
        """Process the shot sequence, to determine the total
        number of shots and the shot vector.

        Args:
            shot_list (Sequence[int]): sequence of non-negative shot integers

        Returns:
            tuple[int, list[.ShotTuple[int]]]: A tuple containing the total number
            of shots, as well as a list of shot tuples.

        **Example**

        >>> shot_list = [3, 1, 2, 2, 2, 2, 6, 1, 1, 5, 12, 10, 10]
        >>> _process_shot_sequence(shot_list)
        (57,
        [ShotTuple(shots=3, copies=1),
        ShotTuple(shots=1, copies=1),
        ShotTuple(shots=2, copies=4),
        ShotTuple(shots=6, copies=1),
        ShotTuple(shots=1, copies=2),
        ShotTuple(shots=5, copies=1),
        ShotTuple(shots=12, copies=1),
        ShotTuple(shots=10, copies=2)])

        The total number of shots (57), and a sparse representation of the shot
        sequence is returned, where tuples indicate the number of times a shot
        integer is repeated.
        """
        if not all(isinstance(shot, int) for shot in shots):
            raise ValueError(
                "Shots must be a single non-negative integer or a sequence of non-negative integers."
            )

        self.shot_list = shots
        if len(set(shots)) == 1:
            # All shots are identical, only require a single shot tuple
            self.shot_vector = [ShotTuple(shots=shots[0], copies=len(shots))]
            self.total_shots = shots[0] * len(shots)
        else:
            # Iterate through the shots, and group consecutive identical shots
            split_at_repeated = np.split(shots, np.diff(shots).nonzero()[0] + 1)
            self.shot_vector = [ShotTuple(shots=i[0], copies=len(i)) for i in split_at_repeated]
            self.total_shots = int(np.sum(np.prod(self.shot_vector, axis=1)))

    def has_partitioned_shots(self):
        """Checks if the device was instructed to perform executions with partitioned shots.

        Returns:
            bool: whether or not shots are partitioned
        """
        return len(self.shot_list) > 1


# TODO: should the set_shots contextmanager move here?
