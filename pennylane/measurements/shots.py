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
"""
previous location of pennylane.core.shots
"""

from pennylane.core.shots import *  # pylint: disable=wildcard-import, unused-import, unused-wildcard-import # tach-ignore


def add_shots(s1: Shots, s2: Shots) -> Shots:
    """Add two :class:`~.Shots` objects by concatenating their shot vectors.

    Args:
        s1 (Shots): a Shots object to add
        s2 (Shots): a Shots object to add

    Returns:
        Shots: a :class:`~.Shots` object built by concatenating the shot vectors of ``s1`` and ``s2``

    Example:
        >>> s1 = Shots((5, (10, 2)))
        >>> s2 = Shots((3, 2, (10, 3)))
        >>> print(qp.measurements.add_shots(s1, s2))
        Shots(total=60, vector=[5 shots, 10 shots x 2, 3 shots, 2 shots, 10 shots x 3])
    """
    return s1 + s2
