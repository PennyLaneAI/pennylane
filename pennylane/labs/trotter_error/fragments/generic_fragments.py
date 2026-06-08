# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wrapper class for generic fragment objects"""

from __future__ import annotations

from numpy.typing import ArrayLike

from trotter_error import AbstractState, Fragment


class NumpyFragment(Fragment):
    def __init__(self, fragment: ArrayLike):
        self.fragment = fragment

    def __add__(self, other: NumpyFragment):
        return NumpyFragment(self.fragment + other.fragment)

    def __sub__(self, other: NumpyFragment):
        return NumpyFragment(self.fragment + (-1) * other.fragment)

    def __mul__(self, scalar: float):
        return NumpyFragment(scalar * self.fragment)

    def __eq__(self, other: NumpyFragment):
        if not isinstance(self.fragment, type(other)):
            return False

        return self.fragment == other.fragment

    __rmul__ = __mul__

    def __matmul__(self, other: NumpyFragment):
        return NumpyFragment(self.fragment @ other.fragment)

    def apply(self, state: NumpyState) -> NumpyState:
        """Apply the fragment to a state using the underlying object's ``__matmul__`` method."""
        return NumpyState(self.fragment @ state.state)

    def expectation(self, left: NumpyState, right: NumpyState) -> float:
        """Compute the expectation value using the underlying object's ``__matmul__`` method."""
        return left.state.T @ self.fragment @ right.state

    def __repr__(self):
        return f"NumpyFragment(type={type(self.fragment)})"


class NumpyState(AbstractState):
    """State wrapper for Numpy objects"""

    def __init__(self, state: ArrayLike):
        self.state = state

    def __add__(self, other: NumpyState):
        return NumpyState(self.state + other.state)

    def __mul__(self, scalar: float):
        return NumpyState(scalar * self.state)

    def dot(self, other: NumpyState):
        return self.state.dot(other.state)
