# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains operations for the default phase space observables"""

from openqml.operation import Expectation



class Fock(Expectation):   # FIXME nondescriptive name
    r"""Returns the photon-number expectation value in the phase space.

    The photon number operator is :math:`n = a^\dagger a = \frac{1}{2\hbar}(x^2 +p^2) -\I/2`.
    """
    def heisenberg_expand(self):
        return np.diag([-0.5, 0.25, 0.25])


class X(Expectation):
    r"""Returns the position expectation value in the phase space.
    """
    def heisenberg_expand(self):
        return np.array([0, 1, 0])


class P(Expectation):
    r"""Returns the momentum expectation value in the phase space.

    """
    def heisenberg_expand(self):
        return np.array([0, 0, 1])


class Homodyne(Expectation):
    r"""Returns the homodyne expectation value in the phase space.

    Args:
        phi (float): axis in the phase space at which to calculate
            the homodyne measurement.
    """
    n_params = 1
    def heisenberg_expand(self):
        phi = self.parameters[0]
        return np.array([0, np.cos(phi), np.sin(phi)])  # TODO check



class Heterodyne(Expectation):
    r"""Returns the displacement expectation value in the phase space.

    """



all_ops = [Heterodyne, Homodyne, Fock, P, X]

__all__ = [cls.__name__ for cls in all_ops]
