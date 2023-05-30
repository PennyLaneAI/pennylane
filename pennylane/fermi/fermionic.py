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
"""The Fermionic representation classes"""
from copy import copy


class FermiSentence(dict):
    """Immutable dictionary used to represent a Fermi sentence, a linear combination of Fermi words, with the keys
    as FermiWord instances and the values correspond to coefficients.

    >>> w1 = FermiWord({(0, 0) : '+', (1, 1) : '-'})
    >>> w2 = FermiWord({(0, 1) : '+', (1, 2) : '-'})
    >>> s = FermiSentence({w1 : 1.2, w2: 3.1})
    >>> s
    1.2 * '0+ 1-' + 3.1 * '1+ 2-'
    """

    @property
    def wires(self):
        """Return wires of the FermiSentence."""
        return set().union(*(fw.wires for fw in self.keys()))

    def __str__(self):
        """String representation of a FermiSentence."""

        return "\n+ ".join(f"{coeff} * '{fw.to_string()}'" for fw, coeff in self.items())

    def __repr__(self):
        """Terminal representation for FermiSentence."""
        return str(self)

    def __missing__(self, key):
        """If the FermiSentence does not contain a FermiWord then the associated value will be 0."""
        return 0.0

    def __add__(self, other):
        """Add two Fermi sentence together by iterating over the smaller one and adding its terms
        to the larger one."""
        smaller_fs, larger_fs = (
            (self, copy(other)) if len(self) < len(other) else (other, copy(self))
        )
        for key in smaller_fs:
            larger_fs[key] += smaller_fs[key]

        return larger_fs

    def __mul__(self, other):
        """Multiply two Fermi sentences by iterating over each sentence and multiplying the Fermi
        words pair-wise"""

        if len(self) == 0:
            return copy(other)

        if len(other) == 0:
            return copy(self)

        keys = [i * j for i in self.keys() for j in other.keys()]
        vals = [i * j for i in self.values() for j in other.values()]

        return FermiSentence(dict(zip(keys, vals)))

    def __pow__(self, value):
        if not isinstance(value, int):
            raise TypeError("The exponent must be integer.")

        operator = FermiSentence({})

        for _ in range(value):
            operator *= self

        return operator

    def simplify(self, tol=1e-8):
        """Remove any FermiWords in the FermiSentence with coefficients less than the threshold
        tolerance."""
        items = list(self.items())
        for fw, coeff in items:
            if abs(coeff) <= tol:
                del self[fw]
