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
# pylint: disable=protected-access
r"""
Classes representing domains
============================

**Module name:** :mod:`pennylane.domain`

.. currentmodule:: pennylane.domain

Operation base classes
----------------------

This module contains classes representing different domains, mostly for specifyting the par_domain of Operations.


Summary
^^^^^^^

.. autosummary::
   Domain
   Scalars
   Complex
   Reals
   Ints
   Interval
   Matrices
   Unitaries
   Hermitians

Code details
^^^^^^^^^^^^
"""

import abc

class Domain():
    def __init__(self):
        pass

    #using this for comparison allows to write something like: if cls.par_domain in qml.domain.Reals(): ...
    def __contains__(self, other):
        return issubclass(type(other), self.__class__)

    def __eq__(self, other):
        return type(other) == type(self)

    @abc.abstractproperty
    def rand():
        r"""
        Produces a random instance from the respective domain.
        """
        raise NotImplementedError

class Scalars(Domain):
    def __init__(self):
        pass

class Complex(Scalars):
    def __init__(self):
        pass

class Reals(Complex):
    def __init__(self, non_negative=False):
        self._non_negative = non_negative

    @property
    def non_negative(self):
        return self._non_negative

class Ints(Reals):
    def __init__(self, non_negative=False):
        super().__init__(non_negative=non_negative)

class Interval(Reals):
    def __init__(self, min, max):
        if(min > max):
            ValueError("min must be smaller than max")
        super().__init__(non_negative=(min >= 0))

class Matrices(Domain):
    r"""qml.domain.Matrices(shape)

    Args:
        shape (callable): a callable taking the number of wires as argument and outputing the shape of the matrix
    """
    def __init__(self, shape):
        raise NotImplementedError

class Unitaries(Matrices):
    def __init__(self, shape, non_negative=False):
        raise NotImplementedError

class Hermitians(Matrices):
    def __init__(self, shape, non_negative=False):
        pass

all_domains = [
    Domain,
    Scalars,
    Complex,
    Reals,
    Ints,
    Interval,
    Matrices,
    Unitaries,
    Hermitians
]

__all__ = [cls.__name__ for cls in all_domains]
