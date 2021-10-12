# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
"""
Contains a utility class ``BooleanFn`` that allows logical composition
of functions with boolean output.
"""

class BooleanFn:
    r"""Wrapper for simple callables with Boolean output that can be
    manipulated and combined with bit-wise operators.

    Args:
        fn (callable): Function to be wrapped. It must have signature
            ``object->bool``.

    **Example**

    Consider functions that filter numbers to lie in a certain domain,
    and wrap them using a ``BooleanFn``:

    >>> bigger_than_4 = qml.BooleanFn(lambda x: x > 4)
    >>> smaller_than_10 = qml.BooleanFn(lambda x: x < 10)
    >>> is_int = qml.BooleanFn(lambda x: isinstance(x, int))
    >>> bigger_than_4(5.2)
    True
    >>> smaller_than_10(20.1)
    False
    >>> is_int(2.3)
    False

    These can then be combined into a single callable:

    >>> between_4_and_10 = bigger_than_4 & smaller_than_10
    >>> between_4_and_10(-3.2)
    False
    >>> between_4_and_10(9.9)
    True
    >>> between_4_and_10(19.7)
    False

    Other operations are ``|``, logical or, and ``~``, logical not:

    >>> smaller_equal_than_4 = ~bigger_than_4
    >>> smaller_than_10_or_int = smaller_than_10 | is_int

    .. warning::

        While the logical composition of multiple ``BooleanFn`` s works as expected,
        and in particular the operations ``|`` and ``&`` are symmetric,
        it might make a difference in which order the composition is written, because
        a ``BooleanFn`` might assume certain object properties to be defined.

        .. code-block:: python

            is_int = StoppingCriterion(lambda x: isinstance(x, int))
            has_bit_length_3 = StoppingCriterion(lambda x: x.bit_length()==3)

        >>> (is_int & has_bit_length_3)(4)
        True
        >>> (is_int & has_bit_length_3)(2.3)
        False
        >>> (has_bit_length_3 & is_int)(2.3)
        AttributeError: 'float' object has no attribute 'bit_length'

    """

    def __init__(self, fn):
        self.fn = fn

    def __and__(self, other):
        return BooleanFn(lambda obj: self.fn(obj) and other.fn(obj))

    def __or__(self, other):
        return BooleanFn(lambda obj: self.fn(obj) or other.fn(obj))

    def __invert__(self):
        return BooleanFn(lambda obj: not self.fn(obj))

    def __call__(self, obj):
        return self.fn(obj)
