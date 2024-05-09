# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterable

import pennylane as qml
from pennylane.ops import Conditional
from pennylane.boolean_fn import BooleanFn


class NoiseConditionals(BooleanFn):
    """Defines a BooleanFn for implementing noise"""

    def __init__(self, fn, repr):
        super().__init__(fn)
        self.repr = repr

    def __and__(self, other):
        return AndConditional(self, other)

    def __or__(self, other):
        return OrConditional(self, other)

    def __repr__(self):
        return self.repr


class AndConditional(NoiseConditionals):
    """Defines a BooleanFn for implementing AND combination of noise fns"""

    def __init__(self, left, right):
        self.l_op = left
        self.r_op = right
        super(NoiseConditionals, self).__init__(super(NoiseConditionals, left).__and__(right))
        self.repr = f"And({left.repr}, {right.repr})"

    def __str__(self):
        return f"And({self.l_op}, {self.r_op})"


class OrConditional(NoiseConditionals):
    """Defines a BooleanFn for implementing OR combination of noise fns"""

    def __init__(self, left, right):
        self.l_op = left
        self.r_op = right
        super(NoiseConditionals, self).__init__(super(NoiseConditionals, left).__or__(right))
        self.repr = f"Or({left.repr}, {right.repr})"

    def __str__(self):
        return f"Or({self.l_op}, {self.r_op})"


def _get_wires(x):
    """Obtain wires as a set from a Wire, Iterable or Operation"""
    wires = getattr(x, "wires", x)
    return set(wires if isinstance(wires, Iterable) else [wires])


def wire_in(wires):
    """BooleanFn for checking if a wire exist in a set of specified wires"""
    return NoiseConditionals(
        lambda x: _get_wires(x).issubset(_get_wires(wires)), f"WiresIn({wires})"
    )


def wire_eq(wires):
    """BooleanFn for checking if a wire is equal to the set of specified wires"""
    return NoiseConditionals(
        lambda x: _get_wires(x).issubset(_get_wires(wires)), f"WiresEq({wires})"
    )


def _get_ops(x):
    """Help deal with arithmetic ops"""
    pass


def op_eq(ops):
    """BooleanFn for checking if an op exist in a set of specified ops"""
    return NoiseConditionals(lambda x: isinstance(x, ops), f"OpEq({ops.__name__})")


def op_in(ops):
    """BooleanFn for checking if an op is equal to the set of specified ops"""
    return NoiseConditionals(
        lambda x: isinstance(x, tuple(ops)), f"OpIn({[op.__name__ for op in ops]})"
    )
