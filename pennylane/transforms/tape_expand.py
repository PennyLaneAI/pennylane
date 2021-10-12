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
"""This module contains tape expansion functions and stopping criteria to
generate such functions from."""

import pennylane as qml


class BooleanFn:
    r"""Wrapper for simple callables with Boolean output that can be
    manipulated and combined with bit-wise operators.

    Args:
        fn (callable): Function to be wrapped. It must have signature
            ``object->bool``.

    **Example**

    Consider functions that filter numbers to lie in a certain domain,
    and wrap them using a ``BooleanFn``:

    >>> bigger_than_4 = qml.transforms.BooleanFn(lambda x: x > 4)
    >>> smaller_than_10 = qml.transforms.BooleanFn(lambda x: x < 10)
    >>> is_int = qml.transforms.BooleanFn(lambda x: isinstance(x, int))
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


has_generator = BooleanFn(
    lambda obj: hasattr(obj, "generator") and obj.generator[0] is not None
)
has_grad_method = BooleanFn(lambda obj: obj.grad_method is not None)
has_multiple_params = BooleanFn(lambda obj: obj.num_params > 1)
has_no_params = BooleanFn(lambda obj: obj.num_params == 0)
has_unitary_generator = BooleanFn(lambda obj: obj.has_unitary_generator)
is_measurement = BooleanFn(lambda obj: isinstance(obj, qml.measure.MeasurementProcess))
is_trainable = BooleanFn(lambda obj: any(qml.math.requires_grad(p) for p in obj.parameters))


def get_expand_fn(depth, stop_at):
    """Create an expansion function using a given depth and stopping criterions,
    wrapping ``tape.expand``.

    Args:
        depth (int): Depth for the expansion
        stop_at (callable): Stopping criterion passed to ``tape.expand``

    Returns:
        callable: Tape expansion function

    **Example**

    Let us construct an expansion function that expands a tape trying to remove
    multi-parameter gates that are trainable. We allow for up to five expansion
    steps, which can be controlled with the argument ``depth``.
    The stopping criterion is easy to write as

    >>> stop_at = ~(qml.transforms.has_multiple_params & qml.transforms.is_trainable)

    Then the expansion function can be obtained via

    >>> expand_fn = qml.transforms.get_expand_fn(5, stop_at)

    We can test the newly generated function on an example tape:

    .. code-block:: python

        with qml.tape.JacobianTape() as tape:
            qml.RX(0.2, wires=0)
            qml.RX(qml.numpy.array(-2.4, requires_grad=True), wires=1)
            qml.Rot(1.7, 0.92, -1.1, wires=0)
            qml.Rot(*qml.numpy.array([-3.1, 0.73, 1.36], requires_grad=True), wires=1)

    >>> new_tape = expand_fn(tape)
    >>> print(*new_tape.operations, sep='\n')
    RX(0.2, wires=[0])
    RX(tensor(-2.4, requires_grad=True), wires=[1])
    Rot(1.7, 0.92, -1.1, wires=[0])
    RZ(tensor(-3.1, requires_grad=True), wires=[1])
    RY(tensor(0.73, requires_grad=True), wires=[1])
    RZ(tensor(1.36, requires_grad=True), wires=[1])

    """
    def expand_fn(tape):
        if not all(stop_at(op) for op in tape.operations):
            tape = tape.expand(depth=depth, stop_at=stop_at)
            params = tape.get_parameters(trainable_only=False)
            tape.trainable_params = qml.math.get_trainable_indices(params)
        return tape

    return expand_fn


expand_multi_par_and_no_gen = get_expand_fn(
    depth=10, stop_at=is_measurement | has_no_params | has_generator
)
expand_multi_par_and_nonunitary_gen = get_expand_fn(
    depth=10, stop_at=is_measurement | has_no_params | (has_generator & has_unitary_generator)
)
