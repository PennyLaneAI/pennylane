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
# pylint: disable=unused-argument
import contextlib

import pennylane as qml
from pennylane.operation import (
    has_gen,
    has_grad_method,
    has_nopar,
    has_unitary_gen,
    is_measurement,
    is_trainable,
    not_tape,
)

# Needed for custom decomposition context manager
from pennylane.transforms.qfunc_transforms import NonQueuingTape

NonQueuingTape = type("NonQueuingTape", (NonQueuingTape, qml.tape.QuantumTape), {})


def _update_trainable_params(tape):
    params = tape.get_parameters(trainable_only=False)
    tape.trainable_params = qml.math.get_trainable_indices(params)


def create_expand_fn(depth, stop_at=None, device=None, docstring=None):
    """Create a function for expanding a tape to a given depth, and
    with a specific stopping criterion. This is a wrapper around
    :meth:`~.QuantumTape.expand`.

    Args:
        depth (int): Depth for the expansion
        stop_at (callable): Stopping criterion. This must be a function with signature
            ``stop_at(obj)``, where ``obj`` is a *queueable* PennyLane object such as
            :class:`~.Operation` or :class:`~.MeasurementProcess`. It must return a
            boolean, indicating if the expansion should stop at this object.
        device (.Device): Ensure that the expanded tape only uses native gates of the
            given device.
        docstring (str): docstring for the generated expansion function

    Returns:
        callable: Tape expansion function. The returned function accepts a :class:`~.QuantumTape`,
        and returns an expanded :class:`~.QuantumTape`.

    **Example**

    Let us construct an expansion function that expands a tape in order to
    decompose trainable multi-parameter gates. We allow for up to five expansion
    steps, which can be controlled with the argument ``depth``.
    The stopping criterion is easy to write as

    >>> stop_at = ~(qml.operation.has_multipar & qml.operation.is_trainable)

    Then the expansion function can be obtained via

    >>> expand_fn = qml.transforms.create_expand_fn(depth=5, stop_at=stop_at)

    We can test the newly generated function on an example tape:

    .. code-block:: python

        with qml.tape.JacobianTape() as tape:
            qml.RX(0.2, wires=0)
            qml.RX(qml.numpy.array(-2.4, requires_grad=True), wires=1)
            qml.Rot(1.7, 0.92, -1.1, wires=0)
            qml.Rot(*qml.numpy.array([-3.1, 0.73, 1.36], requires_grad=True), wires=1)

    >>> new_tape = expand_fn(tape)
    >>> print(tape.draw())
     0: ──RX(0.2)───Rot(1.7, 0.92, -1.1)───┤
     1: ──RX(-2.4)──Rot(-3.1, 0.73, 1.36)──┤
    >>> print(new_tape.draw())
     0: ──RX(0.2)───Rot(1.7, 0.92, -1.1)──────────────────────┤
     1: ──RX(-2.4)──RZ(-3.1)──────────────RY(0.73)──RZ(1.36)──┤

    """
    # pylint: disable=unused-argument
    if device is not None:
        if stop_at is None:
            stop_at = device.stopping_condition
        else:
            stop_at &= device.stopping_condition

    def expand_fn(tape, depth=depth, **kwargs):

        with qml.tape.stop_recording():

            if stop_at is None:
                tape = tape.expand(depth=depth)
            elif not all(stop_at(op) for op in tape.operations):
                tape = tape.expand(depth=depth, stop_at=stop_at)
            else:
                return tape

            _update_trainable_params(tape)

        return tape

    if docstring:
        expand_fn.__doc__ = docstring

    return expand_fn


_expand_multipar_doc = """Expand out a tape so that all its parametrized
operations have a single parameter.

This is achieved by decomposing all parametrized operations that do not have
a generator, up to maximum depth ``depth``.
For a sufficient ``depth``, it should always be possible to obtain a tape containing
only single-parameter operations.

Args:
    tape (.QuantumTape): the input tape to expand
    depth (int) : the maximum expansion depth
    **kwargs: additional keyword arguments are ignored

Returns:
    .QuantumTape: the expanded tape
"""

expand_multipar = create_expand_fn(
    depth=10,
    stop_at=not_tape | is_measurement | has_nopar | has_gen,
    docstring=_expand_multipar_doc,
)

_expand_trainable_multipar_doc = """Expand out a tape so that all its trainable
operations have a single parameter.

This is achieved by decomposing all trainable operations that do not have
a generator, up to maximum depth ``depth``.
For a sufficient ``depth``, it should always be possible to obtain a tape containing
only single-parameter operations.

Args:
    tape (.QuantumTape): the input tape to expand
    depth (int) : the maximum expansion depth
    **kwargs: additional keyword arguments are ignored

Returns:
    .QuantumTape: the expanded tape
"""

expand_trainable_multipar = create_expand_fn(
    depth=10,
    stop_at=not_tape | is_measurement | has_nopar | (~is_trainable) | has_gen,
    docstring=_expand_trainable_multipar_doc,
)


_expand_nonunitary_gen_doc = """Expand out a tape so that all its parametrized
operations have a unitary generator.

This is achieved by decomposing all parametrized operations that either do not have
a generator or have a non-unitary generator, up to maximum depth ``depth``.
For a sufficient ``depth``, it should always be possible to obtain a tape containing
only unitarily generated operations.

Args:
    tape (.QuantumTape): the input tape to expand
    depth (int) : the maximum expansion depth
    **kwargs: additional keyword arguments are ignored

Returns:
    .QuantumTape: the expanded tape
"""

expand_nonunitary_gen = create_expand_fn(
    depth=10,
    stop_at=not_tape | is_measurement | has_nopar | (has_gen & has_unitary_gen),
    docstring=_expand_nonunitary_gen_doc,
)


_expand_invalid_trainable_doc = """Expand out a tape so that it supports differentiation
of requested operations.

This is achieved by decomposing all trainable operations that have
``Operation.grad_method=None`` until all resulting operations
have a defined gradient method, up to maximum depth ``depth``. Note that this
might not be possible, in which case the gradient rule will fail to apply.

Args:
    tape (.QuantumTape): the input tape to expand
    depth (int) : the maximum expansion depth
    **kwargs: additional keyword arguments are ignored

Returns:
    .QuantumTape: the expanded tape
"""

expand_invalid_trainable = create_expand_fn(
    depth=10,
    stop_at=not_tape | is_measurement | (~is_trainable) | has_grad_method,
    docstring=_expand_invalid_trainable_doc,
)


@contextlib.contextmanager
def _custom_decomp_context(custom_decomps):
    """A context manager for applying custom decompositions of operations."""

    # Creates an individual context
    @contextlib.contextmanager
    def _custom_decomposition(obj, fn):
        # Covers the case where the user passes a string to indicate the Operator
        if isinstance(obj, str):
            obj = getattr(qml, obj)

        original_decomp_method = obj.decompose

        # This is the method that will override the operations .decompose method
        def new_decomp_method(self):
            with NonQueuingTape():
                if self.num_params == 0:
                    return fn(self.wires)
                return fn(*self.parameters, self.wires)

        try:
            # Explicitly set the new .decompose method
            obj.decompose = new_decomp_method
            yield

        finally:
            obj.decompose = original_decomp_method

    # Loop through the decomposition dictionary and create all the contexts
    try:
        with contextlib.ExitStack() as stack:
            for obj, fn in custom_decomps.items():
                # We enter a new context for each decomposition the user passes
                stack.enter_context(_custom_decomposition(obj, fn))

            stack = stack.pop_all()

        yield

    finally:
        stack.close()


def create_decomp_expand_fn(custom_decomps, dev, decomp_depth=10):
    """Creates a custom expansion function for a device that applies
    a set of specified custom decompositions.

    Args:
        custom_decomps (Dict[Union(str, qml.operation.Operation), Callable]): Custom
            decompositions to be applied by the device at runtime.
        dev (qml.Device): A quantum device.
        decomp_depth: The maximum depth of the expansion.

    Returns:
        Callable: A custom expansion function that a device can call to expand
        its tapes within a context manager that applies custom decompositions.

    **Example**

    Suppose we would like a custom expansion function that decomposes all CNOTs
    into CZs. We first define a decomposition function:

    .. code-block:: python

        def custom_cnot(wires):
            return [
                qml.Hadamard(wires=wires[1]),
                qml.CZ(wires=[wires[0], wires[1]]),
                qml.Hadamard(wires=wires[1])
            ]

    We then create the custom function (passing a device, in order to pick up any
    additional stopping criteria the expansion should have), and then register the
    result as a custom function of the device:

    >>> custom_decomps = {qml.CNOT : custom_cnot}
    >>> expand_fn = qml.transforms.create_decomp_expand_fn(custom_decomps, dev)
    >>> dev.custom_expand(expand_fn)
    """
    custom_op_names = [op if isinstance(op, str) else op.__name__ for op in custom_decomps.keys()]

    # Create a new expansion function; stop at things that do not have
    # custom decompositions, or that satisfy the regular device stopping criteria
    custom_fn = qml.transforms.create_expand_fn(
        decomp_depth,
        stop_at=qml.BooleanFn(lambda obj: obj.name not in custom_op_names),
        device=dev,
    )

    # Finally, we set the device's custom_expand_fn to a new one that
    # runs in a context where the decompositions have been replaced.
    def custom_decomp_expand(self, circuit, max_expansion=decomp_depth):
        with _custom_decomp_context(custom_decomps):
            return custom_fn(circuit, max_expansion=max_expansion)

    return custom_decomp_expand


@contextlib.contextmanager
def set_decomposition(custom_decomps, dev, decomp_depth=10):
    """Context manager for setting custom decompositions.

    Args:
        custom_decomps (Dict[Union(str, qml.operation.Operation), Callable]): Custom
            decompositions to be applied by the device at runtime.
        dev (qml.Device): A quantum device.
        decomp_depth: The maximum depth of the expansion.

    **Example**

    Suppose we would like a custom expansion function that decomposes all CNOTs
    into CZs. We first define a decomposition function:

    .. code-block:: python

        def custom_cnot(wires):
            return [
                qml.Hadamard(wires=wires[1]),
                qml.CZ(wires=[wires[0], wires[1]]),
                qml.Hadamard(wires=wires[1])
            ]

    This context manager can be used to temporarily change a devices expansion
    function to one that takes into account the custom decompositions.

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, expansion_strategy="device")
        def circuit():
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=0))

    >>> print(qml.draw(circuit)())
     0: ──╭C──┤ ⟨Z⟩
     1: ──╰X──┤

    Now let's set up a context where the custom decomposition will be applied:

    >>> with qml.transforms.set_decomposition({qml.CNOT : custom_cnot}, dev):
    ...     print(qml.draw(circuit)())
     0: ─────╭C─────┤ ⟨Z⟩
     1: ──H──╰Z──H──┤
    """
    original_custom_expand_fn = dev.custom_expand_fn

    # Create a new expansion function; stop at things that do not have
    # custom decompositions, or that satisfy the regular device stopping criteria
    new_custom_expand_fn = qml.transforms.create_decomp_expand_fn(
        custom_decomps, dev, decomp_depth=decomp_depth
    )

    # Set the custom expand function within this context only
    try:
        dev.custom_expand(new_custom_expand_fn)
        yield

    finally:
        dev.custom_expand_fn = original_custom_expand_fn
