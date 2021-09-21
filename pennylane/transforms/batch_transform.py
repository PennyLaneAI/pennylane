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
"""Contains tools and decorators for registering batch transforms."""
# pylint: disable=too-few-public-methods
import functools
import types

import pennylane as qml


class batch_transform:
    r"""Class for registering a tape transform that takes a tape, and outputs
    a batch of tapes to be independently executed on a quantum device.

    Examples of such transforms include quantum gradient shift rules (such
    as finite-differences and the parameter-shift rule) and metrics such as
    the quantum Fisher information matrix.

    Args:
        transform_fn (function): The function to register as the batch tape transform.
            It can have an arbitrary number of arguments, but the first argument
            **must** be the input tape.
        expand_fn (function): An expansion function (if required) to be applied to the
            input tape before the transformation takes place.
        differentiable (bool): Specifies whether the transform is differentiable or
            not. A transform may be non-differentiable for several reasons:

            - It does not use an autodiff framework for its tensor manipulations;
            - It returns a non-differentiable or non-numeric quantity, such as
              a boolean, string, or integer.

            In such a case, setting ``differentiable=False`` instructs the decorator
            to mark the output as 'constant', reducing potential overhead.

    **Example**

    A valid batch tape transform is a function that satisfies the following:

    - The first argument must be a tape.

    - Depending on the structure of this input tape, various quantum operations, functions,
      and templates may be called.

    - Any internal classical processing should use the ``qml.math`` module to ensure
      the transform is differentiable.

    - The transform should return a tuple containing:

      * Multiple transformed tapes to be executed on a device.
      * A classical processing function for post-processing the executed tape results.
        This processing function should have the signature ``f(list[tensor_like]) → Any``.
        If ``None``, no classical processing is applied to the results.

    For example:

    .. code-block:: python

        @qml.batch_transform
        def my_transform(tape, a, b):
            '''Generates two tapes, one with all RX replaced with RY,
            and the other with all RX replaced with RZ.'''

            tape1 = qml.tape.JacobianTape()
            tape2 = qml.tape.JacobianTape()

            # loop through all operations on the input tape
            for op in tape.operations + tape.measurements:
                if op.name == "RX":
                    wires = op.wires
                    param = op.parameters[0]

                    with tape1:
                        qml.RY(a * qml.math.abs(param), wires=wires)

                    with tape2:
                        qml.RZ(b * qml.math.abs(param), wires=wires)
                else:
                    for t in [tape1, tape2]:
                        with t:
                            qml.apply(op)

            def processing_fn(results):
                return qml.math.sum(qml.math.stack(results))

            return [tape1, tape2], processing_fn

    We can apply this transform to a quantum tape:

    >>> with qml.tape.JacobianTape() as tape:
    ...     qml.Hadamard(wires=0)
    ...     qml.RX(-0.5, wires=0)
    ...     qml.expval(qml.PauliX(0))
    >>> tapes, fn = my_transform(tape, 0.65, 2.5)
    >>> print(tapes[0].draw())
     0: ──H──RY(0.325)──┤ ⟨X⟩
    >>> print(tapes[1].draw())
     0: ──H──RZ(1.25)──┤ ⟨X⟩

    We can execute these tapes manually:

    >>> dev = qml.device("default.qubit", wires=1)
    >>> res = qml.execute(tapes, dev, interface="autograd", gradient_fn=qml.gradients.param_shift)
    >>> print(res)
    [tensor([0.94765073], requires_grad=True), tensor([0.31532236], requires_grad=True)]

    Applying the processing function, we retrieve the end result of the transform:

    >>> print(fn(res))
    1.2629730888100839

    Alternatively, we may also transform a QNode directly, using either
    decorator syntax:

    >>> @my_transform(0.65, 2.5)
    ... @qml.qnode(dev)
    ... def circuit(x):
    ...     qml.Hadamard(wires=0)
    ...     qml.RX(x, wires=0)
    ...     return qml.expval(qml.PauliX(0))
    >>> print(circuit(-0.5))
    1.2629730888100839

    or by transforming an existing QNode:

    >>> @qml.qnode(dev)
    ... def circuit(x):
    ...     qml.Hadamard(wires=0)
    ...     qml.RX(x, wires=0)
    ...     return qml.expval(qml.PauliX(0))
    >>> circuit = my_transform(circuit, 0.65, 2.5)
    >>> print(circuit(-0.5))
    1.2629730888100839

    Batch tape transforms are fully differentiable:

    >>> gradient = qml.grad(circuit)(-0.5)
    >>> print(gradient)
    2.5800122591960153
    """

    def __init__(self, transform_fn, expand_fn=None, differentiable=True):
        if not callable(transform_fn):
            raise ValueError(
                f"The batch transform function to register, {transform_fn}, "
                "does not appear to be a valid Python function or callable."
            )

        self.transform_fn = transform_fn
        self.expand_fn = expand_fn
        self.differentiable = differentiable
        self.qnode_wrapper = self.default_qnode_wrapper
        functools.update_wrapper(self, transform_fn)

    def custom_qnode_wrapper(self, fn):
        """Register a custom QNode execution wrapper function
        for the batch transform.

        **Example**

        .. code-block:: python

            def my_transform(tape, *targs, **tkwargs):
                ...
                return tapes, processing_fn

            @my_transform.custom_qnode_wrapper
            def my_custom_qnode_wrapper(self, qnode, targs, tkwargs):
                def wrapper_fn(*args, **kwargs):
                    # construct QNode
                    qnode.construct(args, kwargs)
                    # apply transform to QNode's tapes
                    tapes, processing_fn = self.construct(qnode.qtape, *targs, **tkwargs)
                    # execute tapes and return processed result
                    ...
                    return processing_fn(results)
                return wrapper_fn

        The custom QNode execution wrapper must have arguments
        ``self`` (the batch transform object), ``qnode`` (the input QNode
        to transform and execute), ``targs`` and ``tkwargs`` (the transform
        arguments and keyword arguments respectively).

        It should return a callable object that accepts the *same* arguments
        as the QNode, and returns the transformed numerical result.

        The default :meth:`~.default_qnode_wrapper` method may be called
        if only pre- or post-processing dependent on QNode arguments is required:

        .. code-block:: python

            @my_transform.custom_qnode_wrapper
            def my_custom_qnode_wrapper(self, qnode, targs, tkwargs):
                transformed_qnode = self.default_qnode_wrapper(qnode)

                def wrapper_fn(*args, **kwargs):
                    args, kwargs = pre_process(args, kwargs)
                    res = transformed_qnode(*args, **kwargs)
                    ...
                    return ...
                return wrapper_fn
        """
        self.qnode_wrapper = types.MethodType(fn, self)

    def default_qnode_wrapper(self, qnode, targs, tkwargs):
        """A wrapper method that takes a QNode and transform arguments,
        and returns a function that 'wraps' the QNode execution.

        The returned function should accept the same keyword arguments as
        the QNode, and return the output of the applying the tape transform
        to the QNode's constructed tape.
        """

        def _wrapper(*args, **kwargs):
            qnode.construct(args, kwargs)
            tapes, processing_fn = self.construct(qnode.qtape, *targs, **tkwargs)

            # TODO: work out what to do for backprop
            interface = qnode.interface

            # TODO: extract gradient_fn from QNode
            gradient_fn = getattr(qnode, "gradient_fn", qnode.diff_method)

            if interface is None or not self.differentiable:
                gradient_fn = None

            elif gradient_fn in ("best", "parameter-shift"):
                gradient_fn = qml.gradients.param_shift

            elif gradient_fn == "finite-diff":
                gradient_fn = qml.gradients.finite_diff

            res = qml.execute(
                tapes,
                device=qnode.device,
                gradient_fn=gradient_fn,
                interface=interface,
            )

            return processing_fn(res)

        return _wrapper

    def __call__(self, qnode, *targs, **tkwargs):
        if isinstance(qnode, qml.tape.QuantumTape):
            # Input is a quantum tape.
            # tapes, fn = some_transform(tape, *transform_args)
            return self.construct(qnode, *targs, **tkwargs)

        if isinstance(qnode, (qml.QNode, qml.ExpvalCost)):
            # Input is a QNode:
            # result = some_transform(qnode, *transform_args)(*qnode_args)
            wrapper = self.qnode_wrapper(qnode, targs, tkwargs)
            wrapper = functools.wraps(qnode)(wrapper)

        else:
            # Input is not a QNode nor a quantum tape.
            # Assume Python decorator syntax:
            #
            # result = some_transform(*transform_args)(qnode)(*qnode_args)
            #
            # or
            #
            # @some_transform(*transform_args)
            # @qml.qnode(dev)
            # def circuit(...):
            #     ...
            # result = circuit(*qnode_args)

            # Prepend the input to the transform args,
            # and create a wrapper function.
            targs = (qnode,) + targs

            def wrapper(qnode):
                _wrapper = self.qnode_wrapper(qnode, targs, tkwargs)
                _wrapper = functools.wraps(qnode)(_wrapper)
                return _wrapper

        wrapper.tape_fn = functools.partial(self.transform_fn, *targs, **tkwargs)
        wrapper.expand_fn = self.expand_fn
        wrapper.differentiable = self.differentiable
        return wrapper

    def construct(self, tape, *args, **kwargs):
        """Applies the batch tape transform to an input tape.

        Args:
            tape (.QuantumTape): the tape to be transformed
            *args: positional arguments to pass to the tape transform
            **kwargs: keyword arguments to pass to the tape transform

        Returns:
            tuple[list[tapes], callable]: list of transformed tapes
            to execute and a post-processing function.
        """
        expand = kwargs.pop("_expand", True)

        if expand and self.expand_fn is not None:
            tape = self.expand_fn(tape)

        tapes, processing_fn = self.transform_fn(tape, *args, **kwargs)

        if processing_fn is None:
            processing_fn = lambda x: x

        return tapes, processing_fn
