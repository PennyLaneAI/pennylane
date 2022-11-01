# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This module contains the op_transform decorator.
"""
# pylint: disable=protected-access
import functools
import inspect
import os
import warnings

import pennylane as qml


class OperationTransformError(Exception):
    """Raised when there is an error with the op_transform logic"""


class op_transform:
    r"""Convert a function that applies to operators into a functional transform.

    This allows the operator function to be used across PennyLane
    on both instantiated operators as well as quantum functions.

    By default, this decorator creates functional transforms that
    accept a single operator. However, you can also register how the
    transform acts on multiple operators. Once this is defined,
    the transform can be used anywhere in PennyLane --- at the operator
    level for operator arithmetic, or at the qfunc/QNode level.

    .. warning::

        This is an experimental feature, and is subject to change.

    Args:
        fn (function): The function to register as the operator transform.
            It can have an arbitrary number of arguments, but the first argument
            **must** be the input operator.

    **Example**

    Consider an operator function that computes the trace of an operator:

    .. code-block:: python

        @qml.op_transform
        def trace(op):
            try:
                return qml.math.real(qml.math.sum(op.eigvals()))
            except qml.operation.EigvalsUndefinedError:
                return qml.math.real(qml.math.trace(op.matrix()))

    We can use this function as written:

    >>> op = qml.RX(0.5, wires=0)
    >>> trace(op)
    1.9378248434212895

    By using the ``op_transform`` decorator, we also enable it to be used
    as a functional transform:

    >>> trace(qml.RX)(0.5, wires=0)
    1.9378248434212895

    Note that if we apply our function to an operation that does not define its
    matrix or eigenvalues representation, we get an error:

    >>> weights = np.array([[[0.7, 0.6, 0.5], [0.1, 0.2, 0.3]]])
    >>> trace(qml.StronglyEntanglingLayers(weights, wires=[0, 1]))
    pennylane.operation.EigvalsUndefinedError
    During handling of the above exception, another exception occurred:
    pennylane.operation.MatrixUndefinedError

    The most powerful reason for using ``op_transform`` is the ability to define
    how the transform behaves if applied to a datastructure that supports multiple
    operations, such as a qfunc, tape, or QNode.

    We do this by defining a tape transform:

    .. code-block:: python

        @trace.tape_transform
        def trace(tape):
            tr = qml.math.trace(qml.matrix(tape))
            return qml.math.real(tr)

    We can now apply this transform directly to a qfunc:

    >>> def circuit(x, y):
    ...     qml.RX(x, wires=0)
    ...     qml.Hadamard(wires=1)
    ...     qml.CNOT(wires=[0, 1])
    ...     qml.CRY(y, wires=[1, 0])
    >>> trace(circuit)(0.1, 0.8)
    1.4124461636742214

    Our example above, applying our function to an operation that does not
    define the matrix or eigenvalues, will now work, since PennyLane will
    decompose the operation automatically into multiple operations:

    >>> trace(qml.StronglyEntanglingLayers)(weights, wires=[0, 1])
    0.4253851061350833

    .. note::

        If the operator transform takes additional (optional) transform parameters,
        then the registered tape transform should take the same transform parameters.

        E.g., consider a transform that takes the transform parameter ``lower``:

        .. code-block:: python

            @qml.op_transform
            def name(op, lower=True):
                return op.name().lower() if lower else op.name()

            @name.tape_transform
            def name(tape, lower=True):
                return [name(op, lower=lower) for op in tape.operations]

    If the transformation has purely quantum output, we can register the tape transformation
    as a qfunc transformation in addition:

    .. code-block:: python

        @qml.op_transform
        def simplify_rotation(op):
            if op.name == "Rot":
                params = op.parameters
                wires = op.wires

                if qml.math.allclose(params, 0):
                    return

                if qml.math.allclose(params[1:2], 0):
                    return qml.RZ(params[0], wires)

            return op

        @simplify_rotation.tape_transform
        @qml.qfunc_transform
        def simplify_rotation(tape):
            for op in tape:
                if op.name == "Rot":
                    simplify_rotation(op)
                else:
                    qml.apply(op)

    We can now use this combined operator and quantum function transform in compilation pipelines:

    .. code-block:: python

        @qml.qnode(dev)
        @qml.compile(pipeline=[simplify_rotation])
        def circuit(weights):
            ansatz(weights)
            qml.CNOT(wires=[0, 1])
            qml.Rot(0.0, 0.0, 0.0, wires=0)
            return qml.expval(qml.PauliX(1))
    """

    def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
        if os.environ.get("SPHINX_BUILD") == "1":
            # If called during a Sphinx documentation build,
            # simply return the original function rather than
            # instantiating the object. This allows the signature to
            # be correctly displayed in the documentation.

            warnings.warn(
                "Operator transformations have been disabled, as a Sphinx "
                "build has been detected via SPHINX_BUILD='1'. If this is not the "
                "case, please set the environment variable SPHINX_BUILD='0'.",
                UserWarning,
            )

            args[0].tape_transform = lambda x: x
            return args[0]

        return super().__new__(cls)

    def __init__(self, fn):
        if not callable(fn):
            raise OperationTransformError(
                f"The operator function to register, {fn}, "
                "does not appear to be a valid Python function or callable."
            )

        self._fn = fn
        self._sig = inspect.signature(fn).parameters
        self._tape_fn = None
        functools.update_wrapper(self, fn)

    def __call__(self, *targs, **tkwargs):
        obj = None

        if targs:
            # assume the first argument passed to the transform
            # is the object we wish to transform
            obj, *targs = targs

        if isinstance(obj, (qml.operation.Operator, qml.tape.QuantumScript)) or callable(obj):
            return self._create_wrapper(obj, *targs, **tkwargs)

        # Input is not an operator nor a QNode nor a quantum tape nor a qfunc.
        # Assume Python decorator syntax:
        #
        # op_func = op_transform(op_func)
        # result = op_func(*transform_args)(obj)(*obj_args)
        #
        # or
        #
        # @op_func(*transform_args)
        # @qml.qnode(dev)
        # def circuit(...):
        #     ...
        # result = circuit(*qnode_args)

        # Prepend the input to the transform args,
        # and create a wrapper function.
        if obj is not None:
            targs = (obj,) + tuple(targs)

        def wrapper(obj):
            return self._create_wrapper(obj, *targs, **tkwargs)

        return wrapper

    def fn(self, obj, *args, **kwargs):
        """Evaluate the underlying operator transform function.

        If a corresponding tape transform for the operator has been registered
        using the :attr:`.op_transform.tape_transform` decorator,
        then if an exception is raised while calling the transform function,
        this method will attempt to decompose the provided object for the tape
        transform.

        Args:
            obj (.Operator, pennylane.QNode, .QuantumTape, or Callable): An operator, quantum node, tape,
                or function that applies quantum operations.
            *args: positional arguments to pass to the function
            **kwargs: keyword arguments to pass to the function

        Returns:
            any: the result of evaluating the transform
        """
        try:
            return self._fn(obj, *args, **kwargs)

        except Exception as e1:  # pylint: disable=broad-except

            try:
                # attempt to decompose the operation and call
                # the tape transform function if defined
                return self.tape_fn(obj.expand(), *args, **kwargs)

            except (
                AttributeError,
                qml.operation.OperatorPropertyUndefined,
                OperationTransformError,
            ):
                # if obj.expand() does not exist, a required operation property was not found,
                # or the tape transform function does not exist, simply raise the original exception
                raise e1 from None

    def tape_fn(self, obj, *args, **kwargs):
        """The tape transform function.

        This is the function that is called if a datastructure is passed
        that contains multiple operations.

        Args:
            obj (pennylane.QNode, .QuantumTape, or Callable): A quantum node, tape,
                or function that applies quantum operations.
            *args: positional arguments to pass to the function
            **kwargs: keyword arguments to pass to the function

        Returns:
            any: the result of evaluating the transform

        Raises:
            .OperationTransformError: if no tape transform function is defined

        .. seealso:: :meth:`.op_transform.tape_transform`
        """
        if self._tape_fn is None:
            raise OperationTransformError(
                "This transform does not support tapes or QNodes with multiple operations."
            )

        return self._tape_fn(obj, *args, **kwargs)

    @property
    def is_qfunc_transform(self):
        """bool: Returns ``True`` if the operator transform is also a qfunc transform.
        That is, it maps one or more quantum operations to one or more quantum operations, allowing
        the output of the transform to be used as a quantum function.

        .. seealso:: :func:`~.qfunc_transform`
        """
        return isinstance(getattr(self._tape_fn, "tape_fn", None), qml.single_tape_transform)

    def tape_transform(self, fn):
        """Register a tape transformation to enable the operator transform
        to apply to datastructures containing multiple operations, such as QNodes, qfuncs,
        and tapes.

        .. note::

            The registered tape transform should have the same parameters as the
            original operation transform function.

        .. note::

            If the transformation maps a tape to a tape (or equivalently, a qfunc to a qfunc)
            then the transformation is simultaneously a :func:`~.qfunc_transform`, and
            can be declared as such. This enables additional functionality, for example
            the ability to use the transform in a compilation pipeline.

        Args:
            fn (callable): The function to register as the tape transform. This function
                should accept a :class:`~.QuantumTape` as the first argument.

        **Example**

        .. code-block:: python

            @qml.op_transform
            def name(op, lower=False):
                if lower:
                    return op.name.lower()
                return op.name

            @name.tape_transform
            def name(tape, lower=True):
                return [name(op, lower=lower) for op in tape.operations]

        We can now use this function on a qfunc, tape, or QNode:

        >>> def circuit(x, y):
        ...     qml.RX(x, wires=0)
        ...     qml.Hadamard(wires=1)
        ...     qml.CNOT(wires=[0, 1])
        ...     qml.CRY(y, wires=[1, 0])
        >>> name(circuit, lower=True)(0.1, 0.8)
        ['rx', 'hadamard', 'cnot', 'cry']

        If the transformation has purely quantum output, we can register the tape transformation
        as a qfunc transformation in addition:

        .. code-block:: python

            @qml.op_transform
            def simplify_rotation(op):
                if op.name == "Rot":
                    params = op.parameters
                    wires = op.wires

                    if qml.math.allclose(params, 0):
                        return

                    if qml.math.allclose(params[1:2], 0):
                        return qml.RZ(params[0], wires)

                return op

            @simplify_rotation.tape_transform
            @qml.qfunc_transform
            def simplify_rotation(tape):
                for op in tape:
                    if op.name == "Rot":
                        simplify_rotation(op)
                    else:
                        qml.apply(op)

        We can now use this combined operator and quantum function transform in compilation pipelines:

        .. code-block:: python

            @qml.qnode(dev)
            @qml.compile(pipeline=[simplify_rotation])
            def circuit(weights):
                ansatz(weights)
                qml.CNOT(wires=[0, 1])
                qml.Rot(0.0, 0.0, 0.0, wires=0)
                return qml.expval(qml.PauliX(1))
        """
        self._tape_fn = fn
        return self

    def _create_wrapper(self, obj, *targs, wire_order=None, **tkwargs):
        """Create a wrapper function that, when evaluated, transforms
        ``obj`` according to transform arguments ``*targs`` and ``**tkwargs``
        """

        if isinstance(obj, qml.operation.Operator):
            # Input is a single operation.
            # op_transform(obj, *transform_args)
            if wire_order is not None:
                tkwargs["wire_order"] = wire_order

            wrapper = self.fn(obj, *targs, **tkwargs)

        elif isinstance(obj, qml.tape.QuantumScript):
            # Input is a quantum tape. Get the quantum tape.
            tape, verified_wire_order = self._make_tape(obj, wire_order)

            if wire_order is not None:
                tkwargs["wire_order"] = verified_wire_order

            wrapper = self.tape_fn(tape, *targs, **tkwargs)

        elif callable(obj):
            # Input is a QNode, or qfunc (including single-operation qfuncs).
            # Get the quantum tape.
            def wrapper(*args, **kwargs):
                nonlocal wire_order
                tape, verified_wire_order = self._make_tape(obj, wire_order, *args, **kwargs)

                # HOTFIX: some operator transforms return a tape containing
                # a single transformed operator. As a result, for now we need
                # to treat a tape with a single operation as a single operation.
                # if len(getattr(tape, "operations", [])) == 1 and self._tape_fn is None:
                #    tape = tape.operations[0]

                if wire_order is not None or (
                    "wire_order" in self._sig and isinstance(obj, qml.QNode)
                ):
                    # Use the verified wire order if:
                    # - wire_order was passed to the transform
                    # - The object is a QNode, and the function takes a wire_order argument
                    tkwargs["wire_order"] = verified_wire_order

                if isinstance(tape, qml.operation.Operator):
                    return self.fn(tape, *targs, **tkwargs)

                if self.is_qfunc_transform:
                    # we must evaluate the qfunc transform at the original
                    # function arguments
                    return self.tape_fn(obj, *kwargs, **tkwargs)(*args, **kwargs)

                return self.tape_fn(tape, *targs, **tkwargs)

        else:
            raise OperationTransformError(
                "Input is not an Operator, tape, QNode, or quantum function"
            )

        return wrapper

    @staticmethod
    def _make_tape(obj, wire_order, *args, **kwargs):
        """Given an input object, which may be:

        - an object such as a tape or a operation, or
        - a callable such as a QNode or a quantum function
          (alongside the callable arguments ``args`` and ``kwargs``),

        this function constructs and returns the tape/operation
        represented by the object.

        The ``wire_order`` argument determines whether a custom wire ordering
        should be used. If not provided, the wire ordering defaults to the
        objects wire ordering accessed via ``obj.wires``.

        Returns:
            tuple[.QuantumTape, Wires]: returns the tape and the verified wire order
        """
        if isinstance(obj, qml.QNode):
            # user passed a QNode, get the tape
            obj.construct(args, kwargs)
            tape = obj.qtape
            wires = obj.device.wires

        elif isinstance(obj, qml.tape.QuantumScript):
            # user passed a tape
            tape = obj
            wires = tape.wires

        elif inspect.isclass(obj) and issubclass(obj, qml.operation.Operator):
            with qml.QueuingManager.stop_recording():
                tape = obj(*args, **kwargs)

            wires = tape.wires

        elif callable(obj):
            # user passed something that is callable but not a tape or QNode.
            tape = qml.transforms.make_tape(obj)(*args, **kwargs)
            wires = tape.wires

            # raise exception if it is not a quantum function
            if len(tape.operations) == 0 and len(tape.measurements) == 0:
                raise OperationTransformError("Quantum function contains no quantum operations")

        # if no wire ordering is specified, take wire list from tape/device
        wire_order = wires if wire_order is None else qml.wires.Wires(wire_order)

        # check that all wire labels in the circuit are contained in wire_order
        if not set(tape.wires).issubset(wire_order):
            raise OperationTransformError(
                f"Wires in circuit {tape.wires.tolist()} are inconsistent with "
                f"those in wire_order {wire_order.tolist()}"
            )

        return tape, wire_order
