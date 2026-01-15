# Copyright 2023 Xanadu Quantum Technologies Inc.

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
This module defines the data structure that encapsulates a quantum transform.
"""
from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Sequence
from copy import copy
from functools import lru_cache, partial, singledispatch, update_wrapper, wraps

from pennylane import capture, math
from pennylane.capture import autograph
from pennylane.exceptions import TransformError
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.pytrees import flatten
from pennylane.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch


@lru_cache
def _create_transform_primitive():
    try:
        # pylint: disable=import-outside-toplevel
        from pennylane.capture.custom_primitives import QmlPrimitive
    except ImportError:
        return None

    transform_prim = QmlPrimitive("transform")
    transform_prim.multiple_results = True
    transform_prim.prim_type = "transform"

    # pylint: disable=too-many-arguments, disable=unused-argument
    @transform_prim.def_impl
    def _impl(*all_args, inner_jaxpr, args_slice, consts_slice, **_):
        args = all_args[slice(*args_slice)]
        consts = all_args[slice(*consts_slice)]
        return capture.eval_jaxpr(inner_jaxpr, consts, *args)

    @transform_prim.def_abstract_eval
    def _abstract_eval(*_, inner_jaxpr, **__):
        return [out.aval for out in inner_jaxpr.outvars]

    return transform_prim


def _create_plxpr_fallback_transform(tape_transform):
    # pylint: disable=import-outside-toplevel
    try:
        import jax

        from pennylane.tape import plxpr_to_tape
    except ImportError:
        return None

    def plxpr_fallback_transform(jaxpr, consts, targs, tkwargs, *args):
        # Restore tkwargs from hashable tuple to dict
        tkwargs = dict(tkwargs)

        def wrapper(*inner_args):
            tape = plxpr_to_tape(jaxpr, consts, *inner_args)
            with capture.pause():
                tapes, _ = tape_transform(tape, *targs, **tkwargs)

            if len(tapes) > 1:
                raise TransformError(
                    f"Cannot apply {tape_transform.__name__} transform with program "
                    "capture enabled. Only transforms that return a single QuantumTape "
                    "and null processing function are usable with program capture."
                )

            for op in tapes[0].operations:
                data, struct = jax.tree_util.tree_flatten(op)
                jax.tree_util.tree_unflatten(struct, data)

            out = []
            for mp in tapes[0].measurements:
                data, struct = jax.tree_util.tree_flatten(mp)
                out.append(jax.tree_util.tree_unflatten(struct, data))

            return tuple(out)

        abstracted_axes, abstract_shapes = capture.determine_abstracted_axes(args)
        return jax.make_jaxpr(wrapper, abstracted_axes=abstracted_axes)(*abstract_shapes, *args)

    return plxpr_fallback_transform


def specific_apply_transform(transform, obj, *targs, **tkwargs):
    """The default behavior for Transform._apply_transform. By default, it dispatches to the
    generic registration."""
    return transform.generic_apply_transform(obj, *targs, **tkwargs)


@singledispatch
def generic_apply_transform(obj, transform, *targs, **tkwargs):
    """Apply an generic transform to a specific type of object. A singledispatch function
    used by ``TransformDipsatcher.generic_apply_transform``, but with a different order of arguments
    to allow is to be used by singledispatch.

    When called with an object that is not a valid dispatch target (e.g., not a QNode, tape, etc.),
    this returns a BoundTransform with the supplied args and kwargs. This enables patterns like:

        decompose(gate_set=gate_set) + merge_rotations(1e-6)

    where transforms are called with just configuration parameters and combined into a CompilePipeline
    """
    # If the first argument is not a valid dispatch target, return a BoundTransform
    # with the first argument and any additional args/kwargs stored as transform parameters.
    targs, tkwargs = transform.setup_inputs(obj, *targs, **tkwargs)
    return BoundTransform(transform, args=targs, kwargs=tkwargs)


# pragma: no cover
def _dummy_register(obj):  # just used for sphinx
    if isinstance(obj, type):  # pragma: no cover
        return lambda arg: arg  # pragma: no cover
    return obj  # pragma: no cover


def _default_setup_inputs(*targs, **tkwargs):
    return targs, tkwargs


class Transform:  # pylint: disable=too-many-instance-attributes
    r"""Generalizes a function that transforms tapes to work with additional circuit-like
    objects such as a :class:`~.QNode`.

    ``transform`` should be applied to a function that transforms tapes. Once validated,
    the result will be an object that is able to transform PennyLane's range of circuit-like
    objects: :class:`~.QuantumTape`, quantum function and :class:`~.QNode`. A circuit-like
    object can be transformed either via decoration or by passing it functionally through
    the created transform.

    Args:
        tape_transform (Callable | None): The input quantum transform must be a function
            that satisfies the following requirements:

            * Accepts a :class:`~.QuantumScript` as its first input and returns a sequence
              of :class:`~.QuantumScript` and a processing function.

            * The transform must have the following structure (type hinting is optional):
              ``my_tape_transform(tape: qml.tape.QuantumScript, ...) -> tuple[qml.tape.QuantumScriptBatch, qml.typing.PostprocessingFn]``

        pass_name (str | None): the name of the associated MLIR pass to be applied when
            Catalyst is used. See Usage Details for more information.

    Keyword Args:
        expand_transform=None (Optional[Callable]): An optional transform that is applied directly
            before the input transform. It must be a function that satisfies the same requirements as
            ``tape_transform``.
        classical_cotransform=None (Optional[Callable]): A classical co-transform is a function to
            post-process the classical jacobian and the quantum jacobian and has the signature:
            ``my_cotransform(qjac, cjac, tape) -> tensor_like``
        is_informative=False (bool): Whether or not a transform is informative. If true, the transform
            is queued at the end of the compile pipeline and the tapes or qnode aren't executed.
        final_transform=False (bool): Whether or not the transform is terminal. If true, the transform
            is queued at the end of the compile pipeline. ``is_informative`` supersedes ``final_transform``.
        use_argnum_in_expand=False (bool): Whether to use ``argnum`` of the tape to determine trainable
            parameters during the expansion transform process.
        plxpr_transform=None (Optional[Callable]): Function for transforming plxpr. **Experimental**

    **Example**

    First define an input tape transform with the necessary structure defined above. In this example,
    we copy the tape and sum the results of the execution of the two tapes.

    .. code-block:: python

        from pennylane.tape import QuantumScript, QuantumScriptBatch
        from pennylane.typing import PostprocessingFn

        def my_quantum_transform(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
            tape1 = tape
            tape2 = tape.copy()

            def post_processing_fn(results):
                return qml.math.sum(results)

            return [tape1, tape2], post_processing_fn

    We want to be able to apply this transform on both a ``qfunc`` and a :class:`pennylane.QNode` and will
    use ``transform`` to achieve this. ``transform`` validates the signature of your input quantum transform
    and makes it capable of transforming ``qfunc`` and :class:`pennylane.QNode` in addition to quantum tapes.
    Let's define a circuit as a :class:`pennylane.QNode`:

    .. code-block:: python

        dev = qml.device("default.qubit")

        @qml.qnode(device=dev)
        def qnode_circuit(a):
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.X(0)
            qml.RZ(a, wires=1)
            return qml.expval(qml.Z(0))

    We first apply ``transform`` to ``my_quantum_transform``:

    >>> dispatched_transform = qml.transform(my_quantum_transform)

    Now you can use the dispatched transform directly on a :class:`pennylane.QNode`.

    For :class:`pennylane.QNode`, the dispatched transform populates the ``CompilePipeline`` of your QNode. The
    transform and its processing function are applied in the execution.

    >>> transformed_qnode = dispatched_transform(qnode_circuit)
    >>> transformed_qnode
    <QNode: device='<default.qubit device at ...>', interface='auto', diff_method='best', shots='Shots(total=None)'>

    >>> transformed_qnode.transform_program
    CompilePipeline(my_quantum_transform)

    If we apply ``dispatched_transform`` a second time to the :class:`pennylane.QNode`, we would add
    it to the compile pipeline again and therefore the transform would be applied twice before execution.

    >>> transformed_qnode = dispatched_transform(transformed_qnode)
    >>> transformed_qnode.transform_program
    CompilePipeline(my_quantum_transform, my_quantum_transform)

    When a transformed QNode is executed, the QNode's compile pipeline is applied to the generated tape
    and creates a sequence of tapes to be executed. The execution results are then post-processed in the
    reverse order of the compile pipeline to obtain the final results.

    .. details::
        :title: Setup inputs

        The ``setup_inputs`` function will independently applied prior to any application of
        the transform. This allows for validation of the inputs, separation into positional and
        keyword arguments, and specification of a call signature and docstring for transforms
        without a tape definition.

        .. code-block:: python

            def my_transform_setup(a, b=1, metadata : str = "my_value"):
                "Docstring for my_transform."
                return (a, b), {"metadata": metadata}

            my_transform = qml.transform(pass_name="my_pass", setup_inputs=my_transform_setup)

            @qml.qnode(qml.device('default.qubit', wires=4))
            def circuit():
                return qml.expval(qml.Z(0))

        This allows us to perform eager input validation and set default values.

        >>> my_transform(circuit)
        Traceback (most recent call last):
            ...
        TypeError: my_transform_setup() missing 1 required positional argument: 'a'
        >>> new_circuit = my_transform(circuit, a=2)
        >>> new_circuit.transform_program[0]
        <my_pass(2, 1, metadata=my_value)>

        We will also have a docstring and signature. If a tape transform is present, the signature will
        be determined by that.

        >>> my_transform.__doc__
        'Docstring for my_transform.'
        >>> import inspect
        >>> inspect.signature(my_transform)
        <Signature (a, b=1, metadata: str = 'my_value')>

    .. details::
        :title: Dispatch a transform onto a batch of tapes

        We can compose multiple transforms when working in the tape paradigm and apply them to more than
        one tape. The following example demonstrates how to apply a transform to a batch of tapes.

        **Example**

        In this example, we apply sequentially a transform to a tape and another one to a batch of tapes.
        We then execute the transformed tapes on a device and post-process the results.

        .. code-block:: python

            import pennylane as qml

            H = qml.PauliY(2) @ qml.PauliZ(1) + 0.5 * qml.PauliZ(2) + qml.PauliZ(1)
            measurement = [qml.expval(H)]
            operations = [qml.Hadamard(0), qml.RX(0.2, 0), qml.RX(0.6, 0), qml.CNOT((0, 1))]
            tape = qml.tape.QuantumTape(operations, measurement)

            batch1, function1 = qml.transforms.split_non_commuting(tape)
            batch2, function2 = qml.transforms.merge_rotations(batch1)

            dev = qml.device("default.qubit", wires=3)
            result = dev.execute(batch2)

        The first ``split_non_commuting`` transform splits the original tape, returning a batch of
        tapes ``batch1`` and a processing function ``function1``. The second ``merge_rotations``
        transform is applied to the batch of tapes returned by the first transform. It returns a
        new batch of tapes ``batch2``, each of which has been transformed by the second transform,
        and a processing function ``function2``.

        >>> batch2
        (<QuantumTape: wires=[0, 1, 2], params=1>, <QuantumTape: wires=[0, 1, 2], params=1>)

        >>> type(function2)
        <class 'function'>

        We can combine the processing functions to post-process the results of the execution.

        >>> function1(function2(result))
        np.float64(0.499...)

    .. details::
        :title: Signature of a transform

        A dispatched transform is able to handle several PennyLane circuit-like objects:

        - :class:`pennylane.QNode`
        - a quantum function (callable)
        - :class:`pennylane.tape.QuantumScript`
        - a batch of :class:`pennylane.tape.QuantumScript`
        - :class:`pennylane.devices.Device`.

        For each object, the transform will be applied in a different way, but it always preserves the
        underlying tape-based quantum transform behaviour.

        The return of a dispatched transform depends upon which of the above objects is passed as an input:

        - For a :class:`~.QNode` input, the underlying transform is added to the QNode's
          :class:`~.CompilePipeline` and the return is the transformed :class:`~.QNode`.
          For each execution of the :class:`pennylane.QNode`, it first applies the compile pipeline on
          the original captured circuit. Then the transformed circuits are executed by a device and
          finally the post-processing function is applied on the results.

          When experimental program capture is enabled, transforming a :class:`~.QNode` returns
          a new function to which the transform has been added as a higher-order primitive.

        - For a quantum function (callable) input, the transform builds the tape when the quantum function
          is executed and then applies itself to the tape. The resulting tape is then converted back
          to a quantum function (callable). It therefore returns a transformed quantum function (Callable).
          The limitation is that the underlying transform can only return a sequence containing a single
          tape, because quantum functions only support a single circuit.

          When experimental program capture is enabled, transforming a function (callable) returns a new
          function to which the transform has been added as a higher-order primitive.

        - For a :class:`~.QuantumScript, the underlying quantum transform is directly applied on the
          :class:`~.QuantumScript`. It returns a sequence of :class:`~.QuantumScript` and a processing
          function to be applied after execution.

        - For a batch of :class:`pennylane.tape.QuantumScript`, the quantum transform is mapped across
          all the tapes. It returns a sequence of :class:`~.QuantumScript` and a processing function to
          be applied after execution. Each tape in the sequence is transformed by the transform.

        - For a :class:`~.devices.Device`, the transform is added to the device's compile pipeline
          and a transformed :class:`pennylane.devices.Device` is returned. The transform is added
          to the end of the device program and will be last in the overall compile pipeline.

    .. details::
        :title: Transforms with Catalyst

        If a compilation pass is written in MLIR, using it in a ``qjit``'d workflow requires that
        it have a transform with a matching ``pass_name``. This ensures that the transform is
        properly applied as part of the lower-level compilation.

        For example, we can create a transform that will apply the ``cancel-inverses`` pass, like the
        in-built ``qml.transforms.cancel_inverses`` transform.

        .. code-block:: python

            my_transform = qml.transform(pass_name="cancel-inverses")

            @qml.qjit
            @my_transform
            @qml.qnode(qml.device('lightning.qubit', wires=4))
            def circuit():
                qml.X(0)
                qml.X(0)
                return qml.expval(qml.Z(0))

        We can see that the instruction to apply ``"cancel-inverses"`` is present in the initial MLIR.

        >>> circuit()
        Array(1., dtype=float64)
        >>> print(circuit.mlir[200:600])
        tensor<f64>
        }
        module @module_circuit {
            module attributes {transform.with_named_sequence} {
            transform.named_sequence @__transform_main(%arg0: !transform.op<"builtin.module">) {
                %0 = transform.apply_registered_pass "cancel-inverses" to %arg0 : (!transform.op<"builtin.module">) -> !transform.op<"builtin.module">
                transform.yield
            }
            }
            func.func public @circui

        Transforms can have both tape-based and ``pass_name``-based definitions. For example, the
        transform below called ``my_transform`` has both definitions. In this case, the MLIR pass
        will take precedence when being ``qjit``'d if only MLIR passes can occur after.

        .. code-block:: python

            from functools import partial

            @partial(qml.transform, pass_name="my-pass-name")
            def my_transform(tape):
                return (tape, ), lambda res: res[0]

        Note that any transform with only a ``pass_name`` definition *must* occur after any purely tape-based
        transform, as tape transforms occur prior to lowering to MLIR.

        >>> @qml.qjit
        ... @qml.defer_measurements
        ... @qml.transform(pass_name="cancel-inverses")
        ... @qml.qnode(qml.device('lightning.qubit', wires=4))
        ... def c():
        ...     qml.X(0)
        ...     qml.X(0)
        ...     return qml.expval(qml.Z(0))
        ...
        Traceback (most recent call last):
            ...
        ValueError: <cancel-inverses()> without a tape definition occurs before tape transform <defer_measurements()>.

    .. details::
        :title: Transforms with experimental program capture

        To define a transform that can be applied directly to plxpr without the need to create
        ``QuantumScript``\ s, users must provide the ``plxpr_transform`` argument. If this argument
        is not provided, executing transformed functions is not guaranteed to work. More details
        about this are provided below. The ``plxpr_transform`` argument should be a function that
        applies the respective transform to ``jax.extend.core.Jaxpr`` and returns a transformed
        ``jax.extend.core.ClosedJaxpr``. ``plxpr_transform`` can assume that no transform primitives
        are present in the input plxpr, and its implementation does not need to account for these
        primitives. The exact expected signature of ``plxpr_transform`` is shown in the example below:

        .. code-block:: python

            def dummy_plxpr_transform(
                jaxpr: jax.extend.core.Jaxpr, consts: list, targs: list, tkwargs: dict, *args
            ) -> jax.extend.core.ClosedJaxpr:
                ...

        Once the ``plxpr_transform`` argument is provided, the transform can be easily used with program
        capture enabled! To do so, apply the transform as you normally would:

        .. code-block:: python

            qml.capture.enable()

            @qml.transforms.cancel_inverses
            def circuit():
                qml.X(0)
                qml.S(1)
                qml.X(0)
                qml.adjoint(qml.S(1))
                return qml.expval(qml.Z(1))

        >>> jax.make_jaxpr(circuit)()
        { lambda ; . let
            a:AbstractMeasurement(n_wires=None) = transform[
            args_slice=(0, 0, None)
            consts_slice=(0, 0, None)
            inner_jaxpr={ lambda ; . let
                _:AbstractOperator() = PauliX[n_wires=1] 0:i...[]
                _:AbstractOperator() = S[n_wires=1] 1:i...[]
                _:AbstractOperator() = PauliX[n_wires=1] 0:i...[]
                b:AbstractOperator() = S[n_wires=1] 1:i...[]
                _:AbstractOperator() = Adjoint b
                c:AbstractOperator() = PauliZ[n_wires=1] 1:i...[]
                d:AbstractMeasurement(n_wires=None) = expval_obs c
              in (d,) }
            targs_slice=(0, None, None)
            tkwargs=()
            transform=<transform: cancel_inverses>
          ]
        in (a,) }


        As shown, the transform gets applied as a higher-order primitive, with the jaxpr
        representation of the function being transformed stored in the ``inner_jaxpr``
        parameter of the transform's primitive.

        **Fallback implementation of plxpr transforms:**

        If a transform that does not define a ``plxpr_transform`` is applied to a function,
        a fallback implementation of the transform is used. This fallback implementation converts
        the function into a :func:`~pennylane.tape.QuantumScript`, which is then transformed
        as a traditional tape. However, because of the constraints of program capture, many transforms
        will not be compatible with this fallback implementation:

        * Transforms that return multiple tapes are not compatible.
        * Transforms that require non-trivial post-processing of results are not compatible.
        * Dynamically shaped arrays are not compatible.
        * Functions that are being transformed that contain control flow dependent on dynamic
          parameters are not compatible. This includes:

          * :func:`pennylane.cond` with dynamic parameters as predicates.
          * :func:`pennylane.for_loop` with dynamic parameters for ``start``, ``stop``, or ``step``.
          * :func:`pennylane.while_loop` does not work.

        .. warning::

            Currently, executing a function to which a transform has been applied will raise a
            ``NotImplementedError``. See below for details on how to use functions that are
            transformed.

        To perform the transform, the :func:`pennylane.capture.expand_plxpr_transforms` function
        should be used. This function accepts a function to which transforms have been applied
        as an input, and returns a new function that has been transformed:

        >>> transformed_circuit = qml.capture.expand_plxpr_transforms(circuit)
        >>> jax.make_jaxpr(transformed_circuit)()
        { lambda ; . let
            a:AbstractOperator() = PauliZ[n_wires=1] 1:i...[]
            b:AbstractMeasurement(n_wires=None) = expval_obs a
        in (b,) }
    """

    def __new__(  # pylint: disable=too-many-arguments
        cls,
        tape_transform: Callable | None = None,
        pass_name: None | str = None,
        *,
        setup_inputs: Callable | None = None,
        expand_transform: Callable | None = None,
        classical_cotransform: Callable | None = None,
        is_informative: bool = False,
        final_transform: bool = False,
        use_argnum_in_expand: bool = False,
        plxpr_transform=None,
    ) -> Transform:
        if os.environ.get("SPHINX_BUILD") == "1":
            # If called during a Sphinx documentation build,
            # simply return the original function rather than
            # instantiating the object. This allows the signature to
            # be correctly displayed in the documentation.

            warnings.warn(
                "Transforms have been disabled, as a Sphinx "
                "build has been detected via SPHINX_BUILD='1'. If this is not the "
                "case, please set the environment variable SPHINX_BUILD='0'.",
                UserWarning,
            )

            if tape_transform:
                tape_transform.custom_qnode_transform = lambda x: x
                tape_transform.register = _dummy_register
                return tape_transform
            if setup_inputs:
                setup_inputs.custom_qnode_transform = lambda x: x
                setup_inputs.register = _dummy_register
                return setup_inputs
            raise ValueError("needs at least a tape_transform or setup_inputs for use with sphinx.")

        return super().__new__(cls)

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        tape_transform: Callable | None = None,
        pass_name: None | str = None,
        *,
        setup_inputs: None | Callable = None,
        expand_transform: Callable | None = None,
        classical_cotransform: Callable | None = None,
        is_informative: bool = False,
        final_transform: bool = False,
        use_argnum_in_expand: bool = False,
        plxpr_transform=None,
    ):
        if tape_transform is not None and not callable(tape_transform):
            raise TransformError(
                f"The function to register, {tape_transform}, does "
                "not appear to be a valid Python function or callable."
            )

        if expand_transform is not None and not callable(expand_transform):
            raise TransformError("The expand function must be a valid Python function.")

        if classical_cotransform is not None and not callable(classical_cotransform):
            raise TransformError("The classical co-transform must be a valid Python function.")

        if tape_transform is None and pass_name is None:
            raise ValueError("Transforms must define either a tape transform or a pass_name")

        self._tape_transform = tape_transform
        self._expand_transform = expand_transform
        self._classical_cotransform = classical_cotransform
        self._is_informative = is_informative
        # is_informative supersedes is_final_transform
        self._is_final_transform = is_informative or final_transform
        self._custom_qnode_transform = None
        self._pass_name = pass_name
        self._use_argnum_in_expand = use_argnum_in_expand

        self._setup_inputs = setup_inputs or _default_setup_inputs
        if tape_transform:
            update_wrapper(self, tape_transform)
        elif setup_inputs:
            update_wrapper(
                self,
                setup_inputs,
                assigned=("__module__", "__annotations__", "__type_params__", "__doc__"),
            )

        self._apply_transform = singledispatch(partial(specific_apply_transform, self))
        self._plxpr_transform = plxpr_transform or _create_plxpr_fallback_transform(tape_transform)

    @property
    def pass_name(self) -> None | str:
        """The name of the equivalent MLIR pass."""
        return self._pass_name

    @property
    def register(self):
        """Returns a decorator for registering a specific application behavior for a given transform
        and a new class.

        .. code-block:: python

            @qml.transform
            def printer(tape):
                print("I have a tape: ", tape)
                return (tape, ), lambda x: x[0]

            @printer.register
            def _(obj: qml.operation.Operator, *targs, **tkwargs):
                print("I have an operator:", obj)
                return obj

        >>> printer(qml.X(0))
        I have an operator: X(0)
        X(0)

        """
        return self._apply_transform.register

    def generic_apply_transform(self, obj, *targs, **tkwargs):
        """generic_apply_transform(obj, *targs, **tkwargs)
        Generic application of a transform that forms the default for all transforms.

        Args:
            obj: The object we want to transform
            *targs: The arguments for the transform
            **tkwargs: The keyword arguments for the transform.

        """
        return generic_apply_transform(obj, self, *targs, **tkwargs)

    @staticmethod
    def generic_register(arg):
        """Returns a decorator for registering a default application behavior for a transform for a new class.

        Given a special new class, we can register how transforms should apply to them via:

        .. code-block:: python

            class Subroutine:

                def __repr__(self):
                    return f"<Subroutine: {self.ops}>"

                def __init__(self, ops):
                    self.ops = ops

            from pennylane.transforms.core import Transform

            @Transform.generic_register
            def apply_to_subroutine(obj: Subroutine, transform, *targs, **tkwargs):
                targs, tkwargs = transform.setup_inputs(*targs, **tkwargs)
                tape = qml.tape.QuantumScript(obj.ops)
                batch, _ = transform(tape, *targs, **tkwargs)
                return Subroutine(batch[0].operations)

        >>> qml.transforms.cancel_inverses(Subroutine([qml.Y(0), qml.X(0), qml.X(0)]))
        <Subroutine: [Y(0)]>

        The type can also be explicitly provided like:

        .. code-block:: python

            @Transform.generic_register(Subroutine)
            def apply_to_subroutine(obj: Subroutine, transform, *targs, **tkwargs):
                targs, tkwargs = transform.setup_inputs(*targs, **tkwargs)
                tape = qml.tape.QuantumScript(obj.ops)
                batch, _ = transform(tape, *targs, **tkwargs)
                return Subroutine(batch[0].operations)

        to more explicitly force registration for a given type.

        """
        return generic_apply_transform.register(arg)  # pylint: disable=no-member

    def __call__(self, *args, **kwargs):
        if not args and not kwargs:
            raise TypeError(
                f"{self!r} requires at least one argument. "
                "Provide a tape, qfunc, QNode, or device to transform, "
                "or provide keyword arguments to create a BoundTransform for composition."
            )
        if not args and kwargs:
            args, kwargs = self._setup_inputs(*args, **kwargs)
            return BoundTransform(self, args=args, kwargs=kwargs)
        return self._apply_transform(*args, **kwargs)

    def __repr__(self):
        name = self._tape_transform.__name__ if self._tape_transform else self.pass_name
        return f"<transform: {name}>"

    def __add__(self, other):
        """Add two transforms to create a CompilePipeline."""

        if not isinstance(other, (Transform, BoundTransform)):
            return NotImplemented

        # Technically this is checked in the CompilePipeline dunders but we still
        # do it here to raise a more informative error message.
        if self.is_final_transform and other.is_final_transform:
            raise TransformError(
                f"Both {self} and {other} are final transforms and cannot be combined."
            )

        if self.expand_transform:
            # pylint: disable=import-outside-toplevel
            from .compile_pipeline import CompilePipeline

            return CompilePipeline(self, other)

        # Convert this transform to a BoundTransform (no args/kwargs) and delegate
        return BoundTransform(self) + other

    def __mul__(self, n):
        """Multiply by an integer to create a compile pipeline with this transform repeated."""
        if self.expand_transform:
            # pylint: disable=import-outside-toplevel
            from .compile_pipeline import CompilePipeline

            return CompilePipeline(self) * n

        # Convert to container (no args/kwargs) and delegate
        return BoundTransform(self) * n

    __rmul__ = __mul__

    def setup_inputs(self, *targs, **tkwargs):
        """Call the setup_inputs function."""
        return self._setup_inputs(*targs, **tkwargs)

    @property
    def tape_transform(self):
        """The tape transform."""
        return self._tape_transform

    @property
    def expand_transform(self):
        """The expand transform."""
        return self._expand_transform

    @property
    def classical_cotransform(self):
        """The classical co-transform."""
        return self._classical_cotransform

    @property
    def plxpr_transform(self):
        """Function for transforming plxpr."""
        return self._plxpr_transform

    @property
    def is_informative(self):
        """``True`` if the transform is informative."""
        return self._is_informative

    @property
    def is_final_transform(self):
        """``True`` if the transformed tapes must be executed."""
        return self._is_final_transform

    def custom_qnode_transform(self, fn):
        """Register a custom QNode execution wrapper function for the batch transform.

        **Example**

        .. code-block:: python3

            @transform
            def my_transform(tape, *targs, **tkwargs):
                ...
                return tapes, processing_fn

            @my_transform.custom_qnode_transform
            def my_custom_qnode_wrapper(self, qnode, targs, tkwargs):
                new_tkwargs = dict(tkwargs)
                new_tkwargs['shots'] = 100
                return self.generic_apply_transform(qnode, *targs, **new_tkwargs)

        The custom QNode execution wrapper must have arguments
        ``self`` (the batch transform object), ``qnode`` (the input QNode
        to transform and execute), ``targs`` and ``tkwargs`` (the transform
        arguments and keyword arguments respectively).

        It should return a QNode that accepts the *same* arguments as the
        input QNode with the transform applied.

        The default :meth:`~.generic_apply_transform` method may be called
        if only pre- or post-processing dependent on QNode arguments is required.
        """
        # unfortunately, we don't have access to qml.QNode here, or in the places where
        # transforms are defining custom qnode transforms, so we still need to have this
        # "hold onto until later" approach
        # potentially can remove this patch by moving source code
        self._custom_qnode_transform = fn

    def default_qnode_transform(self, qnode, targs, tkwargs):
        """
        The default method that takes in a QNode and returns another QNode
        with the transform applied.
        """
        # same comment as custom_qnode_transform :(
        qnode = copy(qnode)
        qnode.transform_program.append(BoundTransform(self, args=targs, kwargs=tkwargs))
        return qnode


class BoundTransform:  # pylint: disable=too-many-instance-attributes
    """A transform with bound inputs.

    Args:
        transform: Any transform.
        args (Sequence[Any]): The positional arguments to use with the transform.
        kwargs (Dict | None): The keyword arguments for use with the transform.

    Keyword Args:
        use_argnum (bool): An advanced option used in conjunction with calculating
            classical cotransforms of jax workflows.

    .. seealso:: :func:`~.pennylane.transform`

    >>> bound_t = BoundTransform(qml.transforms.merge_rotations, (), {"atol": 1e-4})
    >>> bound_t
    <merge_rotations(atol=0.0001)>

    The class can also be created by directly calling the transform with its inputs:

    >>> qml.transforms.merge_rotations(atol=1e-4)
    <merge_rotations(atol=0.0001)>

    These objects can now directly applied to anything individual transforms can apply to:

    .. code-block:: python

        @bound_t
        @qml.qnode(qml.device('null.qubit', wires=2))
        def c(x):
            qml.RX(x, 0)
            qml.RX(-x + 1e-6, 0)
            qml.RY(x, 1)
            qml.RY(-x + 1e-2, 1)
            return qml.probs(wires=(0,1))

    If we draw this circuit, we can see that the ``merge_rotations`` transforms was applied with a
    tolerance of ``1e-4``.  The ``RX`` gates sufficiently close to zero disappear, while the ``RY`` gates
    that are further from zero remain.

    >>> print(qml.draw(c)(1.0))
    0: ───────────┤ ╭Probs
    1: ──RY(0.01)─┤ ╰Probs

    Repeated versions of the bound transform can be created with multiplication:

    >>> bound_t * 3
    CompilePipeline(merge_rotations, merge_rotations, merge_rotations)

    And it can be used in conjunction with both individual transforms, bound transforms, and
    compile pipelines.

    >>> bound_t + qml.transforms.cancel_inverses
    CompilePipeline(merge_rotations, cancel_inverses)
    >>> bound_t + qml.transforms.cancel_inverses + bound_t
    CompilePipeline(merge_rotations, cancel_inverses, merge_rotations)

    """

    def __hash__(self):
        hashable_dict = tuple((key, value) for key, value in self.kwargs.items())
        return hash((self.tape_transform, self.pass_name, self.args, hashable_dict))

    def __init__(
        self,
        transform: Transform,
        args: tuple | list = (),
        kwargs: None | dict = None,
        *,
        use_argnum: bool = False,
        **transform_config,
    ):
        if not isinstance(transform, Transform):
            transform = Transform(transform, **transform_config)
        elif transform_config:
            raise ValueError(
                f"transform_config kwargs {transform_config} cannot be passed if a transform is provided."
            )
        self._transform = transform
        self._args = tuple(args)
        self._kwargs = kwargs or {}
        self._use_argnum = use_argnum

    def __repr__(self):
        name = self.tape_transform.__name__ if self.tape_transform else self.pass_name
        arg_str = ", ".join(repr(a) for a in self._args) if self._args else ""
        kwarg_str = (
            ", ".join(f"{key}={value}" for key, value in self._kwargs.items())
            if self._kwargs
            else ""
        )
        if arg_str and kwarg_str:
            total_str = ", ".join([arg_str, kwarg_str])
        elif arg_str:
            total_str = arg_str
        else:
            total_str = kwarg_str
        return f"<{name}({total_str})>"

    def __call__(self, obj):
        return self._transform(obj, *self.args, **self.kwargs)

    def __iter__(self):
        return iter(
            (
                self._transform.tape_transform,
                self._args,
                self._kwargs,
                self._transform.classical_cotransform,
                self._transform.plxpr_transform,
                self._transform.is_informative,
                self._transform.is_final_transform,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoundTransform):
            return False
        return (
            self.args == other.args
            and self.tape_transform == other.tape_transform
            and self.pass_name == other.pass_name
            and self.kwargs == other.kwargs
            and self.classical_cotransform == other.classical_cotransform
            and self.is_informative == other.is_informative
            and self.is_final_transform == other.is_final_transform
        )

    @property
    def tape_transform(self) -> Callable | None:
        """The raw tape transform definition for the transform."""
        return self._transform.tape_transform

    @property
    def transform(self) -> Callable | None:
        """The raw tape transform definition of the transform."""
        # TODO: deprecate this in the next version
        return self.tape_transform

    @property
    def expand_transform(self) -> BoundTransform | None:
        """The expand_transform associated with this transform."""
        if not self._transform.expand_transform:
            return None
        return BoundTransform(
            self._transform.expand_transform,
            args=self.args,
            kwargs=self.kwargs,
            use_argnum=self._transform._use_argnum_in_expand,  # pylint:disable=protected-access
        )

    @property
    def pass_name(self) -> None | str:
        """The name of the corresponding Catalyst pass, if it exists."""
        return self._transform.pass_name

    @property
    def args(self) -> tuple:
        """The stored quantum transform's ``args``."""
        return self._args

    @property
    def kwargs(self) -> dict:
        """The stored quantum transform's ``kwargs``."""
        return self._kwargs

    @property
    def classical_cotransform(self) -> None | Callable:
        """The stored quantum transform's classical co-transform."""
        return self._transform.classical_cotransform

    @property
    def plxpr_transform(self) -> None | Callable:
        """The stored quantum transform's PLxPR transform.

        **UNMAINTAINED AND EXPERIMENTAL**
        """
        return self._transform.plxpr_transform

    @property
    def is_informative(self) -> bool:
        """Whether or not a transform is informative. If true the transform is queued at the end
        of the transform program and the tapes or qnode aren't executed.

        This property is rare, but used by such transforms as ``qml.transforms.commutation_dag``.
        """
        return self._transform.is_informative

    @property
    def is_final_transform(self) -> bool:
        """Whether or not the transform must be the last one to be executed
        in a ``CompilePipeline``.

        This property is ``True`` for most gradient transforms.
        """
        return self._transform.is_final_transform

    def __add__(self, other):
        """Add two transforms to create a CompilePipeline."""

        if not isinstance(other, (Transform, BoundTransform)):
            return NotImplemented

        # Import here to avoid circular import
        # pylint: disable=import-outside-toplevel
        from .compile_pipeline import CompilePipeline

        if self.is_final_transform and other.is_final_transform:
            raise TransformError(
                f"Both {self} and {other} are final transforms and cannot be combined."
            )

        return CompilePipeline(self, other)

    def __mul__(self, n):
        """Multiply by an integer to create a pipeline with this transform repeated."""

        # Import here to avoid circular import
        from .compile_pipeline import CompilePipeline  # pylint: disable=import-outside-toplevel

        if not isinstance(n, int):
            return NotImplemented

        if n < 0:
            raise ValueError("Cannot multiply transform container by negative integer")

        if self.is_final_transform and n > 1:
            raise TransformError(
                f"{self} is a final transform and cannot be applied more than once."
            )

        return CompilePipeline(self) * n

    __rmul__ = __mul__


@Transform.generic_register
def _apply_to_tape(obj: QuantumScript, transform, *targs, **tkwargs):
    if transform.tape_transform is None:
        raise NotImplementedError(
            f"Transform {transform} has no defined tape implementation, "
            "and can only be applied when decorating the entire workflow "
            "with '@qml.qjit' and when it is placed after all transforms "
            "that only have a tape implementation."
        )
    targs, tkwargs = transform.setup_inputs(*targs, **tkwargs)
    if transform.expand_transform:
        expanded_tapes, expand_processing = transform.expand_transform(obj, *targs, **tkwargs)
        transformed_tapes = []
        processing_and_slices = []
        start = 0
        for tape in expanded_tapes:
            intermediate_tapes, post_processing_fn = transform.tape_transform(
                tape, *targs, **tkwargs
            )
            transformed_tapes.extend(intermediate_tapes)
            end = start + len(intermediate_tapes)
            processing_and_slices.append(tuple([post_processing_fn, slice(start, end)]))
            start = end

        def processing_fn(results):
            processed_results = [fn(results[slice]) for fn, slice in processing_and_slices]
            return expand_processing(processed_results)

    else:
        transformed_tapes, processing_fn = transform.tape_transform(obj, *targs, **tkwargs)

    if transform.is_informative:
        return processing_fn(transformed_tapes)

    return transformed_tapes, processing_fn


def _capture_apply(obj, transform, *targs, **tkwargs):
    @autograph.wraps(obj)
    def qfunc_transformed(*args, **kwargs):
        import jax  # pylint: disable=import-outside-toplevel

        flat_qfunc = capture.flatfn.FlatFn(obj)
        jaxpr = jax.make_jaxpr(flat_qfunc)(*args, **kwargs)
        flat_args = jax.tree_util.tree_leaves(args)

        n_args = len(flat_args)
        n_consts = len(jaxpr.consts)
        args_slice = slice(0, n_args)
        consts_slice = slice(n_args, n_args + n_consts)
        targs_slice = slice(n_args + n_consts, None)

        results = _create_transform_primitive().bind(  # pylint: disable=protected-access
            *flat_args,
            *jaxpr.consts,
            *targs,
            inner_jaxpr=jaxpr.jaxpr,
            args_slice=args_slice,
            consts_slice=consts_slice,
            targs_slice=targs_slice,
            tkwargs=tkwargs,
            transform=transform,
        )

        assert flat_qfunc.out_tree is not None
        return jax.tree_util.tree_unflatten(flat_qfunc.out_tree, results)

    return qfunc_transformed


@Transform.generic_register
def apply_to_callable(obj: Callable, transform, *targs, **tkwargs):
    """Apply a transform to a Callable object."""
    if obj.__class__.__name__ == "QJIT":
        raise TransformError(
            "Functions that are wrapped / decorated with qjit cannot subsequently be"
            f" transformed with a PennyLane transform (attempted {transform})."
            f" For the desired affect, ensure that qjit is applied after {transform}."
        )
    targs, tkwargs = transform.setup_inputs(*targs, **tkwargs)

    @wraps(obj)
    def qfunc_transformed(*args, **kwargs):
        if capture.enabled():
            return _capture_apply(obj, transform, *targs, **tkwargs)(*args, **kwargs)

        # removes the argument to the qfuncs from the active queuing context.
        leaves, _ = flatten((args, kwargs), lambda obj: isinstance(obj, Operator))
        for l in leaves:
            if isinstance(l, Operator):
                QueuingManager.remove(l)

        with AnnotatedQueue() as q:
            qfunc_output = obj(*args, **kwargs)

        tape = QuantumScript.from_queue(q)

        with QueuingManager.stop_recording():
            if transform.is_informative:
                transformed_tapes, processing_fn = transform.tape_transform(tape, *targs, **tkwargs)
            else:
                transformed_tapes, processing_fn = transform(tape, *targs, **tkwargs)

        if len(transformed_tapes) != 1:
            raise TransformError(
                "Impossible to dispatch your transform on quantum function, because more than "
                "one tape is returned"
            )

        transformed_tape = transformed_tapes[0]

        if transform.is_informative:
            return processing_fn(transformed_tapes)

        for op in transformed_tape.operations:
            apply(op)

        mps = [apply(mp) for mp in transformed_tape.measurements]

        if not mps:
            return qfunc_output

        if isinstance(qfunc_output, MeasurementProcess):
            return tuple(mps) if len(mps) > 1 else mps[0]

        if isinstance(qfunc_output, (tuple, list)):
            return type(qfunc_output)(mps)

        interface = math.get_interface(qfunc_output)
        return math.asarray(mps, like=interface)

    return qfunc_transformed


@Transform.generic_register
def _apply_to_sequence(obj: Sequence, transform, *targs, **tkwargs):
    if not all(isinstance(t, QuantumScript) for t in obj):
        # not a sequence of quantum script, treat as first argument
        targs, tkwargs = transform.setup_inputs(obj, *targs, **tkwargs)
        return BoundTransform(transform, args=targs, kwargs=tkwargs)
    targs, tkwargs = transform.setup_inputs(*targs, **tkwargs)
    execution_tapes = []
    batch_fns = []
    tape_counts = []

    for t in obj:
        # Preprocess the tapes by applying transforms
        # to each tape, and storing corresponding tapes
        # for execution, processing functions, and list of tape lengths.
        new_tapes, fn = transform(t, *targs, **tkwargs)
        execution_tapes.extend(new_tapes)
        batch_fns.append(fn)
        tape_counts.append(len(new_tapes))

    def processing_fn(res: ResultBatch) -> ResultBatch:
        """Applies a batch of post-processing functions to results.

        Args:
            res (ResultBatch): the results of executing a batch of circuits.

        Returns:
            ResultBatch: results that have undergone classical post processing.

        Closure variables:
            tape_counts: the number of tapes outputted from each application of the transform.
            batch_fns: the post processing functions to apply to each sub-batch.

        """
        count = 0
        final_results = []

        for f, s in zip(batch_fns, tape_counts):
            # apply any batch transform post-processing
            new_res = f(res[count : count + s])
            final_results.append(new_res)
            count += s

        return tuple(final_results)

    return tuple(execution_tapes), processing_fn


TransformContainer = BoundTransform
TransformDispatcher = Transform
