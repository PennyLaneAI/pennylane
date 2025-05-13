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
This module contains the transform function/decorator to make your custom transforms compatible with tapes, quantum
functions and QNodes.
"""
from typing import get_type_hints

from .transform_dispatcher import TransformDispatcher, TransformError


def transform(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    quantum_transform,
    expand_transform=None,
    classical_cotransform=None,
    is_informative=False,
    final_transform=False,
    use_argnum_in_expand=False,
    plxpr_transform=None,
) -> TransformDispatcher:
    r"""Generalizes a function that transforms tapes to work with additional circuit-like objects such as a
    :class:`~.QNode`.

    ``transform`` should be applied to a function that transforms tapes. Once validated, the result will
    be an object that is able to transform PennyLane's range of circuit-like objects:
    :class:`~.QuantumTape`, quantum function and :class:`~.QNode`.
    A circuit-like object can be transformed either via decoration or by passing it functionally through
    the created transform.

    Args:
        quantum_transform (Callable): The input quantum transform must be a function that satisfies the
            following requirements:

            * Accepts a :class:`~.QuantumTape` as its first input and
              returns a sequence of :class:`~.QuantumTape` and a processing function.

            * The transform must have the following structure (type hinting is optional): ``my_quantum_transform(tape:
              qml.tape.QuantumScript, ...) -> tuple[qml.tape.QuantumScriptBatch, qml.typing.PostprocessingFn]``

    Keyword Args:
        expand_transform=None (Optional[Callable]): An optional expand transform is applied directly before the input
            quantum transform. It must be a function that satisfies the same requirements as
            ``quantum_transform``.
        classical_cotransform=None (Optional[Callable]): A classical co-transform is a function to post-process the classical
            jacobian and the quantum jacobian and has the signature: ``my_cotransform(qjac, cjac, tape) -> tensor_like``
        is_informative=False (bool): Whether or not a transform is informative. If true the transform is queued at the end
            of the transform program and the tapes or qnode aren't executed.
        final_transform=False (bool): Whether or not the transform is terminal. If true the transform is queued at the end
            of the transform program. ``is_informative`` supersedes ``final_transform``.
        use_argnum_in_expand=False (bool): Whether or not to use ``argnum`` of the tape to determine trainable parameters
            during the expansion transform process.
        plxpr_transform=None (Optional[Callable]): Function for transforming plxpr. **Experimental**

    Returns:

        .TransformDispatcher: Returns a transform dispatcher object that that can transform any
        circuit-like object in PennyLane.

    **Example**

    First define an input quantum transform with the necessary structure defined above. In this example we copy the
    tape and sum the results of the execution of the two tapes.

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

    >>> dispatched_transform = transform(my_quantum_transform)

    Now you can use the dispatched transform directly on a :class:`pennylane.QNode`.

    For :class:`pennylane.QNode`, the dispatched transform populates the ``TransformProgram`` of your QNode. The
    transform and its processing function are applied in the execution.

    >>> transformed_qnode = dispatched_transform(qnode_circuit)
    <QNode: wires=2, device='default.qubit', interface='auto', diff_method='best'>

    >>> transformed_qnode.transform_program
    TransformProgram(my_quantum_transform)

    If we apply ``dispatched_transform`` a second time to the :class:`pennylane.QNode`, we would add
    it to the transform program again and therefore the transform would be applied twice before execution.

    >>> transformed_qnode = dispatched_transform(transformed_qnode)
    >>> transformed_qnode.transform_program
    TransformProgram(my_quantum_transform, my_quantum_transform)

    When a transformed QNode is executed, the QNode's transform program is applied to the generated tape
    and creates a sequence of tapes to be executed. The execution results are then post-processed in the
    reverse order of the transform program to obtain the final results.

    .. details::
        :title: Dispatch a transform onto a batch of tapes

        We can compose multiple transforms when working in the tape paradigm and apply them to more than one tape.
        The following example demonstrates how to apply a transform to a batch of tapes.

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

        The first ``split_non_commuting`` transform splits the original tape, returning a batch of tapes ``batch1`` and a processing function ``function1``.
        The second ``merge_rotations`` transform is applied to the batch of tapes returned by the first transform.
        It returns a new batch of tapes ``batch2``, each of which has been transformed by the second transform, and a processing function ``function2``.

        >>> batch2
        (<QuantumTape: wires=[0, 1, 2], params=2>,
        <QuantumTape: wires=[0, 1, 2], params=1>)

        >>> type(function2)
        function

        We can combine the processing functions to post-process the results of the execution.

        >>> function1(function2(result))
        [array(0.5)]

    .. details::
        :title: Signature of a transform

        A dispatched transform is able to handle several PennyLane circuit-like objects:

        - :class:`pennylane.QNode`
        - a quantum function (callable)
        - :class:`pennylane.tape.QuantumTape`
        - a batch of :class:`pennylane.tape.QuantumTape`
        - :class:`pennylane.devices.Device`.

        For each object, the transform will be applied in a different way, but it always preserves the underlying
        tape-based quantum transform behaviour.

        The return of a dispatched transform depends upon which of the above objects is passed as an input:

        - For a :class:`~.QNode` input, the underlying transform is added to the QNode's
          :class:`~.TransformProgram` and the return is the transformed :class:`~.QNode`.
          For each execution of the :class:`pennylane.QNode`, it first applies the transform program on the original captured
          circuit. Then the transformed circuits are executed by a device and finally the post-processing function is
          applied on the results.

          When experimental program capture is enabled, transforming a :class:`~.QNode` returns a new function to which the
          transform has been added as a higher-order primitive.

        - For a quantum function (callable) input, the transform builds the tape when the quantum function is
          executed and then applies itself to the tape. The resulting tape is then converted back
          to a quantum function (callable). It therefore returns a transformed quantum function (Callable). The limitation
          is that the underlying transform can only return a sequence containing a single tape, because quantum
          functions only support a single circuit.

          When experimental program capture is enabled, transforming a function (callable) returns a new function to which the
          transform has been added as a higher-order primitive.

        - For a :class:`~.QuantumTape`, the underlying quantum transform is directly applied on the
          :class:`~.QuantumTape`. It returns a sequence of :class:`~.QuantumTape` and a processing
          function to be applied after execution.

        - For a batch of :class:`pennylane.tape.QuantumTape`, the quantum transform is mapped across all the tapes.
          It returns a sequence of :class:`~.QuantumTape` and a processing function to be applied after execution.
          Each tape in the sequence is transformed by the transform.

        - For a :class:`~.devices.Device`, the transform is added to the device's transform program
          and a transformed :class:`pennylane.devices.Device` is returned. The transform is added
          to the end of the device program and will be last in the overall transform program.

    .. details::
        :title: Transforms with experimental program capture

        To define a transform that can be applied directly to plxpr without the need to create ``QuantumScript``\ s, users
        must provide the ``plxpr_transform`` argument. If this argument is not provided, executing transformed functions
        is not guaranteed to work. More details about this are provided below. The ``plxpr_transform`` argument should be a
        function that applies the respective transform to ``jax.extend.core.Jaxpr`` and returns a transformed ``jax.extend.core.ClosedJaxpr``.
        ``plxpr_transform`` can assume that no transform primitives are present in the input plxpr, and its implementation
        does not need to account for these primitives. The exact expected signature of ``plxpr_transform`` is shown in the
        example below:

        .. code-block:: python

            def dummy_plxpr_transform(
                jaxpr: jax.extend.core.Jaxpr, consts: list, targs: list, tkwargs: dict, *args
            ) -> jax.extend.core.ClosedJaxpr:
                ...

        Once the ``plxpr_transform`` argument is provided, the transform can be easily used with program capture
        enabled! To do so, apply the transform as you normally would:

        .. code-block:: python

            qml.capture.enable()

            @qml.transforms.cancel_inverses
            def circuit():
                qml.X(0)
                qml.S(1)
                qml.X(0)
                qml.adjoint(qml.S(1))
                return qml.expval(qml.Z(1))

        >>> qml.capture.make_plxpr(circuit)()
        { lambda ; . let
            a:AbstractMeasurement(n_wires=None) = cancel_inverses_transform[
            args_slice=slice(0, 0, None)
            consts_slice=slice(0, 0, None)
            inner_jaxpr={ lambda ; . let
                _:AbstractOperator() = PauliX[n_wires=1] 0
                _:AbstractOperator() = S[n_wires=1] 1
                _:AbstractOperator() = PauliX[n_wires=1] 0
                b:AbstractOperator() = S[n_wires=1] 1
                _:AbstractOperator() = Adjoint b
                c:AbstractOperator() = PauliZ[n_wires=1] 1
                d:AbstractMeasurement(n_wires=None) = expval_obs c
              in (d,) }
            targs_slice=slice(0, None, None)
            tkwargs={}
            ]
          in (a,) }

        As shown, the transform gets applied as a higher-order primitive, with the jaxpr
        representation of the function being transformed stored in the ``inner_jaxpr``
        parameter of the transform's primitive.

        **Fallback implementation of plxpr transforms:**

        If a transform that does not define a ``plxpr_transform`` is applied to a function,
        a fallback implementation of the transform is used. This fallback implementation converts
        the function into a :func:`~pennylane.tape.QuantumScript`, which is then transformed
        as a traditional tape. However, because of the constraints of program capture, many transforms will not
        be compatible with this fallback implementation:

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
        >>> qml.capture.make_plxpr(transformed_circuit)()
        { lambda ; . let
            a:AbstractOperator() = PauliZ[n_wires=1] 1
            b:AbstractMeasurement(n_wires=None) = expval_obs a
          in (b,) }
    """
    # 1: Checks for the transform
    if not callable(quantum_transform):
        raise TransformError(
            f"The function to register, {quantum_transform}, "
            "does not appear to be a valid Python function or callable."
        )

    signature_transform = get_type_hints(quantum_transform)

    # 2: Checks for the expand transform
    if expand_transform is not None:
        if not callable(expand_transform):
            raise TransformError("The expand function must be a valid Python function.")
        signature_expand_transform = get_type_hints(expand_transform)

        if signature_expand_transform != signature_transform:
            raise TransformError(
                "The expand transform must have the same signature as the transform"
            )

    # 3: Check the classical co-transform
    if classical_cotransform is not None and not callable(classical_cotransform):
        raise TransformError("The classical co-transform must be a valid Python function.")

    return TransformDispatcher(
        quantum_transform,
        expand_transform=expand_transform,
        classical_cotransform=classical_cotransform,
        is_informative=is_informative,
        final_transform=final_transform,
        use_argnum_in_expand=use_argnum_in_expand,
        plxpr_transform=plxpr_transform,
    )
