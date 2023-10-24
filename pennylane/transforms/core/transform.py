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
from typing import get_type_hints, Sequence, List, Tuple, Callable
import pennylane as qml
from .transform_dispatcher import TransformDispatcher, TransformError


def transform(
    quantum_transform,
    expand_transform=None,
    classical_cotransform=None,
    is_informative=None,
    final_transform=False,
):
    """The transform function is to be used to validate and dispatch a quantum transform on PennyLane objects
    (:class:`pennylane.tape.QuantumTape`, quantum function and :class:`pennylane.QNode`). After applying the function on
    a valid quantum transform, the resulted function can be used on all the mentioned objects. The function can be used
    directly as a decorator on quantum function and :class:`pennylane.QNode`.

    Args:
        quantum_transform (callable): A quantum transform is defined as a function that has the following requirements:

            * A quantum transform is a function that takes a :class:`pennylane.tape.QuantumTape` as first input and
              returns a sequence of :class:`pennylane.tape.QuantumTape` and a processing function.

            * The transform must have type hinting of the following form: ``my_quantum_transform(tape:
              qml.tape.QuantumTape, ...) -> ( Sequence[qml.tape.QuantumTape], callable)``

        expand_transform (callable): An expand transform is defined as a function that has the following requirements:

            * An expand transform is a function that is applied before applying the defined quantum transform. It
              takes the same arguments as the transform.

            * The expand transform must have the same type hinting as a quantum transform.

        classical_cotransform (callable): A classical co-transform is a function to post-process the the classical
            jacobian and the quantum jacobian and has the signature: ``my_cotransform(qjac, cjac, tape) -> tensor_like``
        is_informative (bool): Whether or not a transform is informative. If true the transform is queued at the end
            of the transform program and the tapes or qnode aren't executed.
        final_transform (bool): Whether or not the transform is terminal. If true the transform is queued at the end
            of the transform program. ``is_informative`` supersedes ``final_transform``.

    Returns:

        pennylane.transforms.core.TransformDispatcher: It returns a transform dispatcher object that is dispatching
            given the received circuit object.

    **What can a dispatched function accept as argument and what does it return?**

    A dispatched transform is able to handle several PenyLane objects: :class:`pennylane.QNode`, quantum function
    (callable), :class:`pennylane.tape.QuantumTape` and :class:`pennylane.devices.Device`. For each object, the transform
    will be applied in a different way but it always preseves the provided underlying quantum transform. What is
    returned by the transform dispatcher depends on what circuit abstraction is given as argument. Here is provided a
    list of the different behavior and return for the different objects:

    - For a :class:`pennylane.QNode` as argument of the dispatched transform, the underlying transform is added to
      its :class:`pennylane.transform.core.TransformProgram` and the return is the transformed :class:`pennylane.QNode`.
      For each execution of the :class:`pennylane.QNode`, it first applies the transform program on the original captured
      circuit. Then the transformed circuits are executed by a device and finally the post-processing function is
      applied on the results.

    - For a quantum function (callable) as argument of the dispatched transform, it first builds the tape from the
      quantum function, the transform is then applied on the tape. Then the resulting tape is converted back
      to a quantum function (callable). It therefore returns a transformed quantum function (callable). The limitation
      is that the underlying transform can only return a sequence containing a single tape, because quantum
      functions only support single circuit.

    - For a :class:`pennylane.tape.QuantumTape`, the underlying quantum transform is directly applied on the
      :class:`pennylane.tape.QuantumTape`. It returns a sequence of :class:`pennylane.tape.QuantumTape` and a processing
      function to be applied after execution.

    - For a :class:`pennylane.devices.Device`, the transform is added to the device transform program and a transformed
      :class:`pennylane.devices.Device` is returned. The transform is added to the end of the device program and will
      be last in the overall transform program.

    **Example**

    First define your quantum transform, with the necessary type hinting defined above. In this example we copy the
    tape and sum the results of the execution of the two tapes.

    .. code-block:: python

        from typing import Sequence, Callable

        def my_quantum_transform(tape: qml.tape.QuantumTape) -> (Sequence[qml.tape.QuantumTape], Callable):
            tape1 = tape
            tape2 = tape.copy()

            def post_processing_fn(results):
                return qml.math.sum(results)

            return [tape1, tape2], post_processing_fn

    Of course, we want to be able to apply this transform on ``qfunc`` and :class:`pennylane.QNode`. That's where the
    ``transform`` function comes into play. This function validates the signature of your quantum transform and
    dispatches it on the different objects. Let's define a circuit as a quantum function and as a
    :class:`pennylane.QNode`.

        .. code-block:: python

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(device=dev)
            def qnode_circuit(a):
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                qml.PauliX(wires=0)
                qml.RZ(a, wires=1)
                return qml.expval(qml.PauliZ(wires=0))

    We apply the transform on our tranform in order to make it compatible with more PennyLane types.

    >>> dispatched_transform = transform(my_quantum_transform)

    Now you can use the dispatched transform directly on :class:`pennylane.QNode`.

    For :class:`pennylane.QNode`, the dispatched transform populates the ``TransformProgram`` of your QNode. The
    transform and its processing function are applied in the execution.

    >>> transformed_qnode = dispatched_transform(qnode_circuit)
    <QNode: wires=2, device='default.qubit', interface='auto', diff_method='best'>

    >>> transformed_qnode.transform_program
    TransformProgram(my_quantum_transform)

    If we apply a second time the transform on the and :class:`pennylane.QNode`, we would add it to the transform
    program again and therefore the transform would be applied twice before execution.

    >>> transformed_qnode = dispatched_transform(transformed_qnode)
    TransformProgram(my_quantum_transform, my_quantum_transform)

    The transform program is automatically applied on the tapes before sending the tapes to execution. When called the
    transform program applies all the transforms that it contains and create a sequence of tapes, it also produces a single
    post-processing function that containes a reversed concatenation of all the transforms processing functions. After
    executing the tapes, the post-processing function is applied to get the final results.

    """
    # 1: Checks for the transform
    if not callable(quantum_transform):
        raise TransformError(
            f"The function to register, {quantum_transform}, "
            "does not appear to be a valid Python function or callable."
        )

    signature_transform = get_type_hints(quantum_transform)
    # Check signature of transform to force the fn style (tape, ...) - > (Sequence(tape), fn)
    _transform_signature_check(signature_transform)

    # 2: Checks for the expand transform
    if expand_transform is not None:
        if not callable(expand_transform):
            raise TransformError("The expand function must be a valid Python function.")
        signature_expand_transform = get_type_hints(expand_transform)
        # Check the signature of expand_transform to force the fn style tape - > (Sequence(tape), fn)
        _transform_signature_check(signature_expand_transform)

        if signature_expand_transform != signature_transform:
            raise TransformError(
                "The expand transform must have the same signature as the transform"
            )

    # 3: CHeck the classical co-transform
    if classical_cotransform is not None:
        if not callable(classical_cotransform):
            raise TransformError("The classical co-transform must be a valid Python function.")

    dispatcher = TransformDispatcher(
        quantum_transform,
        expand_transform=expand_transform,
        classical_cotransform=classical_cotransform,
        is_informative=is_informative,
        final_transform=final_transform,
    )
    return dispatcher


def _transform_signature_check(signature):
    """Check the signature of a quantum transform: (tape, ...) - > (Sequence(tape), fn)"""
    # Check that the arguments of the transforms follows: (tape: qml.tape.QuantumTape, ...)
    tape = signature.get("tape", None)

    if tape is None:
        raise TransformError("The first argument of a transform must be tape.")

    if tape != qml.tape.QuantumTape:
        raise TransformError("The type of the tape argument must be a QuantumTape.")

    # Check return is (qml.tape.QuantumTape, callable):
    ret = signature.get("return", None)

    if ret is None or not isinstance(ret, tuple):
        raise TransformError(
            "The return of a transform must match (collections.abc.Sequence["
            "pennylane.tape.tape.QuantumTape], <built-in function callable>)"
        )

    if ret[0] not in (
        Sequence[qml.tape.QuantumTape],
        List[qml.tape.QuantumTape],
        Tuple[qml.tape.QuantumTape],
    ):  # pylint:disable=unsubscriptable-object
        raise TransformError(
            "The first return of a transform must be a sequence of tapes: collections.abc.Sequence["
            "pennylane.tape.tape.QuantumTape]"
        )

    if ret[1] != Callable:
        raise TransformError(
            "The second return of a transform must be a callable: <built-in function callable>"
        )
