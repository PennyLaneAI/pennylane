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
This module contains the transform function to make your custom transforms compatible with qfunc and QNodes.
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
    """The transform function is to be used to validate and dispatch a quantum transform on PennyLane objects (tape,
    qfunc and Qnode). It can be used directly as a decorator on qfunc and qnodes.

    Args:
        quantum_transform (callable): A quantum transform is defined as a function that has the following requirements:

            * A quantum transform is a function that takes a quantum tape as first input and returns a sequence of tapes
              and a processing function.

            * The transform must have type hinting of the following form: my_quantum_transform(tape:
              qml.tape.QuantumTape, ...) -> ( Sequence[qml.tape.QuantumTape], callable)

        expand_transform (callable): An expand transform is defined as a function that has the following requirements:

            * An expand transform is a function that is applied before applying the defined quantum transform. It
              takes the same arguments as the transform and returns a single tape in a sequence with a dummy processing
              function.

            * The expand transform must have the same type hinting as a quantum transform.

        classical_cotransform (callable): A classical co-transform. NOT YET SUPPORTED.
        is_informative (bool): Whether or not a transform is informative. If true the transform is queued at the end
            of the transform program and the tapes or qnode aren't executed.
        final_transform (bool): Whether or not the transform is terminal. If true the transform is queued at the end
            of the transform program. ``is_informative`` supersedes ``final_transform``.

    **Example**

    First define your quantum_transform, with the necessary type hinting defined above. In this example we copy the
    tape and sum the results of the execution of the two tapes.

    .. code-block:: python

        def my_quantum_transform(tape: qml.tape.QuantumTape) -> (Sequence[qml.tape.QuantumTape], callable):
            tape1 = tape
            tape2 = tape.copy()

            def post_processing_fn(results):
                return qml.math.sum(results)

            return [tape1, tape2], post_processing_fn

    Of course, we want to be able to apply this transform on ``qfunc`` and ``qnodes``. That's where the ``transform`` function
    comes into play. This function validates the signature of your quantum transform and dispatches it on the different
    object. Let's define a circuit as a qfunc and as a qnode.

        .. code-block:: python

            def qfunc_circuit(a):
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                qml.PauliX(wires=0)
                qml.RZ(a, wires=1)
                return qml.expval(qml.PauliZ(wires=0))

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(device=dev)
            def qnode_circuit(a):
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                qml.PauliX(wires=0)
                qml.RZ(a, wires=1)
                return qml.expval(qml.PauliZ(wires=0))

    >>> dispatched_transform = transform(my_quantum_transform)

    Now you can use the dispatched transform directly on qfunc and qnodes.

    For QNodes, the dispatched transform populates the ``TransformProgram`` of your QNode. The transform and its
    processing function are applied in the execution.

    >>> transformed_qnode = dispatched_transform(qfunc_circuit)
    <QNode: wires=2, device='default.qubit', interface='auto', diff_method='best'>

    One subtlety here, this transform would not work for a qfunc because our transform return more than one case. If
    it was not the case you would be able to dispatch on quantum functions.
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
        raise NotImplementedError("Classical cotransforms are not yet integrated.")
        # TODO: Add more verification in a future PR
        # if not callable(classical_cotransform):
        #    raise TransformError("The classical co-transform must be a valid Python function.")

    return TransformDispatcher(
        quantum_transform,
        expand_transform=expand_transform,
        classical_cotransform=classical_cotransform,
        is_informative=is_informative,
        final_transform=final_transform,
    )


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
