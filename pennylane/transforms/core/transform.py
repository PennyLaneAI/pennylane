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


def transform(
    quantum_transform,
    expand_transform=None,
    classical_cotransform=None,
    is_informative=False,
    final_transform=False,
    use_argnum_in_expand=False,
):  # pylint: disable=too-many-arguments
    """Generalizes a function that transforms tapes to work with additional circuit-like objects such as a
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
              qml.tape.QuantumTape, ...) -> ( Sequence[qml.tape.QuantumTape], Callable)``

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

    Returns:

        .TransformDispatcher: Returns a transform dispatcher object that that can transform any
        circuit-like object in PennyLane.

    **Example**

    First define an input quantum transform with the necessary structure defined above. In this example we copy the
    tape and sum the results of the execution of the two tapes.

    .. code-block:: python

        from typing import Sequence, Callable

        def my_quantum_transform(tape: qml.tape.QuantumTape) -> (Sequence[qml.tape.QuantumTape], Callable):
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
        :title: Signature of a transform

        A dispatched transform is able to handle several PennyLane circuit-like objects:

        - :class:`pennylane.QNode`
        - a quantum function (callable)
        - :class:`pennylane.tape.QuantumTape`
        - :class:`pennylane.devices.Device`.

        For each object, the transform will be applied in a different way, but it always preserves the underlying
        tape-based quantum transform behaviour.

        The return of a dispatched transform depends upon which of the above objects is passed as an input:

        - For a :class:`~.QNode` input, the underlying transform is added to the QNode's
          :class:`~.TransformProgram` and the return is the transformed :class:`~.QNode`.
          For each execution of the :class:`pennylane.QNode`, it first applies the transform program on the original captured
          circuit. Then the transformed circuits are executed by a device and finally the post-processing function is
          applied on the results.

        - For a quantum function (callable) input, the transform builds the tape when the quantum function is
          executed and then applies itself to the tape. The resulting tape is then converted back
          to a quantum function (callable). It therefore returns a transformed quantum function (Callable). The limitation
          is that the underlying transform can only return a sequence containing a single tape, because quantum
          functions only support a single circuit.

        - For a :class:`~.QuantumTape`, the underlying quantum transform is directly applied on the
          :class:`~.QuantumTape`. It returns a sequence of :class:`~.QuantumTape` and a processing
          function to be applied after execution.

        - For a :class:`~.devices.Device`, the transform is added to the device's transform program
          and a transformed :class:`pennylane.devices.Device` is returned. The transform is added
          to the end of the device program and will be last in the overall transform program.
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
    )
