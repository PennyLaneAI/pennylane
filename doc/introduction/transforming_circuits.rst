.. role:: html(raw)
   :format: html

.. _intro_ref_transform_circuits:

Transforming circuits
=====================

A quantum transform is a crucial concept in PennyLane, and refers to mapping a quantum
circuit to one or more circuits, alongside a classical post-processing function.
Once a transform is registered with PennyLane, the transformed circuits will be executed,
and the classical post-processing function automatically applied to the outputs.
This becomes particularly valuable when a transform generates multiple circuits,
requiring a method to aggregate or reduce the results (e.g.,
applying the parameter-shift rule or computing the expectation value of a Hamiltonian 
term-by-term).

.. note::

    For examples of built-in transforms that come with PennyLane, see the
    :doc:`/introduction/compiling_circuits` documentation.

Creating your own transform
---------------------------

To streamline the creation of transforms and ensure their versatility across
various circuit abstractions in PennyLane, the
:func:`pennylane.transform` is available.

This decorator registers transforms that accept a :class:`~.QuantumTape`
as its primary input, and returns a sequence of :class:`~.QuantumTape`
and an associated processing function.

To illustrate the process of creating a quantum transform, let's consider a straightforward example. Suppose we want
a transform that removes all :class:`~.RX` operations from a given circuit. In this case, we merely need to filter the
original :class:`~.QuantumTape` and return a new one without the filtered operations. As we don't require a specific processing
function in this scenario, we include a null function that simply returns the result.

.. code-block:: python

    from typing import Sequence, Callable
    from pennylane.tape import QuantumTape

    def remove_rx(tape: QuantumTape) -> (Sequence[QuantumTape], Callable):

        operations = filter(lambda op: op.name != "RX", tape.operations)
        new_tape = type(tape)(operations, tape.measurements, shots=tape.shots)

        def null_postprocessing(results):
            return results[0]

        return [new_tape], null_postprocessing

To make your transform applicable to both :class:`~.QNode` and quantum functions, you can use the :func:`pennylane.transform` decorator.

>>> dispatched_transform = qml.transform(remove_rx)

For a more advanced example, let's consider a transform that sums a circuit with its adjoint. We define the adjoint
of the tape operations, create a new tape, and return both tapes. The processing function then sums the results.

.. code-block:: python

    from typing import Sequence, Callable
    from pennylane.tape import QuantumTape

    @qml.transform
    def sum_circuit_and_adjoint(tape: QuantumTape) -> (Sequence[QuantumTape], Callable):

        operations = [qml.adjoint(op) for op in tape.operation]
        new_tape = type(tape)(operations, tape.measurements, shots=tape.shots)

        def null_postprocessing(results):
            return qml.sum(results)

        return [tape, shifted_tape], null_postprocessing

Composability of transforms
---------------------------

Transforms are inherently composable on a :class:`~.QNode`, meaning that transforms with compatible post-processing
functions can be successively applied to QNodes. For example, this allows for the application of multiple compilation
passes on a QNode to maximize gate reduction before execution.

.. code-block:: python

        dev = qml.device("default.qubit", wires=1)
        @qml.merge_rotations
        @qml.cancel_inverses
        @qml.qnode(device=dev):
        def circuit(x, y):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=0)
            qml.RX(x, wires=0)
            qml.RY(y, wires=0)
            qml.RZ(y, wires=0)
            qml.RY(x, wires=0)
            return qml.expval(qml.PauliZ(wires=0))

In this example, inverses are canceled, leading to the removal of two Hadamard gates. Subsequently, rotations are
merged into a single :class:`qml.Rot` gate. Consequently, two transforms are successfully applied to the circuit.

Additional information
----------------------

Explore practical examples of transforms focused on compiling circuits in the :doc:`compiling circuits documentation </introduction/compiling_circuits>`.
For gradient transforms, refer to the examples in :doc:`gradients documentation <../code/qml_gradients>`.
Discover quantum information transformations in the :doc:`quantum information documentation <../code/qml_qinfo>`. Finally,
for a comprehensive overview of transforms and core functionalities, consult the :doc:`transforms documentation <../code/qml_transforms>`.