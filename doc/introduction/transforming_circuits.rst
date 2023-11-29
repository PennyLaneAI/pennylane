.. role:: html(raw)
   :format: html

.. _intro_ref_transform_circuits:

Transforming circuits
=====================

A quantum transform is a function that takes a circuit as input and returns one or more transformed circuits with a
post processing function. The post processing functions is applied on the results after execution of the resulting
circuit. This function is mainly useful when the transform returns multiple circuits, as we need a recipe to gather
the results of the executed circuits (e.g. parameter shift or hamiltonian expansion).

This requirements translates in PennyLane as follows:

* A transform accepts a :class:`~.QuantumTape` as its first input and
  returns a sequence of :class:`~.QuantumTape` and a processing function.

In order to simplify the creation of transforms and to make them powerful tools that work on all different circuit
abstraction in PennyLane (:class:`~.QuantumTape`, :class:`~.QNode` and quantum functions). We have created a simple
decorator :func:`~.transform` that can be applied on your quantum transform respecting the above contract.

How to create your own transform?
---------------------------------

First let's create a quantum transform that respects the structure requirements from above. Let's say we want a very
simple transform that removes the :class:`~.RX` operations from the circuit. We simply need to filter the original
:class:`~.QuantumTape` and return a new one without the filter operations. In this case, we do not need any specific
processing function, therefore we add a null function that just returns the result.

.. code-block:: python

    from typing import Sequence, Callable
    from pennylane.tape import QuantumTape

    def remove_rx(tape: QuantumTape) -> (Sequence[QuantumTape], Callable):

        operations = filter(lambda op: op.name != "RX", tape.operations)
        new_tape = type(tape)(operations, tape.measurements, shots=tape.shots)

        def null_postprocessing(results):
            return results[0]

        return [new_tape], null_postprocessing

Now if you want your transform to work on :class:`~.QNode` and quantum functions, you simply need to use
:func:`~.transform` as a decorator.

>>> dispatched_transform = qml.transform(remove_rx)

Let's now consider an example where the post processing is not a dummy function. Let's we want to take a circuit and
sum it to its adjoint. We simply define the adjoint of the tape operations, create a new tape and return both tapes.
The processing function simply sums the results.

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

The transforms are by principle composable on :class:`~.QNode`, it means that transforms with compatible post processing
function can be applied on QNodes successively. For example you want to apply multiple compilation passes on your QNode
to increase the reduction of gates before execution then it is possible:

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

First inverses are cancelled and therefore the two Hadamard gates are removed, then the rotation are merged as a single
:class:`qml.Rot` gate. Therefore we successfully applied two transforms on this circuit very easily!

Relevant links
--------------

Good examples of transforms with the purpose of compiling can be found in :doc:`inspecting circuits </introduction/compiling_circuits>`.
For gradient transforms, you can find examples in :doc:`gradients documentation <../code/api/pennylane.gradients>`.
Quantum information transformations can be found the :doc:`qinfo documentation <../code/api/pennylane.qinfo>`. Finally,
the rest of the transforms and the core functionalities can be found in the:doc:`qinfo documentation <../code/api/pennylane.transforms>`.