.. role:: html(raw)
   :format: html

Architectural overview
======================

PennyLane is a framework for optimization and machine learning of quantum and
hybrid quantum-classical computations. The library provides a unified
architecture for near-term quantum computing devices, supporting various
quantum information paradigms.

PennyLane's core feature is the ability to compute gradients of variational
quantum circuits in a way that is compatible with classical techniques such as
backpropagation. PennyLane thus extends the automatic differentiation
algorithms common in optimization and machine learning to include quantum and
:doc:`hybrid computations <glossary/hybrid_computation>`.
A plugin system makes the framework compatible with many quantum
simulators or hardware devices, remote or local.

Components
##########

The central object in PennyLane is a **QNode**, which represents a
"node" performing a quantum computation. Several quantum nodes may be
part of a larger classical computation. QNodes run **quantum circuits** on **devices**.
These devices may be simulators built into PennyLane, or external devices
provided by plugins. The user specifies a quantum circuit by defining a "quantum function",
which is a Python function that contains quantum gates, observables and measurements
represented by the ``Operation``, ``Observable`` and ``MeasurementProcess`` classes, respectively).


.. image:: pl_overview.png
    :width: 800px

The power of a QNode lies in the fact that it can be run in a "forward" fashion to
execute the quantum circuit, or in a "backward" fashion in which it provides
gradients (or, more precisely, jacobian-vector products).

Let's go through these components one-by-one.

Operator
********

Quantum operators are represented by the :class:`~.Operator` class which
contains one or more representations of the operator and the wires they act on.
Devices interpret this information to implement the operator.

For example, the PauliX operator ``qml.PauliX(wires=0)``
provides its representation as a matrix, ...

[TODO: Examples PauliX, RX and Hamiltonian, used to describe both, ops and obs].

[TODO: Explain templates]

[TODO: overview graphic of operator representations]

MeasurementProcess
******************

While the ``Operator`` class describes a physical system and its dynamics,
the ``MeasurementProcess`` class describes how we extract information from the quantum system.

[TODO: Add more once we know it]

QuantumTape
***********

Quantum operators and measurement processes can be used to build a quantum circuit.
The user does this by defining a quantum function.

CODE EXAMPLE.

Internally, a quantum function is translated to a quantum tape, which is
the representation of a quantum circuit. The tape inherits from :class:`~.pennylane.QueuingContext`,
and creating operations inside a tape context adds them to the queue
by having :meth:`.Operator.__init__` call the :meth:`.Operator.queue` method.
Measurement processes such as :func:`~.pennylane.expval` are responsible for queuing observables.

EXAMPLE

The relevant parts of the queue can then be accessed via ``tape.operations``,
``tape.observables`` and ``tape.measurements``.

.. note::

    Tapes can represent parts of quantum circuits and do not necessarily need to define a measurement.
    They can also be nested. [TODO: explain more]

[TODO: explain tape expansion]

Devices
*******

In PennyLane, the abstraction of a quantum computation device is encompassed
within the :class:`~.Device` class. The main job of devices is to
interpret and execute tapes. The most important method is

.. code-block:: python

    device.batch_execute([tape1, tape2,...])


There are also device subclasses available, containing shared logic for
particular types of devices.  For example, qubit-based devices can inherit from
the :class:`~.QubitDevice` class, easing development.

To register a new device with PennyLane, they must register an `entry point
<https://packaging.python.org/specifications/entry-points/>`__ under the `pennylane.plugins`
namespace using Setuptools. Once registered, the device can be instantiated using the :func:`~.device`
loader function.

A Python package that registers one or more PennyLane devices is known as a *plugin*. For more details
on plugins and devices, see :doc:`/development/plugins`.

QNodes
******

This is where it all comes together: A **QNode** (represented by a subclass of
:class:`~.BaseQNode`) is an encapsulation of a function
:math:`f(x;\theta)=R^m\rightarrow R^n` that is executed using quantum
information processing on a quantum device. It is created by a quantum function and a device.
Users don't typically instantiate QNodes directly---instead, the :func:`~pennylane.qnode` decorator or
:func:`~pennylane.QNode` constructor function automates the process of creating a QNode from a provided
quantum function and device.

Internally, the QNode translates the quantum function into one or more quantum tapes
and classical processing routines that, taken together, execute the quantum computation.

The crucial property of a QNode is that it is differentiable by classical autodifferentiation
frameworks such as autograd, jax, TensorFlow and PyTorch. The next section will look at
differentiation workflows in more detail.

Workflow
########

Autodifferentiation frameworks may run QNodes in "forward mode" 
to compute the result of a quantum circuit, or in "backward mode" to compute 
the gradient of a qnode with respect to some trainable parameters.

The internal workflow in the QNode is surprisingly similar in both cases, and 
consists of three steps: to construct one or more tapes using the quantum function,
to run the tapes on the device, and to post-process the results.


.. image:: pl_workflow.png
    :width: 800px

The fact that multiple tapes may be constructed from one quantum function may be 
surprising at first, but there are many situations in which the evaluation of a quantum circuit 
practically requires many circuits to be evaluated, for example:

* When the observable is a Hamiltonian represented as a linear combination of Pauli words, the device may 
  instruct the QNode to create one circuit for each Pauli word, and to compute their linear combination 
  during post-processing.
* When a gradient of the QNode is requested, and parameter-shift rules have to be used. The QNode 
  constructs tapes in which parameters are shifted, and recombines the result to return a gradient.

Interfaces
**********

The construction of tapes, as well as post-processing are classical computations, and they 
are "tracked" by the autodifferentiation framework (marked in red above). 
In other words, these steps can invoke differentiable classical computations, such as:

* The decomposition of a user-defined gate into other gates that take some 
  function of the original gate's parameters
* The linear re-combination of Hamiltonian terms with trainable coefficients.

There are some devices where the execution of the quantum circuit is also tracked by the 
autodifferentiation framework. This is possible if the device is a simulator that is 
coded entiely in the framework's language (such as a TensorFlow quantum simulator).

.. image:: pl_backprop-device.png
    :width: 300px

Most devices, however, are blackboxes with regards to the autodifferentiation framework. 
This means that when the execution on the device begins, autograd, jax, PyTorch and TensorFlow 
tensors need to be converted to formats that the device understands - which is in most cases 
a representation as Numpy arrays. Likewise, the results of the execution have to be translated 
back to differentiable tensors. These two conversions happen at what PennyLane calls the
"interface", and you can specify this interface in the QNode with the ``interface`` keyword argument.  

Gradients
*********

[TODO: Be a bit more precise here]






