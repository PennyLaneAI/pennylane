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
A plugin system makes the framework compatible with any gate-based quantum
simulator or hardware.

Using PennyLane, quantum computing *devices* can be used to
evaluate quantum nodes (*QNodes*) and to return statistics of the results. To
process the classical information obtained, *interfaces* allow using
accelerated machine learning libraries.

In most cases, computations with PennyLane are performed on a local machine
using a simulator. Through using PennyLane plugins, one can, however, also
utilize remote quantum devices and simulators.

Components of PennyLane
#######################

.. image:: architecture_diagram.png
    :width: 800px

Devices
*******

In PennyLane, the abstraction of a quantum computation device is encompassed
within the :class:`~.Device` class, making it one of the basic components of
the library. The :class:`~.Device` class provides a common API for accessing quantum
devices, independent of both the type of device (both simulators and hardware are supported),
as well as the quantum information model used.

In particular, the :class:`~.Device` class provides a common API for:

* Initializing various device parameters such as the number of samples/shots,

* Executing quantum circuits,

* Retrieving measurement and state results (including samples, measurement
  statistics, and probabilities).

There are also device subclasses available, containing shared logic for
particular types of devices.  For example, qubit-based devices can inherit from
the :class:`~.QubitDevice` class, easing development.

To register a new device with PennyLane, they must register an `entry point
<https://packaging.python.org/specifications/entry-points/>`__ under the `pennylane.plugins`
namespace using Setuptools. Once registered, the device can be instantiated using the :func:`~.device`
loader function.

A Python package that registers one or more PennyLane devices is known as a *plugin*. For more details
on plugins and devices, see :doc:`/development/plugins`.

The purpose of the :class:`Device` class can be summarized as:

* Providing a common API to execute a quantum circuit and request
  the measurement of the associated observable.
* Providing an easy way of developing a new device for PennyLane

QNodes
******

A  quantum node or QNode (represented by a subclass of
:class:`~.BaseQNode`) is an encapsulation of a function
:math:`f(x;\theta)=R^m\rightarrow R^n` that is executed using quantum
information processing on a quantum device.

Users don't typically instantiate QNodes directly---instead, the :func:`~pennylane.qnode` decorator or
:func:`~pennylane.QNode` constructor function automates the process of creating a QNode from a provided
quantum function and device. The constructor attempts to determine the ``"best"``
differentiation method for the provided device and interface. For more fine-grained control,
the differentiation method can be specified directly via the ``diff_method`` option.

Tapes
*****

Internally, QNodes store the details of the quantum processing using a datastructure called the
*tape*. Apart from encapsulating quantum processing, tapes also provide custom quantum
differentiation rules. Examples include the :doc:`parameter-shift rule <glossary/parameter_shift>`,
where the derivative of a tape can be expressed by a linear combination of other tapes plus
classical post-processing. As these rules allow quantum gradients to be obtained from tapes, hybrid
computations may include QNodes as part of training deep learning models.

A common representation of quantum circuits is a `Directed Acyclic Graph (DAG)
<https://pennylane.ai/qml/glossary/hybrid_computation.html#directed-acyclic-graphs>`__ where quantum
operations are nodes within the graph. Each tape builds such a DAG using a :class:`~.CircuitGraph`
instance.

For further details on tapes, and a full list of tapes with their custom
differentiation rules, refer to the :doc:`/code/qml_tape` module.

Interfaces
**********

The integration between classical and quantum computations is encompassed by
interfaces. QNodes that provide black-box gradient rules are 'wrapped' by an interface function.
These wrappers further transform
the ``QNode`` such that the quantum gradient rules of the QNodes are registered
to the machine learning interface via a custom gradient class or function.

Typically, an interface integrates QNodes with external libraries as follows:

* It wraps the QNode, returning a QNode that accepts and returns the core data
  structure of the classical machine learning library (e.g., a TensorFlow or PyTorch
  tensor, Autograd NumPy array, etc).

* It unwraps the input data structures to simple NumPy arrays, so that the
  quantum device can execute the user's quantum function.

* It registers the ``QNode.jacobian()`` method as a custom gradient method, so that
  the machine learning library can 'backpropagate' across the QNode, when
  integrated into a classical computation.

We refer to the :ref:`intro_interfaces` page for a more in-depth introduction
and a list of available interfaces.

Key design details
##################

The following are key design details related to how PennyLane works internally.

Operators
*********

Quantum operators are incorporated by the :class:`~.Operator` class which
contains basic information about the operator (e.g. number of parameters,
number of wires it acts on, etc.) and further convenience methods (e.g.
:attr:`~.Operator.matrix`, :attr:`~.Operator.eigvals`).

Two important subclasses of the ``Operator`` class are:

* the :class:`~.Operation` class representing quantum gates,
* the :class:`~.Observable` representing quantum observables specified for
  measurement.

Together with ``Operator``, these classes serve as base classes for quantum
operators.

Certain operators can serve as both quantum gates and observables (e.g.
:class:`~.PauliZ`, :class:`~.PauliX`, etc.). Such classes inherit from both
``Operation`` and ``Observable`` classes.

Quantum operators are used to build quantum functions
which are evaluated by a ``QNode`` on a bound device. Users can define such quantum
functions by creating regular Python functions and instantiating ``Operator``
instances in temporal order, one per line.

The following is an example of this using the :func:`~pennylane.qnode` decorator and a
valid pre-defined device (``dev``).

.. code-block:: python

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(0)
        return qml.expval(qml.PauliZ(0))

Queuing of operators
********************

In PennyLane, the construction of quantum gates is separated from the specific
QNode that they belong to. QNode circuit construction happens only when the
QNode is evaluated. On QNode evaluation, the quantum function is executed.

Operators are queued to the QNode on instantiation, by having :meth:`.Operator.__init__`
call the :meth:`.Operator.queue` method. The operators themselves queue themselves to
the surrounding :class:`~.pennylane.QueuingContext`.

Measurement functions such as :func:`~.pennylane.expval` are responsible for queuing observables.

For further details, refer to the description in :class:`~.pennylane.QueuingContext`.
