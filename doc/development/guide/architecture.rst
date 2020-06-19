.. role:: html(raw)
   :format: html

Architectural overview
======================

PennyLane allows optimization and machine learning of quantum and hybrid
quantum-classical computations. The library provides a unified architecture for
near-term quantum computing devices, supporting various quantum information
paradigms.

PennyLane's core feature is the ability to compute gradients of variational
quantum circuits in a way that is compatible with classical techniques such as
backpropagation. PennyLane thus extends the automatic differentiation
algorithms common in optimization and machine learning to include quantum and
`hybrid computations
<https://pennylane.ai/qml/glossary/hybrid_computation.html#backpropagation-through-hybrid-computations>`__.
A plugin system makes the framework compatible with any gate-based quantum
simulator or hardware.

Using PennyLane, quantum computing devices (``Devices``) can be used to
evaluate quantum nodes (``QNodes``) and to return statistics of the results. To
process the classical information obtained, ``Interfaces`` allow using
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
<https://packaging.python.org/specifications/entry-points/>`__under the `pennylane.plugins`
namespace using Setuptools. Once registered, the device can be instantiated using the :func:`~.device`
loader function.

A Python package that registers one or more PennyLane device is known as a *plugin*. For more details
on plugins and devices, see :doc:`/development/plugins`.

The purpose of the ``Device`` class can be summarized as:

* Providing a common API to execute a quantum circuit and request
  the measurement of the associated observable.
* Providing an easy way of developing a new device for PennyLane

QNodes
******

A  quantum node or QNode (represented by a subclass of
:class:`~.BaseQNode`) is an encapsulation of a function
:math:`f(x;\theta)=R^m\rightarrow R^n` that is executed using quantum
information processing on a quantum device.

Apart from incorporating quantum functions, QNodes also offer custom
quantum differentiation rules. Using the so-called `parameter-shift rules
<https://pennylane.ai/qml/glossary/parameter_shift.html>`__, many quantum
functions can be expressed through the linear combination of other quantum
functions. As these rules allow quantum gradients to be obtained from
QNodes, hybrid computations may include QNodes as part of training deep
learnings models.

These QNode types are available to users through the :func:`~.qnode` decorator by
passing the user-facing ``diff_method`` option. This decorator then uses the
:func:`~.QNode` constructor function to create the specific type of qnode based on
the device, interface, and quantum function. If ``diff_method`` option is not
provided, the QNode constructor function attempts to determine the ``"best"``
differentiation method, based on the available device and interface.

A widespread representation of quantum circuits is by creating a `Directed
Acyclic Graph (DAG)
<https://pennylane.ai/qml/glossary/hybrid_computation.html#directed-acyclic-graphs>`__
and representing quantum operations within such a graph. Each ``QNode``
represents the quantum circuit by building such a DAG by creating a
:class:`~.CircuitGraph` instance.

For further details on QNodes, and a full list of QNodes with their custom
differentiation rule, refer to the :doc:`/code/qml_qnodes` module.

Interfaces
**********

The integration between classical and quantum computations is encompassed by
interfaces. QNodes that provide black-box gradient rules are 'wrapped' by an interface function.
that provide a 'wrapper' around QNodes such. These wrappers further transform
the ``QNode`` such that the quantum gradient rules of the QNodes are registered
to the machine learning interface via a custom gradient class or function.

An interface integrates QNodes with external libraries by the following:

* It wraps the QNode, returning a QNode that accepts and returns the core data
  structure of the classical machine learning library (e.g., a TF tensor, Torch
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

* the :class:`~.Operation` class representing quantum gates and
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

The following is an example of this using the :func:`~.qnode` decorator and a
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
the surrounding :class:`~.QueuingContext`.

Measurement functions such as :func:`~.expval` are responsible for queuing observables.

For further details, refer to the description in :class:`~.QueuingContext`.

Variables
*********

Circuit parameters in PennyLane are tracked and updated using
:class:`~.Variable`. They play a key role in the evaluation of ``QNode`` gradients, as
the symbolic parameters are substituted with numeric values. The ``Variable`` class plays
an important role in book-keeping, allowing PennyLane to keep track of which parameters are
used in which operations, and automatically perform the product and chain rule where required.

We refer to the :doc:`/code/qml_variable` page for a more in-depth description of how
``Variables`` are used during execution.
