.. role:: html(raw)
   :format: html

Architectural overview
======================

PennyLane allows optimization and machine learning of quantum and hybrid
quantum-classical computations. The library provides a unified architecture for
near-term quantum computing devices, supporting various quantum information
paradigms.

PennyLane’s core feature is the ability to compute gradients of variational
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

* Retrieving measurement and state results (including samples, measurement statistics, and probabilities).

There are also device subclasses available, containing shared logic for particular types of devices.
For example, qubit-based devices can inherit from the :class:`~.QubitDevice` class, easing development.

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

Qubit based devices can use shared utilities by using the
:class:`~.QubitDevice`.

QNodes
******

A  quantum node or QNode (represented by a subclass of
:class:`~.BaseQNode`) is an encapsulation of a function
:math:`f(x;\theta)=R^m\rightarrow R^n` that is executed using quantum
information processing on a quantum device.

Apart from incorporating quantum functions, ``QNodes`` also offer custom
quantum differentiation rules. Using the so-called `parameter-shift rules
<https://pennylane.ai/qml/glossary/parameter_shift.html>`__, many quantum
functions can be expressed through the linear combination of other quantum
functions. As these rules allow quantum gradients to be obtained from
``QNodes``, hybrid computations may include ``QNodes`` as part of training deep
learnings models.

PennyLane offers the following qnode types and differentiation rules:

* :class:`~.QubitQNode`: qubit parameter-shift rule
* :class:`~.CVQNode`: CV parameter-shift rule
* :class:`~.JacobianQNode`: finite differences
* :class:`~.DeviceJacobianQNode`: queries the device directly for the gradient
* :class:`~.PassthruQNode`: classical backpropagation
* :class:`~.ReversibleQNode`: reversible backpropagation

These QNode types are available to users through the :func:`~.qnode` decorator by
passing the user-facing ``diff_method`` option. This decorator then uses the
:func:`~.QNode` constructor function to create the specific type of qnode based on
the device, interface, and quantum function.

A widespread representation of quantum circuits is by creating a `Directed
Acyclic Graph (DAG)
<https://pennylane.ai/qml/glossary/hybrid_computation.html#directed-acyclic-graphs>`__
and representing quantum operations within such a graph. Each ``QNode``
represents the quantum circuit by building such a DAG by creating a
:class:`~.CircuitGraph` instance.

For further details on QNodes, and a full list of QNodes, refer to the
:doc:`/code/qml_qnodes` module.

Interfaces
**********

The integration between classical and quantum computations is encompassed by
interfaces. QNodes that provide black-box gradient rules are 'wrapped' by an interface function.
that provide a 'wrapper' around QNodes such. These wrappers further transform
the ``QNode`` such that the quantum gradient rules of the QNodes are registered
to the machine learning interface via a custom gradient class or function.

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
:attr:`~.Operator.matrix`, :attr:`~.Operator.eigvals`.

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

This syntax of PennyLane results in the Operator instances not being "recorded"
by the Python function. What is more, there is actually *no composition logic*
in between the ``QNode`` that is being created and the ``Operator`` instances
(``qml.PauliX(0)`` and ``qml.PauliZ(0)``), as there are merely class
instantiations happening upon calling the ``circuit`` quantum function.

How does then PennyLane keep track of which ``Operations`` and ``Observables``
were instantiated within a quantum function? The solution to this problem is
related to the queuing of operators.

Queuing of operators
********************

In PennyLane, the construction of quantum gates is separated from the specific
quantum node (:class:`~.BaseQNode`) that they belong to. The reason for this is
that, as mentioned before, the syntax of how ``Operator`` instances are
included in quantum functions does not rely on or use ``QNode`` instances.
Therefore, ``QNodes`` cannot keep track by default, which operators were
included in the quantum function. A high-level object that holds information
about the relationship between quantum gates and a quantum node can, however,
help with this.

The :class:`~.QueuingContext` class realizes this by providing access to the
current QNode using the concept of a Python context manager. This is achieved
by subclassing the ``BaseQNode`` class from ``QueuingContext``.  Furthermore, it
provides the flexibility to have multiple objects record the creation of
quantum gates.

Details of queueing and specific operators can be described as follows:

* ``Operators`` in a quantum function are queued on initialization.  This
  happens via a call to :meth:`Operator.queue` which then interacts with the
  ``QueuingContext``;
* ``Observables`` do not queue themselves; the measurement functions
  :meth:`~.measure.expval`, :meth:`~.measure.var` etc. do the queueing. Hence, observables
  can be instantiated before queueing;
* Already instantiated operators cannot be re-queued unless the ``op.queue``
  method is manually called within the ``QueuingContext``.

The ``QueuingContext`` class both acts as the abstract base class for all
classes that expose a queue for Operations (so-called contexts), as well as the
interface to these queues. The active contexts contain maximally one QNode and
an arbitrary number of other contexts like the :class:`~.OperationRecorder`.

``OperationRecorders`` are context managers that help record the quantum
operations instantiated within the context and can play a key role in testing.

Variables
*********

Circuit parameters in PennyLane are tracked and updated using
:class:`~.Variable`. They play a key role in the evaluation of a ``QNode`` as
the symbolic parameters are substituted with numeric values.

We refer to the :ref:`qml_variable` page for a more in-depth description of how
``Variables`` are used during execution.
