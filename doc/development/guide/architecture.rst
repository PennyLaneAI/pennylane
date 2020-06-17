.. role:: html(raw)
   :format: html

Architectural overview
======================

PennyLane allows optimization and machine learning of quantum and hybrid
quantum-classical computations by integrating several key components. A
*quantum device* can be matched with a *quantum node* to return statistics upon
evaluation. To process these, *interfaces* allow using familiar classical
frameworks.

In most cases, computations with PennyLane are performed on a local machine
using a simulator. Through using PennyLane plugins, one can, however, also
utilize remote quantum devices and simulators.

Components of PennyLane
#######################

.. image:: architecture_diagram.png
    :width: 800px

Devices
*******

In PennyLane, the abstraction of a quantum device is encompassed within the
:class:`~.Device` class, making it one of the basic components of the
library. It includes basic functionality that is shared for quantum
devices, independent of the qubit and CV models. PennyLane gives access to
multiple simulators and hardware chips through its plugins; each of these
devices is implemented as a custom class. These classes have the
``Device`` class as their parent class.

The purpose of the ``Device`` class can be summarized as:

* Providing a common API to execute a quantum circuit and request
  the measurement of the associated observable.
* Providing an easy way of developing a new device for PennyLane

Qubit based devices can use shared utilities by using the
:class:`~.QubitDevice`.


QNodes
******

A  quantum  node or ``QNode`` (represented by a subclass to
:class:`~.BaseQNode`) is an encapsulation of a function :math:`f(x;\theta)=R^m\rightarrow R^n`
that is executed using quantum information processing on a quantum
device.

Each ``QNode`` represents the quantum circuit by building a
:class:`~.CircuitGraph` instance, but the way differentiation is done is custom
to the differentiation method offered by the ``QNode``.

For further details on QNodes, and for a full list of QNodes, refer to the 
:doc:`/code/qml_qnodes` module.

Interfaces
**********

The integration between classical and quantum computations is encompassed by
interfaces.

We refer to the :ref:`intro_interfaces` page for a more in-depth introduction
and a list of available interfaces.

Key design details
##################

The following are key design details related to how PennyLane works internally.

Queuing of operators
********************

In PennyLane, the construction of quantum gates is separated from the specific
quantum node (:class:`~.BaseQNode`) that they belong to. However, including
logic for this when creating an instance of :class:`~.Operator` does not align
with the current architecture. Therefore, there is a need to use a high-level
object that holds information about the relationship between quantum gates and
a quantum node.

The :class:`~.QueuingContext` class realizes this by providing access to the current
QNode.  Furthermore, it provides the flexibility to have multiple objects
record the creation of quantum gates.

The ``QueuingContext`` class both acts as the abstract base class for all
classes that expose a queue for Operations (so-called contexts), as well as the
interface to said queues. The active contexts contain maximally one QNode and
an arbitrary number of other contexts like the :class:`~.OperationRecorder`.

Variables
*********

Circuit parameters in PennyLane are tracked and updated using
:class:`~.Variable`. They play a key role in the evaluation of a ``QNode``.

We refer to the :ref:`qml_variable` page for a more in-depth description of how
``Variables`` are used during execution.
