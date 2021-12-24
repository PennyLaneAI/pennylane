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
part of a larger hybrid quantum-classical computation.

QNodes run **quantum circuits** on **devices**.
These devices may be simulators built into PennyLane, or external devices
provided by plugins. The user specifies a quantum circuit by defining a "quantum function",
which is a Python function that contains quantum operations and measurements
represented by the :class:`~.Operator` and :class:`~.MeasurementProcess` classes,
respectively.


.. image:: pl_overview.png
    :width: 800px

The power of a QNode lies in the fact that it can be run in a "forward" fashion to
execute the quantum circuit, or in a "backward" fashion in which it provides
gradients (or, more precisely, *jacobian-vector products*).

Let's go through these components one-by-one.

Operator
********

Quantum operators are represented by the :class:`~.Operator` class.

Operators are uniquely defined by their name, their (trainable) parameters,
their (non-trainable) hyperparameters, and the wires they act on.

These four properties are accessible for all operators:

.. code-block:: python

    >>> from jax import numpy as jnp
    >>> op = qml.PauliRot(jnp.array(0.2), "XY", wires=["a", "b"])
    >>> op.name
    PauliRot
    >>> op.parameters
    [DeviceArray(0.2, dtype=float32, weak_type=True)]
    >>> op.hyperparameters
    {'pauli_word': 'XY'}
    >>> op.wires
    <Wires = ['a', 'b']>

Furthermore, operators can optionally define the transformation they implement via
symbolic or numerical representations, such as:

.. code-block:: python

    >>> # representation as a product of operators
    >>> op = qml.Rot(0.1, 0.2, 0.3, wires=["a"])
    >>> op.decomposition()
    [RZ(0.1, wires=['a']), RY(0.2, wires=['a']), RZ(0.3, wires=['a'])]

    >>> # representation as a linear combination of operators
    >>> op = qml.Hamiltonian([1., 2.], [qml.PauliX(0), qml.PauliZ(0)])
    >>> op.terms()
    ((1.0, 2.0), [PauliX(wires=[0]), PauliZ(wires=[0])])

    >>> # representation by the eigenvalue decomposition
    >>> op = qml.PauliX(0)
    >>> op.diagonalizing_gates()
    [Hadamard(wires=[0])]
    >>> op.eigvals()
    [ 1 -1]

    >>> # representation as a matrix
    >>> op = qml.PauliRot(0.2, "X", wires=["b"])
    >>> op.matrix()
    [[9.95004177e-01-2.25761781e-18j 2.72169462e-17-9.98334214e-02j]
     [2.72169462e-17-9.98334214e-02j 9.95004177e-01-2.25761781e-18j]]

    >>> # representation as a sparse matrix
    >>> from scipy.sparse.coo import coo_matrix
    >>> row = np.array([0, 1])
    >>> col = np.array([1, 0])
    >>> data = np.array([1, -1])
    >>> mat = coo_matrix((data, (row, col)), shape=(4, 4))
    >>> op = qml.SparseHamiltonian(mat, wires=["a"])
    >>> op.sparse_matrix()
    (0, 1)   1
    (1, 0) - 1

If a representation is not defined, a custom error (such as a ``DecompositionUndefinedError``)
is raised.

Devices use the information provided by the properties and representations
to implement the operator.

MeasurementProcess
******************

While the :class:`~.Operator` class describes a physical system and its dynamics,
the :class:`~.MeasurementProcess` class describes how we extract information from the quantum system.
The object returned by a quantum function, such as :func:`~.expval` creates an instance of this class.

The class takes a return type upon initialization, which specifies the kind of measurement performed.
PennyLane supports the following return types: Expectation, Variance, Probability, State, Sample.

QuantumTape
***********

Quantum operators and measurement processes can be used to build a quantum circuit.
The user defines the circuit by constructing a quantum function.

.. code-block:: python

    def qfunc(params):
        qml.RX(params[0], wires='b')
        qml.CNOT(wires=['a', 'b'])
        qml.RY(params[1], wires='a')
        return qml.expval(qml.PauliZ(wires='b'))

Internally, a quantum function is translated to a quantum tape, which is
the central representation of a quantum circuit. The tape is a context manager that stores lists
of :class:`~.Operator` and :class:`~.MeasurementProcesses` instances.
Creating operations inside a tape context adds them to these lists.

For example, if we call the quantum function in a tape context, the
gates are stored in the tape's ``operation`` property, while the
measurement processes such as :func:`~.expval` are responsible for adding observables
to the tape's ``measurement`` property.

.. code-block:: python

    >>> with qml.tape.QuantumTape() as tape:
    ...	    qfunc(params)

    >>> tape.operations
    [RX(DeviceArray(0.5, dtype=float32), wires=['b']),
     CNOT(wires=['a', 'b']),
     RY(DeviceArray(0.2, dtype=float32), wires=['a'])]

    >>> tape.measurements
    [expval(PauliZ(wires=['b']))]

These two "queues" are used by devices to retrieve a circuit.

.. note::

    Tapes can represent parts of quantum circuits and do not necessarily need to define a measurement.
    They can also be nested.

Devices
*******

In PennyLane, the abstraction of a quantum computation device is encompassed
within the :class:`~.Device` class. The main job of devices is to
interpret and execute tapes. The most important method is ``batch_execute``,
which executes a list of tapes, such as the one created above:

.. code-block:: python

    >>> device = qml.device("default.qubit", wires=['a', 'b'], shots=None)
    >>> device.batch_execute([tape])
    [array([0.87758256])]

There are also device subclasses available, containing shared logic for
particular types of devices.  For example, qubit-based devices can inherit from
the :class:`~.QubitDevice` class, easing development.

To register a new device with PennyLane, a device subclass has to be created and registered
as an `entry point <https://packaging.python.org/specifications/entry-points/>`__ under the `pennylane.plugins`
namespace using Setuptools. Once registered, the device can be instantiated using the :func:`~.device`
loader function, using the device's name.

A Python package that registers one or more PennyLane devices is known as a *plugin*. For more details
on plugins and devices, see :doc:`/development/plugins`.

QNodes
******

This is where it all comes together: A **QNode** is an encapsulation of a function
:math:`f(x;\theta)=R^m\rightarrow R^n` that is executed using quantum
information processing on a quantum device. It is created by a quantum function and a device.

.. code-block:: python

    >>> import jax
    >>> from jax import numpy as jnp
    >>> params = jnp.array([0.5, 0.2])

    >>> qnode = qml.QNode(qfunc, device, interface='jax')
    >>> qnode(params)
    0.8776

    >>> jax.grad(qnode)
    [-0.4794  0.]

    # transforms create new functions from qnodes
    >>> qnode_drawer = qml.transforms.draw(qnode)
    >>> qnode_drawer(params)
    a: ───────────╭C──RY(0.2)──┤
    b: ──RX(0.5)──╰X───────────┤ ⟨Z⟩


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
