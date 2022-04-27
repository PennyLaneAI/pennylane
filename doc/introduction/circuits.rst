 .. role:: html(raw)
   :format: html


.. _intro_vcircuits:

Quantum circuits
================


.. image:: ../_static/qnode.png
    :align: right
    :width: 180px
    :target: javascript:void(0);


In PennyLane, quantum computations are represented as *quantum node* objects. A quantum node is used to
declare the quantum circuit, and also ties the computation to a specific device that executes it.
Quantum nodes can be easily created by using the :ref:`qnode <intro_vcirc_decorator>` decorator.

QNodes can interface with any of the supported numerical and machine learning libraries---:doc:`NumPy <interfaces/numpy>`,
:doc:`PyTorch <interfaces/torch>`, :doc:`TensorFlow <interfaces/tf>`, and
`JAX <https://github.com/google/jax>`__---indicated by providing an optional ``interface`` argument
when creating a QNode. Each interface allows the quantum circuit to integrate seamlessly with
library-specific data structures (e.g., NumPy arrays, or Pytorch/TensorFlow tensors) and
:doc:`optimizers <optimizers>`.

By default, QNodes use the NumPy interface. The other PennyLane interfaces are
introduced in more detail in the section on :doc:`interfaces <interfaces>`.


.. _intro_vcirc_qfunc:

Quantum functions
-----------------

A quantum circuit is constructed as a special Python function, a
*quantum circuit function*, or *quantum function* in short.
For example:

.. code-block:: python

    import pennylane as qml

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(y, wires=1)
        return qml.expval(qml.PauliZ(1))

.. note::

    PennyLane uses the term *wires* to refer to a quantum subsystem---for most
    devices, this corresponds to a qubit. For continuous-variable
    devices, a wire corresponds to a quantum mode.

Quantum functions are a restricted subset of Python functions, adhering to the following
constraints:

* The quantum function accepts classical inputs, and consists of
  :doc:`quantum operations <operations>` or sequences of operations called :doc:`templates`,
  using one instruction per line.

* The function can contain classical flow control structures such as ``for`` loops,
  but in general they must not depend on the parameters of the function.

* The quantum function must always return either a single or a tuple of
  *measured observable values*, by applying a :doc:`measurement function <measurements>`
  to a :ref:`qubit observable <intro_ref_ops_qobs>` or :ref:`continuous-value observable <intro_ref_ops_cvobs>`.

.. note::

    Measured observables **must** come after all other operations at the end
    of the circuit function as part of the return statement, and cannot appear in the middle.

.. note::

    Quantum functions can only be evaluated on a device from within a QNode.



.. _intro_vcirc_device:

Defining a device
-----------------

To run---and later optimize---a quantum circuit, one needs to first specify a *computational device*.

The device is an instance of the :class:`~.pennylane.Device`
class, and can represent either a simulator or hardware device. They can be
instantiated using the :func:`device <pennylane.device>` loader.

.. code-block:: python

    dev = qml.device('default.qubit', wires=2, shots=1000)

PennyLane offers some basic devices such as the ``'default.qubit'`` and ``'default.gaussian'``
simulators; additional devices can be installed as plugins (see
`available plugins <https://pennylane.ai/plugins.html>`_ for more details). Note that the
choice of a device significantly determines the speed of your computation, as well as
the available options that can be passed to the device loader.

Device options
^^^^^^^^^^^^^^

When loading a device, the name of the device must always be specified.
Further options can then be passed as keyword arguments; these options can differ based
on the device. For the in-built ``'default.qubit'`` and ``'default.gaussian'``
devices, the options are:

* ``wires`` (*int* or *Iterable*): Number of subsystems represented by the device,
  or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
  and/or strings (``['ancilla', 'q1', 'q2']``).

* ``shots`` (*None* or *int* or *list[int]*): How many times the circuit should be evaluated (or
  sampled) to estimate statistical quantities. On some supported simulator devices,
  ``shots=None`` indicates to compute measurement statistics *exactly*. Note that this
  argument can be temporarily overwritten when a QNode is called. For example, ``my_qnode(shots=3)``
  will temporarily evaluate ``my_qnode`` using three shots.

For a plugin device, refer to the plugin documentation for available device options.

Shot batches
^^^^^^^^^^^^

Batches of shots can be specified by passing a list, allowing measurement statistics
to be course-grained with a single QNode evaluation.

Consider

>>> shots_list = [5, 10, 1000]
>>> dev = qml.device("default.qubit", wires=2, shots=shots_list)

When QNodes are executed on this device, a single execution of 1015 shots will be submitted.
However, three sets of measurement statistics will be returned; using the first 5 shots,
second set of 10 shots, and final 1000 shots, separately.

For example:

.. code-block:: python

    @qml.qnode(dev)
    def circuit(x):
      qml.RX(x, wires=0)
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))

Executing this, we will get an output of size ``(3, 2)``:

>>> circuit(0.5)
tensor([[ 1.   ,  1.   ],
        [ 0.2  ,  1.   ],
        [-0.022,  0.876]], requires_grad=True)

Custom wire labels
^^^^^^^^^^^^^^^^^^

When you create a device by passing an integer to the ``wires`` argument, the integer defines the *number of wires*
that you can address by consecutive integer labels ``0, 1, 2, ...``.

But you can define your own wire labels instead, which may be handy if wires have "meanings" like an
ancilla or garbage register, if they are arranged in a non-linear fashion like a grid, or if there are wires
that you want to skip because they do not work on a hardware device.

This is done by passing an iterable of wire labels to the ``wires`` argument:

.. code-block:: python

    dev = qml.device('default.qubit', wires=['wire1', 'wire2'])

In the quantum function you can now use your own labels to address wires:

.. code-block:: python

    def my_quantum_function(x, y):
        qml.RZ(x, wires='wire1')
        qml.CNOT(wires=['wire1' ,'wire2'])
        qml.RY(y, wires='wire2')
        return qml.expval(qml.PauliZ('wire2'))

Allowed wire labels can be of the following types:

* *strings* like ``wires=['a', 'd', 'b', ...]`` or ``wires=['auxiliary', 'q1', 'q2', ...]``,

* *integers* like ``wires=[0, 4, 7]`` or even ``wires=[-1, 0, 4]``

* *floats* and other *numbers* like ``wires=[1., 2., 4.]``

* *mixed types* like ``wires=['auxiliary', -1, 0, 'q3']``

.. note::

    Some devices, such as hardware chips, may have a fixed number of wires.
    The iterable of labels passed to the device's ``wires``
    argument must match this expected number of wires.

.. _intro_vcirc_qnode:

Creating a quantum node
-----------------------

Together, a quantum function and a device are used to create a *quantum node* or
:class:`~.pennylane.QNode` object, which wraps the quantum function and binds it to the device.

A QNode can be explicitly created as follows:

.. code-block:: python

    circuit = qml.QNode(my_quantum_function, dev)

The QNode can be used to compute the result of a quantum circuit as if it was a standard Python
function. It takes the same arguments as the original quantum function:

>>> circuit(np.pi/4, 0.7)
tensor(0.764, requires_grad=True)

To view the quantum circuit given specific parameter values, we can use the :func:`~.pennylane.draw`
transform:

>>> print(qml.draw(circuit)(np.pi/4, 0.7))
wire1: ──RZ(0.79)─╭C───────────┤     
wire2: ───────────╰X──RY(0.70)─┤  <Z>

.. _intro_vcirc_decorator:

The QNode decorator
-------------------

A more convenient---and in fact the recommended---way for creating QNodes is the provided
``qnode`` decorator. This decorator converts a Python function containing PennyLane quantum
operations to a :class:`~.pennylane.QNode` circuit that will run on a quantum device.

.. note::
    The decorator completely replaces the Python-based quantum function with
    a :class:`~.pennylane.QNode` of the same name---as such, the original
    function is no longer accessible.

For example:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(x):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(x, wires=1)
        return qml.expval(qml.PauliZ(1))

    result = circuit(0.543)


Collections of QNodes
---------------------

Sometimes you may need multiple QNodes that only differ in the measurement observable
(like in VQE), or in the device they are run on (for example, if you benchmark different devices),
or even the quantum circuit that is evaluated. While these QNodes can be defined manually
"by hand", PennyLane offers **QNode collections** as a convenient way to define and run
families of QNodes.

QNode collections are a sequence of QNodes that:

1. Have the same function signature, and

2. Can be evaluated independently (that is, the input of any QNode in the collection
   does not depend on the output of another).

Consider the following two quantum nodes:


.. code-block:: python

    @qml.qnode(dev1)
    def x_rotations(params):
        qml.RX(params[0], wires=0)
        qml.RX(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev2)
    def y_rotations(params):
        qml.RY(params[0], wires=0)
        qml.RY(params[1], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.Hadamard(0))

As the QNodes in the collection have the same signature, and we can can construct a
:class:`~.QNodeCollection` and therefore feed them the same parameters:

>>> qnodes = qml.QNodeCollection([x_rotations, y_rotations])
>>> len(qnodes)
2
>>> qnodes([0.2, 0.1])
array([0.98006658, 0.70703636])

PennyLane also provides some high-level tools for creating and evaluating
QNode collections. For example, :func:`~.map` allows a single
function of quantum operations (or :doc:`template <templates>`) to be mapped across
multiple observables or devices.

For example, consider the following quantum function ansatz:

.. code-block:: python

    def my_ansatz(params, **kwargs):
        qml.RX(params[0], wires=0)
        qml.RX(params[1], wires=1)
        qml.CNOT(wires=[0, 1])

We can define a list of observables, and two devices:

>>> obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliX(1)]
>>> qpu1 = qml.device("forest.qvm", device="Aspen-4-4Q-D") # requires PennyLane-Forest
>>> qpu2 = qml.device("forest.qvm", device="Aspen-7-4Q-B") # requires PennyLane-Forest

.. note::

    The two devices above require the `PennyLane-Forest plugin <https://pennylane-forest.rtfd.io>`_
    be installed, as well as the Forest QVM. You can also try replacing them with alternate devices.

Mapping the template across the observables and devices creates a :class:`~.QNodeCollection`:

>>> qnodes = qml.map(my_ansatz, obs_list, [qpu1, qpu2], measure="expval")
>>> type(qnodes)
pennylane.collections.qnode_collection.QNodeCollection
>>> params = [0.54, 0.12]
>>> qnodes(params)
array([-0.02854835  0.99280864])
Functions are available to process QNode collections, including :func:`~.pennylane.collections.dot`,
:func:`~.pennylane.collections.sum`, and :func:`~.pennylane.collections.apply`:

>>> cost_fn = qml.sum(qnodes)
>>> cost_fn(params)
0.906

.. note::

    QNode collections support an experimental parallel execution mode. See
    the :class:`~.QNodeCollection` documentation for more details.


Importing circuits from other frameworks
----------------------------------------

PennyLane supports creating customized PennyLane templates imported from other
frameworks. By loading your existing quantum code as a PennyLane template, you
add the ability to perform analytic differentiation, and interface with machine
learning libraries such as PyTorch and TensorFlow. Currently, ``QuantumCircuit``
objects from Qiskit, OpenQASM files, pyQuil ``programs``, and Quil files can
be loaded by using the following functions:

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.from_qiskit
    ~pennylane.from_qasm
    ~pennylane.from_qasm_file
    ~pennylane.from_pyquil
    ~pennylane.from_quil
    ~pennylane.from_quil_file

:html:`</div>`

.. note::

    To use these conversion functions, the latest version of the PennyLane-Qiskit
    and PennyLane-Forest plugins need to be installed.

Objects for quantum circuits can be loaded outside or directly inside of a
:class:`~.pennylane.QNode`. Circuits that contain unbound parameters are also
supported. Parameter binding may happen by passing a dictionary containing the
parameter-value pairs.

Once a PennyLane template has been created from such a quantum circuit, it can
be used similarly to other :doc:`templates <templates>` in PennyLane. One important thing to note
is that custom templates must always be executed
within a :class:`~.pennylane.QNode` (similar to pre-defined templates).

.. note::
    Certain instructions that are specific to the external frameworks might be
    ignored when loading an external quantum circuit. Warning messages will
    be emitted for ignored instructions.

The following is an example of loading and calling a parametrized Qiskit ``QuantumCircuit`` object
while using the :class:`~.pennylane.QNode` decorator:

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    import numpy as np

    dev = qml.device('default.qubit', wires=2)

    theta = Parameter('θ')

    qc = QuantumCircuit(2)
    qc.rz(theta, [0])
    qc.rx(theta, [0])
    qc.cx(0, 1)

    @qml.qnode(dev)
    def quantum_circuit_with_loaded_subcircuit(x):
        qml.from_qiskit(qc)({theta: x})
        return qml.expval(qml.PauliZ(0))

    angle = np.pi/2
    result = quantum_circuit_with_loaded_subcircuit(angle)

Furthermore, loaded templates can be used with any supported device, any number of times.
For instance, in the following example a template is loaded from a QASM string,
and then used multiple times on the ``forest.qpu`` device provided by PennyLane-Forest:

.. code-block:: python

    import pennylane as qml

    dev = qml.device('forest.qpu', wires=2)

    hadamard_qasm = 'OPENQASM 2.0;' \
                    'include "qelib1.inc";' \
                    'qreg q[1];' \
                    'h q[0];'

    apply_hadamard = qml.from_qasm(hadamard_qasm)

    @qml.qnode(dev)
    def circuit_with_hadamards():
        apply_hadamard(wires=[0])
        apply_hadamard(wires=[1])
        qml.Hadamard(wires=[1])
        return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

    result = circuit_with_hadamards()
