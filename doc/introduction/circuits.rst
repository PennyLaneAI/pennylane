 .. role:: html(raw)
   :format: html


.. _intro_vcircuits:

Quantum circuits
================


.. image:: ../_static/qnode.png
    :align: right
    :width: 180px
    :target: javascript:void(0);


In PennyLane, quantum computations, which involve the execution of one or more quantum circuits,
are represented as *quantum node* objects. A quantum node is used to
declare the quantum circuit, and also ties the computation to a specific device that executes it.

QNodes can interface with any of the supported numerical and machine learning libraries---:doc:`NumPy <interfaces/numpy>`,
:doc:`PyTorch <interfaces/torch>`, :doc:`TensorFlow <interfaces/tf>`, and
:doc:`JAX <interfaces/jax>`---indicated by providing an optional ``interface`` argument
when creating a QNode. Each interface allows the quantum circuit to integrate seamlessly with
library-specific data structures (e.g., NumPy and JAX arrays or Pytorch/TensorFlow tensors) and
:doc:`optimizers <interfaces>`.

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
  :doc:`quantum operators <operations>` or sequences of operators called :doc:`templates`,
  using one instruction per line.

* The function can contain classical flow control structures such as ``for`` loops or ``if`` statements.

* The quantum function must always return either a single or a tuple of
  *measured observable values*, by applying a :doc:`measurement function <measurements>`
  to a :ref:`qubit observable <intro_ref_ops_qobs>` or :ref:`continuous-value observable <intro_ref_ops_cvobs>`.

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

PennyLane offers some basic devices such as the ``'default.qubit'``, ``'default.mixed'``, ``lightning.qubit``,
and ``'default.gaussian'`` simulators; additional devices can be installed as plugins (see
`available plugins <https://pennylane.ai/plugins.html>`_ for more details). Note that the
choice of a device significantly determines the speed of your computation, as well as
the available options that can be passed to the device loader.

.. note::

    For example, check out the ``'lightning.qubit'`` `plugin <https://github.com/PennyLaneAI/pennylane-lightning>`_,
    which is a fast state-vector simulator supporting GPUs.

.. note::

    For details on saving device configurations, please visit the
    :doc:`configurations page</introduction/configuration>`.

Device options
^^^^^^^^^^^^^^

When loading a device, the name of the device must always be specified.
Further options can then be passed as keyword arguments, and can differ based
on the device. For a plugin device, refer to the plugin documentation for available device options.

The two most important device options are the ``wires`` and ``shots`` arguments.

Wires
*****

The wires argument can either be an integer that defines the *number of wires*
that you can address by consecutive integer labels ``0, 1, 2, ...``.

.. code-block:: python

    dev = qml.device('default.qubit', wires=3)

Alternatively, you can use custom labels by passing an iterable that contains unique labels for the subsystems:

.. code-block:: python

    dev = qml.device('default.qubit', wires=['aux', 'q1', 'q2'])

In the quantum function you can now use your own labels to address wires:

.. code-block:: python

    def my_quantum_function(x, y):
        qml.RZ(x, wires='q1')
        qml.CNOT(wires=['aux' ,'q1'])
        qml.RY(y, wires='q2')
        return qml.expval(qml.PauliZ('q2'))

Allowed wire labels can be of any type that is hashable, which allows two wires to be uniquely distinguished.

.. note::

    Some devices, such as hardware chips, may have a fixed number of wires.
    The iterable of labels passed to the device's ``wires``
    argument must match this expected number of wires.

Shots
*****

The ``shots`` argument is an integer that defines how many times the circuit should be evaluated (or "sampled")
to estimate statistical quantities. On some supported simulator devices, ``shots=None`` computes
measurement statistics *exactly*.

Note that this argument can be temporarily overwritten when a QNode is called. For example, ``my_qnode(shots=3)``
will temporarily evaluate ``my_qnode`` using three shots.

It is sometimes useful to retrieve the result of a computation for different shot numbers without evaluating a
QNode several times ("shot batching"). Batches of shots can be specified by passing a list of integers,
allowing measurement statistics to be course-grained with a single QNode evaluation.

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
transform,

>>> print(qml.draw(circuit)(np.pi/4, 0.7))
aux: ───────────╭●─┤     
 q1: ──RZ(0.79)─╰X─┤     
 q2: ──RY(0.70)────┤  <Z>

or the :func:`~.pennylane.draw_mpl` transform:

>>> import matplotlib.pyplot as plt
>>> fig, ax = qml.draw_mpl(circuit)(np.pi/4, 0.7)
>>> plt.show()

.. image:: ../_static/draw_mpl.png
    :align: center
    :width: 300px
    :target: javascript:void(0);

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
