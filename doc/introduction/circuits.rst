 .. role:: html(raw)
   :format: html


.. _intro_vcircuits:

Quantum Circuits
================


.. image:: ../_static/qnode.png
    :align: right
    :width: 180px
    :target: javascript:void(0);


In PennyLane, quantum circuits are represented as *quantum node* objects. A quantum node is used to
declare the quantum circuit, and also ties the computation to a specific device that executes it.
Quantum nodes can be easily created by using the :ref:`qnode <intro_vcirc_decorator>` decorator.

QNodes can interface with any of the supported numerical and machine learning libraries---:ref:`NumPy <numpy_interf>`, :ref:`PyTorch <torch_interf>`, and :ref:`TensorFlow <tf_interf>`---indicated by providing an optional ``interface`` argument when creating a QNode. Each interface allows the quantum circuit to integrate seamlessly with library-specific data structures (e.g., NumPy arrays, or Pytorch/TensorFlow tensors) and :ref:`optimizers <intro_ref_opt>`.

By default, QNodes use the NumPy interface. The other PennyLane interfaces are
introduced in more detail in the section on :ref:`interfaces <intro_interfaces>`.


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

    PennyLane uses the term '*wires*' to refer to a quantum subsystem---for most
    devices, this corresponds to a qubit. For continuous-variable
    devices, a wire corresponds to a quantum mode.

Quantum functions are a restricted subset of Python functions, adhering to the following
constraints:

* The quantum function accepts classical inputs, and consists of
  :doc:`operations` or sequences of operations called :doc:`templates`,
  using one instruction per line.

* The quantum function must always return either a single or a tuple of
  *measured observable values*, by applying a :ref:`measurement function <intro_ref_meas>`
  to a :ref:`qubit <intro_ref_ops_qobs>` or :ref:`continuous-value observable <intro_ref_ops_cvobs>`.

* Classical processing of function arguments, either by arithmetic operations
  or external functions, is not allowed. One current exception is simple scalar
  multiplication.

.. note::

    Quantum operations can only be executed on a device from within a QNode.

.. note::

    Measured observables **must** come after all other operations at the end
    of the circuit function as part of the return statement, and cannot appear in the middle.


.. _intro_vcirc_device:

Defining a device
-----------------

To run---and later optimize---a quantum circuit, one needs to first specify a *computational device*.

The device is an instance of the :class:`~.pennylane.Device`
class, and can represent either a simulator or hardware device. They can be
instantiated using the :func:`device <pennylane.device>` loader.

.. code-block:: python

    dev = qml.device('default.qubit', wires=2, shots=1000, analytic=False)

PennyLane offers some basic devices such as the ``'default.qubit'`` and ``'default.gaussian'``
simulators; additional devices can be installed as plugins (see
`available plugins <https://pennylane.ai/plugins.html>`_ for more details). Note that the
choice of a device significantly determines the speed of your computation, as well as
the available options that can be passed to the device loader.

Device options
~~~~~~~~~~~~~~

When loading a device, the name of the device must always be specified.
Further options can then be passed as keyword arguments; these options can differ based
on the device. For the in-built ``'default.qubit'`` and ``'default.gaussian'``
devices, the options are:

* ``wires`` (*int*): The number of wires to initialize the device with.

* ``analytic`` (*bool*): Indicates if the device should calculate expectations
  and variances analytically. Only possible with simulator devices. Defaults to ``True``.

* ``shots`` (*int*): How many times the circuit should be evaluated (or sampled) to estimate
  the expectation values. Defaults to 1000 if not specified.

For a plugin device, refer to the plugin documentation for available device options.

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
0.7648421872844883


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
        return qml.expval(qml.PauliZ(0))

    result = circuit(0.543)
