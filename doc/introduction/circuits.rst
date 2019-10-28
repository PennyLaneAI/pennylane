 .. role:: html(raw)
   :format: html


.. _intro_vcircuits:

Quantum Circuits
================


.. image:: ../_static/qnode.png
    :align: right
    :width: 180px
    :target: javascript:void(0);


In PennyLane, variational quantum circuits are represented as *quantum node* objects. A quantum node
is a combination of a quantum function that composes the circuit,
and a device that runs the computation. One can conveniently create quantum nodes using
the quantum node decorator.

Each classical :ref:`interface <intro_interfaces>` uses a different version of a quantum node,
and we will introduce the standard QNode to use with the NumPy interface here.
NumPy-interfacing quantum nodes take NumPy datastructures,
such as floats and arrays, and return Numpy data structures.
They can be optimized using NumPy-based :ref:`optimization methods <intro_ref_opt>`.
Quantum nodes for other PennyLane interfaces like :ref:`PyTorch <torch_interf>` and
:ref:`TensorFlow's Eager mode <tf_interf>` are introduced in the section on :ref:`interfaces <intro_interfaces>`.


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

    The quantum operations cannot be used outside of a quantum circuit function, as all
    :class:`Operations <pennylane.operation.Operation>` must be executed from within a QNode.

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
simulator; additional devices can be installed as plugins (see
`available plugins <https://pennylane.ai/plugins.html>`_ for more details). Note that the
choice of a device significantly determines the speed of your computation, as well as
the available options that can be passed to the device loader.

Device options
~~~~~~~~~~~~~~

When loading a device, there is one argument that must always be specified:

* ``name`` (*str*): the name of the device.

Further options can be passed as keyword arguments; these options can differ based
on the device. For the in-built ``'default.qubit'`` and ``'default.gaussian'``
devices, the options are:

* ``wires`` (*int*): The number of wires to initialize the device with.

* ``shots=1000`` (*int*): How many times the circuit should be evaluated (or sampled) to estimate
    the expectation values. Defaults to 1000 if not specified.

    If ``analytic == True``, then the number of shots is ignored in the calculation of
    expectation values and variances, and only controls the number of samples returned
    by ``sample``.

* ``analytic=True`` (*bool*): Indicates if the device should calculate expectations
    and variances analytically.

.. _intro_vcirc_qnode:

Creating a quantum node
-----------------------

Together, a quantum function and a device are used to create a *quantum node* or
:class:`~.pennylane.QNode` object, which wraps the quantum function and binds it to the device.

A QNode can be explicitly created as follows:

.. code-block:: python

    qnode = qml.QNode(my_quantum_function, dev)

The QNode can be used to compute the result of a quantum circuit as if it was a standard Python
function. It takes the same arguments as the original quantum function:

>>> qnode(np.pi/4, 0.7)
0.7648421872844883


.. _intro_vcirc_decorator:

The QNode decorator
-------------------

A more convenient---and in fact the recommended---way for creating QNodes is the provided
quantum node decorator. This decorator converts a quantum function containing PennyLane quantum
operations to a :class:`~.pennylane.QNode` that will run on a quantum device.

.. note::
    The decorator completely replaces the Python-based quantum function with
    a :class:`~.pennylane.QNode` of the same name---as such, the original
    function is no longer accessible.

For example:

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def qfunc(x):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(x, wires=1)
        return qml.expval(qml.PauliZ(0))

    result = qfunc(0.543)
