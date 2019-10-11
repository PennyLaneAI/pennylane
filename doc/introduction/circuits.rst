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
is a combination of a :ref:`quantum function <intro_vcirc_qfunc>` that composes the circuit,
and a :ref:`device <intro_vcirc_device>` that runs the computation. One can conveniently create quantum nodes using
the quantum node :ref:`decorator <intro_vcirc_decorator>`.

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

A quantum circuit is constructed as a special Python function, a *quantum circuit function*, or *quantum function* in short.
For example:

.. code-block:: python

    import pennylane as qml

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(y, wires=1)
        return qml.expval(qml.PauliZ(1))


Quantum functions are a restricted subset of Python functions, adhering to the following
constraints:

* The body of the function must consist of only supported PennyLane
  :ref:`operations <intro_ref_ops>` or sequences of operations called :ref:`templates <intro_ref_temp>`,
  using one instruction per line.

* The function must always return either a single or a tuple of
  *measured observable values*, by applying a :ref:`measurement function <intro_ref_meas>`
  to a :ref:`qubit <intro_ref_ops_qobs>` or :ref:`continuous-value observable <intro_ref_ops_cvobs>`.

* Classical processing of function arguments, either by arithmetic operations
  or external functions, is not allowed. One current exception is simple scalar
  multiplication.

.. note::

    The quantum operations cannot be used outside of a quantum circuit function, as all
    :class:`Operations <pennylane.operation.Operation>` require a QNode in order to perform queuing on initialization.

.. note::

    Measured observables **must** come after all other operations at the end
    of the circuit function as part of the return statement, and cannot appear in the middle.


.. _intro_vcirc_device:

Defining a device
-----------------

To run - and later optimize - a quantum circuit, one needs to first specify a *computational device*.

The device is an instance of the :class:`~_device.Device`
class, and can represent either a simulator or hardware device. They can be
instantiated using the :func:`device <pennylane.device>` loader.

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)

PennyLane offers some basic devices such as the ``'default.qubit'`` simulator; additional devices can be installed
as plugins (see :ref:`plugins <plugins>` for more details). Note that the choice of a device significantly
determines the speed of your computation.

.. _intro_vcirc_qnode:

Creating a quantum node
-----------------------

Together, a quantum function and a device are used to create a *quantum node* or
:class:`QNode <pennylane.qnode.QNode>` object, which wraps the quantum function and binds it to the device.

A `QNode` can be explicitly created as follows:

.. code-block:: python

    qnode = qml.QNode(my_quantum_function, dev)

The `QNode` can be used to compute the result of a quantum circuit as if it was a standard Python
function. It takes the same arguments as the original quantum function:

>>> qnode(np.pi/4, 0.7)
0.7648421872844883


.. _intro_vcirc_decorator:

The QNode decorator
-------------------

A more convenient - and in fact the recommended - way for creating `QNodes` is the provided
quantum node decorator. This decorator converts a quantum function containing PennyLane quantum
operations to a :class:`QNode <pennylane.qnode.QNode>` that will run on a quantum device.

.. note::
    The decorator completely replaces the Python-based quantum function with
    a :mod:`QNode <pennylane.qnode.QNode>` of the same name - as such, the original
    function is no longer accessible (but is accessible via the ``func`` attribute).

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





