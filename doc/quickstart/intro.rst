 .. role:: html(raw)
   :format: html

.. _pl_intro:

Introduction
============

The :ref:`Key Concepts <overview>` section introduced the idea of *variational*
or *parametrized quantum circuits* which can be optimized for a given task.

This section is an overview of how these concepts are implemented in PennyLane.
It shows how to:

* Construct quantum circuits via **quantum functions**

* Create **computational devices**

* Encapsulate quantum functions and devices in a **quantum node**

* Conveniently create quantum nodes using the **decorator**

* Compute **gradients** of quantum nodes

* **Optimize** hybrid computations that contain quantum nodes

* Save **configurations** for PennyLane

More information about PennyLane's code base can be found in the :ref:`User Documentation <user_docs>`.

Quantum functions
-----------------

A quantum circuit is constructed as a special Python function, a *quantum (circuit) function*.
For example:

.. code-block:: python

    def my_quantum_function(x, y):
        qml.RZ(x, wires=0)
        qml.CNOT(wires=[0,1])
        qml.RY(y, wires=1)
        return qml.expval(qml.PauliZ(1))


Quantum circuit functions are a restricted subset of Python functions, adhering to the following
constraints:

* The body of the function must consist of only supported PennyLane
  :mod:`operations <pennylane.ops>`, one per line.

* The function must always return either a single or a tuple of
  *measured observable values*, by applying a :mod:`measurement function <pennylane.measure>`
  to an :mod:`observable <pennylane.ops>`.

* Classical processing of function arguments, either by arithmetic operations
  or external functions, is not allowed. One current exception is simple scalar
  multiplication.

.. note::

    The quantum operations cannot be used outside of a quantum circuit function, as all
    :class:`Operations <pennylane.operation.Operation>` require a QNode in order to perform queuing on initialization.

.. note::

    Measured observables **must** come after all other operations at the end
    of the circuit function as part of the return statement, and cannot appear in the middle.


Defining a device
-----------------

To run - and later optimize - a quantum circuit, one needs to first specify a *computational device*.

The device is an instance of the :class:`~_device.Device`
class, and can represent either a simulator or hardware device. They can be
instantiated using the :func:`~device` loader. PennyLane comes included with
some basic devices; additional devices can be installed as plugins
(see :ref:`plugins` for more details).

.. code-block:: python

    dev = qml.device('default.qubit', wires=2)


Quantum nodes
-------------

Together, a quantum function and a device are used to create a *quantum node* or
:class:`QNode` object, which wraps the quantum function and binds it to the device.
A quantum node is a subroutine executed by a quantum computer, which is part of a
larger :ref:`hybrid computation <_hybrid_computation>`.

A `QNode` can be explicitly created as follows:

.. code-block:: python

    qnode = qml.QNode(my_quantum_function, dev)

The `QNode` can be used to compute the result of a quantum circuit as if it was a standard Python
function. It takes the same arguments as the original quantum function:

>>> qnode(np.pi/4, 0.7)

One or more :class:`QNodes` can be combined in standard python functions:

.. code-block:: python

    def my_quantum_function2(x, y):
        qml.Displacement(x, 0, wires=0)
        qml.Beamsplitter(y, 0, wires=[0, 1])
        return qml.expval(qml.NumberOperator(0))

    dev2 = qml.device('default.gaussian', wires=2)

    qnode2 = qml.QNode(my_quantum_function2, dev2)

    def hybrid_computation(x, y):
        return np.sin(qnode1(y))*np.exp(-qnode2(x+y, x)**2)


Note that `hybrid_computation` contains results from two different devices, one being a qubit-based
and the other a continuous-variable device.

The QNode decorator
-------------------

A more convenient - and in fact the recommended - way for creating `QNodes` is the provided
`qnode` decorator. This decorator converts a quantum circuit function containing PennyLane quantum
operations to a :mod:`QNode <pennylane.qnode>` that will run on a quantum device.

.. note::
    The decorator completely replaces the Python-defined function with
    a :mod:`QNode <pennylane.qnode>` of the same name - as such, the original
    function is no longer accessible (but is accessible via the :attr:`~.QNode.func` attribute).

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


Quantum gradients
-----------------

The gradient of the `QNodes` from above can be computed as follows:

.. code-block:: python

    g1 = qml.grad(qnode, [0, 1])
    g2 = qml.grad(qnode1, [0])
    g3 = qml.grad(qfunc, [1])

The first argument of :func:`grad` is the quantum node, and the second is a list of indices of the parameters
we want to derive for. The result is a new function which computes gradients for specific values of the parameters,
for example:

>>> x = 1.1
>>> y = -2.2
>>> g1(x, y)
(array(0.56350015), array(0.17825313))
>>> g2(x, y)
(array(0.56350015), array(0.17825313))
>>> g3(x, y)
(array(0.56350015), array(0.17825313))

We can also compute gradients of *functions of qnodes*:

.. code-block:: python

    g4 = qml.grad(hybrid_computation, [0, 1])

and evaluate

>>> g4(1.1, -2.2)
(array(0.56350015), array(0.17825313))

Optimization
------------

PennyLane comes with a collection of optimizers for a basic `QNode`. They
can be found in the :mod:`pennylane.optimize` module.




Configuration
-------------

The settings for PennyLane can be stored in a configuration file for ease of use.

Behaviour
*********

On first import, PennyLane attempts to load the configuration file `config.toml`, by
scanning the following three directories in order of preference:

1. The current directory
2. The path stored in the environment variable ``PENNYLANE_CONF``
3. The default user configuration directory:

   * On Linux: ``~/.config/pennylane``
   * On Windows: ``~C:\Users\USERNAME\AppData\Local\Xanadu\pennylane``
   * On MacOS: ``~/Library/Preferences/pennylane``

If no configuration file is found, a warning message will be displayed in the logs,
and all device parameters will need to be passed as keyword arguments when
loading the device.

The user can access the initialized configuration via `pennylane.config`, view the
loaded configuration filepath, print the configurations options, access and modify
them via keys (i.e., ``pennylane.config['main.shots']``), and save/load new configuration files.

Configuration files
*******************

The configuration file `config.toml` uses the `TOML standard <https://github.com/toml-lang/toml>`_,
and has the following format:

.. code-block:: toml

    [main]
    # Global PennyLane options.
    # Affects every loaded plugin if applicable.
    shots = 0

    [strawberryfields.global]
    # Options for the Strawberry Fields plugin
    hbar = 1
    shots = 100

      [strawberryfields.fock]
      # Options for the Strawberry Fields Fock plugin
      cutoff_dim = 10
      hbar = 0.5

      [strawberryfields.gaussian]
      # Indentation doesn't matter in TOML files,
      # but helps provide clarity.

    [projectq.global]
    # Options for the Project Q plugin

      [projectq.simulator]
      gate_fusion = true

      [projectq.ibm]
      user = "johnsmith"
      password = "secret123"
      use_hardware = true
      device = "ibmqx4"
      num_runs = 1024

Main PennyLane options, that are passed to all loaded devices, are provided under the ``[main]``
section. Alternatively, options can be specified on a per-plugin basis, by setting the options under
``[plugin.global]``.

For example, in the above configuration file, the Strawberry Fields
devices will be loaded with a default of ``shots = 100``, rather than ``shots = 0``. Finally,
you can also specify settings on a device-by-device basis, by placing the options under the
``[plugin.device]`` settings.