.. role:: html(raw)
   :format: html

For adding a plugin that inherits from the legacy interface, see :doc:`/development/plugins_legacy`.

.. _plugin_overview:

Building a plugin
=================

Writing your own PennyLane plugin, to allow an external quantum library to take advantage of the
automatic differentiation ability of PennyLane, is a simple and easy process. In this section,
we will walk through the steps for creating your own PennyLane plugin. The :class:`~.devices.DefaultQubit`
device provides a pure-python reference.

Creating your device
--------------------

In order to define a custom device, you only need to override the :meth:`~.devices.Device.execute` method.

.. code-block:: python

    from pennylane.devices import Device, DefaultExecutionConfig

    class MyDevice(Device):
        """My Documentation."""

        def execute(self, circuits: "QuantumTape_or_Batch", execution_config: "ExecutionConfig" = DefaultExecutionConfig):
            # your implementation here.

For example:

.. code-block:: python

    class MyDevice(Device):
        """My Documentation."""

        def execute(self, circuits: "QuantumTape_or_Batch", execution_config: "ExecutionConfig" = DefaultExecutionConfig):
            return 0.0 if isinstance(circuits, qml.tape.QuantumScript) else tuple(0.0 for c in circuits)

    dev = MyDevice()

    @qml.qnode(dev)
    def circuit():
        return qml.state()

    circuit()

The device ``name`` defaults to the name of the class, but we recommend a name of the form ``pluginname.devicename``.
For example, ``default.qubit``, ``lightning.qubit``, or ``qiskit.aer``.

Preprocessing
-------------

Shots
-----

While the workflow default for shots is specified by :attr:`~.Device.shots`, the device itself should use
the number of shots specified in the :attr:`~.QuantumScript.shots` property for each quantum tape.
By pulling shots dynamically for each circuit, users can efficiently distribute a shot budget across batch of
circuits.

>>> tape0 = qml.tape.QuantumScript([], [qml.sample(wires=0)], shots=5)
>>> tape1 = qml.tape.QuantumScript([], [qml.sample(wires=0)], shots=10)
>>> dev = qml.device('default.qubit')
>>> dev.execute((tape0, tape1))
(array([0, 0, 0, 0, 0]), array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

The :class:`~.measurements.Shots` class describes the shots. Users can optionally specify a shot vector, or
different numbers of shots to use when calculating the final expecation value.

>>> tape0 = qml.tape.QuantumScript([], [qml.expval(qml.PauliX(0))], shots=(5, 500, 1000))
>>> tape0.shots.shot_vector
(ShotCopies(5 shots x 1),
 ShotCopies(500 shots x 1),
 ShotCopies(1000 shots x 1))
>>> dev.execute(tape0)
(0.2, -0.052, -0.014)

The first number ``0.2`` is calculated with 5 shots, the second ``-0.052`` is calculated with 500 shots, and
``-0.014`` is calculated with 1000 shots.  All 1,505 shots can be requested together in one batch, but the post
processing into the expecation value is done with shots ``0:5``, ``5:505``, and ``505:1505`` respectively.

Wires
-----

Devices can now either:
1) Strictly use wires provided by the user on initialization ``device(name, wires=wires)``
2) Infer the number and ordering of wires provided by the submitted circuit.

Option 2 allows workflows to change the number and labeling of wires over time, but sometimes users want
to enfore a wire convention and labels. If a user does provide wires, the :method:`~.devices.Device.preprocess` should
validate that submitted circuits only have wires in the requested range.

>>> dev = qml.device('default.qubit', wires=1)
>>> circuit = qml.tape.QuantumScript([qml.CNOT((0,1))], [qml.state()])
>>> dev.preprocess()[0]((circuit, ))
WireError: Cannot run circuit(s) of default.qubit as they contain wires not found on the device.

PennyLane wires can be any hashable object, where wire labels are distinguished by their equality and hash.
If working with successive integers (``0``, ``1``, ``2``, ...) is preferred internally,
the :method:`~.QuantumScript.map_to_standard_wires` method can be used inside of 
the :method:`~.devices.Device.execute` method.

Measurement Defaults
--------------------

PennyLane has a wide variety of measurement processes, from the basic :class:`~.SampleMP` to the more complicated
quantum info measurements like :class:`~.PurityMP`. Users can potentially even define their own measurement processes.
While devices can use their own custom implementation for handling a measurement process,
the :method:`~.measurements.StateMeasurement.process_state` and :method:`~.measurements.SampleMeasurement.process_samples` methods
provide defaults for calculating the value from a state or array of samples respectively.

>>> mp = qml.measurements.PurityMP(wires=(0,1))
>>> state = np.zeros(8)
>>> state[0] = 1.0
>>> mp.process_state(state, wire_order=(0,1,2))
1.0
>>> mp = qml.measurements.CountsMP(all_outcomes=True)
>>> samples = np.zeros((8,2)) # 8 samples, 2 wires
>>> mp.process_samples(samples, wire_order=(0, 1))
{'00': 8, '01': 0, '10': 0, '11': 0}

Execution Config
----------------

The execution config stores two kinds of information:

1) Information about how the device should perform the execution. Examples include ``device_options`` and ``gradient_method``.
2) Information about how PennyLane should interact with the device. Examples include ``use_device_gradient`` and ``grad_on_execution``.


Device tracker support
----------------------

The device tracker stores and records information when tracking mode is turned on. Devices can store data like
the number of executions, number of shots, number of batches, or remote simulator cost for users to interact with
in a customizable way.

Three aspects of the :class:`~.Tracker` class are relevant to plugin designers:

* The boolean ``active`` attribute that denotes whether or not to update and record
* ``update`` method which accepts keyword-value pairs and stores the information
* ``record`` method which users can customize to log, print, or otherwise do something with the stored information

To gain any of the device tracker functionality, a device should initialize with a placeholder 
:class:`~.Tracker` instance. Users can overwrite this attribute by initializing a new instance with
the device as an argument.

We recommend placing the following code near the end of the ``execute`` method:

.. code-block:: python

  if self.tracker.active:
    self.tracker.update(batches=1, executions=len(circuits))
    for c in circuits:
        self.tracker.update(shots=c.shots)
    self.tracker.record()

If the device provides differentiation logic, we also recommend tracking the number of derivative batches,
number of execute and derivative batches, and number of derivatives.

While this is the recommended usage, the ``update`` and ``record`` methods can be called at any location
within the device. While the above example tracks executions, shots, and batches, the 
:meth:`~.Tracker.update` method can accept any combination of
keyword-value pairs.  For example, a device could also track cost and a job ID via:

.. code-block:: python

  price_for_execution = 0.10
  job_id = "abcde"
  self.tracker.update(price=price_for_execution, job_id=job_id)

Identifying and installing your device
--------------------------------------

When performing a hybrid computation using PennyLane, one of the first steps is often to
initialize the quantum device(s). PennyLane identifies the devices via their ``name``,
which allows the device to be initialized in the following way:

.. code-block:: python

    import pennylane as qml
    dev1 = qml.device(name)

where ``name`` is a string that uniquely identifies the device. The ``name``
should have the form ``pluginname.devicename``, using periods for delimitation.

PennyLane uses a setuptools ``entry_points`` approach to plugin discovery/integration.
In order to make the devices of your plugin accessible to PennyLane, simply provide the
following keyword argument to the ``setup()`` function in your ``setup.py`` file:

.. code-block:: python

    devices_list = [
            'example.mydevice1 = MyModule.MySubModule:MyDevice1'
            'example.mydevice2 = MyModule.MySubModule:MyDevice2'
        ],
    setup(entry_points={'pennylane.plugins': devices_list})

where

* ``devices_list`` is a list of devices you would like to register,

* ``example.mydevice1`` is the name of the device, and

* ``MyModule.MySubModule`` is the path to your Device class, ``MyDevice1``.

To ensure your device is working as expected, you can install it in developer mode using
``pip install -e pluginpath``, where ``pluginpath`` is the location of the plugin. It will
then be accessible via PennyLane.

Supporting custom operators
---------------------------

If you would like to support an operator (such as a gate or observable) that is not currently supported by
PennyLane, you can subclass the :class:`~.Operator` class. Detailed information can be found in the
section :doc:`/development/adding_operators`.

Users can then import this operator directly from your plugin, and use it when defining a QNode:

.. code-block:: python

    import pennylane as qml
    from MyModule.MySubModule import CustomGate

    @qnode(dev1)
    def my_qfunc(phi):
        qml.Hadamard(wires=0)
        CustomGate(phi, theta, wires=0)
        return qml.expval(qml.PauliZ(0))

.. warning::

    If you are providing custom operators not natively supported by PennyLane, it is recommended
    that the plugin unit tests provide tests to ensure that PennyLane returns the correct
    gradient for the custom operations.

If the custom operator is diagonal in the computational basis, it can be added to the
``diagonal_in_z_basis`` attribute in ``pennylane.ops.qubit.attributes``. Devices can use this
information to implement faster simulations.