.. role:: html(raw)
   :format: html

For adding a plugin that inherits from the legacy interface, see :doc:`/development/legacy_plugins`.

.. _plugin_overview:

Building a plugin
=================

Writing your own PennyLane plugin, to allow an external quantum library to take advantage of the
automatic differentiation ability of PennyLane, is a simple and easy process. In this section, we discuss
the methods and concepts involved in the device interface. For a walkthrough implementing a
minimal device, see :doc:`/development/minimal_dev`.

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
>>> list(tape0.shots)
[5, 500, 1000]
>>> list(tape0.shots.bins())
[(0, 5), (5, 505), (505, 1505)]
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
3) Strictly require specific wire labels

Option 2 allows workflows to change the number and labeling of wires over time, but sometimes users want
to enfore a wire convention and labels. If a user does provide wires, the :meth:`~.devices.Device.preprocess` should
validate that submitted circuits only have wires in the requested range.

>>> dev = qml.device('default.qubit', wires=1)
>>> circuit = qml.tape.QuantumScript([qml.CNOT((0,1))], [qml.state()])
>>> dev.preprocess()[0]((circuit, ))
WireError: Cannot run circuit(s) of default.qubit as they contain wires not found on the device.

PennyLane wires can be any hashable object, where wire labels are distinguished by their equality and hash.
If working with successive integers (``0``, ``1``, ``2``, ...) is preferred internally,
the :meth:`~.QuantumScript.map_to_standard_wires` method can be used inside of 
the :meth:`~.devices.Device.execute` method.  The :class:`~.map_wires` transform can also 
map the wires of the submitted circuit to internal labels.

Sometimes hardware qubit labels cannot be arbitrarily mapped without a changing in behaviour.
Connectivity, noise, performance, and
other constraints can make it so that operations on qubit one cannot be arbitrarily exchanged with the same operation
of qubit two.  In such a situation, the device can hard code a list of the only acceptable wire labels. In such a case, it
will be on the user to deliberately map wires if they wish such a thing to occur.

>>> qml.device('my_hardware').wires
<Wires = [0, 1, 2, 3]>
>>> qml.device('my_hardware', wires=(10, 11, 12, 13))
DeviceError: Device my_hardware cannot internally map wires, as the labels 0, 1, 2, 3 indicate unique qubits


Preprocessing
-------------

The preprocessing method has two main responsibilities:

1) Create a :class:`~.TransformProgram` capable of turning an arbitrary :class:`~.QuantumScript` into one supported by the device
2) Setup the :class:`~.ExecutionConfig` dataclass by filling in device options and making decisions about differentiation

These two tasks can be extracted into private methods or helper functions if that improves source code organization.

See the section on the **Execution Config** below for more information on step 2.

Devices may define their own transforms following the description in the ``transforms`` module,
or can include in-built transforms such as:

.. currentmodule:: pennylane.devices.preprocess
.. autosummary::
    :toctree: api

    decompose
    validate_observables
    validate_measurements
    validate_device_wires
    validate_multiprocessing_workers
    validate_adjoint_trainable_params
    no_sampling


.. currentmodule:: pennylane
.. autosummary::
    :toctree: api

    defer_measurements
    dynamic_one_shot
    transforms.broadcast_expand
    transforms.split_non_commuting
    transforms.transpile

Once a program is created, an individual transform can be added to the program with the
:meth:`~.TransformProgram.add_transform` method.

.. code-block:: python

    from pennylane.devices.preprocess import validate_device_wires

    program = qml.transforms.core.TransformProgram()
    program.add_transform(validate_device_wires, wires=qml.wires.Wires((0,1,2)), name="my_device")

Preprocessing and validation can also exist inside the :meth:`~devices.Device.execute` method, but placing them
in the preprocessing program has several benefits. Validation can happen earlier, leading to fewer resources
spent before the error is raised.  Users can inspect, draw, and spec out the tapes at different stages throughout
preprocessing. This provides users a better awareness of what the device is actually executing. When device
gradients are used, the preprocessing transforms are tracked by the machine learning interfaces.  With the
ML framework tracking the classical component of preprocessing, the device does not need to manually track the
classical component of any decompositions or compilation. 

Even with these benefits, devices can still opt to
place some transforms inside the ``execute`` method.  For example, ``default.qubit`` maps wires to simulation indices
inside ``execute`` instead of in ``preprocess``.

The :meth:`~.devices.Device.execute` can assume that device preprocessing has been run on the input
tapes, and has no obligation to re-validate the input or provide sensible error messages.

>>> op = qml.Permute([2,1,0], wires=(0,1,2))
>>> tape = qml.tape.QuantumScript([op], [qml.probs(wires=(0,1))])
>>> qml.device('default.qubit').execute(tape)
MatrixUndefinedError:
>>> tape = qml.tape.QuantumScript([], [qml.density_matrix(wires=0)], shots=50)
>>> qml.device('default.qubit').execute(tape)
AttributeError: 'DensityMatrixMP' object has no attribute 'process_samples'


Execution Config
----------------

The execution config stores two kinds of information:

1) Information about how the device should perform the execution. Examples include ``device_options`` and ``gradient_method``.
2) Information about how PennyLane should interact with the device. Examples include ``use_device_gradient`` and ``grad_on_execution``.

**Device options:**

Device options are any device specific options used to configure the behavior of an execution. For example, ``default.qubit``
has ``max_workers``, ``rng``, and ``prng_key``. ``default.tensor`` has ``contract``, ``cutoff``, ``dtype``, ``method``, and ``max_bond_dim``.
These options are often set with default values on initialization. These values should be placed into the ``ExecutionConfig.device_options``
dictionary on preprocessing.

>>> dev = qml.device('default.tensor', wires=2, max_bond_dim=4, contract="nonlocal", dtype=np.complex64)
>>> dev.preprocess()[1].device_options
{'contract': 'nonlocal',
 'cutoff': 1.1920929e-07,
 'dtype': numpy.complex64,
 'method': 'mps',
 'max_bond_dim': 4}

Even if the property is stored as an attribute on the device, execution should pull the
value of these properties from the config instead of from the device instance.
While not yet integrated at the top user level, we aim to allow dynamic configuration of the device.

>>> dev = qml.device('default.qubit')
>>> config = qml.devices.ExecutionConfig(device_options={"rng": 42})
>>> tape = qml.tape.QuantumTape([qml.Hadamard(0)], [qml.sample(wires=0)], shots=10)
>>> dev.execute(tape, config)
array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0])
>>> dev.execute(tape, config)
array([1, 0, 1, 1, 0, 1, 1, 1, 0, 0])

By pulling options from this dictionary instead of from device properties,
we unlock two key pieces of functionality.

1) Track and specify the exact configuration of the execution by only inspecting the ``ExecutionConfig`` object
2) Dynamically configure the device over the course of a workflow.


**Workflow Configuration:**

Note that these properties are only applicable to devices that provided derivatives or VJPs. If your device
does not provide derivatives, you can safely ignore these properties.

The workflow options are ``use_device_gradient``, ``use_device_jacobian_product``, and ``grad_on_execution``. 
``use_device_gradient=True`` indicates that workflow should request derivatives from the device. 
``grad_on_execution=True`` indicates a preference
to use ``execute_and_compute_derivatives`` instead of ``execute`` followed by ``compute_derivatives``.  And finally
``use_device_jacobian_product`` indicates a request to call ``compute_vjp`` instead of ``compute_derivatives``.  Note that 
if ``use_device_jacobian_product`` is ``True``, this takes precedence over calculating the full jacobian.

>>> config = qml.devices.ExecutionConfig(gradient_method="adjoint")
>>> processed_config = qml.device('default.qubit').preprocess(config)[1]
>>> processed_config.use_device_jacobian_product
True
>>> processed_config.use_device_gradient
True
>>> processed_config.grad_on_execution
True

Execution
---------

For documentation on the expected result type output, please refer to :ref:`ReturnTypeSpec`.

The device API allows individual devices to calculate results in whatever way makes sense for
the individual device. With this freedom over the implementation does come more responsibility
to handle each stage in the process.


Mid circuit measurement support
-------------------------------

TODO

Device tracker support
----------------------

The device tracker stores and records information when tracking mode is turned on. Devices can store data like
the number of executions, number of shots, number of batches, or remote simulator cost for users to interact with
in a customizable way.

Three aspects of the :class:`~.Tracker` class are relevant to plugin designers:

* The boolean ``active`` attribute that denotes whether or not to update and record
* ``update`` method which accepts keyword-value pairs and stores the information
* ``record`` method which users can customize to log, print, or otherwise do something with the stored information

To gain simulation-like tracking behavior, the :func:`~.devices.modifiers.simulator_tracking` decorator can be added
to the device:

.. code-block:: python

    @qml.devices.modifiers.simulator_tracking
    class MyDevice(Device):
        ...


``simulator_tracking`` is useful when the device can simulataneously measure non-commuting measurements or
handle parameter-broadcasting, as it both tracks simulations and the corresponding number of QPU-like
circuits.

To implement your own tracking, we recommend placing the following code in the ``execute`` method:

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
