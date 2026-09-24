:orphan:

.. role:: html(raw)
   :format: html

Building a legacy plugin
========================

For adding a plugin that inherits from the new device interface, see :doc:`/development/plugins`.

PennyLane plugins allow an external quantum library to take advantage of the automatic differentiation ability of PennyLane.
Writing your own plugin is a simple and easy process. In this section, we will walk through the steps for creating your own
PennyLane plugin within the legacy device API.


What a plugin provides
----------------------

Here's a quick primer on PennyLane plugins:

* A plugin is an external Python package that provides additional quantum *devices* to PennyLane.

* Each plugin may provide one or more devices that are accessible directly through
  PennyLane, as well as any additional private functions or classes.

* Depending on the scope of the plugin, you may wish to provide additional (custom)
  quantum operations and observables that the user can import.

.. important::

    In your plugin module, **standard NumPy** (*not* the wrapped Autograd version of NumPy, ``pennylane.numpy``)
    should be imported in all places (i.e., ``import numpy as np``).


Creating your device
--------------------

The first step in creating your PennyLane plugin is to create your device class.
This is as simple as importing the abstract base class :class:`pennylane.devices.LegacyDevice` from PennyLane,
and subclassing it:

.. code-block:: python

    from pennylane.devices import LegacyDevice

    class MyDevice(LegacyDevice):
        """MyDevice docstring"""
        name = 'My custom device'
        short_name = 'example.mydevice'
        pennylane_requires = '0.1.0'
        version = '0.0.1'
        author = 'Ada Lovelace'

.. note::

    Most devices inherit from a subclass of :class:`pennylane.devices.LegacyDevice` called :class:`~pennylane.devices.QubitDevice`,
    which contains a lot of functionality specific to computations based on qubits. We will
    take a deeper look at this important case below.

.. warning::

    The API of PennyLane devices is currently being updated to follow a new interface defined by
    the :class:`pennylane.devices.Device` class. This guide describes
    how to create a device with the :class:`pennylane.devices.LegacyDevice` and
    :class:`pennylane.devices.QubitDevice` base classes, and will be updated as we continue switching
    to the new API. In the meantime, please reach out to the PennyLane team if you would like help with
    building a plugin, either by creating an `issue <https://github.com/PennyLaneAI/pennylane/issues>`_
    or by posting in our `discussion forum <https://discuss.pennylane.ai/>`_.

Here, we have begun defining some important class attributes that allow PennyLane to identify
and use the device. These include:

* :attr:`pennylane.devices.LegacyDevice.name`: a string containing the official name of the device

* :attr:`pennylane.devices.LegacyDevice.short_name`: the string used to identify and load the device by users of PennyLane

* :attr:`pennylane.devices.LegacyDevice.pennylane_requires`: the PennyLane version this device supports.
  Note that this class attribute supports pip *requirements.txt* style version ranges,
  for example:

  - ``pennylane_requires = "2"`` to support PennyLane version 2.x.x
  - ``pennylane_requires = ">=0.1.5,<0.6"`` to support a range of PennyLane versions

* :attr:`pennylane.devices.LegacyDevice.version`: the version number of the device

* :attr:`pennylane.devices.LegacyDevice.author`: the author of the device

**Defining all of the attributes above is mandatory.**


Device capabilities
-------------------

Furthermore, you must tell PennyLane about the operations that your device supports, 
as well as potential further capabilities, by providing the following class attributes/properties:

* :attr:`pennylane.devices.LegacyDevice.stopping_condition`: This :class:`~.BooleanFn` should return ``True`` for supported
  operations and measurement processes, and ``False`` otherwise. Note that this function is called on
  **both** ``Operator`` and ``MeasurementProcess`` classes. Though this function must accept both ``Operator``
  and ``MeasurementProcess`` classes, it does not affect whether a ``MeasurementProcess`` is supported or not.

  .. code-block:: python

      @property
      def stopping_condition(self):
          def accepts_obj(obj):
              return obj.name in {'CNOT', 'PauliX', 'PauliY', 'PauliZ'}
          return qp.BooleanFn(accepts_obj)

  Supported operations can also be determined by the :attr:`pennylane.devices.LegacyDevice.operations` property.
  This property is a list of string names for supported operations.

  .. code-block:: python

      operations = {"CNOT", "PauliX"}

  See :doc:`/introduction/operations` for a full list of operations
  supported by PennyLane.

  If your device does not natively support an operation that has the
  :meth:`~.Operation.decomposition` static method defined, PennyLane will
  attempt to decompose the operation before calling the device. For example,
  the :class:`~.Rot` `decomposition method <../_modules/pennylane/ops/qubit.html#Rot.decomposition>`_ will
  decompose the single-qubit rotation gate to :class:`~.RZ` and :class:`~.RY` gates.

  .. note::

      If the convention differs between the built-in PennyLane operation
      and the corresponding operation in the targeted framework, ensure that the
      conversion between the two conventions takes place automatically
      by the plugin device.

* :func:`pennylane.devices.LegacyDevice.capabilities`: A class method which returns the dictionary of capabilities of a device. A
  new device should override this method to retrieve the parent classes' capabilities dictionary, make a copy,
  and update and/or add capabilities before returning the copy.

  Examples of capabilities are:

  * ``'model'`` (*str*): either ``'qubit'`` or ``'cv'``.

  * ``'returns_state'`` (*bool*): ``True`` if the device returns the quantum state via ``dev.state``.

  * ``'supports_inverse_operations'`` (*bool*): ``True`` if the device supports
    applying the inverse of operations. Operations which should be inverted
    have the property ``operation.inverse == True``.

  *  ``'supports_tensor_observables'`` (*bool*): ``True`` if the device supports observables composed from tensor
     products such as ``PauliZ(wires=0) @ PauliZ(wires=1)``.

  *  ``'supports_tracker'`` (*bool*): ``True`` if it has a device tracker attribute and updates information with
     it.

  Some capabilities are queried by PennyLane core to make decisions on how to best run computations, while others are used
  by external apps built on top of the device ecosystem.

  To find out which capabilities are (possibly automatically) defined for your device, ``dev = qp.device('my.device', *args, **kwargs)``,
  check the output of ``dev.capabilities()``.

Adding arguments to your device
--------------------------------

.. important::

    PennyLane supports both qubit and continuous-variable (CV) devices. However, from
    here onwards, we will demonstrate plugin development focusing on qubit-based devices
    inheriting from the :class:`~pennylane.devices.QubitDevice` class.

Defining the ``__init__`` method of a custom device is not necessary; by default,
the :class:`~pennylane.devices.QubitDevice` initialization will be called, where the user can pass the
following arguments:

* ``wires`` (*int* or *Iterable[Number, str]*): The number of subsystems represented by the device,
  or an iterable that contains unique labels for the subsystems as numbers (e.g., ``[-1, 0, 2]``)
  and/or strings (``['auxiliary', 'q1', 'q2']``).

* ``shots=1000`` (*None*, *int* or *List[int]*): number of circuit
  evaluations/random samples used to estimate probabilities, expectation
  values, variances  of observables in non-analytic mode. If ``shots=None``, the device
  calculates probability, expectation values, and variances analytically. If `shots` is an
  integer, it specifies the number of samples to estimate these quantities. If a
  list of integers is passed, the circuit evaluations are batched over the list
  of shots.

To add your own device arguments, or to override any of the above defaults, simply
overwrite the ``__init__`` method. For example, here is a device where the number
of wires is fixed to ``24``, that cannot be used in analytic mode, and that can accept a dictionary
of low-level hardware control options:

.. code-block:: python3

    class CustomDevice(QubitDevice):
        name = 'My custom device'
        short_name = 'example.mydevice'
        pennylane_requires = '0.1.0'
        version = '0.0.1'
        author = 'Ada Lovelace'

        operations = {"PauliX", "RX", "CNOT"}
        observables = {"PauliZ", "PauliX", "PauliY"}

        def __init__(self, shots=1024, hardware_options=None):
            super().__init__(wires=24, shots=shots)
            self.hardware_options = hardware_options or hardware_defaults

Note that we have also overridden the default shot number.

The user can now pass any of these arguments to the PennyLane device loader:

>>> dev = qp.device("example.mydevice", hardware_options={"t2": 0.1})
>>> dev.hardware_options
{"t2": 0.1}


Device execution
----------------

Once all of the class attributes are defined, it is necessary to define some required class
methods to allow PennyLane to apply operations and measure observables on your device.

To execute operations on the device, the following methods **must** be defined:

.. currentmodule:: pennylane.devices

.. autosummary::

    ~QubitDevice.apply

If the device is a statevector simulator (it can perform analytic computations when ``shots=None``)
then it **must** also overwrite:

.. autosummary::

    ~QubitDevice.analytic_probability

The :class:`~pennylane.devices.QubitDevice` class
provides the following convenience methods that may be used by the plugin:

.. autosummary::

    ~QubitDevice.active_wires
    ~QubitDevice.marginal_prob

In addition, if your qubit device generates its own computational basis samples for measured wires
after execution, you need to overwrite the following method:

.. autosummary::

    ~QubitDevice.generate_samples

:meth:`~pennylane.devices.QubitDevice.generate_samples` should return samples with shape ``(dev.shots, dev.num_wires)``.
Furthermore, PennyLane uses the convention :math:`|q_0,q_1,\dots,q_{N-1}\rangle` where
:math:`q_0` is the most significant bit.

And thats it! The device has inherited :meth:`~pennylane.devices.QubitDevice.expval`, :meth:`~pennylane.devices.QubitDevice.var`,
and :meth:`~pennylane.devices.QubitDevice.sample` methods, each of which accepts an observable (or tensor product of
observables) and returns the corresponding measurement statistic.



:html:`<div class="caution admonition" id="aside1"><a data-toggle="collapse" data-parent="#aside1" href="#content1" class="collapsed"><p class="first admonition-title">Advanced execution control (click to expand) <i class="fas fa-chevron-circle-down"></i></p></a><div id="content1" class="collapse" data-parent="#aside1" style="height: 0px;">`

Additional flexibility is sometimes required for interfacing with more
complicated frameworks.

When PennyLane needs to evaluate a QNode, it accesses the :meth:`~pennylane.devices.QubitDevice.execute` method of
your plugin which, by default, performs the following process:

.. code-block:: python

    self.check_validity(circuit.operations, circuit.observables)

    # apply all circuit operations
    self.apply(circuit.operations, rotations=circuit.diagonalizing_gates)

    # generate computational basis samples
    if self.shots is not None or circuit.is_sampled:
        self._samples = self.generate_samples()

    # compute the required statistics
    results = self.statistics(circuit)

    return self._asarray(results)

Here,

* ``circuit`` is a :class:`~.CircuitGraph` object

* :attr:`circuit.operations <pennylane.CircuitGraph.operations>` are the user-provided
  operations to be executed

* :attr:`circuit.observables <pennylane.CircuitGraph.observables>` are the user-provided
  observables to be measured

* :attr:`circuit.diagonalizing_gates <pennylane.CircuitGraph.diagonalizing_gates>` are the
  gates that rotate the circuit prior to measurement so that computational basis
  measurements are performed in the eigenbasis of the requested observables

* :meth:`~pennylane.devices.QubitDevice.statistics` returns the results of :meth:`~pennylane.devices.QubitDevice.expval`,
  :meth:`~pennylane.devices.QubitDevice.var`, or :meth:`~pennylane.devices.QubitDevice.sample` depending on the type
  of observable.

In advanced cases, the :meth:`~pennylane.devices.QubitDevice.execute` method, as well as
:meth:`~pennylane.devices.QubitDevice.statistics`, may be overwritten directly.
This provides full flexibility for handling the device execution yourself. However,
this may have unintended side-effects and is not recommended.


:html:`</div></div>`


Wire handling
-------------

PennyLane uses the :class:`~.wires.Wires` class for the internal representation of wires. :class:`~.wires.Wires`
inherits from Python's ``Sequence``, and represents an ordered set of unique wire labels.
The ``labels`` attribute stores a tuple of the wire labels.
Indexing a ``Wires`` instance with an integer will return the corresponding label.
Indexing with a ``slice`` will return a ``Wires`` instance.

For example:

.. code-block:: python

    from pennylane.wires import Wires

    wires = Wires(['auxiliary', 0, 1])
    print(wires.labels) # ('auxiliary', 0, 1)
    print(wires[0]) # 'auxiliary'
    print(wires[0:1]) # Wires(['auxiliary'])

As shown in the section on :doc:`/introduction/circuits`, a device can be created with custom wire labels:

.. code-block:: python

    from pennylane import *

    dev = device('my.device', wires=['q11', 'q12', 'q21', 'q22'])

    @qnode(dev)
    def circuit():
       Gate1(wires='q22')
       Gate2(wires=['q21','q11'])
       Gate1(wires=['q21'])
       return expval(Obs(wires='q11') @ Obs(wires='q12'))

Behind the scenes, when ``my.device`` gets created it turns ``['q11', 'q12', 'q21', 'q22']`` into a
:class:`~.wires.Wires` object and stores it in the device's ``wires`` attribute. Likewise, when gates and
observables get created they turn their ``wires`` argument into a :class:`~.wires.Wires`
object and store it in their ``wires`` attribute.

.. code-block:: python

    print(dev.wires) #  Wires(['q11', 'q12', 'q21', 'q22'])

    op = Gate2(wires=['q21','q11'])
    print(op.wires) # Wires(['q21', 'q11'])

When the device applies operations, it needs to translate
``op.wires`` into wire labels that the backend "understands". This can be done with the
:meth:`pennylane.devices.LegacyDevice.map_wires` method, which maps ``Wires`` objects to other ``Wires`` objects and changes the labels according to the ``wire_map`` attribute of the device which defines the translation.

.. code-block:: python

    # inside the class defining 'my.device', which inherits from the base Device class
    device_wires = self.map_wires(op.wires)
    print(device_wires) # Wires([2, 0])

By default, the map translates the custom labels ``'q11'``, ``'q12'``, ``'q21'``, ``'q22'`` to
consecutive integers ``0``, ``1``, ``2``, ``3``. If a device uses a different wire labeling,
such as non-consecutive wires ``0``, ``4``, ``7``, ``12``, the :meth:`pennylane.devices.LegacyDevice.define_wire_map` method
has to be overwritten accordingly.

The ``device_wires`` can then be further processed, for example, by extracting the actual labels as a tuple,
list or array, or by getting the number of wires:

.. code-block:: python

    device_wires.labels # (2, 0)

    device_wires.tolist() # [2, 0]

    device_wires.toarray() # ndarray([2, 0])

    len(device_wires) # 2

The ``Wires`` class also offers set functionality like identifying the unique or shared wires between several ``Wires``
object.

As a convention, devices should do the translation and unpacking as late as possible in the function tree, and
where possible pass the original :class:`~.wires.Wires` objects around.

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

We recommend placing the following code near the end of the ``execute`` method,

.. code-block:: python

  if self.tracker.active:
    self.tracker.update(executions=1, shots=self._shots)
    self.tracker.record()

and similar code in the ``batch_execute`` method:

.. code-block:: python

  if self.tracker.active:
    self.tracker.update(batches=1, batch_len=len(circuits))
    self.tracker.record()

These functions are called in base :class:`pennylane.devices.LegacyDevice` and :class:`~.QubitDevice` devices. Unless you are
overriding the ``execute`` and ``batch_execute`` methods or want to customize the stored
information, you do not need to add any new code.

While this is the recommended usage, the ``update`` and ``record`` methods can be called at any location
within the device. While the above example tracks executions, shots, and batches, the 
:meth:`~.Tracker.update` method can accept any combination of
keyword-value pairs.  For example, a device could also track cost and a job ID via:

.. code-block:: python

  price_for_execution = 0.10
  job_id = "abcde"
  self.tracker.update(price=price_for_execution, job_id=job_id)

.. _installing_plugin:

Identifying and installing your device
--------------------------------------

When performing a hybrid computation using PennyLane, one of the first steps is often to
initialize the quantum device(s). PennyLane identifies the devices via their ``short_name``,
which allows the device to be initialized in the following way:

.. code-block:: python

    import pennylane as qp
    dev1 = qp.device(short_name, wires=2)

where ``short_name`` is a string that uniquely identifies the device. The ``short_name``
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

* ``example.mydevice1`` is the short name of the device, and

* ``MyModule.MySubModule`` is the path to your Device class, ``MyDevice1``.

To ensure your device is working as expected, you can install it in developer mode using
``pip install -e pluginpath``, where ``pluginpath`` is the location of the plugin. It will
then be accessible via PennyLane.

Testing
-------

All plugins should come with extensive unit tests, to ensure that each logical unit of the
device has correct execution.

Integration tests to check that the probabilities, expectation values, variance, and samples are
correct for various circuits and observables are provided as part of the PennyLane device
test utility:


.. code-block:: console

    pl-device-test --device device_shortname --shots 10000

In general, as all supported operations have their gradient formula defined and tested by
PennyLane, testing that your device calculates the correct gradients is not required.
For more details on the PennyLane device test utility, see :mod:`pennylane.devices.tests`.

Supporting custom operators
---------------------------

If you would like to support an operator (such as a gate or observable) that is not currently supported by
PennyLane, you can subclass the :class:`~.Operator` class. Detailed information can be found in the
section :doc:`/development/adding_operators`.

Users can then import this operator directly from your plugin, and use it when defining a QNode:

.. code-block:: python

    import pennylane as qp
    from MyModule.MySubModule import CustomGate

    @qnode(dev1)
    def my_qfunc(phi):
        qp.Hadamard(wires=0)
        CustomGate(phi, theta, wires=0)
        return qp.expval(qp.PauliZ(0))

.. warning::

    If you are providing custom operators not natively supported by PennyLane, it is recommended
    that the plugin unit tests provide tests to ensure that PennyLane returns the correct
    gradient for the custom operations.

If the custom operator is diagonal in the computational basis, it can be added to the
``diagonal_in_z_basis`` attribute in ``pennylane.ops.qubit.attributes``. Devices can use this
information to implement faster simulations.