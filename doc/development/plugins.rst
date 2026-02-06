.. role:: html(raw)
   :format: html

Building a plugin
=================

For adding a plugin that inherits from the legacy interface, see :doc:`/development/legacy_plugins`.

.. _plugin_overview:

.. important::

    In your plugin module, **standard NumPy** (*not* the wrapped Autograd version of NumPy)
    should be imported in all places (i.e., ``import numpy as np``).

PennyLane allows external quantum libraries to take advantage of the automatic differentiation
ability of PennyLane via plugins. Writing your own plugin is a simple and easy process. In this section, we discuss
the methods and concepts involved in the device interface. To see an implementation of a
minimal device, we recommend looking at the implementation in ``pennylane/devices/reference_qubit.py``.

Creating your device
--------------------

In order to define a custom device, you only need to override the :meth:`~.devices.Device.execute` method.

.. code-block:: python

    from pennylane.devices import Device, ExecutionConfig
    from pennylane.tape import QuantumScript, QuantumScriptOrBatch

    class MyDevice(Device):
        """My Documentation."""

        def execute(
            self,
            circuits: QuantumScriptOrBatch,
            execution_config: ExecutionConfig | None = None
        ):
            # your implementation here.

For example:

.. code-block:: python

    class MyDevice(Device):
        """My Documentation."""

        def execute(
            self,
            circuits: QuantumScriptOrBatch,
            execution_config: ExecutionConfig | None = None
        )
            return 0.0 if isinstance(circuits, qml.tape.QuantumScript) else tuple(0.0 for c in circuits)

    dev = MyDevice()

    @qml.qnode(dev)
    def circuit():
        return qml.state()

    circuit()

This execute method works in tandem with the optional :meth:`Device.preprocess_transforms <pennylane.devices.Device.preprocess_transforms>`
and :meth:`Device.setup_execution_config`, described below in more detail. Preprocessing transforms
turns generic circuits into ones supported by the device or raises an error if the circuit is invalid.
Execution produces numerical results from those supported circuits.

In a more minimal example, for any initial batch of quantum tapes and a config object, we expect to be able to do:

.. code-block:: python

    execution_config = dev.setup_execution_config(initial_config)
    compile_pipeline = dev.preprocess_transforms(execution_config)
    circuit_batch, postprocessing = compile_pipeline(initial_circuit_batch)
    results = dev.execute(circuit_batch, execution_config)
    final_results = postprocessing(results)

Shots
-----

While the workflow default for shots is specified by :attr:`Device.shots <pennylane.devices.Device.shots>`, the device itself should use
the number of shots specified in the :attr:`~.QuantumScript.shots` property for each quantum tape.
By pulling shots dynamically for each circuit, users can efficiently distribute a shot budget across batch of
circuits.

>>> tape0 = qml.tape.QuantumScript([], [qml.sample(wires=0)], shots=5)
>>> tape1 = qml.tape.QuantumScript([], [qml.sample(wires=0)], shots=10)
>>> dev = qml.device('default.qubit')
>>> dev.execute((tape0, tape1))
(array([[0],
        [0],
        [0],
        [0],
        [0]]),
 array([[0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
        [0]]))

The :class:`~.measurements.Shots` class describes the shots. Users can optionally specify a shot vector, or
different numbers of shots to use when calculating the final expectation value.

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
processing into the expectation value is done with shots ``0:5``, ``5:505``, and ``505:1505`` respectively.

``shots.total_shots is None`` indicates an analytic execution (infinite shots). ``bool(shots)`` also
can be used to detect the difference between finite shots and analytic executions. If ``shots`` is truthy,
then finite shots exist. If ``shots`` is falsy, then an analytic execution should be performed.

Preprocessing
-------------

There are two components of preprocessing circuits for device execution:

1) Create a :class:`~.CompilePipeline` capable of turning an arbitrary batch of :class:`~.QuantumScript`\ s into a new batch of tapes supported by the ``execute`` method.
2) Setup the :class:`~.ExecutionConfig` dataclass by filling in device options and making decisions about differentiation.

These two tasks are performed by :meth:`~.devices.Device.setup_execution_config` and :meth:`~.devices.Device.preprocess_transforms`
respectively. Once the compile pipeline has been applied to a batch of circuits, the result
circuit batch produced by the program should be run via ``Device.execute`` without error:

.. code-block:: python

    execution_config = dev.setup_execution_config(initial_config)
    compile_pipeline = dev.preprocess_transforms(execution_config)
    batch, fn = compile_pipeline(initial_batch)
    fn(dev.execute(batch, execution_config))

This section will focus on :meth:`~.devices.Device.preprocess_transforms`, see the section on the :ref:`**Execution Config** <execution_config>`
below for more information on :meth:`~.devices.Device.setup_execution_config`.

PennyLane can potentially provide a default implementation of a compile pipeline through :meth:`~.devices.Device.preprocess_transforms`,
which should be sufficient for most plugin devices. This requires that a TOML-formatted configuration
file is defined for your device. The details of this configuration file is described :ref:`the next section <device_capabilities>`.
The default preprocessing program will be constructed based on what is declared in this file if provided.

You could override the :meth:`~.devices.Device.preprocess_transforms` method with a completely
customized implementation, or extend the default behaviour by adding new transforms.

The :meth:`~.devices.Device.preprocess_transforms` method should start with creating a compile pipeline:

.. code-block:: python

    program = qml.CompilePipeline()

Once a program is created, individual transforms can be added to the program with the :meth:`~.CompilePipeline.add_transform` method.

.. code-block:: python

    from pennylane.devices.preprocess import validate_device_wires, validate_measurements, decompose

    program.add_transform(validate_device_wires, wires=qml.wires.Wires((0,1,2)), name="my_device")
    program.add_transform(validate_measurements, name="my_device")
    program.add_transform(qml.defer_measurements)
    program.add_transform(qml.transforms.split_non_commuting)

    def supports_operation(op): 
        return getattr(op, "name", None) in operation_names
        
    program.add_transform(decompose, stopping_condition=supports_operation, name="my_device")
    program.add_transform(qml.transforms.broadcast_expand)

Preprocessing and validation can also exist inside the :meth:`~devices.Device.execute` method, but placing them
in the preprocessing program has several benefits. Validation can happen earlier, leading to fewer resources
spent before the error is raised. Users can inspect, draw, and spec out the tapes at different stages throughout
preprocessing. This provides users a better awareness of what the device is actually executing. When device
gradients are used, the preprocessing transforms are tracked by the machine learning interfaces. With the
ML framework tracking the classical component of preprocessing, the device does not need to manually track the
classical component of any decompositions or compilation. For example,

>>> @qml.qnode(qml.device('reference.qubit', wires=2))
... def circuit(x):
...     qml.IsingXX(x, wires=(0,1))
...     qml.CH((0,1))
...     return qml.expval(qml.X(0))
>>> print(qml.draw(circuit, level="device")(0.5))
0: ─╭●──RX(0.50)─╭●────────────╭●──RY(-1.57)─┤  <Z>
1: ─╰X───────────╰X──RY(-0.79)─╰Z──RY(0.79)──┤     

Allows the user to see that both ``IsingXX`` and ``CH`` are decomposed by the device, and that
the diagonalizing gates for ``qml.expval(qml.X(0))`` are applied.

Even with these benefits, devices can still opt to place some transforms inside the ``execute``
method. For example, ``default.qubit`` maps wires to simulation indices inside ``execute`` instead
of in ``preprocess_transforms``.

The :meth:`~.devices.Device.execute` method can assume that device preprocessing has been performed on the input
tapes, and has no obligation to re-validate the input or provide sensible error messages. In the below example,
we see that ``default.qubit`` errors out when unsupported operations and unsupported measurements are present.

>>> op = qml.Permute([2,1,0], wires=(0,1,2))
>>> tape = qml.tape.QuantumScript([op], [qml.probs(wires=(0,1))])
>>> qml.device('default.qubit').execute(tape)
MatrixUndefinedError:
>>> tape = qml.tape.QuantumScript([], [qml.density_matrix(wires=0)], shots=50)
>>> qml.device('default.qubit').execute(tape)
AttributeError: 'DensityMatrixMP' object has no attribute 'process_samples'

Devices may define their own transforms following the description in the :ref:`transforms` module,
or can include in-built transforms such as:

* :func:`pennylane.defer_measurements`
* :func:`pennylane.dynamic_one_shot`
* :func:`pennylane.transforms.broadcast_expand`
* :func:`pennylane.transforms.split_non_commuting`
* :func:`pennylane.transforms.transpile`
* :func:`pennylane.transforms.diagonalize_measurements`
* :func:`pennylane.transforms.split_to_single_terms`
* :func:`pennylane.devices.preprocess.decompose`
* :func:`pennylane.devices.preprocess.validate_observables`
* :func:`pennylane.devices.preprocess.validate_measurements`
* :func:`pennylane.devices.preprocess.validate_device_wires`
* :func:`pennylane.devices.preprocess.validate_multiprocessing_workers`
* :func:`pennylane.devices.preprocess.validate_adjoint_trainable_params`
* :func:`pennylane.devices.preprocess.no_sampling`
  
Custom Device Decompositions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`pennylane.devices.preprocess.decompose` transform is typically required as part of the compile pipeline that
decomposes unsupported operations to the device's native gate set. To define this transform a stopping condition needs
to be specified. This is a function mapping an operator to a boolean that determines whether the operator should be decomposed.

For example, for a device supporting the ``CNOT``, ``RX`` and ``RZ`` gates, the stopping condition and the decompose transform
can be specified like so:

.. code-block:: python

    from pennylane.devices.preprocess import decompose

    def stopping_condition(op):
        return op.name in {"CNOT", "RX", "RZ"}
    
    program.add_transform(decompose, stopping_condition=stopping_condition, name="my_device")

However, if the device native gate set is unreachable with the default decompositions defined in PennyLane,
an error will be raised. In this case, you may need to override the decompositions of certain operators
via the ``decomposer`` argument. 

For example, consider a device with ``RX``, ``RY`` and ``IsingXX`` as native gates but we want
to execute a circuit written in terms of ``CNOT`` s. Then, we can define a decomposition for ``CNOT`` 
(e.g., ``custom_decomposer``) and pass it to the decomposer kwarg:

.. code-block:: python

    def stopping_condition(op):
        return op.name in {"IsingXX", "RX", "RY"}

    def custom_decomposer(op):
        if isinstance(op, qml.CNOT):
            wires = op.wires
            return [
                qml.RY(np.pi/2, wires=wires[0]),
                qml.IsingXX(np.pi/2, wires=wires),
                qml.RX(-np.pi/2, wires=wires[0]),
                qml.RY(-np.pi/2, wires=wires[0]),
                qml.RY(-np.pi/2, wires=wires[1])
            ]
        return op.decomposition()
    
    program.add_transform(
        decompose,
        stopping_condition=stopping_condition,
        decomposer=custom_decomposer,
        name="my_device"
    )

There is also an experimental graph-based decomposition algorithm (activated via
:func:`qml.decomposition.enable_graph() <pennylane.decomposition.enable_graph>`) that can
be leveraged when overriding the decompositions of certain operators. To make your device
compatible with this new system, the ``target_gates`` kwarg in the :func:`pennylane.devices.preprocess.decompose` transform
needs to be specified as part of the compile pipeline. Note that the stopping condition function
defines whether an operator should be decomposed, while the ``target_gates`` defines the set of operator
types that the graph-based decomposition algorithm needs to target.

In this case, the decomposition for the CNOT needs to be specified as a quantum function, ``decompose_cnot``, and
registered with ``qml.add_decomps``:

.. code-block:: python

    @qml.register_resources({qml.RY: 3, qml.RX: 1, qml.IsingXX: 1})
    def decompose_cnot(wires, **_):
        qml.RY(np.pi/2, wires=wires[0])
        qml.IsingXX(np.pi/2, wires=wires)
        qml.RX(-np.pi/2, wires=wires[0])
        qml.RY(-np.pi/2, wires=wires[0])
        qml.RY(-np.pi/2, wires=wires[1])

    qml.add_decomps(qml.CNOT, decompose_cnot)

    program.add_transform(
        decompose,
        stopping_condition=stopping_condition,
        decomposer=custom_decomposer,
        device_wires=[0, 1],
        target_gates={qml.IsingXX, "RX", "RY"},
        name="my_device"
    )

.. _device_capabilities:

Device Capabilities
-------------------

Optionally, you can add a ``config_filepath`` class variable pointing to your configuration file.
This file should be a `toml file <https://toml.io/en/>`_ that describes which gates and features are
supported by your device, i.e., what the :meth:`~pennylane.devices.Device.execute` method accepts.

.. code-block:: python

    from os import path
    from pennylane.devices import Device

    class MyDevice(Device):
        """My Documentation."""

        config_filepath = path.join(path.dirname(__file__), "relative/path/to/config.toml")

This configuration file will be loaded into another class variable :attr:`~pennylane.devices.Device.capabilities`
that is used in the default implementation of :meth:`~pennylane.devices.Device.preprocess_transforms`
if you choose not to override it yourself as described above. Note that this file must be declared as
package data as instructed at the end of :ref:`this section <packaging>`.

Below is an example configuration file defining all accepted fields, with inline descriptions of
how to fill these fields. All headers and fields are generally required, unless stated otherwise.

.. code-block:: toml

    schema = 3

    # The set of all gate types supported at the runtime execution interface of the
    # device, i.e., what is supported by the `execute` method. The gate definitions
    # should have the following format:
    #
    #   GATE = { properties = [ PROPS ], conditions = [ CONDS ] }
    #
    # where PROPS and CONS are zero or more comma separated quoted strings.
    #
    # PROPS: additional support provided for each gate.
    #        - "controllable": if a controlled version of this gate is supported.
    #        - "invertible": if the adjoint of this operation is supported.
    #        - "differentiable": if device gradient is supported for this gate.
    # CONDS: constraints on the support for each gate.
    #        - "analytic" or "finiteshots": if this operation is only supported in
    #          either analytic execution or with shots, respectively.
    #
    [operators.gates]

    PauliX = { properties = ["controllable", "invertible"] }
    PauliY = { properties = ["controllable", "invertible"] }
    PauliZ = { properties = ["controllable", "invertible"] }
    RY = { properties = ["controllable", "invertible", "differentiable"] }
    RZ = { properties = ["controllable", "invertible", "differentiable"] }
    CRY = { properties = ["invertible", "differentiable"] }
    CRZ = { properties = ["invertible", "differentiable"] }
    CNOT = { properties = ["invertible"] }

    # Observables supported by the device for measurements. The observables defined
    # in this section should have the following format:
    #
    #   OBSERVABLE = { conditions = [ CONDS ] }
    #
    # where CONDS is zero or more comma separated quoted strings, same as above.
    #
    # CONDS: constraints on the support for each observable.
    #        - "analytic" or "finiteshots": if this observable is only supported in
    #          either analytic execution or with shots, respectively.
    #        - "terms-commute": if a composite operator is only supported under the
    #          condition that its terms commute.
    #
    [operators.observables]

    PauliX = { }
    PauliY = { }
    PauliZ = { }
    Hamiltonian = { conditions = [ "terms-commute" ] }
    Sum = { conditions = [ "terms-commute" ] }
    SProd = { }
    Prod = { }

    # Types of measurement processes supported on the device. The measurements in
    # this section should have the following format:
    #
    #   MEASUREMENT_PROCESS = { conditions = [ CONDS ] }
    #
    # where CONDS is zero or more comma separated quoted strings, same as above.
    #
    # CONDS: constraints on the support for each measurement process.
    #        - "analytic" or "finiteshots": if this measurement is only supported
    #          in either analytic execution or with shots, respectively.
    #
    [measurement_processes]

    ExpectationMP = { }
    SampleMP = { }
    CountsMP = { conditions = ["finiteshots"] }
    StateMP = { conditions = ["analytic"] }

    # Additional support that the device may provide that informs the compilation
    # process. All accepted fields and their default values are listed below.
    [compilation]

    # Whether the device is compatible with qjit.
    qjit_compatible = false

    # Whether the device requires run time generation of the quantum circuit.
    runtime_code_generation = false

    # Whether the device supports allocating and releasing qubits during execution.
    dynamic_qubit_management = false

    # Whether simultaneous measurements on overlapping wires is supported.
    overlapping_observables = true

    # Whether simultaneous measurements of non-commuting observables is supported.
    # If false, a circuit with multiple non-commuting measurements will have to be
    # split into multiple executions for each subset of commuting measurements.
    non_commuting_observables = false

    # Whether the device supports initial state preparation.
    initial_state_prep = false

    # The methods of handling mid-circuit measurements that the device supports,
    # e.g., "one-shot", "tree-traversal", "device", etc. An empty list indicates
    # that the device does not support mid-circuit measurements.
    supported_mcm_methods = [ ]

This TOML configuration file is optional for PennyLane but required for Catalyst integration,
i.e., compatibility with ``qml.qjit``. For more details, see `Custom Devices <https://docs.pennylane.ai/projects/catalyst/en/stable/dev/custom_devices.html>`_.

Mid Circuit Measurements
~~~~~~~~~~~~~~~~~~~~~~~~

PennyLane supports :ref:`mid-circuit measurements <mid_circuit_measurements>`, i.e., measurements
in the middle of a quantum circuit used to shape the structure of the circuit dynamically, and to
gather information about the quantum state during the circuit execution. This might not be natively
supported by all devices.

If your device does not support mid-circuit measurements, the :ref:`deferred measurements <deferred_measurements>`
method will be applied. On the other hand, if your device is able to evaluate dynamic circuits by
executing them one shot at a time, sampling a dynamic execution path for each shot, you should
include ``"one-shot"`` as one of the ``supported_mcm_methods`` in your configuration file. When the
``"one-shot"`` method is requested on the ``QNode``, the :ref:`dynamic one-shot <one_shot_transform>`
method will be applied.

Both methods mentioned above involve compile pipelines to be applied on the circuits that prepare
them for device execution and post-processing functions to aggregate the results. Alternatively, if
your device natively supports all mid-circuit measurement features provided in PennyLane, you should
include ``"device"`` as one of the ``supported_mcm_methods``.

Wires
-----

Devices can now either:

1) Strictly use wires provided by the user on initialization: ``device(name, wires=wires)``
2) Infer the number and order of wires provided by the submitted circuit
3) Strictly require specific wire labels

Option 2 allows workflows to change the number and labeling of wires over time, but sometimes users want
to enforce a wire convention and labels. If a user does provide wires, :meth:`~.devices.Device.preprocess_transforms` should
validate that submitted circuits only have wires in the requested range.

>>> dev = qml.device('default.qubit', wires=1)
>>> circuit = qml.tape.QuantumScript([qml.CNOT((0,1))], [qml.state()])
>>> dev.preprocess_transforms()((circuit,))
WireError: Cannot run circuit(s) of default.qubit as they contain wires not found on the device.

PennyLane wires can be any hashable object, where wire labels are distinguished by their equality and hash.
If working with successive integers (``0``, ``1``, ``2``, ...) is preferred internally,
the :meth:`~.QuantumScript.map_to_standard_wires` method can be used inside of 
the :meth:`~.devices.Device.execute` method. The :class:`~.map_wires` transform can also 
map the wires of the submitted circuit to internal labels.

Sometimes, hardware qubit labels cannot be arbitrarily mapped without a change in behaviour.
Connectivity, noise, performance, and
other constraints can make it so that operations on qubit 1 cannot be arbitrarily exchanged with the same operation
on qubit 2. In such a situation, the device can hard code a list of the only acceptable wire labels. In such a case, it
will be on the user to deliberately map wires if they wish such a thing to occur.

>>> qml.device('my_hardware').wires
<Wires = [0, 1, 2, 3]>
>>> qml.device('my_hardware', wires=(10, 11, 12, 13))
TypeError: MyHardware.__init__() got an unexpected keyword argument 'wires'

To implement such validation, a device developer can simply leave ``wires`` from the initialization
call signature and hard code the ``wires`` property. They should additionally make sure to include
``validate_device_wires`` in the compile pipeline.

.. code-block:: python

    class MyDevice(qml.devices.Device):

        def __init__(self, shots=None):
            super().__init__(shots=shots)

        @property
        def wires(self):
            return qml.wires.Wires((0,1,2,3))

.. _execution_config:

Execution Config
----------------

The execution config stores two kinds of information:

1) Information about how the device should perform the execution. Examples include ``device_options`` and ``gradient_method``.
2) Information about how PennyLane should interact with the device. Examples include ``use_device_gradient`` and ``grad_on_execution``.

**Device options:**

Device options are any device specific options used to configure the behavior of an execution. For
example, ``default.qubit`` has ``max_workers``, ``rng``, and ``prng_key``. ``default.tensor`` has
``contract``, ``contraction_optimizer``, ``cutoff``, ``c_dtype``, ``local_simplify``, ``method``, and ``max_bond_dim``. These options are often set
with default values on initialization. These values should be placed into the ``ExecutionConfig.device_options``
dictionary in :meth:`~.devices.Device.setup_execution_config`. Note that we do provide a default
implementation of this method, but you will most likely need to override it yourself.

>>> dev = qml.device('default.tensor', wires=2, max_bond_dim=4, contract="nonlocal", c_dtype=np.complex64)
>>> dev.setup_execution_config().device_options
{'contract': 'nonlocal',
 'contraction_optimizer': 'auto-hq',
 'cutoff': None,
 'c_dtype': numpy.complex64,
 'local_simplify': 'ADCRS',
 'max_bond_dim': 4,
 'method': 'mps'}

Even if the property is stored as an attribute on the device, execution should pull the value of
these properties from the config instead of from the device instance. While not yet integrated at
the top user level, we aim to allow dynamic configuration of the device.

>>> dev = qml.device('default.qubit')
>>> config = qml.devices.ExecutionConfig(device_options={"rng": 42})
>>> tape = qml.tape.QuantumTape([qml.Hadamard(0)], [qml.sample(wires=0)], shots=10)
>>> dev.execute(tape, config)
array([[1],
       [1],
       [0],
       [1],
       [0],
       [1],
       [0],
       [1],
       [0],
       [0]])
>>> dev.execute(tape, config)
array([[0],
       [1],
       [0],
       [0],
       [0],
       [1],
       [1],
       [0],
       [0],
       [0]])

By pulling options from this dictionary instead of from device properties, we unlock two key
pieces of functionality:

1) Track and specify the exact configuration of the execution by only inspecting the ``ExecutionConfig`` object
2) Dynamically configure the device over the course of a workflow.

**Workflow Configuration:**

Note that these properties are only applicable to devices that provided derivatives or VJPs. If your device
does not provide derivatives, you can safely ignore these properties.

The workflow options are ``use_device_gradient``, ``use_device_jacobian_product``, ``grad_on_execution``,
and ``convert_to_numpy``. 
``use_device_gradient=True`` indicates that workflow should request derivatives from the device. 
``grad_on_execution=True`` indicates a preference to use ``execute_and_compute_derivatives`` instead
of ``execute`` followed by ``compute_derivatives``. ``use_device_jacobian_product`` indicates
a request to call ``compute_vjp`` instead of ``compute_derivatives``. Note that if ``use_device_jacobian_product``
is ``True``, this takes precedence over calculating the full jacobian. If the device can accept ML framework parameters, like
jax, ``convert_to_numpy=False`` should be specified. Then the parameters will not be converted, and special
interface-specific processing (like executing inside a ``jax.pure_callback`` when using ``jax.jit``) will be needed.

>>> config = qml.devices.ExecutionConfig(gradient_method="adjoint")
>>> processed_config = qml.device('default.qubit').setup_execution_config(config)
>>> processed_config.use_device_jacobian_product
True
>>> processed_config.use_device_gradient
True
>>> processed_config.grad_on_execution
True
>>> processed_config.convert_to_numpy
True

Execution
---------

For documentation on the expected result type output, please refer to :ref:`ReturnTypeSpec`.

The device API allows individual devices to calculate results in whatever way makes sense for
the individual device. With this freedom over the implementation does come more responsibility
to handle each stage in the process. 

PennyLane does provide some helper functions to assist in executing
circuits. Any ``StateMeasurement`` has ``process_state`` and ``process_density_matrix`` methods for
classical post-processing of a state vector or density matrix, and ``SampleMeasurement``'s implement
both ``process_samples`` and ``process_counts``. The ``pennylane.devices.qubit`` module also contains
functions that implement parts of a Python-based statevector simulation.

Suppose you are accessing hardware that can only return raw samples. Here, we use the ``mp.process_samples``
methods to process the subsamples into the requested final result object. Note that we need
to squeeze out singleton dimensions when we have no shot vector or a single measurement.

.. code-block:: python

    def single_tape_execution(tape) -> qml.typing.Result:
        samples = get_samples(tape)
        results = []
        for lower, upper in tape.shots.bins():
            sub_samples = samples[lower:upper]
            results.append(
                tuple(mp.process_samples(sub_samples, tape.wires) for mp in tape.measurements)
            )
        if len(tape.measurements) == 1:
            results = tuple(res[0] for res in results)
        if tape.shots.has_partitioned_shots:
            results = results[0]
        return results
    

Device Modifiers
----------------

PennyLane currently provides two device modifiers.

* :func:`pennylane.devices.modifiers.single_tape_support`
* :func:`pennylane.devices.modifiers.simulator_tracking`

For example, with a custom device we can add simulator-style tracking and the ability
to handle a single circuit. See the documentation for each modifier for more details.

.. code-block:: python

    @simulator_tracking
    @single_tape_support
    class MyDevice(qml.devices.Device):

        def execute(self, circuits, execution_config: ExecutionConfig | None = None):
            return tuple(0.0 for _ in circuits)

>>> dev = MyDevice()
>>> tape = qml.tape.QuantumTape([qml.S(0)], [qml.expval(qml.X(0))])
>>> with dev.tracker:
...     out = dev.execute(tape)
>>> out
0.0
>>> dev.tracker.history
{'batches': [1],
 'simulations': [1],
 'executions': [1],
 'results': [0.0],
 'resources': [Resources(num_wires=1, num_gates=1,
 gate_types=defaultdict(<class 'int'>, {'S': 1}),
 gate_sizes=defaultdict(<class 'int'>, {1: 1}), depth=1,
 shots=Shots(total_shots=None, shot_vector=()))]}


Device tracker support
----------------------

The device tracker stores and records information when tracking mode is turned on. Devices can store data like
the number of executions, number of shots, number of batches, or remote simulator cost for users to interact with
in a customizable way.

Three aspects of the :class:`~pennylane.Tracker` class are relevant to plugin designers:

* The boolean ``active`` attribute that denotes whether or not to update and record
* ``update`` method which accepts keyword-value pairs and stores the information
* ``record`` method which users can customize to log, print, or otherwise do something with the stored information

To gain simulation-like tracking behavior, the :func:`~.devices.modifiers.simulator_tracking` decorator can be added
to the device:

.. code-block:: python

    @qml.devices.modifiers.simulator_tracking
    class MyDevice(Device):
        ...


``simulator_tracking`` is useful when the device can simultaneously measure non-commuting measurements or
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
:meth:`~pennylane.devices.Tracker.update` method can accept any combination of
keyword-value pairs.  For example, a device could also track cost and a job ID via:

.. code-block:: python

  price_for_execution = 0.10
  job_id = "abcde"
  self.tracker.update(price=price_for_execution, job_id=job_id)


.. _packaging:

Identifying and installing your device
--------------------------------------

When performing a hybrid computation using PennyLane, one of the first steps is often to
initialize the quantum device(s). PennyLane identifies the devices via their ``name``,
which allows the device to be initialized in the following way:

.. code-block:: python

    import pennylane as qp
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

If a :ref:`configuration file <device_capabilities>` is defined for your device, you will need
to declare it as package data in ``setup.py``:

.. code-block:: python

    from setuptools import setup, find_packages

    setup(
        ...
        include_package_data=True,
        package_data={
            'package_name' : ['path/to/config/device_name.toml'],
        },
        ...
    )

Alternatively, with ``include_package_data=True``, you can also declare the file in a ``MANIFEST.in``:

.. code-block::

    include path/to/config/device_name.toml

See `packaging data files <https://setuptools.pypa.io/en/stable/userguide/datafiles.html>`_
for a detailed explanation. This will ensure that PennyLane can correctly load the device and its
associated capabilities.
