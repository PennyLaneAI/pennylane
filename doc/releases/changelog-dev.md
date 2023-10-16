:orphan:

# Release 0.33.0-dev (development release)

<h3>New features since last release</h3>

<h4>Decompose circuits into the Clifford+T gateset 🧩</h4>

* A new transform called `clifford_t_decomposition` is available, which decomposes
  circuits into the Clifford+T gate set. 
  [(#ABCD)]()

  The Clifford+T universal gate set — `Hadamard`, `S`, `CNOT` and `T` — is paramount
  to the implementation of many fault-tolerant protocols on quantum computers. With 
  the new `clifford_t_decomposition` transform, circuits can be decomposed into 
  a basis consisting of Clifford, `RZ` and `GlobalPhase` operations. Here, Clifford 
  gates include `Identity`, `PauliX`, `PauliY`, `PauliZ`, `SX`, `S`, `Hadamard`, 
  `CNOT`, `CY`, `CZ`, `SWAP` and `ISWAP`. 

  ```python
  dev = qml.device("default.qubit", wires=2)

  def circuit():
      qml.SingleExcitation(0.2, [1, 0])
      return qml.state()

  clifford_circuit = qml.QNode(qml.transforms.clifford_t_decomposition(circuit), dev)
  print(qml.draw(clifford_circuit)())
  ```

  ```pycon
  0: ──RZ(-2.36)──H────╭●──S†─────────H──RZ(0.10)──H─╭●──S──────────H──S──T──GlobalPhase(-2.36)─┤
  1: ──RZ(-0.79)──H──S─╰X──RZ(-0.10)─────────────────╰X──RZ(-1.57)──H──T─────GlobalPhase(-2.36)─┤
  ```

  `RZ` gates are approximated in the Clifford+T basis using the method 
  described in [Ross and Selinger (2016)](https://arxiv.org/abs/1403.2975). The 
  `epsilon` argument in `clifford_t_decomposition` allows for the error in approximating
  `RZ` gates to be adjusted.

<h4>Postselection and statistics in mid-circuit measurements 📌</h4>

* TODO postselection

* Measurement statistics can now be collected for mid-circuit measurements.
  [(#4544)](https://github.com/PennyLaneAI/pennylane/pull/4544)

  ```python
  dev = qml.device("default.qubit")

  @qml.qnode(dev)
  def circ(x, y):
      qml.RX(x, wires=0)
      qml.RY(y, wires=1)
      m0 = qml.measure(1)
      return qml.expval(qml.PauliZ(0)), qml.expval(m0), qml.sample(m0)
  ```

  ```pycon
  >>> circ(1.0, 2.0, shots=10000)
  (0.5606, 0.7089, array([0, 1, 1, ..., 1, 1, 1]))
  ```
  
  Support is provided for both
  [finite-shot and analytic modes](https://docs.pennylane.ai/en/stable/introduction/circuits.html#shots)
  and devices default to using the
  [deferred measurement](https://docs.pennylane.ai/en/stable/code/api/pennylane.defer_measurements.html)
  principle to enact the mid-circuit measurements.

  In future releases, we will be exploring the ability to combine and manipulate mid-circuit
  measurements such as `qml.expval(m0 @ m1)` or `qml.expval(m0 @ qml.PauliZ(0))`.

<h4>Exponentiate Hamiltonians with flexible Trotter products 🤩</h4>

<h4>New device capabilities, integration with Catalyst, and more! ⚗️</h4>

* `default.qubit` now uses the new `qml.devices.Device` API and supporting functionality in
  `qml.devices.qubit`. If you experience any issues with the updated `default.qubit`, please let us
  know by [posting an issue](https://github.com/PennyLaneAI/pennylane/issues/new/choose). 
  The old version of the device is still
  accessible by the short name `default.qubit.legacy`, or directly via `qml.devices.DefaultQubitLegacy`.
  [(#4594)](https://github.com/PennyLaneAI/pennylane/pull/4594)
  [(#4436)](https://github.com/PennyLaneAI/pennylane/pull/4436)
  [(#4620)](https://github.com/PennyLaneAI/pennylane/pull/4620)
  [(#4632)](https://github.com/PennyLaneAI/pennylane/pull/4632)

  This changeover has a number of benefits for `default.qubit`, including:

  * The number of wires is now optional - simply having `qml.device("default.qubit")` is valid! If
    wires are not provided at instantiation, the device automatically infers the required number of
    wires for each circuit provided for execution.

  * `default.qubit` is no longer silently swapped out with an interface-appropriate device when the
    backpropagation differentiation method is used. For example, consider:

    ```python
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, diff_method="backprop")
    def f(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliX(0))
    
    f(jax.numpy.array(0.2))
    ```
    
    In previous versions of PennyLane, the device will be swapped for the JAX equivalent:

    ```pycon
    >>> f.device
    <DefaultQubitJax device (wires=1, shots=None) at 0x7f8c8bff50a0>
    >>> f.device == dev
    False
    ```
    
    Now, `default.qubit` can itself dispatch to all of the interfaces in a backprop-compatible way
    and hence does not need to be swapped:

    ```pycon
    >>> f.device
    <default.qubit device (wires=1) at 0x7f20d043b040>
    >>> f.device == dev
    True
    ```

* Added support for drawing a QNode that has been decorated with `qjit` from PennyLane's
  [Catalyst](https://docs.pennylane.ai/projects/catalyst) library for just-in-time hybrid
  compilation.
  [(#4609)](https://github.com/PennyLaneAI/pennylane/pull/4609)

  ```python
  import catalyst

  @catalyst.qjit
  @qml.qnode(qml.device("lightning.qubit", wires=3))
  def circuit(x, y, z, c):
      """A quantum circuit on three wires."""

      @catalyst.for_loop(0, c, 1)
      def loop(i):
          qml.Hadamard(wires=i)

      qml.RX(x, wires=0)
      loop()  # pylint: disable=no-value-for-parameter
      qml.RY(y, wires=1)
      qml.RZ(z, wires=2)
      return qml.expval(qml.PauliZ(0))
  
  draw = qml.draw(circuit, decimals=None)(1.234, 2.345, 3.456, 1)
  ```
  
  ```pycon
  >>>draw
  "0: ──RX──H──┤  <Z>\n1: ──H───RY─┤     \n2: ──RZ─────┤     "
  ```
  
  Stay tuned for more integration of Catalyst into PennyLane!

* Extended ``qml.qchem.import_state`` to import more quantum chemistry wavefunctions, from MPS DMRG
  and SHCI classical calculations performed with the Block2 and Dice libraries.
  [#4523](https://github.com/PennyLaneAI/pennylane/pull/4523)
  [#4524](https://github.com/PennyLaneAI/pennylane/pull/4524)
  [#4626](https://github.com/PennyLaneAI/pennylane/pull/4626)
  [#4634](https://github.com/PennyLaneAI/pennylane/pull/4634)

  Check out our [how-to guide](https://pennylane.ai/qml/demos/tutorial_initial_state_preparation)
  to learn more about how PennyLane integrates with your favourite quantum chemistry libraries.

<h3>Improvements 🛠</h3>

<h4>More PyTrees!</h4>

* `MeasurementProcess` and `QuantumScript` objects are now registered as JAX PyTrees.
  [(#4607)](https://github.com/PennyLaneAI/pennylane/pull/4607)
  [(#4608)](https://github.com/PennyLaneAI/pennylane/pull/4608)

  TODO

<h4>Transforms</h4>

* All quantum functions transforms are update to the new transform program system.
  [(#4439)](https://github.com/PennyLaneAI/pennylane/pull/4439)

* All batch transforms are updated to the new transform program system.
  [(#4440)](https://github.com/PennyLaneAI/pennylane/pull/4440)

* Quantum information transforms are updated to the new transform program system.
  [(#4569)](https://github.com/PennyLaneAI/pennylane/pull/4569)

* Operator transforms `qml.matrix`, `qml.eigvals`, `qml.generator`, and `qml.transforms.to_zx` are updated
  to the new transform program system.
  [(#4573)](https://github.com/PennyLaneAI/pennylane/pull/4573)

* Add the method ``add_transform`` and ``insert_front_transform`` transform in the ``TransformProgram``.
  [(#4559)](https://github.com/PennyLaneAI/pennylane/pull/4559)

* Dunder ``__add__`` method is added to the ``TransformProgram`` class, therefore two programs can be added using ``+`` .
  [(#4549)](https://github.com/PennyLaneAI/pennylane/pull/4549)

* Transforms can be applied on devices following the new device API.
 [(#4667)](https://github.com/PennyLaneAI/pennylane/pull/4667)

* All gradient transforms are updated to the new transform program system.
 [(#4595)](https://github.com/PennyLaneAI/pennylane/pull/4595)

* `pennylane.defer_measurements` will now exit early if the input does not contain mid circuit measurements.
  [(#4659)](https://github.com/PennyLaneAI/pennylane/pull/4659)

<h4>Improving QChem and existing algorithms</h4>

* The qchem ``fermionic_dipole`` and ``particle_number`` functions are updated to use a
  ``FermiSentence``. The deprecated features for using tuples to represent fermionic operations are
  removed.
  [(#4546)](https://github.com/PennyLaneAI/pennylane/pull/4546)
  [(#4556)](https://github.com/PennyLaneAI/pennylane/pull/4556)

* Tensor-network template `qml.MPS` now supports changing `offset` between subsequent blocks for more flexibility.
  [(#4531)](https://github.com/PennyLaneAI/pennylane/pull/4531)

* Improve builtin types support with `qml.pauli_decompose`.
  [(#4577)](https://github.com/PennyLaneAI/pennylane/pull/4577)

* `AmplitudeEmbedding` now inherits from `StatePrep`, allowing for it to not be decomposed
  when at the beginning of a circuit, thus behaving like `StatePrep`.
  [(#4583)](https://github.com/PennyLaneAI/pennylane/pull/4583)

<h4>Next-generation device API</h4>

* `default.qubit` now tracks the number of equivalent qpu executions and total shots
  when the device is sampling. Note that `"simulations"` denotes the number of simulation passes, where as
  `"executions"` denotes how many different computational bases need to be sampled in. Additionally, the
  new `default.qubit` also tracks the results of `device.execute`.
  [(#4628)](https://github.com/PennyLaneAI/pennylane/pull/4628)
  [(#4649)](https://github.com/PennyLaneAI/pennylane/pull/4649)

* `DefaultQubit2` can now accept a `jax.random.PRNGKey` as a `seed`, to set the key for the JAX pseudo random 
  number generator when using the JAX interface. This corresponds to the `prng_key` on 
  `DefaultQubitJax` in the old API.
  [(#4596)](https://github.com/PennyLaneAI/pennylane/pull/4596)

* `DefaultQubit2` dispatches to a faster implementation for applying `ParametrizedEvolution` to a state
  when it is more efficient to evolve the state than the operation matrix.
  [(#4598)](https://github.com/PennyLaneAI/pennylane/pull/4598)
  [(#4620)](https://github.com/PennyLaneAI/pennylane/pull/4620)

* Wires can be provided to the new device API.
  [(#4538)](https://github.com/PennyLaneAI/pennylane/pull/4538)
  [(#4562)](https://github.com/PennyLaneAI/pennylane/pull/4562)

* `qml.sample()` in the new device API now returns a `np.int64` array instead of `np.bool8`.
  [(#4539)](https://github.com/PennyLaneAI/pennylane/pull/4539)

* The new device API now has a `repr()`
  [(#4562)](https://github.com/PennyLaneAI/pennylane/pull/4562)

* `DefaultQubit2` now works as expected with measurement processes that don't specify wires.
  [(#4580)](https://github.com/PennyLaneAI/pennylane/pull/4580)

* Various changes to measurements to improve feature parity between the legacy `default.qubit` and
  the new `DefaultQubit2`. This includes not trying to squeeze batched `CountsMP` results and implementing
  `MutualInfoMP.map_wires`.
  [(#4574)](https://github.com/PennyLaneAI/pennylane/pull/4574)

* `devices.qubit.simulate` now accepts an interface keyword argument. If a QNode with `DefaultQubit2`
  specifies an interface, the result will be computed with that interface.
  [(#4582)](https://github.com/PennyLaneAI/pennylane/pull/4582)

* `ShotAdaptiveOptimizer` has been updated to pass shots to QNode executions instead of overriding
  device shots before execution. This makes it compatible with the new device API.
  [(#4599)](https://github.com/PennyLaneAI/pennylane/pull/4599)

* `pennylane.devices.preprocess` now offers the transforms `decompose`, `validate_observables`, `validate_measurements`,
  `validate_device_wires`, `validate_multiprocessing_workers`, `warn_about_trainable_observables`,
  and `no_sampling` to assist in the construction of devices under the new `devices.Device` API.
  [(#4659)](https://github.com/PennyLaneAI/pennylane/pull/4659)

<h4>Other improvements</h4>

* The `JacobianProductCalculator` abstract base class and implementation `TransformJacobianProducts`
  have been added to `pennylane.interfaces.jacobian_products`.
  [(#4435)](https://github.com/PennyLaneAI/pennylane/pull/4435)

* The density matrix aspects of `StateMP` have been split into their own measurement
  process, `DensityMatrixMP`.
  [(#4558)](https://github.com/PennyLaneAI/pennylane/pull/4558)

* The `StateMP` measurement now accepts a wire order (e.g., a device wire order). The `process_state`
  method will re-order the given state to go from the inputted wire-order to the process's wire-order.
  If the process's wire-order contains extra wires, it will assume those are in the zero-state.
  [(#4570)](https://github.com/PennyLaneAI/pennylane/pull/4570)
  [(#4602)](https://github.com/PennyLaneAI/pennylane/pull/4602)

* `StateMeasurement.process_state` now assumes the input is flat. `ProbabilityMP.process_state` has
  been updated to reflect this assumption and avoid redundant reshaping.
  [(#4602)](https://github.com/PennyLaneAI/pennylane/pull/4602)

* `qml.exp` returns a more informative error message when decomposition is unavailable for non-unitary operator.
  [(#4571)](https://github.com/PennyLaneAI/pennylane/pull/4571)

* Added `qml.math.get_deep_interface` to get the interface of a scalar hidden deep in lists or tuples.
  [(#4603)](https://github.com/PennyLaneAI/pennylane/pull/4603)

* Updated `qml.math.ndim` and `qml.math.shape` to work with built-in lists/tuples that contain
  interface-specific scalar data, eg `[(tf.Variable(1.1), tf.Variable(2.2))]`.
  [(#4603)](https://github.com/PennyLaneAI/pennylane/pull/4603)

* When decomposing a unitary matrix with `one_qubit_decomposition`, and opting to include the `GlobalPhase` 
  in the decomposition, the phase is no longer cast to `dtype=complex`.
  [(#4653)](https://github.com/PennyLaneAI/pennylane/pull/4653)

* `qml.cut_circuit` is now compatible with circuits that compute the expectation values of Hamiltonians 
  with two or more terms.
  [(#4642)](https://github.com/PennyLaneAI/pennylane/pull/4642)


* `_qfunc_output` has been removed from `QuantumScript`, as it is no longer necessary. There is
  still a `_qfunc_output` property on `QNode` instances.
  [(#4651)](https://github.com/PennyLaneAI/pennylane/pull/4651)

* `qml.data.load` properly handles parameters that come after `'full'`
  [(#4663)](https://github.com/PennyLaneAI/pennylane/pull/4663)

* The `qml.jordan_wigner` function has been modified to optionally remove the imaginary components
  of the computed qubit operator, if imaginary components are smaller than a threshold. 
  [(#4639)](https://github.com/PennyLaneAI/pennylane/pull/4639)


<h3>Breaking changes 💔</h3>

* `default.qubit` now implements the new device API. If you initialize a device
  with `qml.device("default.qubit")`, all methods and properties that were tied to the old
  device API will no longer be on the device. The legacy version can still be accessed with
  `qml.device("default.qubit.legacy", wires=n_wires)`.
  [(#4436)](https://github.com/PennyLaneAI/pennylane/pull/4436)

* The `__eq__` and `__hash__` methods of `Operator` and `MeasurementProcess` no longer rely on the
  object's address is memory. Using `==` with operators and measurement processes will now behave the
  same as `qml.equal`, and objects of the same type with the same data and hyperparameters will have
  the same hash.
  [(#4536)](https://github.com/PennyLaneAI/pennylane/pull/4536)

  In the following scenario, the second and third code blocks show the previous and current behaviour
  of operator and measurement process equality, determined by the `__eq__` dunder method:

  ```python
  op1 = qml.PauliX(0)
  op2 = qml.PauliX(0)
  op3 = op1
  ```
  Old behaviour:
  ```pycon
  >>> op1 == op2
  False
  >>> op1 == op3
  True
  ```
  New behaviour:
  ```pycon
  >>> op1 == op2
  True
  >>> op1 == op3
  True
  ```

  The `__hash__` dunder method defines the hash of an object. The default hash of an object
  is determined by the objects memory address. However, the new hash is determined by the
  properties and attributes of operators and measurement processes. Consider the scenario below.
  The second and third code blocks show the previous and current behaviour.

  ```python
  op1 = qml.PauliX(0)
  op2 = qml.PauliX(0)
  ```
  Old behaviour:
  ```pycon
  >>> print({op1, op2})
  {PauliX(wires=[0]), PauliX(wires=[0])}
  ```
  New behaviour:
  ```pycon
  >>> print({op1, op2})
  {PauliX(wires=[0])}
  ```

* The old return type and associated functions ``qml.enable_return`` and ``qml.disable_return`` are removed.
  [(#4503)](https://github.com/PennyLaneAI/pennylane/pull/4503)

* The ``mode`` keyword argument in ``QNode`` is removed. Please use ``grad_on_execution`` instead.
  [(#4503)](https://github.com/PennyLaneAI/pennylane/pull/4503)

* The CV observables ``qml.X`` and ``qml.P`` are removed. Please use ``qml.QuadX`` and ``qml.QuadP`` instead.
  [(#4533)](https://github.com/PennyLaneAI/pennylane/pull/4533)

* The ``sampler_seed`` argument of ``qml.gradients.spsa_grad`` has been removed.
  Instead, the ``sampler_rng`` argument should be set, either to an integer value, which will be used
  to create a PRNG internally, or to a NumPy pseudo-random number generator (PRNG) created via
  ``np.random.default_rng(seed)``.
  [(#4550)](https://github.com/PennyLaneAI/pennylane/pull/4550)

* The ``QuantumScript.set_parameters`` method and the ``QuantumScript.data`` setter have
  been removed. Please use ``QuantumScript.bind_new_parameters`` instead.
  [(#4548)](https://github.com/PennyLaneAI/pennylane/pull/4548)

* The method ``tape.unwrap()`` and corresponding ``UnwrapTape`` and ``Unwrap`` classes are removed.
  Instead of ``tape.unwrap()``, use ``qml.transforms.convert_to_numpy_parameters``.
  [(#4535)](https://github.com/PennyLaneAI/pennylane/pull/4535)

* `MeasurementProcess.eigvals()` now raises an `EigvalsUndefinedError` if the measurement observable
  does not have eigenvalues.
  [(#4544)](https://github.com/PennyLaneAI/pennylane/pull/4544)

* The device test suite now converts device kwargs to integers or floats if they can be converted to integers or floats.
  [(#4640)](https://github.com/PennyLaneAI/pennylane/pull/4640)

* The ``RandomLayers.compute_decomposition`` keyword argument ``ratio_imprivitive`` has been changed to
  ``ratio_imprim`` to match the call signature of the operation.
  [(#4552)](https://github.com/PennyLaneAI/pennylane/pull/4552)

* The private `TmpPauliRot` operator used for `SpecialUnitary` no longer decomposes to nothing
  when the theta value is trainable.
  [(#4585)](https://github.com/PennyLaneAI/pennylane/pull/4585)

* `ProbabilityMP.marginal_prob` has been removed. Its contents have been moved into `process_state`,
  which effectively just called `marginal_prob` with `np.abs(state) ** 2`.
  [(#4602)](https://github.com/PennyLaneAI/pennylane/pull/4602)

<h3>Deprecations 👋</h3>

* The following decorator syntax for transforms has been deprecated and will raise a warning:
  [(#4457)](https://github.com/PennyLaneAI/pennylane/pull/4457/)

  ```python
  @transform_fn(**transform_kwargs)
  @qml.qnode(dev)
  def circuit():
      ...
  ```
  
  If you are using a transform that has supporting `transform_kwargs`, please call the
  transform directly using `circuit = transform_fn(circuit, **transform_kwargs)`,
  or use `functools.partial`:

  ```python
  @functools.partial(transform_fn, **transform_kwargs)
  @qml.qnode(dev)
  def circuit():
      ...
  ```

* The ``prep`` keyword argument in ``QuantumScript`` is deprecated and will be removed from `QuantumScript`.
  ``StatePrepBase`` operations should be placed at the beginning of the `ops` list instead.
  [(#4554)](https://github.com/PennyLaneAI/pennylane/pull/4554)

* `qml.gradients.pulse_generator` becomes `qml.gradients.pulse_odegen` to adhere to paper naming conventions. During v0.33, `pulse_generator`
  is still available but raises a warning.
  [(#4633)](https://github.com/PennyLaneAI/pennylane/pull/4633)

<h3>Documentation 📝</h3>

* Add a warning section in DefaultQubit's docstring regarding the start method used in multiprocessing.
  This may help users circumvent issues arising in Jupyter notebooks on macOS for example.
  [(#4622)](https://github.com/PennyLaneAI/pennylane/pull/4622)

* Minor documentation improvements to the new device API. The documentation now correctly states that interface-specific
  parameters are only passed to the device for backpropagation derivatives. 
  [(#4542)](https://github.com/PennyLaneAI/pennylane/pull/4542)

* Add functions for qubit-simulation to the `qml.devices` sub-page of the "Internal" section.
  Note that these functions are unstable while device upgrades are underway.
  [(#4555)](https://github.com/PennyLaneAI/pennylane/pull/4555)

* Minor documentation improvement to the usage example in the `qml.QuantumMonteCarlo` page. Integral was missing the differential dx with respect to which the integration is being performed. [(#4593)](https://github.com/PennyLaneAI/pennylane/pull/4593)  

<h3>Bug fixes 🐛</h3>

* Providing `work_wires=None` to `qml.GroverOperator` no longer interprets `None` as a wire.
  [(#4668)](https://github.com/PennyLaneAI/pennylane/pull/4668)

* Fixed issue where `__copy__` method of the `qml.Select()` operator attempted to access un-initialized data.
[(#4551)](https://github.com/PennyLaneAI/pennylane/pull/4551)

* Fix `skip_first` option in `expand_tape_state_prep`.
  [(#4564)](https://github.com/PennyLaneAI/pennylane/pull/4564)

* `convert_to_numpy_parameters` now uses `qml.ops.functions.bind_new_parameters`. This reinitializes the operation and
  makes sure everything references the new numpy parameters.

* `tf.function` no longer breaks `ProbabilityMP.process_state` which is needed by new devices.
  [(#4470)](https://github.com/PennyLaneAI/pennylane/pull/4470)

* Fix mocking in the unit tests for `qml.qchem.mol_data`.
  [(#4591)](https://github.com/PennyLaneAI/pennylane/pull/4591)

* Fix `ProbabilityMP.process_state` so it allows for proper Autograph compilation. Without this,
  decorating a QNode that returns an `expval` with `tf.function` would fail when computing the
  expectation.
  [(#4590)](https://github.com/PennyLaneAI/pennylane/pull/4590)

* The `torch.nn.Module` properties are now accessible on a `pennylane.qnn.TorchLayer`.
  [(#4611)](https://github.com/PennyLaneAI/pennylane/pull/4611)

* `qml.math.take` with torch now returns `tensor[..., indices]` when the user requests
  the last axis (`axis=-1`). Without the fix, it would wrongly return `tensor[indices]`.
  [(#4605)](https://github.com/PennyLaneAI/pennylane/pull/4605)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Utkarsh Azad,
Thomas Bromley,
Isaac De Vlugt,
Stepan Fomichev,
Joana Fraxanet,
Diego Guala,
Soran Jahangiri,
Edward Jiang,
Korbinian Kottmann,
Christina Lee,
Lillian M. A. Frederiksen,
Vincent Michaud-Rioux,
Romain Moyard,
Daniel F. Nino,
Mudit Pandey,
Matthew Silverman,
Jay Soni.
