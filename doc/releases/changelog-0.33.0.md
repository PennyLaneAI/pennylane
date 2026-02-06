
# Release 0.33.0

<h3>New features since last release</h3>

<h4>Postselection and statistics in mid-circuit measurements 📌</h4>

* It is now possible to request postselection on a mid-circuit measurement.
  [(#4604)](https://github.com/PennyLaneAI/pennylane/pull/4604)

  This can be achieved by specifying the `postselect` keyword argument in `qml.measure` as either
  `0` or `1`, corresponding to the basis states.

  ```python
  import pennylane as qp

  dev = qml.device("default.qubit")

  @qml.qnode(dev, interface=None)
  def circuit():
      qml.Hadamard(wires=0)
      qml.CNOT(wires=[0, 1])
      qml.measure(0, postselect=1)
      return qml.expval(qml.PauliZ(1)), qml.sample(wires=1)
  ```
  
  This circuit prepares the :math:`| \Phi^{+} \rangle` Bell state and postselects on measuring
  :math:`|1\rangle` in wire `0`. The output of wire `1` is then also :math:`|1\rangle`
  at all times:

  ```pycon
  >>> circuit(shots=10)
  (-1.0, array([1, 1, 1, 1, 1, 1]))
  ```

  Note that the number of shots is less than the requested amount because we have thrown away the
  samples where :math:`|0\rangle` was measured in wire `0`.

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

<h4>Exponentiate Hamiltonians with flexible Trotter products 🐖</h4>

* Higher-order Trotter-Suzuki methods are now easily accessible through a new operation
  called `TrotterProduct`.
  [(#4661)](https://github.com/PennyLaneAI/pennylane/pull/4661)

  Trotterization techniques are an affective route towards accurate and efficient
  Hamiltonian simulation. The Suzuki-Trotter product formula allows for the ability
  to express higher-order approximations to the matrix exponential of a Hamiltonian, 
  and it is now available to use in PennyLane via the `TrotterProduct` operation. 
  Simply specify the `order` of the approximation and the evolution `time`.

  ```python
  coeffs = [0.25, 0.75]
  ops = [qml.PauliX(0), qml.PauliZ(0)]
  H = qml.dot(coeffs, ops)

  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def circuit():
      qml.Hadamard(0)
      qml.TrotterProduct(H, time=2.4, order=2)
      return qml.state()
  ```

  ```pycon
  >>> circuit()
  [-0.13259524+0.59790098j  0.        +0.j         -0.13259524-0.77932754j  0.        +0.j        ]
  ```

* Approximating matrix exponentiation with random product formulas, qDrift, is now available with the new `QDrift`
  operation.
  [(#4671)](https://github.com/PennyLaneAI/pennylane/pull/4671)

  As shown in [1811.08017](https://arxiv.org/pdf/1811.08017.pdf), qDrift is a Markovian process that can provide
  a speedup in Hamiltonian simulation. At a high level, qDrift works by randomly sampling from the Hamiltonian 
  terms with a probability that depends on the Hamiltonian coefficients. This method for Hamiltonian
  simulation is now ready to use in PennyLane with the `QDrift` operator. Simply specify the evolution `time`
  and the number of samples drawn from the Hamiltonian, `n`:
  
  ```python
  coeffs = [0.25, 0.75]
  ops = [qml.PauliX(0), qml.PauliZ(0)]
  H = qml.dot(coeffs, ops)

  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def circuit():
      qml.Hadamard(0)
      qml.QDrift(H, time=1.2, n = 10)
      return qml.probs()
  ```

  ```pycon
  >>> circuit()
  array([0.61814334, 0.        , 0.38185666, 0.        ])
  ```

<h4>Building blocks for quantum phase estimation 🧱</h4>

* A new operator called `CosineWindow` has been added to prepare an initial state based on a cosine wave function.
  [(#4683)](https://github.com/PennyLaneAI/pennylane/pull/4683)

  As outlined in [2110.09590](https://arxiv.org/pdf/2110.09590.pdf), the cosine tapering window is part of a modification
  to quantum phase estimation that can provide a cubic improvement to the algorithm's error rate. Using `CosineWindow` will 
  prepare a state whose amplitudes follow a cosinusoidal distribution over the computational basis.

  ```python
  import matplotlib.pyplot as plt

  dev = qml.device('default.qubit', wires=4)

  @qml.qnode(dev)
  def example_circuit():
        qml.CosineWindow(wires=range(4))
        return qml.state()
  output = example_circuit()

  plt.style.use("pennylane.drawer.plot")
  plt.bar(range(len(output)), output)
  plt.show()
  ```

  <img src="https://docs.pennylane.ai/en/stable/_images/cosine_window.png" width=50%/>

* Controlled gate sequences raised to decreasing powers, a sub-block in quantum phase estimation, can now be created with the new 
  `ControlledSequence` operator.
  [(#4707)](https://github.com/PennyLaneAI/pennylane/pull/4707/)

  To use `ControlledSequence`, specify the controlled unitary operator and the control wires, `control`:

  ```python
  dev = qml.device("default.qubit", wires = 4)

  @qml.qnode(dev)
  def circuit():
      for i in range(3):
          qml.Hadamard(wires = i)
      qml.ControlledSequence(qml.RX(0.25, wires = 3), control = [0, 1, 2])
      qml.adjoint(qml.QFT)(wires = range(3))
      return qml.probs(wires = range(3))
  ```

  ```pycon
  >>> print(circuit())
  [0.92059345 0.02637178 0.00729619 0.00423258 0.00360545 0.00423258 0.00729619 0.02637178]
  ```
  
<h4>New device capabilities, integration with Catalyst, and more! ⚗️</h4>

* `default.qubit` now uses the new `qml.devices.Device` API and functionality in
  `qml.devices.qubit`. If you experience any issues with the updated `default.qubit`, please let us
  know by [posting an issue](https://github.com/PennyLaneAI/pennylane/issues/new/choose). 
  The old version of the device is still
  accessible by the short name `default.qubit.legacy`, or directly via `qml.devices.DefaultQubitLegacy`.
  [(#4594)](https://github.com/PennyLaneAI/pennylane/pull/4594)
  [(#4436)](https://github.com/PennyLaneAI/pennylane/pull/4436)
  [(#4620)](https://github.com/PennyLaneAI/pennylane/pull/4620)
  [(#4632)](https://github.com/PennyLaneAI/pennylane/pull/4632)

  This changeover has a number of benefits for `default.qubit`, including:

  * The number of wires is now optional — simply having `qml.device("default.qubit")` is valid! If
    wires are not provided at instantiation, the device automatically infers the required number of
    wires for each circuit provided for execution.

    ```python
    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def circuit():
        qml.PauliZ(0)
        qml.RZ(0.1, wires=1)
        qml.Hadamard(2)
        return qml.state()
    ```

    ```pycon
    >>> print(qml.draw(circuit)())
    0: ──Z────────┤  State
    1: ──RZ(0.10)─┤  State
    2: ──H────────┤  State
    ```

  * `default.qubit` is no longer silently swapped out with an interface-appropriate device when the
    backpropagation differentiation method is used. For example, consider:

    ```python
    import jax

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, diff_method="backprop")
    def f(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))
    
    f(jax.numpy.array(0.2))
    ```
    
    In previous versions of PennyLane, the device will be swapped for the JAX equivalent:

    ```pycon
    >>> f.device
    <DefaultQubitJax device (wires=1, shots=None) at 0x7f8c8bff50a0>
    >>> f.device == dev
    False
    ```
    
    Now, `default.qubit` can itself dispatch to all the interfaces in a backprop-compatible way
    and hence does not need to be swapped:

    ```pycon
    >>> f.device
    <default.qubit device (wires=1) at 0x7f20d043b040>
    >>> f.device == dev
    True
    ```

* A QNode that has been decorated with `qjit` from PennyLane's
  [Catalyst](https://docs.pennylane.ai/projects/catalyst/en/stable/index.html) library for just-in-time hybrid
  compilation is now compatible with `qml.draw`.
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
      loop()  
      qml.RY(y, wires=1)
      qml.RZ(z, wires=2)
      return qml.expval(qml.PauliZ(0))
  
  draw = qml.draw(circuit, decimals=None)(1.234, 2.345, 3.456, 1)
  ```
  
  ```pycon
  >>> print(draw)
  0: ──RX──H──┤  <Z>
  1: ──H───RY─┤
  2: ──RZ─────┤
  ```

<h3>Improvements 🛠</h3>

<h4>More PyTrees!</h4>

* `MeasurementProcess` and `QuantumScript` objects are now registered as JAX PyTrees.
  [(#4607)](https://github.com/PennyLaneAI/pennylane/pull/4607)
  [(#4608)](https://github.com/PennyLaneAI/pennylane/pull/4608)

  It is now possible to JIT-compile functions with arguments that are a `MeasurementProcess` or
  a `QuantumScript`:

  ```python
  import jax

  tape0 = qml.tape.QuantumTape([qml.RX(1.0, 0), qml.RY(0.5, 0)], [qml.expval(qml.PauliZ(0))])
  dev = qml.device('lightning.qubit', wires=5)

  execute_kwargs = {"device": dev, "gradient_fn": qml.gradients.param_shift, "interface":"jax"}

  jitted_execute = jax.jit(qml.execute, static_argnames=execute_kwargs.keys())
  jitted_execute((tape0, ), **execute_kwargs)
  ```

<h4>Improving QChem and existing algorithms</h4>

* Computationally expensive functions in `integrals.py`, `electron_repulsion` and `_hermite_coulomb`, have
  been modified to replace indexing with slicing for better compatibility with JAX.
  [(#4685)](https://github.com/PennyLaneAI/pennylane/pull/4685)

* `qml.qchem.import_state` has been extended to import more quantum chemistry wavefunctions, 
  from MPS, DMRG and SHCI classical calculations performed with the Block2 and Dice libraries.
  [#4523](https://github.com/PennyLaneAI/pennylane/pull/4523)
  [#4524](https://github.com/PennyLaneAI/pennylane/pull/4524)
  [#4626](https://github.com/PennyLaneAI/pennylane/pull/4626)
  [#4634](https://github.com/PennyLaneAI/pennylane/pull/4634)

  Check out our [how-to guide](https://pennylane.ai/qml/demos/tutorial_initial_state_preparation)
  to learn more about how PennyLane integrates with your favourite quantum chemistry libraries.

* The qchem `fermionic_dipole` and `particle_number` functions have been updated to use a
  `FermiSentence`. The deprecated features for using tuples to represent fermionic operations are
  removed.
  [(#4546)](https://github.com/PennyLaneAI/pennylane/pull/4546)
  [(#4556)](https://github.com/PennyLaneAI/pennylane/pull/4556)

* The tensor-network template `qml.MPS` now supports changing the `offset` between subsequent blocks for more flexibility.
  [(#4531)](https://github.com/PennyLaneAI/pennylane/pull/4531)

* Builtin types support with `qml.pauli_decompose` have been improved.
  [(#4577)](https://github.com/PennyLaneAI/pennylane/pull/4577)

* `AmplitudeEmbedding` now inherits from `StatePrep`, allowing for it to not be decomposed
  when at the beginning of a circuit, thus behaving like `StatePrep`.
  [(#4583)](https://github.com/PennyLaneAI/pennylane/pull/4583)

* `qml.cut_circuit` is now compatible with circuits that compute the expectation values of Hamiltonians 
  with two or more terms.
  [(#4642)](https://github.com/PennyLaneAI/pennylane/pull/4642)

<h4>Next-generation device API</h4>

* `default.qubit` now tracks the number of equivalent qpu executions and total shots
  when the device is sampling. Note that `"simulations"` denotes the number of simulation passes, whereas
  `"executions"` denotes how many different computational bases need to be sampled in. Additionally, the
  new `default.qubit` tracks the results of `device.execute`.
  [(#4628)](https://github.com/PennyLaneAI/pennylane/pull/4628)
  [(#4649)](https://github.com/PennyLaneAI/pennylane/pull/4649)

* `DefaultQubit` can now accept a `jax.random.PRNGKey` as a `seed` to set the key for the JAX pseudo random 
  number generator when using the JAX interface. This corresponds to the `prng_key` on 
  `default.qubit.jax` in the old API.
  [(#4596)](https://github.com/PennyLaneAI/pennylane/pull/4596)

* The `JacobianProductCalculator` abstract base class and implementations `TransformJacobianProducts`
  `DeviceDerivatives`, and `DeviceJacobianProducts` have been added to `pennylane.interfaces.jacobian_products`.
  [(#4435)](https://github.com/PennyLaneAI/pennylane/pull/4435)
  [(#4527)](https://github.com/PennyLaneAI/pennylane/pull/4527)
  [(#4637)](https://github.com/PennyLaneAI/pennylane/pull/4637)

* `DefaultQubit` dispatches to a faster implementation for applying `ParametrizedEvolution` to a state
  when it is more efficient to evolve the state than the operation matrix.
  [(#4598)](https://github.com/PennyLaneAI/pennylane/pull/4598)
  [(#4620)](https://github.com/PennyLaneAI/pennylane/pull/4620)

* Wires can be provided to the new device API.
  [(#4538)](https://github.com/PennyLaneAI/pennylane/pull/4538)
  [(#4562)](https://github.com/PennyLaneAI/pennylane/pull/4562)

* `qml.sample()` in the new device API now returns a `np.int64` array instead of `np.bool8`.
  [(#4539)](https://github.com/PennyLaneAI/pennylane/pull/4539)

* The new device API now has a `repr()` method.
  [(#4562)](https://github.com/PennyLaneAI/pennylane/pull/4562)

* `DefaultQubit` now works as expected with measurement processes that don't specify wires.
  [(#4580)](https://github.com/PennyLaneAI/pennylane/pull/4580)

* Various improvements to measurements have been made for feature parity between `default.qubit.legacy` and
  the new `DefaultQubit`. This includes not trying to squeeze batched `CountsMP` results and implementing
  `MutualInfoMP.map_wires`.
  [(#4574)](https://github.com/PennyLaneAI/pennylane/pull/4574)

* `devices.qubit.simulate` now accepts an interface keyword argument. If a QNode with `DefaultQubit`
  specifies an interface, the result will be computed with that interface.
  [(#4582)](https://github.com/PennyLaneAI/pennylane/pull/4582)

* `ShotAdaptiveOptimizer` has been updated to pass shots to QNode executions instead of overriding
  device shots before execution. This makes it compatible with the new device API.
  [(#4599)](https://github.com/PennyLaneAI/pennylane/pull/4599)

* `pennylane.devices.preprocess` now offers the transforms `decompose`, `validate_observables`, `validate_measurements`,
  `validate_device_wires`, `validate_multiprocessing_workers`, `warn_about_trainable_observables`,
  and `no_sampling` to assist in constructing devices under the new device API.
  [(#4659)](https://github.com/PennyLaneAI/pennylane/pull/4659)

* Updated `qml.device`, `devices.preprocessing` and the `tape_expand.set_decomposition` context 
  manager to bring `DefaultQubit` to feature parity with `default.qubit.legacy` with regards to 
  using custom decompositions. The `DefaultQubit` device can now be included in a `set_decomposition` 
  context or initialized with a `custom_decomps` dictionary, as well as a custom `max_depth` for 
  decomposition.
  [(#4675)](https://github.com/PennyLaneAI/pennylane/pull/4675)

<h4>Other improvements</h4>

* The `StateMP` measurement now accepts a wire order (e.g., a device wire order). The `process_state`
  method will re-order the given state to go from the inputted wire-order to the process's wire-order.
  If the process's wire-order contains extra wires, it will assume those are in the zero-state.
  [(#4570)](https://github.com/PennyLaneAI/pennylane/pull/4570)
  [(#4602)](https://github.com/PennyLaneAI/pennylane/pull/4602)

* Methods called `add_transform` and `insert_front_transform`  have been added to `TransformProgram`.
  [(#4559)](https://github.com/PennyLaneAI/pennylane/pull/4559)

* Instances of the `TransformProgram` class can now be added together.
  [(#4549)](https://github.com/PennyLaneAI/pennylane/pull/4549)

* Transforms can now be applied to devices following the new device API.
 [(#4667)](https://github.com/PennyLaneAI/pennylane/pull/4667)

* All gradient transforms have been updated to the new transform program system.
 [(#4595)](https://github.com/PennyLaneAI/pennylane/pull/4595)

* Multi-controlled operations with a single-qubit special unitary target can now automatically decompose.
  [(#4697)](https://github.com/PennyLaneAI/pennylane/pull/4697)

* `pennylane.defer_measurements` will now exit early if the input does not contain mid circuit measurements.
  [(#4659)](https://github.com/PennyLaneAI/pennylane/pull/4659)

* The density matrix aspects of `StateMP` have been split into their own measurement
  process called `DensityMatrixMP`.
  [(#4558)](https://github.com/PennyLaneAI/pennylane/pull/4558)

* `StateMeasurement.process_state` now assumes that the input is flat. `ProbabilityMP.process_state` has
  been updated to reflect this assumption and avoid redundant reshaping.
  [(#4602)](https://github.com/PennyLaneAI/pennylane/pull/4602)

* `qml.exp` returns a more informative error message when decomposition is unavailable for non-unitary operators.
  [(#4571)](https://github.com/PennyLaneAI/pennylane/pull/4571)

* Added `qml.math.get_deep_interface` to get the interface of a scalar hidden deep in lists or tuples.
  [(#4603)](https://github.com/PennyLaneAI/pennylane/pull/4603)

* Updated `qml.math.ndim` and `qml.math.shape` to work with built-in lists or tuples that contain
  interface-specific scalar dat (e.g., `[(tf.Variable(1.1), tf.Variable(2.2))]`).
  [(#4603)](https://github.com/PennyLaneAI/pennylane/pull/4603)

* When decomposing a unitary matrix with `one_qubit_decomposition` and opting to include the `GlobalPhase` 
  in the decomposition, the phase is no longer cast to `dtype=complex`.
  [(#4653)](https://github.com/PennyLaneAI/pennylane/pull/4653)

* `_qfunc_output` has been removed from `QuantumScript`, as it is no longer necessary. There is
  still a `_qfunc_output` property on `QNode` instances.
  [(#4651)](https://github.com/PennyLaneAI/pennylane/pull/4651)

* `qml.data.load` properly handles parameters that come after `'full'`
  [(#4663)](https://github.com/PennyLaneAI/pennylane/pull/4663)

* The `qml.jordan_wigner` function has been modified to optionally remove the imaginary components
  of the computed qubit operator, if imaginary components are smaller than a threshold. 
  [(#4639)](https://github.com/PennyLaneAI/pennylane/pull/4639)

* `qml.data.load` correctly performs a full download of the dataset after a partial download of the
  same dataset has already been performed.
  [(#4681)](https://github.com/PennyLaneAI/pennylane/pull/4681)
  
* The performance of `qml.data.load()` has been improved when partially loading a dataset
  [(#4674)](https://github.com/PennyLaneAI/pennylane/pull/4674)

* Plots generated with the `pennylane.drawer.plot` style of `matplotlib.pyplot` now have black
  axis labels and are generated at a default DPI of 300.
  [(#4690)](https://github.com/PennyLaneAI/pennylane/pull/4690)

* Shallow copies of the `QNode` now also copy the `execute_kwargs` and transform program. When applying
  a transform to a `QNode`, the new qnode is only a shallow copy of the original and thus keeps the same
  device.
  [(#4736)](https://github.com/PennyLaneAI/pennylane/pull/4736)

* `QubitDevice` and `CountsMP` are updated to disregard samples containing failed hardware measurements
  (record as `np.NaN`) when tallying samples, rather than counting failed measurements as ground-state 
  measurements, and to display `qml.counts` coming from these hardware devices correctly.
  [(#4739)](https://github.com/PennyLaneAI/pennylane/pull/4739)

<h3>Breaking changes 💔</h3>

* `qml.defer_measurements` now raises an error if a transformed circuit measures `qml.probs`,
  `qml.sample`, or `qml.counts` without any wires or observable, or if it measures `qml.state`.
  [(#4701)](https://github.com/PennyLaneAI/pennylane/pull/4701)

* The device test suite now converts device keyword arguments to integers or floats if possible.
  [(#4640)](https://github.com/PennyLaneAI/pennylane/pull/4640)

* `MeasurementProcess.eigvals()` now raises an `EigvalsUndefinedError` if the measurement observable
  does not have eigenvalues.
  [(#4544)](https://github.com/PennyLaneAI/pennylane/pull/4544)

* The `__eq__` and `__hash__` methods of `Operator` and `MeasurementProcess` no longer rely on the
  object's address in memory. Using `==` with operators and measurement processes will now behave the
  same as `qml.equal`, and objects of the same type with the same data and hyperparameters will have
  the same hash.
  [(#4536)](https://github.com/PennyLaneAI/pennylane/pull/4536)

  In the following scenario, the second and third code blocks show the previous and current behaviour
  of operator and measurement process equality, determined by `==`:

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

* The old return type and associated functions `qml.enable_return` and `qml.disable_return` have been removed.
  [(#4503)](https://github.com/PennyLaneAI/pennylane/pull/4503)

* The `mode` keyword argument in `QNode` has been removed. Please use `grad_on_execution` instead.
  [(#4503)](https://github.com/PennyLaneAI/pennylane/pull/4503)

* The CV observables `qml.X` and `qml.P` have been removed. Please use `qml.QuadX` and `qml.QuadP` instead.
  [(#4533)](https://github.com/PennyLaneAI/pennylane/pull/4533)

* The `sampler_seed` argument of `qml.gradients.spsa_grad` has been removed.
  Instead, the `sampler_rng` argument should be set, either to an integer value, which will be used
  to create a PRNG internally, or to a NumPy pseudo-random number generator (PRNG) created via
  `np.random.default_rng(seed)`.
  [(#4550)](https://github.com/PennyLaneAI/pennylane/pull/4550)

* The `QuantumScript.set_parameters` method and the `QuantumScript.data` setter have
  been removed. Please use `QuantumScript.bind_new_parameters` instead.
  [(#4548)](https://github.com/PennyLaneAI/pennylane/pull/4548)

* The method `tape.unwrap()` and corresponding `UnwrapTape` and `Unwrap` classes have been removed.
  Instead of `tape.unwrap()`, use `qml.transforms.convert_to_numpy_parameters`.
  [(#4535)](https://github.com/PennyLaneAI/pennylane/pull/4535)

* The `RandomLayers.compute_decomposition` keyword argument `ratio_imprivitive` has been changed to
  `ratio_imprim` to match the call signature of the operation.
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

* The `prep` keyword argument in `QuantumScript` has been deprecated and will be removed from `QuantumScript`.
  `StatePrepBase` operations should be placed at the beginning of the `ops` list instead.
  [(#4554)](https://github.com/PennyLaneAI/pennylane/pull/4554)

* `qml.gradients.pulse_generator` has been renamed to `qml.gradients.pulse_odegen` to adhere to paper naming conventions. During v0.33, `pulse_generator`
  is still available but raises a warning.
  [(#4633)](https://github.com/PennyLaneAI/pennylane/pull/4633)

<h3>Documentation 📝</h3>

* A warning section in the docstring for `DefaultQubit` regarding the start method used in multiprocessing has been added.
  This may help users circumvent issues arising in Jupyter notebooks on macOS for example.
  [(#4622)](https://github.com/PennyLaneAI/pennylane/pull/4622)

* Documentation improvements to the new device API have been made. The documentation now correctly states that interface-specific
  parameters are only passed to the device for backpropagation derivatives. 
  [(#4542)](https://github.com/PennyLaneAI/pennylane/pull/4542)

* Functions for qubit-simulation to the `qml.devices` sub-page of the "Internal" section have been added.
  Note that these functions are unstable while device upgrades are underway.
  [(#4555)](https://github.com/PennyLaneAI/pennylane/pull/4555)

* A documentation improvement to the usage example in the `qml.QuantumMonteCarlo` page has been made.
  An integral was missing the differential :math:`dx`.
  [(#4593)](https://github.com/PennyLaneAI/pennylane/pull/4593)  

* A documentation improvement for the use of the `pennylane` style of `qml.drawer` and the
  `pennylane.drawer.plot` style of `matplotlib.pyplot` has been made by clarifying the use of the default font.
  [(#4690)](https://github.com/PennyLaneAI/pennylane/pull/4690)

<h3>Bug fixes 🐛</h3>

* Jax jit now works when a probability measurement is broadcasted onto all wires.
  [(#4742)](https://github.com/PennyLaneAI/pennylane/pull/4742)

* Fixed `LocalHilbertSchmidt.compute_decomposition` so that the template can be used in a QNode.
  [(#4719)](https://github.com/PennyLaneAI/pennylane/pull/4719)

* Fixes `transforms.transpile` with arbitrary measurement processes.
  [(#4732)](https://github.com/PennyLaneAI/pennylane/pull/4732)

* Providing `work_wires=None` to `qml.GroverOperator` no longer interprets `None` as a wire.
  [(#4668)](https://github.com/PennyLaneAI/pennylane/pull/4668)

* Fixed an issue where the `__copy__` method of the `qml.Select()` operator attempted to access un-initialized data.
  [(#4551)](https://github.com/PennyLaneAI/pennylane/pull/4551)

* Fixed the `skip_first` option in `expand_tape_state_prep`.
  [(#4564)](https://github.com/PennyLaneAI/pennylane/pull/4564)

* `convert_to_numpy_parameters` now uses `qml.ops.functions.bind_new_parameters`. This reinitializes the operation and
  makes sure everything references the new NumPy parameters.
  [(#4540)](https://github.com/PennyLaneAI/pennylane/pull/4540)

* `tf.function` no longer breaks `ProbabilityMP.process_state`, which is needed by new devices.
  [(#4470)](https://github.com/PennyLaneAI/pennylane/pull/4470)

* Fixed unit tests for `qml.qchem.mol_data`.
  [(#4591)](https://github.com/PennyLaneAI/pennylane/pull/4591)

* Fixed `ProbabilityMP.process_state` so that it allows for proper Autograph compilation. Without this,
  decorating a QNode that returns an `expval` with `tf.function` would fail when computing the
  expectation.
  [(#4590)](https://github.com/PennyLaneAI/pennylane/pull/4590)

* The `torch.nn.Module` properties are now accessible on a `pennylane.qnn.TorchLayer`.
  [(#4611)](https://github.com/PennyLaneAI/pennylane/pull/4611)

* `qml.math.take` with Pytorch now returns `tensor[..., indices]` when the user requests
  the last axis (`axis=-1`). Without the fix, it would wrongly return `tensor[indices]`.
  [(#4605)](https://github.com/PennyLaneAI/pennylane/pull/4605)

* Ensured the logging `TRACE` level works with gradient-free execution.
  [(#4669)](https://github.com/PennyLaneAI/pennylane/pull/4669)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso,
Utkarsh Azad,
Thomas Bromley,
Isaac De Vlugt,
Jack Brown,
Stepan Fomichev,
Joana Fraxanet,
Diego Guala,
Soran Jahangiri,
Edward Jiang,
Korbinian Kottmann,
Ivana Kurečić
Christina Lee,
Lillian M. A. Frederiksen,
Vincent Michaud-Rioux,
Romain Moyard,
Daniel F. Nino,
Lee James O'Riordan,
Mudit Pandey,
Matthew Silverman,
Jay Soni.
