:orphan:

# Release 0.37.0-dev (development release)

<h3>New features since last release</h3>

* The `default.tensor` device now supports the `tn` method to simulate quantum circuits using exact tensor networks.
  [(#5786)](https://github.com/PennyLaneAI/pennylane/pull/5786)

* QROM template is added. This template allows you to enter classic data in the form of bitstrings.
  [(#5688)](https://github.com/PennyLaneAI/pennylane/pull/5688)

  ```python
  # a list of bitstrings is defined
  bitstrings = ["010", "111", "110", "000"]

  dev = qml.device("default.qubit", shots = 1)

  @qml.qnode(dev)
  def circuit():

      # the third index is encoded in the control wires [0, 1]
      qml.BasisEmbedding(2, wires = [0,1])

      qml.QROM(bitstrings = bitstrings,
              control_wires = [0,1],
              target_wires = [2,3,4],
              work_wires = [5,6,7])

      return qml.sample(wires = [2,3,4])
  ```
   ```pycon
  >>> print(circuit())
  [1 1 0]
  ```

* The `default.tensor` device is introduced to perform tensor network simulations of quantum circuits using the `mps` (Matrix Product State) method.
  [(#5699)](https://github.com/PennyLaneAI/pennylane/pull/5699)

* Added `from_openfermion` to convert openfermion `FermionOperator` objects to PennyLane `FermiWord` or
`FermiSentence` objects.
[(#5808)](https://github.com/PennyLaneAI/pennylane/pull/5808)

  ```python
  of_op = openfermion.FermionOperator('0^ 2')
  pl_op = qml.from_openfermion(of_op)

  ```
  ```pycon
  >>> print(pl_op)
  a⁺(0) a(2)
  ```

* A new `qml.noise` module which contains utililty functions for building `NoiseModels`.
  [(#5674)](https://github.com/PennyLaneAI/pennylane/pull/5674)
  [(#5684)](https://github.com/PennyLaneAI/pennylane/pull/5684)

  ```python
  fcond = qml.noise.op_eq(qml.X) | qml.noise.op_eq(qml.Y)
  noise = qml.noise.partial_wires(qml.AmplitudeDamping, 0.4)
  ```

  ```pycon
  >>> qml.NoiseModel({fcond: noise}, t1=0.04)
  NoiseModel({
    OpEq(PauliX) | OpEq(PauliY) = AmplitudeDamping(gamma=0.4)
  }, t1 = 0.04)
  ```

<h3>Improvements 🛠</h3>

* Add operation and measurement specific routines in `default.tensor` to improve scalability.
  [(#5795)](https://github.com/PennyLaneAI/pennylane/pull/5795)
  
* `param_shift` with the `broadcast=True` option now supports shot vectors and multiple measurements.
  [(#5667)](https://github.com/PennyLaneAI/pennylane/pull/5667)

* `default.clifford` now supports arbitrary state-based measurements with `qml.Snapshot`.
  [(#5794)](https://github.com/PennyLaneAI/pennylane/pull/5794)

* `qml.TrotterProduct` is now compatible with resource tracking by inheriting from `ResourcesOperation`. 
   [(#5680)](https://github.com/PennyLaneAI/pennylane/pull/5680)

* The wires for the `default.tensor` device are selected at runtime if they are not provided by user.
  [(#5744)](https://github.com/PennyLaneAI/pennylane/pull/5744)

* Added `packaging` in the required list of packages.
  [(#5769)](https://github.com/PennyLaneAI/pennylane/pull/5769).

* Logging now allows for an easier opt-in across the stack, and also extends control support to `catalyst`.
  [(#5528)](https://github.com/PennyLaneAI/pennylane/pull/5528).

* A number of templates have been updated to be valid pytrees and PennyLane operations.
  [(#5698)](https://github.com/PennyLaneAI/pennylane/pull/5698)

* `ctrl` now works with tuple-valued `control_values` when applied to any already controlled operation.
  [(#5725)](https://github.com/PennyLaneAI/pennylane/pull/5725)

* Add support for 3 new pytest markers: `unit`, `integration` and `system`.
  [(#5517)](https://github.com/PennyLaneAI/pennylane/pull/5517)

* The sorting order of parameter-shift terms is now guaranteed to resolve ties in the absolute value with the sign of the shifts.
  [(#5582)](https://github.com/PennyLaneAI/pennylane/pull/5582)

* `qml.transforms.split_non_commuting` can now handle circuits containing measurements of multi-term observables.
  [(#5729)](https://github.com/PennyLaneAI/pennylane/pull/5729)
  [(#5853)](https://github.com/PennyLaneAI/pennylane/pull/5838)

* The qchem module has dedicated functions for calling `pyscf` and `openfermion` backends.
  [(#5553)](https://github.com/PennyLaneAI/pennylane/pull/5553)

* `qml.from_qasm` now supports the ability to convert mid-circuit measurements from `OpenQASM 2` code, and it can now also take an
   optional argument to specify a list of measurements to be performed at the end of the circuit, just like `from_qiskit`.
   [(#5818)](https://github.com/PennyLaneAI/pennylane/pull/5818)

<h4>Mid-circuit measurements and dynamic circuits</h4>

* `qml.QNode` and `qml.qnode` now accept two new keyword arguments: `postselect_mode` and `mcm_method`.
  These keyword arguments can be used to configure how the device should behave when running circuits with
  mid-circuit measurements.
  [(#5679)](https://github.com/PennyLaneAI/pennylane/pull/5679)
  [(#5833)](https://github.com/PennyLaneAI/pennylane/pull/5833)
  [(#5850)](https://github.com/PennyLaneAI/pennylane/pull/5850)

  * `postselect_mode="hw-like"` will indicate to devices to discard invalid shots when postselecting
    mid-circuit measurements. Use `postselect_mode="fill-shots"` to unconditionally sample the postselected
    value, thus making all samples valid. This is equivalent to sampling until the number of valid samples
    matches the total number of shots.
  * `mcm_method` will indicate which strategy to use for running circuits with mid-circuit measurements.
    Use `mcm_method="deferred"` to use the deferred measurements principle, or `mcm_method="one-shot"`
    to execute once for each shot. If using `qml.jit` with the Catalyst compiler, `mcm_method="single-branch-statistics"`
    is also available. Using this method, a single branch of the execution tree will be randomly explored.

* The `dynamic_one_shot` transform is made compatible with the Catalyst compiler.
  [(#5766)](https://github.com/PennyLaneAI/pennylane/pull/5766)
  
* Rationalize MCM tests, removing most end-to-end tests from the native MCM test file,
  but keeping one that validates multiple mid-circuit measurements with any allowed return
  and interface end-to-end tests.
  [(#5787)](https://github.com/PennyLaneAI/pennylane/pull/5787)

* The `dynamic_one_shot` transform uses a single auxiliary tape with a shot vector and `default.qubit` implements the loop over shots with `jax.vmap`.
  [(#5617)](https://github.com/PennyLaneAI/pennylane/pull/5617)

* The `dynamic_one_shot` transform can be compiled with `jax.jit`.
  [(#5557)](https://github.com/PennyLaneAI/pennylane/pull/5557)

* When using `defer_measurements` with postselecting mid-circuit measurements, operations
  that will never be active due to the postselected state are skipped in the transformed
  quantum circuit. In addition, postselected controls are skipped, as they are evaluated
  at transform time. This optimization feature can be turned off by setting `reduce_postselected=False`
  [(#5558)](https://github.com/PennyLaneAI/pennylane/pull/5558)

  Consider a simple circuit with three mid-circuit measurements, two of which are postselecting,
  and a single gate conditioned on those measurements:

  ```python
  @qml.qnode(qml.device("default.qubit"))
  def node(x):
      qml.RX(x, 0)
      qml.RX(x, 1)
      qml.RX(x, 2)
      mcm0 = qml.measure(0, postselect=0, reset=False)
      mcm1 = qml.measure(1, postselect=None, reset=True)
      mcm2 = qml.measure(2, postselect=1, reset=False)
      qml.cond(mcm0+mcm1+mcm2==1, qml.RX)(0.5, 3)
      return qml.expval(qml.Z(0) @ qml.Z(3))
  ```

  Without the new optimization, we obtain three gates, each controlled on the three measured
  qubits. They correspond to the combinations of controls that satisfy the condition
  `mcm0+mcm1+mcm2==1`:

  ```pycon
  >>> print(qml.draw(qml.defer_measurements(node, reduce_postselected=False))(0.6))
  0: ──RX(0.60)──|0⟩⟨0|─╭●─────────────────────────────────────────────┤ ╭<Z@Z>
  1: ──RX(0.60)─────────│──╭●─╭X───────────────────────────────────────┤ │
  2: ──RX(0.60)─────────│──│──│───|1⟩⟨1|─╭○────────╭○────────╭●────────┤ │
  3: ───────────────────│──│──│──────────├RX(0.50)─├RX(0.50)─├RX(0.50)─┤ ╰<Z@Z>
  4: ───────────────────╰X─│──│──────────├○────────├●────────├○────────┤
  5: ──────────────────────╰X─╰●─────────╰●────────╰○────────╰○────────┤
  ```

  If we do not explicitly deactivate the optimization, we obtain a much simpler circuit:

  ```pycon
  >>> print(qml.draw(qml.defer_measurements(node))(0.6))
  0: ──RX(0.60)──|0⟩⟨0|─╭●─────────────────┤ ╭<Z@Z>
  1: ──RX(0.60)─────────│──╭●─╭X───────────┤ │
  2: ──RX(0.60)─────────│──│──│───|1⟩⟨1|───┤ │
  3: ───────────────────│──│──│──╭RX(0.50)─┤ ╰<Z@Z>
  4: ───────────────────╰X─│──│──│─────────┤
  5: ──────────────────────╰X─╰●─╰○────────┤
  ```

  There is only one controlled gate with only one control wire.

* `qml.devices.LegacyDevice` is now an alias for `qml.Device`, so it is easier to distinguish it from
  `qml.devices.Device`, which follows the new device API.
  [(#5581)](https://github.com/PennyLaneAI/pennylane/pull/5581)

* The `dtype` for `eigvals` of `X`, `Y`, `Z` and `Hadamard` is changed from `int` to `float`, making them
  consistent with the other observables. The `dtype` of the returned values when sampling these observables
  (e.g. `qml.sample(X(0))`) is also changed to `float`.
  [(#5607)](https://github.com/PennyLaneAI/pennylane/pull/5607)

* Sets up the framework for the development of an `assert_equal` function for testing operator comparison.
  [(#5634)](https://github.com/PennyLaneAI/pennylane/pull/5634)

* `qml.sample` can now be used on Boolean values representing mid-circuit measurement results in
  traced quantum functions. This feature is used with Catalyst to enable the pattern
  `m = measure(0); qml.sample(m)`.
  [(#5673)](https://github.com/PennyLaneAI/pennylane/pull/5673)

* PennyLane operators, measurements, and QNodes can now automatically be captured as instructions in JAXPR.
  [(#5564)](https://github.com/PennyLaneAI/pennylane/pull/5564)
  [(#5511)](https://github.com/PennyLaneAI/pennylane/pull/5511)
  [(#5708)](https://github.com/PennyLaneAI/pennylane/pull/5708)
  [(#5523)](https://github.com/PennyLaneAI/pennylane/pull/5523)
  [(#5686)](https://github.com/PennyLaneAI/pennylane/pull/5686)

* The `decompose` transform has an `error` kwarg to specify the type of error that should be raised,
  allowing error types to be more consistent with the context the `decompose` function is used in.
  [(#5669)](https://github.com/PennyLaneAI/pennylane/pull/5669)

* The `qml.pytrees` module now has `flatten` and `unflatten` methods for serializing pytrees.
  [(#5701)](https://github.com/PennyLaneAI/pennylane/pull/5701)

* Empty initialization of `PauliVSpace` is permitted.
  [(#5675)](https://github.com/PennyLaneAI/pennylane/pull/5675)

* `MultiControlledX` can now be decomposed even when no `work_wires` are provided. The implementation returns $\mathcal{O}(\text{len(control\_wires)}^2)$ operations, and is applicable for any multi controlled unitary gate.
  [(#5735)](https://github.com/PennyLaneAI/pennylane/pull/5735)

* Single control unitary now includes the correct global phase.
  [(#5735)](https://github.com/PennyLaneAI/pennylane/pull/5735)

* Single control `GlobalPhase` has now a decomposition, i.e. relative phase on control wire.
  [(#5735)](https://github.com/PennyLaneAI/pennylane/pull/5735)

* `QuantumScript` properties are only calculated when needed, instead of on initialization. This decreases the classical overhead by >20%.
  `par_info`, `obs_sharing_wires`, and `obs_sharing_wires_id` are now public attributes.
  [(#5696)](https://github.com/PennyLaneAI/pennylane/pull/5696)

* `qml.ops.Conditional` now inherits from `qml.ops.SymbolicOp`, thus it inherits several useful common functionalities. Other properties such as adjoint and diagonalizing gates have been added using the `base` properties.
  [(##5772)](https://github.com/PennyLaneAI/pennylane/pull/5772)

* New dispatches for `qml.ops.Conditional` and `qml.MeasurementValue` have been added to `qml.equal`.
  [(##5772)](https://github.com/PennyLaneAI/pennylane/pull/5772)

* The `qml.qchem.Molecule` object is now the central object used by all qchem functions.
  [(#5571)](https://github.com/PennyLaneAI/pennylane/pull/5571)

* The `qml.qchem.Molecule` class now supports Angstrom as a unit.
  [(#5694)](https://github.com/PennyLaneAI/pennylane/pull/5694)

* The `qml.qchem.Molecule` class now supports open-shell systems.
  [(#5655)](https://github.com/PennyLaneAI/pennylane/pull/5655)

* The `qml.qchem.molecular_hamiltonian` function now supports parity and Bravyi-Kitaev mappings.
  [(#5657)](https://github.com/PennyLaneAI/pennylane/pull/5657/)

* The qchem docs are updated with the new qchem improvements.
  [(#5758)](https://github.com/PennyLaneAI/pennylane/pull/5758/)
  [(#5638)](https://github.com/PennyLaneAI/pennylane/pull/5638/)

* Device preprocess transforms now happen inside the ml boundary.
  [(#5791)](https://github.com/PennyLaneAI/pennylane/pull/5791)

* `qml.qchem.molecular_dipole` function is added for calculating the dipole operator using "dhf" and "openfermion" backends.
  [(#5764)](https://github.com/PennyLaneAI/pennylane/pull/5764)

* Transforms applied to callables now use `functools.wraps` to preserve the docstring and call signature of the original function.
  [(#5857)](https://github.com/PennyLaneAI/pennylane/pull/5857)

<h4>Community contributions 🥳</h4>

* Implemented kwargs (`check_interface`, `check_trainability`, `rtol` and `atol`) support in `qml.equal` for the operators `Pow`, `Adjoint`, `Exp`, and `SProd`.
  [(#5668)](https://github.com/PennyLaneAI/pennylane/issues/5668)
  
* `qml.QutritDepolarizingChannel` has been added, allowing for depolarizing noise to be simulated on the `default.qutrit.mixed` device.
  [(#5502)](https://github.com/PennyLaneAI/pennylane/pull/5502)
 
* Implement support in `assert_equal` for `Operator`, `Controlled`, `Adjoint`, `Pow`, `Exp`, `SProd`, `ControlledSequence`, `Prod`, `Sum`, `Tensor` and `Hamiltonian`
 [(#5780)](https://github.com/PennyLaneAI/pennylane/pull/5780)

* `qml.QutritChannel` has been added, enabling the specification of noise using a collection of (3x3) Kraus matrices on the `default.qutrit.mixed` device.
  [(#5793)](https://github.com/PennyLaneAI/pennylane/issues/5793)

* `qml.QutritAmplitudeDamping` channel has been added, allowing for noise processes modelled by amplitude damping to be simulated on the `default.qutrit.mixed` device.
  [(#5503)](https://github.com/PennyLaneAI/pennylane/pull/5503)
  [(#5757)](https://github.com/PennyLaneAI/pennylane/pull/5757)
  [(#5799)](https://github.com/PennyLaneAI/pennylane/pull/5799)
  
* `qml.TritFlip` has been added, allowing for trit flip errors, such as misclassification, 
  to be simulated on the `default.qutrit.mixed` device.
  [(#5784)](https://github.com/PennyLaneAI/pennylane/pull/5784)

<h3>Breaking changes 💔</h3>

* Passing `shots` as a keyword argument to a `QNode` initialization now raises an error, instead of ignoring the input.
  [(#5748)](https://github.com/PennyLaneAI/pennylane/pull/5748)

* A custom decomposition can no longer be provided to `QDrift`. Instead, apply the operations in your custom
  operation directly with `qml.apply`.
  [(#5698)](https://github.com/PennyLaneAI/pennylane/pull/5698)

* Sampling observables composed of `X`, `Y`, `Z` and `Hadamard` now returns values of type `float` instead of `int`.
  [(#5607)](https://github.com/PennyLaneAI/pennylane/pull/5607)

* `qml.is_commuting` no longer accepts the `wire_map` argument, which does not bring any functionality.
  [(#5660)](https://github.com/PennyLaneAI/pennylane/pull/5660)

* `qml.from_qasm_file` has been removed. The user can open files and load their content using `qml.from_qasm`.
  [(#5659)](https://github.com/PennyLaneAI/pennylane/pull/5659)

* `qml.load` has been removed in favour of more specific functions, such as `qml.from_qiskit`, etc.
  [(#5654)](https://github.com/PennyLaneAI/pennylane/pull/5654)

* `qml.transforms.convert_to_numpy_parameters` is now a proper transform and its output signature has changed,
  returning a list of `QuantumTape`s and a post-processing function instead of simply the transformed circuit.
  [(#5693)](https://github.com/PennyLaneAI/pennylane/pull/5693)

* `Controlled.wires` does not include `self.work_wires` anymore. That can be accessed separately through `Controlled.work_wires`.
  Consequently, `Controlled.active_wires` has been removed in favour of the more common `Controlled.wires`.
  [(#5728)](https://github.com/PennyLaneAI/pennylane/pull/5728)

<h3>Deprecations 👋</h3>

* The `simplify` argument in `qml.Hamiltonian` and `qml.ops.LinearCombination` is deprecated.
  Instead, `qml.simplify()` can be called on the constructed operator.
  [(#5677)](https://github.com/PennyLaneAI/pennylane/pull/5677)

* `qml.transforms.map_batch_transform` is deprecated, since a transform can be applied directly to a batch of tapes.
  [(#5676)](https://github.com/PennyLaneAI/pennylane/pull/5676)

<h3>Documentation 📝</h3>

* The documentation for the `default.tensor` device has been added.
  [(#5719)](https://github.com/PennyLaneAI/pennylane/pull/5719)

* A small typo was fixed in the docstring for `qml.sample`.
  [(#5685)](https://github.com/PennyLaneAI/pennylane/pull/5685)

* Typesetting for some of the documentation was fixed, (use of left/right delimiters, fractions, and fix of incorrectly set up commands)
  [(#5804)](https://github.com/PennyLaneAI/pennylane/pull/5804)

* The `qml.Tracker` examples are updated.
  [(#5803)](https://github.com/PennyLaneAI/pennylane/pull/5803)

<h3>Bug fixes 🐛</h3>

* Fixes a bug in the wire handling on special controlled ops.
  [(#5856)](https://github.com/PennyLaneAI/pennylane/pull/5856)

* Fixes a bug where `Sum`'s with repeated identical operations ended up with the same hash as
  `Sum`'s with different numbers of repeats.
  [(#5851)](https://github.com/PennyLaneAI/pennylane/pull/5851)

* `qml.qaoa.cost_layer` and `qml.qaoa.mixer_layer` can now be used with `Sum` operators.
  [(#5846)](https://github.com/PennyLaneAI/pennylane/pull/5846)

* Fixes a bug where `MottonenStatePreparation` produces wrong derivatives at special parameter values.
  [(#5774)](https://github.com/PennyLaneAI/pennylane/pull/5774)

* Fixes a bug where fractional powers and adjoints of operators were commuted, which is
  not well-defined/correct in general. Adjoints of fractional powers can no longer be evaluated.
  [(#5835)](https://github.com/PennyLaneAI/pennylane/pull/5835)

* `qml.qnn.TorchLayer` now works with tuple returns.
  [(#5816)](https://github.com/PennyLaneAI/pennylane/pull/5816)

* An error is now raised if a transform is applied to a catalyst qjit object.
  [(#5826)](https://github.com/PennyLaneAI/pennylane/pull/5826)

* `KerasLayer` and `TorchLayer` no longer mutate the input `QNode`'s interface.
  [(#5800)](https://github.com/PennyLaneAI/pennylane/pull/5800)

* Disable Docker builds on PR merge.
  [(#5777)](https://github.com/PennyLaneAI/pennylane/pull/5777)

* The validation of the adjoint method in `DefaultQubit` correctly handles device wires now.
  [(#5761)](https://github.com/PennyLaneAI/pennylane/pull/5761)

* `QuantumPhaseEstimation.map_wires` on longer modifies the original operation instance.
  [(#5698)](https://github.com/PennyLaneAI/pennylane/pull/5698)

* The decomposition of `AmplitudeAmplification` now correctly queues all operations.
  [(#5698)](https://github.com/PennyLaneAI/pennylane/pull/5698)

* Replaced `semantic_version` with `packaging.version.Version`, since the former cannot
  handle the metadata `.post` in the version string.
  [(#5754)](https://github.com/PennyLaneAI/pennylane/pull/5754)

* The `dynamic_one_shot` transform now has expanded support for the `jax` and `torch` interfaces.
  [(#5672)](https://github.com/PennyLaneAI/pennylane/pull/5672)

* The decomposition of `StronglyEntanglingLayers` is now compatible with broadcasting.
  [(#5716)](https://github.com/PennyLaneAI/pennylane/pull/5716)

* `qml.cond` can now be applied to `ControlledOp` operations when deferring measurements.
  [(#5725)](https://github.com/PennyLaneAI/pennylane/pull/5725)

* The legacy `Tensor` class can now handle a `Projector` with abstract tracer input.
  [(#5720)](https://github.com/PennyLaneAI/pennylane/pull/5720)

* Fixed a bug that raised an error regarding expected vs actual `dtype` when using `JAX-JIT` on a circuit that
  returned samples of observables containing the `qml.Identity` operator.
  [(#5607)](https://github.com/PennyLaneAI/pennylane/pull/5607)

* The signature of `CaptureMeta` objects (like `Operator`) now match the signature of the `__init__` call.
  [(#5727)](https://github.com/PennyLaneAI/pennylane/pull/5727)

* Use vanilla NumPy arrays in `test_projector_expectation` to avoid differentiating `qml.Projector` with respect to the state attribute.
  [(#5683)](https://github.com/PennyLaneAI/pennylane/pull/5683)

* `qml.Projector` is now compatible with jax-jit.
  [(#5595)](https://github.com/PennyLaneAI/pennylane/pull/5595)

* Finite shot circuits with a `qml.probs` measurement, both with a `wires` or `op` argument, can now be compiled with `jax.jit`.
  [(#5619)](https://github.com/PennyLaneAI/pennylane/pull/5619)

* `param_shift`, `finite_diff`, `compile`, `insert`, `merge_rotations`, and `transpile` now
  all work with circuits with non-commuting measurements.
  [(#5424)](https://github.com/PennyLaneAI/pennylane/pull/5424)
  [(#5681)](https://github.com/PennyLaneAI/pennylane/pull/5681)

* A correction is added to `bravyi_kitaev` to call the correct function for a FermiSentence input.
  [(#5671)](https://github.com/PennyLaneAI/pennylane/pull/5671)

* Fixes a bug where `sum_expand` produces incorrect result dimensions when combining shot vectors,
  multiple measurements, and parameter broadcasting.
  [(#5702)](https://github.com/PennyLaneAI/pennylane/pull/5702)

* Fixes a bug in `qml.math.dot` that raises an error when only one of the operands is a scalar.
  [(#5702)](https://github.com/PennyLaneAI/pennylane/pull/5702)

* `qml.matrix` is now compatible with qnodes compiled by catalyst.qjit.
  [(#5753)](https://github.com/PennyLaneAI/pennylane/pull/5753)

* `CNOT` and `Toffoli` now have an `arithmetic_depth` of `1`, as they are controlled operations.
  [(#5797)](https://github.com/PennyLaneAI/pennylane/pull/5797)

* Fixes a bug where the gradient of `ControlledSequence`, `Reflection`, `AmplitudeAmplification`, and `Qubitization` is incorrect on `default.qubit.legacy` with `parameter_shift`.
  [(#5806)](https://github.com/PennyLaneAI/pennylane/pull/5806)

* Fixed a bug where `split_non_commuting` raises an error when the circuit contains measurements of observables that are not pauli words.
  [(#5827)](https://github.com/PennyLaneAI/pennylane/pull/5827)

* Simplify method for `Exp` now returns an operator with the correct number of Trotter steps, i.e. equal to the one from the pre-simplified operator.
  [(#5831)](https://github.com/PennyLaneAI/pennylane/pull/5831)

* Fix bug where `CompositeOp.overlapping_ops` sometimes puts overlapping ops in different groups, leading to incorrect results returned by `LinearCombination.eigvals()`
  [(#5847)](https://github.com/PennyLaneAI/pennylane/pull/5847)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Tarun Kumar Allamsetty,
Guillermo Alonso-Linaje,
Utkarsh Azad,
Lillian M. A. Frederiksen,
Gabriel Bottrill,
Astral Cai,
Ahmed Darwish,
Isaac De Vlugt,
Diksha Dhawan,
Pietropaolo Frisoni,
Emiliano Godinez,
Austin Huang,
David Ittah,
Soran Jahangiri,
Rohan Jain,
Mashhood Khan,
Korbinian Kottmann,
Christina Lee,
Vincent Michaud-Rioux,
Lee James O'Riordan,
Mudit Pandey,
Kenya Sakka,
Jay Soni,
Kazuki Tsuoka,
Haochen Paul Wang,
David Wierichs.
