:orphan:

# Release 0.36.0 (current release)

<h3>New features since last release</h3>

* Support for entanglement entropy computation is added. `qml.math.vn_entanglement_entropy` computes the von Neumann entanglement entropy from a density matrix, and a QNode transform `qml.qinfo.vn_entanglement_entropy` is also added.
  [(#5306)](https://github.com/PennyLaneAI/pennylane/pull/5306)

<h4>Estimate errors in a quantum circuit 🧮</h4>

* Added `error` method to `QuantumPhaseEstimation` template.
  [(#5278)](https://github.com/PennyLaneAI/pennylane/pull/5278)

* Added new `SpectralNormError` class to the new error tracking functionality.
  [(#5154)](https://github.com/PennyLaneAI/pennylane/pull/5154)

* The `qml.TrotterProduct` operator now supports error estimation functionality. 
  [(#5384)](https://github.com/PennyLaneAI/pennylane/pull/5384)

  ```pycon
  >>> hamiltonian = qml.dot([1.0, 0.5, -0.25], [qml.X(0), qml.Y(0), qml.Z(0)])
  >>> op = qml.TrotterProduct(hamiltonian, time=0.01, order=2)
  >>> op.error(method="one-norm")
  SpectralNormError(8.039062500000003e-06)
  >>>
  >>> op.error(method="commutator")
  SpectralNormError(6.166666666666668e-06)
  ```

* `qml.specs` and `qml.Tracker` now return information about algorithmic errors for the qnode as well.
  [(#5464)](https://github.com/PennyLaneAI/pennylane/pull/5464)
  [(#5465)](https://github.com/PennyLaneAI/pennylane/pull/5465)


<h4>Access an extended arsenal of quantum algorithms 🏹</h4>

* The `FABLE` template is added for efficient block encoding of matrices. Users can now call FABLE to efficiently construct circuits according to a user-set approximation level. 
  [(#5107)](https://github.com/PennyLaneAI/pennylane/pull/5107)

* Create the `qml.Reflection` operator, useful for amplitude amplification and its variants.
  [(#5159)](https://github.com/PennyLaneAI/pennylane/pull/5159)

  ```python
  @qml.prod
  def generator(wires):
        qml.Hadamard(wires=wires)

  U = generator(wires=0)

  dev = qml.device('default.qubit')
  @qml.qnode(dev)
  def circuit():

        # Initialize to the state |1>
        qml.PauliX(wires=0)

        # Apply the reflection
        qml.Reflection(U)

        return qml.state()

  ```

  ```pycon
  >>> circuit()
  tensor([1.+6.123234e-17j, 0.-6.123234e-17j], requires_grad=True)
  ```
  
* The `qml.AmplitudeAmplification` operator is introduced, which is a high-level interface for amplitude amplification and its variants.
  [(#5160)](https://github.com/PennyLaneAI/pennylane/pull/5160)

  ```python
  @qml.prod
  def generator(wires):
      for wire in wires:
          qml.Hadamard(wires=wire)

  U = generator(wires=range(3))
  O = qml.FlipSign(2, wires=range(3))

  dev = qml.device("default.qubit")

  @qml.qnode(dev)
  def circuit():

      generator(wires=range(3))
      qml.AmplitudeAmplification(U, O, iters=5, fixed_point=True, work_wire=3)

      return qml.probs(wires=range(3))

  ```
  
  ```pycon
  >>> print(np.round(circuit(), 3))
  [0.013, 0.013, 0.91, 0.013, 0.013, 0.013, 0.013, 0.013]

  ```

<h4>Make use of more methods to map from molecules 🗺️</h4>

* Added new function `qml.bravyi_kitaev` to map fermionic Hamiltonians to qubit Hamiltonians.
  [(#5390)](https://github.com/PennyLaneAI/pennylane/pull/5390)

  ```python
  import pennylane as qml
  fermi_ham = qml.fermi.from_string('0+ 1-')

  qubit_ham = qml.bravyi_kitaev(fermi_ham, n=6)
  ```

  ```pycon
  >>> print(qubit_ham)
  -0.25j * Y(0.0) + (-0.25+0j) * X(0) @ Z(1.0) + (0.25+0j) * X(0.0) + 0.25j * Y(0) @ Z(1.0)
  ```

* The `qml.qchem.hf_state` function is upgraded to be compatible with the parity and Bravyi-Kitaev bases.
  [(#5472)](https://github.com/PennyLaneAI/pennylane/pull/5472)


* Added `qml.Qubitization` operator. This operator encodes a Hamiltonian into a suitable unitary operator. 
  When applied in conjunction with QPE, allows computing the eigenvalue of an eigenvector of the Hamiltonian.
  [(#5500)](https://github.com/PennyLaneAI/pennylane/pull/5500)

  ```python
  H = qml.dot([0.1, 0.3, -0.3], [qml.Z(0), qml.Z(1), qml.Z(0) @ qml.Z(2)])

  @qml.qnode(qml.device("default.qubit"))
  def circuit():

      # initialize the eigenvector
      qml.PauliX(2)

      # apply QPE
      measurements = qml.iterative_qpe(
                    qml.Qubitization(H, control = [3,4]), ancilla = 5, iters = 3
                    )
      return qml.probs(op = measurements)
  
  output = circuit()
  
  # post-processing 
  lamb = sum([abs(c) for c in H.terms()[0]])
  ```
  
  ```pycon
  >>> print("eigenvalue: ", lamb * np.cos(2 * np.pi * (np.argmax(output)) / 8))
  eigenvalue: 0.7
  ```


* A new `qml.lie_closure` function to compute the Lie closure of a list of operators.
  [(#5161)](https://github.com/PennyLaneAI/pennylane/pull/5161)
  [(#5169)](https://github.com/PennyLaneAI/pennylane/pull/5169)

  The Lie closure, pronounced "Lee closure", is a way to compute the so-called dynamical Lie algebra (DLA) of a set of operators.
  For a list of operators `ops = [op1, op2, op3, ..]`, one computes all nested commutators between `ops` until no new operators are generated from commutation.
  All these operators together form the DLA, see e.g. section IIB of [arXiv:2308.01432](https://arxiv.org/abs/2308.01432).

  Take for example the following ops

  ```python
  ops = [X(0) @ X(1), Z(0), Z(1)]
  ```

  A first round of commutators between all elements yields the new operators `Y(0) @ X(1)` and `X(0) @ Y(1)` (omitting scalar prefactors).

  ```python
  >>> qml.commutator(X(0) @ X(1), Z(0))
  -2j * (X(1) @ Y(0))
  >>> qml.commutator(X(0) @ X(1), Z(1))
  -2j * (Y(1) @ X(0))
  ```

  A next round of commutators between all elements further yields the new operator `Y(0) @ Y(1)`.

  ```python
  >>> qml.commutator(X(0) @ Y(1), Z(0))
  -2j * (Y(1) @ Y(0))
  ```

  After that, no new operators emerge from taking nested commutators and we have the resulting DLA.
  This can now be done in short via `qml.lie_closure` as follows.

  ```python
  >>> ops = [X(0) @ X(1), Z(0), Z(1)]
  >>> dla = qml.lie_closure(ops)
  >>> print(dla)
  [1.0 * X(1) @ X(0),
   1.0 * Z(0),
   1.0 * Z(1),
   -1.0 * X(1) @ Y(0),
   -1.0 * Y(1) @ X(0),
   -1.0 * Y(1) @ Y(0)]
  ```

* We can compute the structure constants (the adjoint representation) of a dynamical Lie algebra.
  [(5406)](https://github.com/PennyLaneAI/pennylane/pull/5406)

  For example, we can compute the adjoint representation of the transverse field Ising model DLA.

  ```python
  >>> dla = [X(0) @ X(1), Z(0), Z(1), Y(0) @ X(1), X(0) @ Y(1), Y(0) @ Y(1)]
  >>> structure_const = qml.structure_constants(dla)
  >>> structure_constp.shape
  (6, 6, 6)
  ```

* We can compute the center of a dynamical Lie algebra.
  [(#5477)](https://github.com/PennyLaneAI/pennylane/pull/5477)

  Given a DLA `g`, we can now compute its center. The `center` is the collection of operators that commute with _all_ other operators in the DLA.

  ```pycon
  >>> g = [X(0), X(1) @ X(0), Y(1), Z(1) @ X(0)]
  >>> qml.center(g)
  [X(0)]
  ```

<h4>Simulate mixed-state qutrit systems 3️⃣</h4>

* Functions `measure_with_samples` and `sample_state` have been added to the new `qutrit_mixed` module found in
 `qml.devices`. These functions are used to sample device-compatible states, returning either the final measured state or value of an observable.
  [(#5082)](https://github.com/PennyLaneAI/pennylane/pull/5082)

* Fixed differentiability for Hamiltonian measurements in new `qutrit_mixed` module. 
  [(#5186)](https://github.com/PennyLaneAI/pennylane/pull/5186)

* Added `simulate` function to the new `qutrit_mixed` module in `qml.devices`. This allows for simulation of a 
  noisy qutrit circuit with measurement and sampling.
  [(#5213)](https://github.com/PennyLaneAI/pennylane/pull/5213)

 * Created the `DefaultQutritMixed` class, which inherits from `qml.devices.Device`, with an implementation 
  for `preprocess`.
  [(#5451)](https://github.com/PennyLaneAI/pennylane/pull/5451)

 * Implemented `execute` on `qml.devices.DefaultQutritMixed` device, `execute` can be used to simulate noisy qutrit based circuits.
  [(#5495)](https://github.com/PennyLaneAI/pennylane/pull/5495)

<h3>Improvements 🛠</h3>

* Fixed typo and string formatting in error message in `ClassicalShadow._convert_to_pauli_words` when the input is not a valid pauli.
  [(#5572)](https://github.com/PennyLaneAI/pennylane/pull/5572)

<h4>Community contributions 🥳</h4>

* Implemented the method `process_counts` in `ExpectationMP`, `VarianceMP`, `CountsMP`, and `SampleMP`
  [(#5256)](https://github.com/PennyLaneAI/pennylane/pull/5256)
  [(#5395)](https://github.com/PennyLaneAI/pennylane/pull/5395)

* Add type hints for unimplemented methods of the abstract class `Operator`.
  [(#5490)](https://github.com/PennyLaneAI/pennylane/pull/5490)

* Implement `Shots.bins()` method.
  [(#5476)](https://github.com/PennyLaneAI/pennylane/pull/5476)

<h4>Updated operators</h4>

* `qml.ops.Sum` now supports storing grouping information. Grouping type and method can be
  specified during construction using the `grouping_type` and `method` keyword arguments of
  `qml.dot`, `qml.sum`, or `qml.ops.Sum`. The grouping indices are stored in `Sum.grouping_indices`.
  [(#5179)](https://github.com/PennyLaneAI/pennylane/pull/5179)

  ```python
  import pennylane as qml

  a = qml.X(0)
  b = qml.prod(qml.X(0), qml.X(1))
  c = qml.Z(0)
  obs = [a, b, c]
  coeffs = [1.0, 2.0, 3.0]

  op = qml.dot(coeffs, obs, grouping_type="qwc")
  ```

  ```pycon
  >>> op.grouping_indices
  ((2,), (0, 1))
  ```

  Additionally, grouping type and method can be set or changed after construction using
  `Sum.compute_grouping()`:

  ```python
  import pennylane as qml

  a = qml.X(0)
  b = qml.prod(qml.X(0), qml.X(1))
  c = qml.Z(0)
  obs = [a, b, c]
  coeffs = [1.0, 2.0, 3.0]

  op = qml.dot(coeffs, obs)
  ```

  ```pycon
  >>> op.grouping_indices is None
  True
  >>> op.compute_grouping(grouping_type="qwc")
  >>> op.grouping_indices
  ((2,), (0, 1))
  ```

  Note that the grouping indices refer to the lists returned by `Sum.terms()`, not `Sum.operands`.

* Added new function `qml.operation.convert_to_legacy_H` to convert `Sum`, `SProd`, and `Prod` to `Hamiltonian` instances.
  [(#5309)](https://github.com/PennyLaneAI/pennylane/pull/5309)

* The `qml.is_commuting` function now accepts `Sum`, `SProd`, and `Prod` instances.
  [(#5351)](https://github.com/PennyLaneAI/pennylane/pull/5351)

* Operators can now be left multiplied `x * op` by numpy arrays.
  [(#5361)](https://github.com/PennyLaneAI/pennylane/pull/5361)

* A new class `qml.ops.LinearCombination` is introduced. In essence, this class is an updated equivalent of `qml.ops.Hamiltonian`
  but for usage with new operator arithmetic.
  [(#5216)](https://github.com/PennyLaneAI/pennylane/pull/5216)

* The generators in the source code return operators consistent with the global setting for
  `qml.operator.active_new_opmath()` wherever possible. `Sum`, `SProd` and `Prod` instances
  will be returned even after disabling the new operator arithmetic in cases where they offer
  additional functionality not available using legacy operators.
  [(#5253)](https://github.com/PennyLaneAI/pennylane/pull/5253)
  [(#5410)](https://github.com/PennyLaneAI/pennylane/pull/5410)
  [(#5411)](https://github.com/PennyLaneAI/pennylane/pull/5411)
  [(#5421)](https://github.com/PennyLaneAI/pennylane/pull/5421)

* A new `Prod.obs` property is introduced to smoothen the transition of the new operator arithmetic system.
  In particular, this aims at preventing breaking code that uses `Tensor.obs`. This is immediately deprecated.
  Moving forward, we recommend using `op.operands`.
  [(#5539)](https://github.com/PennyLaneAI/pennylane/pull/5539)
  
* `ApproxTimeEvolution` is now compatible with any operator that defines a `pauli_rep`.
  [(#5362)](https://github.com/PennyLaneAI/pennylane/pull/5362)

* `Hamiltonian.pauli_rep` is now defined if the hamiltonian is a linear combination of paulis.
  [(#5377)](https://github.com/PennyLaneAI/pennylane/pull/5377)

* `Prod.eigvals()` is now compatible with Qudit operators.
  [(#5400)](https://github.com/PennyLaneAI/pennylane/pull/5400)

* `qml.transforms.hamiltonian_expand` can now handle multi-term observables with a constant offset.
  [(#5414)](https://github.com/PennyLaneAI/pennylane/pull/5414)

* `taper_operation` method is compatible with new operator arithmetic.
  [(#5326)](https://github.com/PennyLaneAI/pennylane/pull/5326)

* Removed the warning that an observable might not be hermitian in `qnode` executions. This enables jit-compilation.
  [(#5506)](https://github.com/PennyLaneAI/pennylane/pull/5506)

* `qml.transforms.split_non_commuting` will now work with single-term operator arithmetic.
  [(#5314)](https://github.com/PennyLaneAI/pennylane/pull/5314)

* `LinearCombination` and `Sum` now accept `_grouping_indices` on initialization.
  [(#5524)](https://github.com/PennyLaneAI/pennylane/pull/5524)

<h4>Mid-circuit measurements and dynamic circuits</h4>

* The `QubitDevice` class and children classes support the `dynamic_one_shot` transform provided that they support `MidMeasureMP` operations natively.
  [(#5317)](https://github.com/PennyLaneAI/pennylane/pull/5317)

* The `dynamic_one_shot` transform is introduced enabling dynamic circuit execution on circuits with shots and devices that support `MidMeasureMP` operations natively.
  [(#5266)](https://github.com/PennyLaneAI/pennylane/pull/5266)

* Added a qml.capture module that will contain PennyLane's own capturing mechanism for hybrid
  quantum-classical programs.
  [(#5509)](https://github.com/PennyLaneAI/pennylane/pull/5509)

<h4>Performance and broadcasting</h4>

* Gradient transforms may now be applied to batched/broadcasted QNodes, as long as the
  broadcasting is in non-trainable parameters.
  [(#5452)](https://github.com/PennyLaneAI/pennylane/pull/5452)

* Improve the performance of computing the matrix of `qml.QFT`
  [(#5351)](https://github.com/PennyLaneAI/pennylane/pull/5351)

* `qml.transforms.broadcast_expand` now supports shot vectors when returning `qml.sample()`.
  [(#5473)](https://github.com/PennyLaneAI/pennylane/pull/5473)

* `LightningVJPs` is now compatible with Lightning devices using the new device API.
  [(#5469)](https://github.com/PennyLaneAI/pennylane/pull/5469)

<h4>Other improvements</h4>

* Calculating the dense, differentiable matrix for `PauliSentence` and operators with pauli sentences
  is now faster.
  [(#5578)](https://github.com/PennyLaneAI/pennylane/pull/5578)

* `DefaultQubit` now uses the provided seed for sampling mid-circuit measurements with finite shots.
  [(#5337)](https://github.com/PennyLaneAI/pennylane/pull/5337)

* `qml.draw` and `qml.draw_mpl` will now attempt to sort the wires if no wire order
  is provided by the user or the device.
  [(#5576)](https://github.com/PennyLaneAI/pennylane/pull/5576)

* `qml.ops.Conditional` now stores the `data`, `num_params`, and `ndim_param` attributes of
  the operator it wraps.
  [(#5473)](https://github.com/PennyLaneAI/pennylane/pull/5473)

* The `molecular_hamiltonian` function calls `PySCF` directly when `method='pyscf'` is selected.
  [(#5118)](https://github.com/PennyLaneAI/pennylane/pull/5118)

* Upgraded `null.qubit` to the new device API. Also, added support for all measurements and various modes of differentiation.
  [(#5211)](https://github.com/PennyLaneAI/pennylane/pull/5211)

* Obtaining classical shadows using the `default.clifford` device is now compatible with
  [stim](https://github.com/quantumlib/Stim) `v1.13.0`.
  [(#5409)](https://github.com/PennyLaneAI/pennylane/pull/5409)

* `qml.transforms.hamiltonian_expand` and `qml.transforms.sum_expand` can now handle multi-term observables with a constant offset.
  [(#5414)](https://github.com/PennyLaneAI/pennylane/pull/5414)
  [(#5543)](https://github.com/PennyLaneAI/pennylane/pull/5543)

* `default.mixed` has improved support for sampling-based measurements with non-numpy interfaces.
  [(#5514)](https://github.com/PennyLaneAI/pennylane/pull/5514)
  [(#5530)](https://github.com/PennyLaneAI/pennylane/pull/5530)

* `default.mixed` now supports arbitrary state-based measurements with `qml.Snapshot`.
  [(#5552)](https://github.com/PennyLaneAI/pennylane/pull/5552)

* Replaced `cache_execute` with an alternate implementation based on `@transform`.
  [(#5318)](https://github.com/PennyLaneAI/pennylane/pull/5318)

* The `QNode` now defers `diff_method` validation to the device under the new device api `qml.devices.Device`.
  [(#5176)](https://github.com/PennyLaneAI/pennylane/pull/5176)

* Extend the device test suite to cover gradient methods, templates and arithmetic observables.
  [(#5273)](https://github.com/PennyLaneAI/pennylane/pull/5273)
  [(#5518)](https://github.com/PennyLaneAI/pennylane/pull/5518)

* A clear error message is added in `KerasLayer` when using the newest version of TensorFlow with Keras 3 
  (which is not currently compatible with `KerasLayer`), linking to instructions to enable Keras 2.
  [(#5488)](https://github.com/PennyLaneAI/pennylane/pull/5488)

<h3>Breaking changes 💔</h3>

* Applying a `gradient_transform` to a QNode directly now gives the same shape and type independent
  of whether there is classical processing in the node.
  [(#4945)](https://github.com/PennyLaneAI/pennylane/pull/4945)
  
* State measurements preserve `dtype`.
  [(#5547)](https://github.com/PennyLaneAI/pennylane/pull/5547)

* Use `SampleMP`s in the `dynamic_one_shot` transform to get back the values of the mid-circuit measurements.
  [(#5486)](https://github.com/PennyLaneAI/pennylane/pull/5486)

* Operator dunder methods now combine like-operator arithmetic classes via `lazy=False`. This reduces the chance of `RecursionError` and makes nested
  operators easier to work with.
  [(#5478)](https://github.com/PennyLaneAI/pennylane/pull/5478)

* The private functions `_pauli_mult`, `_binary_matrix` and `_get_pauli_map` from the `pauli` module have been removed. The same functionality can be achieved using newer features in the ``pauli`` module.
  [(#5323)](https://github.com/PennyLaneAI/pennylane/pull/5323)
  
* `DefaultQubit` uses a pre-emptive key-splitting strategy to avoid reusing JAX PRNG keys throughout a single `execute` call. 
  [(#5515)](https://github.com/PennyLaneAI/pennylane/pull/5515)

* `qml.matrix()` called on the following will raise an error if `wire_order` is not specified:
  * tapes with more than one wire.
  * quantum functions.
  * Operator class where `num_wires` does not equal to 1
  * QNodes if the device does not have wires specified.
  * PauliWords and PauliSentences with more than one wire.
  [(#5328)](https://github.com/PennyLaneAI/pennylane/pull/5328)
  [(#5359)](https://github.com/PennyLaneAI/pennylane/pull/5359)

* `qml.pauli.pauli_mult` and `qml.pauli.pauli_mult_with_phase` are now removed. Instead, you  should use `qml.simplify(qml.prod(pauli_1, pauli_2))` to get the reduced operator.
  [(#5324)](https://github.com/PennyLaneAI/pennylane/pull/5324)

  ```pycon
  >>> op = qml.simplify(qml.prod(qml.PauliX(0), qml.PauliZ(0)))
  >>> op
  -1j*(PauliY(wires=[0]))
  >>> [phase], [base] = op.terms()
  >>> phase, base
  (-1j, PauliY(wires=[0]))
  ```

* `MeasurementProcess.name` and `MeasurementProcess.data` have been removed. Use `MeasurementProcess.obs.name` and `MeasurementProcess.obs.data` instead.
  [(#5321)](https://github.com/PennyLaneAI/pennylane/pull/5321)

* `Operator.validate_subspace(subspace)` has been removed. Instead, you should use `qml.ops.qutrit.validate_subspace(subspace)`.
  [(#5311)](https://github.com/PennyLaneAI/pennylane/pull/5311)

* The contents of `qml.interfaces` is moved inside `qml.workflow`. The old import path no longer exists.
  [(#5329)](https://github.com/PennyLaneAI/pennylane/pull/5329)

* `single_tape_transform`, `batch_transform`, `qfunc_transform`, `op_transform`, `gradient_transform`
  and `hessian_transform` are removed. Instead, switch to using the new `qml.transform` function. Please refer to
  `the transform docs <https://docs.pennylane.ai/en/stable/code/qml_transforms.html#custom-transforms>`_
  to see how this can be done.
  [(#5339)](https://github.com/PennyLaneAI/pennylane/pull/5339)

* Attempting to multiply `PauliWord` and `PauliSentence` with `*` will raise an error. Instead, use `@` to conform with the PennyLane convention.
  [(#5341)](https://github.com/PennyLaneAI/pennylane/pull/5341)

* When new operator arithmetic is enabled, `qml.Hamiltonian` is now an alias for `qml.ops.LinearCombination`.
  `Hamiltonian` will still be accessible as `qml.ops.Hamiltonian`.
  [(#5393)](https://github.com/PennyLaneAI/pennylane/pull/5393)

* Since `default.mixed` does not support snapshots with measurements, attempting to do so will result in a `DeviceError` instead of getting the density matrix.
  [(#5416)](https://github.com/PennyLaneAI/pennylane/pull/5416)

* `LinearCombination._obs_data` is removed. You can still use `LinearCombination.compare` to check mathematical equivalence between a `LinearCombination` and another operator.
  [(#5504)](https://github.com/PennyLaneAI/pennylane/pull/5504)

<h3>Deprecations 👋</h3>

* `qml.load` is deprecated. Instead, please use the functions outlined in the *Importing workflows* quickstart guide, such as `qml.from_qiskit`.
  [(#5312)](https://github.com/PennyLaneAI/pennylane/pull/5312)

* Specifying `control_values` with a bit string to `qml.MultiControlledX` is deprecated. Instead, use a list of booleans or 1s and 0s.
  [(#5352)](https://github.com/PennyLaneAI/pennylane/pull/5352)

* `qml.from_qasm_file` is deprecated. Instead, please open the file and then load its content using `qml.from_qasm`.
  [(#5331)](https://github.com/PennyLaneAI/pennylane/pull/5331)

  ```pycon
  >>> with open("test.qasm", "r") as f:
  ...     circuit = qml.from_qasm(f.read())
  ```

* Accessing `qml.ops.Hamiltonian` with new operator arithmetic is deprecated. Using `qml.Hamiltonian` with new operator arithmetic enabled now
  returns a `LinearCombination` instance. Some functionality may not work as expected. To continue using the `Hamiltonian` class, you can use
  `qml.operation.disable_new_opmath()` to disable the new operator arithmetic.
  [(#5393)](https://github.com/PennyLaneAI/pennylane/pull/5393)

<h3>Documentation 📝</h3>

* Adds a page explaining the shapes and nesting of result objects.
  [(#5418)](https://github.com/PennyLaneAI/pennylane/pull/5418)

* Removed some redundant documentation for the `evolve` function.
  [(#5347)](https://github.com/PennyLaneAI/pennylane/pull/5347)

* Updated the final example in the `compile` docstring to use transforms correctly.
  [(#5348)](https://github.com/PennyLaneAI/pennylane/pull/5348)

* A link to the demos for using `qml.SpecialUnitary` and `qml.QNGOptimizer` has been added to their respective docstrings.
  [(#5376)](https://github.com/PennyLaneAI/pennylane/pull/5376)

* A code example in the `qml.measure` docstring has been added that showcases returning mid-circuit measurement statistics from QNodes.
  [(#5441)](https://github.com/PennyLaneAI/pennylane/pull/5441)

* The computational basis convention used for `qml.measure` — 0 and 1 rather than ±1 — has been clarified in its docstring.
  [(#5474)](https://github.com/PennyLaneAI/pennylane/pull/5474)

* A new *Release news* section has been added to the table of contents, containing release notes,
  deprecations, and other pages focusing on recent changes.
  [(#5548)](https://github.com/PennyLaneAI/pennylane/pull/5548)

<h3>Bug fixes 🐛</h3>

* `null.qubit` now automatically supports any operation without a decomposition.
  [(#5582)](https://github.com/PennyLaneAI/pennylane/pull/5582)

* Fixed a bug where the shape and type of derivatives obtained by applying a gradient transform to
  a QNode differed, based on whether the QNode uses classical coprocessing.
  [(#4945)](https://github.com/PennyLaneAI/pennylane/pull/4945)

* `ApproxTimeEvolution`, `CommutingEvolution`, `QDrift`, and `TrotterProduct` 
  now de-queue their input observable.
  [(#5524)](https://github.com/PennyLaneAI/pennylane/pull/5524)

* (In)equality of `qml.HilbertSchmidt` instances is now reported correctly by `qml.equal`.
  [(#5538)](https://github.com/PennyLaneAI/pennylane/pull/5538)

* `qml.ParticleConservingU1` and `qml.ParticleConservingU2` no longer raise an error when the initial state is not specified but default to the all-zeros state.
  [(#5535)](https://github.com/PennyLaneAI/pennylane/pull/5535)

* `qml.counts` no longer returns negative samples when measuring 8 or more wires.
  [(#5544)](https://github.com/PennyLaneAI/pennylane/pull/5544)
  [(#5556)](https://github.com/PennyLaneAI/pennylane/pull/5556)

* The `dynamic_one_shot` transform now works with broadcasting.
  [(#5473)](https://github.com/PennyLaneAI/pennylane/pull/5473)

* Diagonalize the state around `ProbabilityMP` measurements in `statistics` when executing on a Lightning device.
  [(#5529)](https://github.com/PennyLaneAI/pennylane/pull/5529)

* `two_qubit_decomposition` no longer diverges at a special case of unitary matrix.
  [(#5448)](https://github.com/PennyLaneAI/pennylane/pull/5448)

* The `qml.QNSPSAOptimizer` now correctly handles optimization for legacy devices that do not follow the new API design.
  [(#5497)](https://github.com/PennyLaneAI/pennylane/pull/5497)

* Operators applied to all wires are now drawn correctly in a circuit with mid-circuit measurements.
  [(#5501)](https://github.com/PennyLaneAI/pennylane/pull/5501)

* Fix a bug where certain unary mid-circuit measurement expressions would raise an uncaught error.
  [(#5480)](https://github.com/PennyLaneAI/pennylane/pull/5480)

* The probabilities now sum to one using the `torch` interface with `default_dtype` set to `torch.float32`. 
  [(#5462)](https://github.com/PennyLaneAI/pennylane/pull/5462)

* Tensorflow can now handle devices with float32 results but float64 input parameters.
  [(#5446)](https://github.com/PennyLaneAI/pennylane/pull/5446)

* Fix a bug where the `argnum` kwarg of `qml.gradients.stoch_pulse_grad` references the wrong parameters in a tape,
  creating an inconsistency with other differentiation methods and preventing some use cases.
  [(#5458)](https://github.com/PennyLaneAI/pennylane/pull/5458)

* Avoid bounded value failures due to numerical noise with calls to `np.random.binomial`.
  [(#5447)](https://github.com/PennyLaneAI/pennylane/pull/5447)

* Using `@` with legacy Hamiltonian instances now properly de-queues the previously existing operations.
  [(#5454)](https://github.com/PennyLaneAI/pennylane/pull/5455)

* The `QNSPSAOptimizer` now properly handles differentiable parameters, resulting in being able to use it for more than one optimization step.
  [(#5439)](https://github.com/PennyLaneAI/pennylane/pull/5439)

* The `QNode` interface now resets if an error occurs during execution.
  [(#5449)](https://github.com/PennyLaneAI/pennylane/pull/5449)

* Fix failing tests due to changes with Lightning's adjoint diff pipeline.
  [(#5450)](https://github.com/PennyLaneAI/pennylane/pull/5450)

* Fix Torch tensor locality with autoray-registered coerce method.
  [(#5438)](https://github.com/PennyLaneAI/pennylane/pull/5438)

* `jax.jit` now works with `qml.sample` with a multi-wire observable.
  [(#5422)](https://github.com/PennyLaneAI/pennylane/pull/5422)

* `qml.qinfo.quantum_fisher` now works with non-`default.qubit` devices.
  [(#5423)](https://github.com/PennyLaneAI/pennylane/pull/5423)

* We no longer perform unwanted dtype promotion in the `pauli_rep` of `SProd` instances when using tensorflow.
  [(#5246)](https://github.com/PennyLaneAI/pennylane/pull/5246)

* Fixed `TestQubitIntegration.test_counts` in `tests/interfaces/test_jax_qnode.py` to always produce counts for all
  outcomes.
  [(#5336)](https://github.com/PennyLaneAI/pennylane/pull/5336)

* Fixed `PauliSentence.to_mat(wire_order)` to support identities with wires.
  [(#5407)](https://github.com/PennyLaneAI/pennylane/pull/5407)

* `CompositeOp.map_wires` now correctly maps the `overlapping_ops` property.
  [(#5430)](https://github.com/PennyLaneAI/pennylane/pull/5430)

* Update `DefaultQubit.supports_derivatives` to correctly handle circuits containing `MidMeasureMP` with adjoint
  differentiation.
  [(#5434)](https://github.com/PennyLaneAI/pennylane/pull/5434)

* `SampleMP`, `ExpectationMP`, `CountsMP`, `VarianceMP` constructed with ``eigvals`` can now properly process samples.
  [(#5463)](https://github.com/PennyLaneAI/pennylane/pull/5463)

* Fixes a bug in `hamiltonian_expand` that produces incorrect output dimensions when shot vectors are combined with parameter broadcasting.
  [(#5494)](https://github.com/PennyLaneAI/pennylane/pull/5494)

* Allows `default.qubit` to measure Identity on no wires, and observables containing Identity on
  no wires.
  [(#5570)](https://github.com/PennyLaneAI/pennylane/pull/5570/)

* Fixes a bug where `TorchLayer` does not work with shot vectors.
  [(#5492)](https://github.com/PennyLaneAI/pennylane/pull/5492)

* Fixes a bug where the output shape of a qnode returning a list containing a single measurement is incorrect when combined with shot vectors.
  [(#5492)](https://github.com/PennyLaneAI/pennylane/pull/5492)

* Fixes a bug in `qml.math.kron` that makes torch incompatible with numpy.
  [(#5540)](https://github.com/PennyLaneAI/pennylane/pull/5540)

* Fixes a bug in `_group_measurements` that fails to group measurements with commuting observables when they are operands of `Prod`.
  [(#5512)](https://github.com/PennyLaneAI/pennylane/issues/5512)

* `qml.equal` can now be used with sums and products that contain operators on no wires like `I` and `GlobalPhase`.
  [(#5562)](https://github.com/PennyLaneAI/pennylane/pull/5562)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Tarun Kumar Allamsetty,
Guillermo Alonso,
Mikhail Andrenkov,
Utkarsh Azad,
Gabriel Bottrill,
Astral Cai,
Diksha Dhawan,
Isaac De Vlugt,
Amintor Dusko,
Pietropaolo Frisoni,
Lillian M. A. Frederiksen,
Austin Huang,
Soran Jahangiri,
Korbinian Kottmann,
Christina Lee,
Vincent Michaud-Rioux,
Mudit Pandey,
Kenya Sakka,
Jay Soni,
Matthew Silverman,
David Wierichs.
