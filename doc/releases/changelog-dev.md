:orphan:

# Release 0.37.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

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

* The `molecular_hamiltonian` function now works with Molecule as the central object.
  [(#5571)](https://github.com/PennyLaneAI/pennylane/pull/5571)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
