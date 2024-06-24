:orphan:

# Release 0.38.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* The `default.tensor` device now preserves the order of wires if the initial MPS is created from a dense state vector.
  [(#5892)](https://github.com/PennyLaneAI/pennylane/pull/5892)

* Fixes a bug where `hadamard_grad` returned a wrong shape for `qml.probs()` without wires.
  [(#5860)](https://github.com/PennyLaneAI/pennylane/pull/5860)

* An error is now raised on processing an `AnnotatedQueue` into a `QuantumScript` if the queue
  contains something other than an `Operator`, `MeasurementProcess`, or `QuantumScript`.
  [(#5866)](https://github.com/PennyLaneAI/pennylane/pull/5866)

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

* `qml.snapshots` raises an error when a measurement other than `qml.state` is requested from `default.qubit.legacy` instead of silently returning the statevector.
  [(#5805)](https://github.com/PennyLaneAI/pennylane/pull/5805)

* Fixes a bug where `default.qutrit` is falsely determined to be natively compatible with `qml.snapshots`.
  [(#5805)](https://github.com/PennyLaneAI/pennylane/pull/5805)

* Fixes a bug where the measurement of a `qml.Snapshot` instance is not passed on during the `qml.adjoint` and `qml.ctrl` operations.
  [(#5805)](https://github.com/PennyLaneAI/pennylane/pull/5805)

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

* Implement the correct decomposition for a `qml.PauliRot` with an identity as `pauli_word`, i.e. returns a `qml.GlobalPhase` with half the angle.
  [(#5875)](https://github.com/PennyLaneAI/pennylane/pull/5875)

* `qml.pauli_decompose` now works in a jit-ted context, such as `jax.jit` and `catalyst.qjit`.
  [(#5878)](https://github.com/PennyLaneAI/pennylane/pull/5878)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
