:orphan:

# Release 0.31.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* The `qchem.molecular_hamiltonian` function is upgraded to support custom wires for constructing
  differentiable Hamiltonians.
  [(4050)](https://github.com/PennyLaneAI/pennylane/pull/4050)

* An error is now raised by `qchem.molecular_hamiltonian` when the `dhf` method is used for an 
  open-shell system. This duplicates a similar error in `qchem.Molecule` but makes it easier to
  inform the users that the `pyscf` backend can be used for open-shell calculations.
  [(4058)](https://github.com/PennyLaneAI/pennylane/pull/4058)

* Added a `shots` property to `QuantumScript`. This will allow shots to be tied to executions instead of devices more
  concretely.
  [(#4067)](https://github.com/PennyLaneAI/pennylane/pull/4067)

* `qml.specs` is compatible with custom operations that have `depth` bigger than 1.
  [(#4033)](https://github.com/PennyLaneAI/pennylane/pull/4033)

* `qml.prod` now accepts a single qfunc input for creating new `Prod` operators.
  [(#4011)](https://github.com/PennyLaneAI/pennylane/pull/4011)

* Added a function `measure_with_samples` that returns a sample-based measurement result given a state
  [(#4083)](https://github.com/PennyLaneAI/pennylane/pull/4083)

<h3>Breaking changes 💔</h3>

* `pennylane.collections`, `pennylane.op_sum`, and `pennylane.utils.sparse_hamiltonian` are removed.

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

* The description of `mult` in the `qchem.Molecule` docstring now correctly states the value
  of `mult` that is supported.
  [(4058)](https://github.com/PennyLaneAI/pennylane/pull/4058)

<h3>Bug fixes 🐛</h3>

* Removes a patch in `interfaces/autograd.py` that checks for the `strawberryfields.gbs` device.  That device
  is pinned to PennyLane <= v0.29.0, so that patch is no longer necessary.

<<<<<<< HEAD
* The `qchem.jordan_wigner` function is extended to support more fermionic operator orders.
  [(#3754)](https://github.com/PennyLaneAI/pennylane/pull/3754)
  [(#3751)](https://github.com/PennyLaneAI/pennylane/pull/3751)

* `AdaptiveOptimizer` is updated to use non-default user-defined qnode arguments.
  [(#3765)](https://github.com/PennyLaneAI/pennylane/pull/3765)

* Adds logic to `qml.devices.qubit.measure` to compute the expectation values of `Hamiltonian` and `Sum `
  in a backpropagation compatible way.
  [(#3862)](https://github.com/PennyLaneAI/pennylane/pull/3862/)

* Use `TensorLike` type in `Operator` dunder methods.
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)

* The `apply_operation` function added to `devices/qubit` now supports broadcasting.
  [(#3852)](https://github.com/PennyLaneAI/pennylane/pull/3852)

* `qml.QubitStateVector.state_vector` now supports broadcasting.
  [(#3852)](https://github.com/PennyLaneAI/pennylane/pull/3852)
  
* `pennylane.devices.qubit.preprocess` now allows circuits with non-commuting observables.
  [(#3857)](https://github.com/PennyLaneAI/pennylane/pull/3857)

* When using Jax-jit with gradient transforms the trainable parameters are correctly set (instead of every parameter 
  to be set as trainable), and therefore the derivatives are computed more efficiently.
  [(#3697)](https://github.com/PennyLaneAI/pennylane/pull/3697)

* `qml.SparseHamiltonian` can now be applied to any wires in a circuit rather than being restricted to all wires
  in the circuit.
  [(#3888)](https://github.com/PennyLaneAI/pennylane/pull/3888)

* Added `max_distance` keyword argument to `qml.pulse.rydberg_interaction` to allow removal of negligible contributions
  from atoms beyond `max_distance`from each other.
  [(#3889)](https://github.com/PennyLaneAI/pennylane/pull/3889)

* 3 new decomposition algorithms are added for n-controlled operations with single-qubit target,
  and are selected automatically when they produce a better result. They can be accessed via
  `ops.op_math.ctrl_decomp_bisect`.
  [(#3851)](https://github.com/PennyLaneAI/pennylane/pull/3851)

* `repr` for `MutualInfoMP` now displays the distribution of the wires between the two subsystems.
  [(#3898)](https://github.com/PennyLaneAI/pennylane/pull/3898)

* Changed `Operator.num_wires` from an abstract value to `AnyWires`.
  [(#3919)](https://github.com/PennyLaneAI/pennylane/pull/3919)

* Do not run `qml.transforms.sum_expand` in `Device.batch_transform` if the device supports Sum observables.
  [(#3915)](https://github.com/PennyLaneAI/pennylane/pull/3915)

* `CompositeOp` now overrides `Operator._check_batching`, providing a significant performance improvement.
  `Hamiltonian` also overrides this method and does nothing, because it does not support batching.
  [(#3915)](https://github.com/PennyLaneAI/pennylane/pull/3915)

* If a `Sum` operator has a pre-computed Pauli representation, `is_hermitian` now checks that all coefficients
  are real, providing a significant performance improvement.
  [(#3915)](https://github.com/PennyLaneAI/pennylane/pull/3915)

  * The type of `n_electrons` in `qml.qchem.Molecule` is set to `int`.
  [(#3885)](https://github.com/PennyLaneAI/pennylane/pull/3885)

* Added explicit errors to `QutritDevice` if `classical_shadow` or `shadow_expval` are measured.
  [(#3934)](https://github.com/PennyLaneAI/pennylane/pull/3934)

* `DefaultQutrit` supports the new return system.
  [(#3934)](https://github.com/PennyLaneAI/pennylane/pull/3934)

* `QubitDevice` now defines the private `_get_diagonalizing_gates(circuit)` method and uses it when executing circuits.
  This allows devices that inherit from `QubitDevice` to override and customize their definition of diagonalizing gates.
  [(#3938)](https://github.com/PennyLaneAI/pennylane/pull/3938)

<h3>Breaking changes</h3>

* Both JIT interfaces are not compatible with Jax `>0.4.3`, we raise an error for those versions.
  [(#3877)](https://github.com/PennyLaneAI/pennylane/pull/3877)

* An operation that implements a custom `generator` method, but does not always return a valid generator, also has
  to implement a `has_generator` property that reflects in which scenarios a generator will be returned.
  [(#3875)](https://github.com/PennyLaneAI/pennylane/pull/3875)
 
* Trainable parameters for the Jax interface are the parameters that are `JVPTracer`, defined by setting
  `argnums`. Previously, all JAX tracers, including those used for JIT compilation, were interpreted to be trainable.
  [(#3697)](https://github.com/PennyLaneAI/pennylane/pull/3697)

* The keyword argument `argnums` is now used for gradient transform using Jax, instead of `argnum`.
  `argnum` is automatically converted to `argnums` when using JAX, and will no longer be supported in v0.31.
  [(#3697)](https://github.com/PennyLaneAI/pennylane/pull/3697)
  [(#3847)](https://github.com/PennyLaneAI/pennylane/pull/3847)

* Made `qml.OrbitalRotation` and consequently `qml.GateFabric` consistent with the interleaved Jordan-Wigner ordering.
  Previously, they were consistent with the sequential Jordan-Wigner ordering.
  [(#3861)](https://github.com/PennyLaneAI/pennylane/pull/3861)

* Some `MeasurementProcess` classes can now only be instantiated with arguments that they will actually use.
  For example, you can no longer create `StateMP(qml.PauliX(0))` or `PurityMP(eigvals=(-1,1), wires=Wires(0))`.
  [(#3898)](https://github.com/PennyLaneAI/pennylane/pull/3898)

<h3>Deprecations</h3>

<h3>Documentation</h3>

* A typo was corrected in the documentation for introduction to `inspecting_circuits` and `chemistry`.
[(#3844)](https://github.com/PennyLaneAI/pennylane/pull/3844)

<h3>Bug fixes</h3>

* Fixed a bug where the time parameter of `ParametrizedEvolution` as function argument would break
  JIT compiling for some inputs.
  [(#3962)](https://github.com/PennyLaneAI/pennylane/pull/3962)

* Fixed a bug where calling `Evolution.generator` with `coeff` being a complex ArrayBox raised an error.
  [(#3796)](https://github.com/PennyLaneAI/pennylane/pull/3796)
  
* `MeasurementProcess.hash` now uses the hash property of the observable. The property now depends on all
  properties that affect the behaviour of the object, such as `VnEntropyMP.log_base` or the distribution of wires between
  the two subsystems in `MutualInfoMP`.
  [(#3898)](https://github.com/PennyLaneAI/pennylane/pull/3898)

* The enum `measurements.Purity` is added so that `PurityMP.return_type` is defined. `str` and `repr` for `PurityMP` are now defined.
  [(#3898)](https://github.com/PennyLaneAI/pennylane/pull/3898)

* `Sum.hash` and `Prod.hash` are slightly changed
  to work with non-numeric wire labels.  `sum_expand` should now return correct results and not treat some products as the same
  operation.
  [(#3898)](https://github.com/PennyLaneAI/pennylane/pull/3898)
  
* Fixed bug where the coefficients where not ordered correctly when summing a `ParametrizedHamiltonian`
  with other operators.
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)
  [(#3902)](https://github.com/PennyLaneAI/pennylane/pull/3902)

* The metric tensor transform is fully compatible with Jax and therefore users can provide multiple parameters.
  [(#3847)](https://github.com/PennyLaneAI/pennylane/pull/3847)

* Registers `math.ndim` and `math.shape` for built-ins and autograd to accomodate Autoray 0.6.1.
  [#3864](https://github.com/PennyLaneAI/pennylane/pull/3865)

* Ensure that `qml.data.load` returns datasets in a stable and expected order.
  [(#3856)](https://github.com/PennyLaneAI/pennylane/pull/3856)

* The `qml.equal` function now handles comparisons of `ParametrizedEvolution` operators.
  [(#3870)](https://github.com/PennyLaneAI/pennylane/pull/3870)

* Made `qml.OrbitalRotation` and consequently `qml.GateFabric` consistent with the interleaved Jordan-Wigner ordering.
  [(#3861)](https://github.com/PennyLaneAI/pennylane/pull/3861)

* `qml.devices.qubit.apply_operation` catches the `tf.errors.UnimplementedError` that occurs when `PauliZ` or `CNOT` gates
  are applied to a large (>8 wires) tensorflow state. When that occurs, the logic falls back to the tensordot logic instead.
  [(#3884)](https://github.com/PennyLaneAI/pennylane/pull/3884/)

* Fixed parameter broadcasting support with `qml.counts` in most cases, and introduced explicit errors otherwise.
  [(#3876)](https://github.com/PennyLaneAI/pennylane/pull/3876)

* An error is now raised if a `QNode` with Jax-jit in use returns `counts` while having trainable parameters
  [(#3892)](https://github.com/PennyLaneAI/pennylane/pull/3892)

* A correction is added to the reference values in `test_dipole_of` to account for small changes
  (~2e-8) in the computed dipole moment values, resulting from the new [PySCF 2.2.0](https://github.com/pyscf/pyscf/releases/tag/v2.2.0) release.
  [(#3908)](https://github.com/PennyLaneAI/pennylane/pull/3908)

* `SampleMP.shape` is now correct when sampling only occurs on a subset of the device wires.
  [(#3921)](https://github.com/PennyLaneAI/pennylane/pull/3921)

<h3>Contributors</h3>
=======
<h3>Contributors ✍️</h3>
>>>>>>> f0cabe85f86355a4dc0f132121db85d166949dbe

This release contains contributions from (in alphabetical order):

Isaac De Vlugt,
Soran Jahangiri,
Christina Lee,
Mudit Pandey,
Matthew Silverman,
Jay Soni
