:orphan:

# Release 0.30.0-dev (development release)

<h3>New features since last release</h3>

* The `sample_state` function is added to `devices/qubit` that returns a series of samples based on a given
  state vector and a number of shots.
  [(#3720)](https://github.com/PennyLaneAI/pennylane/pull/3720)

* Added the needed functions and classes to simulate an ensemble of Rydberg atoms:
  * A new internal `RydbergHamiltonian` class is added, which contains the Hamiltonian of an ensemble of
    Rydberg atoms.
  * A new user-facing `rydberg_interaction` function is added, which returns a `RydbergHamiltonian` containing
    the Hamiltonian of the interaction of all the Rydberg atoms.
  * A new user-facing `rydberg_drive` function is added, which returns a `RydbergHamiltonian` containing
    the Hamiltonian of the interaction between a driving laser field and a group of atoms.
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)

* Added `Operation.__truediv__` dunder method to be able to divide operators.
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)

* The `simulate` function added to `devices/qubit` now supports measuring expectation values of large observables such as
  `qml.Hamiltonian`, `qml.SparseHamiltonian`, `qml.Sum`.
  [(#3759)](https://github.com/PennyLaneAI/pennylane/pull/3759)

<h3>Improvements</h3>

* `Operator` now has a `has_generator` attribute that returns whether or not the operator
  has a generator defined. It is used in `qml.operation.has_gen`, improving its performance.
  [(#3875)](https://github.com/PennyLaneAI/pennylane/pull/3875)

* The custom JVP rules in PennyLane now also support non-scalar and mixed-shape tape parameters as
  well as multi-dimensional tape return types, like broadcasted `qml.probs`, for example.
  [(#3766)](https://github.com/PennyLaneAI/pennylane/pull/3766)

* The `qchem.jordan_wigner` function is extended to support more fermionic operator orders.
  [(#3754)](https://github.com/PennyLaneAI/pennylane/pull/3754)
  [(#3751)](https://github.com/PennyLaneAI/pennylane/pull/3751)

* `AdaptiveOptimizer` is updated to use non-default user-defined qnode arguments.
  [(#3765)](https://github.com/PennyLaneAI/pennylane/pull/3765)

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

* Added `max_distance` keyword argument to `qml.pulse.rydberg_interaction` to allow removal of negligible contributions
  from atoms beyond `max_distance`from each other.
  [(#3889)](https://github.com/PennyLaneAI/pennylane/pull/3889)

* 3 new decomposition algorithms are added for n-controlled operations with single-qubit target,
  and are selected automatically when they produce a better result. They can be accessed via
  `ops.op_math.ctrl_decomp_bisect`.
  [(#3851)](https://github.com/PennyLaneAI/pennylane/pull/3851)

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

<h3>Deprecations</h3>

<h3>Documentation</h3>

* A typo was corrected in the documentation for introduction to `inspecting_circuits` and `chemistry`.
[(#3844)](https://github.com/PennyLaneAI/pennylane/pull/3844)

<h3>Bug fixes</h3>

* Fixed bug where the coefficients where not ordered correctly when summing a `ParametrizedHamiltonian`
  with other operators.
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)

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

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Komi Amiko
Utkarsh Azad
Lillian M. A. Frederiksen
Soran Jahangiri
Christina Lee
Vincent Michaud-Rioux
Albert Mitjans
Romain Moyard
Mudit Pandey
Matthew Silverman
Jay Soni
David Wierichs
