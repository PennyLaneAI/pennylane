:orphan:

# Release 0.35.0-dev (development release)

<h3>New features since last release</h3>

* Adjoint device VJP's are now supported with `jax.jacobian`. `device_vjp=True` is
  is now strictly faster for jax.
  [(#4963)](https://github.com/PennyLaneAI/pennylane/pull/4963)

<h3>Improvements 🛠</h3>

* Improve the performance of circuit-cutting workloads with large numbers of generated tapes.
  [(#5005)](https://github.com/PennyLaneAI/pennylane/pull/5005)

* Update `tests/ops/functions/conftest.py` to ensure all operator types are tested for validity.
  [(#4978)](https://github.com/PennyLaneAI/pennylane/pull/4978)

* Upgrade Pauli arithmetic:
  You can now multiply `PauliWord` and `PauliSentence` instances by scalars, e.g. `0.5 * PauliWord({0:"X"})` or `0.5 * PauliSentence({PauliWord({0:"X"}): 1.})`.
  You can now intuitively add together 
  `PauliWord` and `PauliSentence` as well as scalars, which are treated implicitly as identities.
  For example `ps1 + pw1 + 1.` for some Pauli word `pw1 = PauliWord({0: "X", 1: "Y"})` and Pauli
  sentence `ps1 = PauliSentence({pw1: 3.})`.
  You can now subtract `PauliWord` and `PauliSentence` instances, as well as scalars, from each other. For example `ps1 - pw1 - 1`.
  Overall, you can now intuitively construct `PauliSentence` operators like `0.5 * pw1 - 1.5 * ps1 + 2`.
  [(#4989)](https://github.com/PennyLaneAI/pennylane/pull/4989)
  [(#5001)](https://github.com/PennyLaneAI/pennylane/pull/5001)
  [(#5003)](https://github.com/PennyLaneAI/pennylane/pull/5003)
  [(#5017)](https://github.com/PennyLaneAI/pennylane/pull/5017)

* Improve efficiency of matrix calculation when operator is symmetric over wires
   [(#3601)](https://github.com/PennyLaneAI/pennylane/pull/3601)

* PennyLane can now use lightning provided VJPs by selecting `device_vjp=True` on the QNode.
  [(#4914)](https://github.com/PennyLaneAI/pennylane/pull/4914)

* A new `pennylane.workflow` module is added. This module now contains `qnode.py`, `execution.py`, `set_shots.py`, `jacobian_products.py`, and the submodule `interfaces`.
  [(#5023)](https://github.com/PennyLaneAI/pennylane/pull/5023)

* Composite operations (eg. those made with `qml.prod` and `qml.sum`) and `SProd` operations convert `Hamiltonian` and
  `Tensor` operands to `Sum` and `Prod` types, respectively. This helps avoid the mixing of
  incompatible operator types.
  [(#5031)](https://github.com/PennyLaneAI/pennylane/pull/5031)
  [(#5063)](https://github.com/PennyLaneAI/pennylane/pull/5063)

* Raise a more informative error when calling `adjoint_jacobian` with trainable state-prep operations.
  [(#5026)](https://github.com/PennyLaneAI/pennylane/pull/5026)

<h4>Community contributions 🥳</h4>

* The transform `split_non_commuting` now accepts measurements of type `probs`, `sample` and `counts` which accept both wires and observables. 
  [(#4972)](https://github.com/PennyLaneAI/pennylane/pull/4972)

<h3>Breaking changes 💔</h3>

* `gradient_analysis_and_validation` is now renamed to `find_and_validate_gradient_methods`. Instead of returning a list, it now returns a dictionary of gradient methods for each parameter index, and no longer mutates the tape.
  [(#5035)](https://github.com/PennyLaneAI/pennylane/pull/5035)

* Passing additional arguments to a transform that decorates a QNode must be done through the use
  of `functools.partial`.
  [(#5046)](https://github.com/PennyLaneAI/pennylane/pull/5046)

* `Observable.return_type` has been removed. Instead, you should inspect the type
  of the surrounding measurement process.
  [(#5044)](https://github.com/PennyLaneAI/pennylane/pull/5044)

* `ClassicalShadow.entropy()` no longer needs an `atol` keyword as a better
  method to estimate entropies from approximate density matrix reconstructions
  (with potentially negative eigenvalues) has been implemented.
  [(#5048)](https://github.com/PennyLaneAI/pennylane/pull/5048)

<h3>Deprecations 👋</h3>

* Matrix and tensor products between `PauliWord` and `PauliSentence` instances are done using the `@` operator, `*` will be used only for scalar multiplication.
  [(#4989)](https://github.com/PennyLaneAI/pennylane/pull/4989)

* `MeasurementProcess.name` and `MeasurementProcess.data` are now deprecated, as they contain dummy
  values that are no longer needed.
  [(#5047)](https://github.com/PennyLaneAI/pennylane/pull/5047)
  [(#5071)](https://github.com/PennyLaneAI/pennylane/pull/5071)

* Calling `qml.matrix` without providing a `wire_order` on objects where the wire order could be
  ambiguous now raises a warning. In the future, the `wire_order` argument will be required in
  these cases.
  [(#5039)](https://github.com/PennyLaneAI/pennylane/pull/5039)

* `qml.pauli.pauli_mult` and `qml.pauli.pauli_mult_with_phase` are now deprecated. Instead, you
  should use `qml.simplify(qml.prod(pauli_1, pauli_2))` to get the reduced operator.
  [(#5057)](https://github.com/PennyLaneAI/pennylane/pull/5057)

* The private functions `_pauli_mult`, `_binary_matrix` and `_get_pauli_map` from the
  `pauli` module have been deprecated, as they are no longer used anywhere and the same
  functionality can be achieved using newer features in the `pauli` module.
  [(#5057)](https://github.com/PennyLaneAI/pennylane/pull/5057)

<h3>Documentation 📝</h3>

* The module documentation for `pennylane.tape` now explains the difference between `QuantumTape` and `QuantumScript`.
  [(#5065)](https://github.com/PennyLaneAI/pennylane/pull/5065)

* A typo in a code example in the `qml.transforms` API has been fixed.
  [(#5014)](https://github.com/PennyLaneAI/pennylane/pull/5014)

* Clarification for the definition of `argnum` added to gradient methods
  [(#5035)](https://github.com/PennyLaneAI/pennylane/pull/5035)

* A typo in the code example for `qml.qchem.dipole_of` has been fixed.
  [(#5036)](https://github.com/PennyLaneAI/pennylane/pull/5036) 

<h3>Bug fixes 🐛</h3>

* If `argnum` is provided to a gradient transform, only the parameters specified in `argnum` will have their gradient methods validated.
  [(#5035)](https://github.com/PennyLaneAI/pennylane/pull/5035)

* `StatePrep` operations expanded onto more wires are now compatible with backprop.
  [(#5028)](https://github.com/PennyLaneAI/pennylane/pull/5028)

* The return value of `Controlled.generator` now contains a projector that projects onto the correct subspace based on the control value specified.
  [(#5068)](https://github.com/PennyLaneAI/pennylane/pull/5068)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Abhishek Abhishek,
Astral Cai,
Pablo Antonio Moreno Casares,
Isaac De Vlugt,
Korbinian Kottmann,
Christina Lee,
Xiaoran Li,
Lee J. O'Riordan,
Mudit Pandey,
Matthew Silverman.
