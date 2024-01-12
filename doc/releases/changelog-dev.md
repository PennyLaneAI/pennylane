:orphan:

# Release 0.35.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Update `tests/ops/functions/conftest.py` to ensure all operator types are tested for validity.
  [(#4978)](https://github.com/PennyLaneAI/pennylane/pull/4978)

* Upgrade Pauli arithmetic with multiplying by scalars, e.g. `0.5 * PauliWord({0:"X"})` or `0.5 * PauliSentence({PauliWord({0:"X"}): 1.})`.
  [(#4989)](https://github.com/PennyLaneAI/pennylane/pull/4989)

* Upgrade Pauli arithmetic addition. You can now intuitively add together 
  `PauliWord` and `PauliSentence` as well as scalars, which are treated implicitly as identities.
  For example `ps1 + pw1 + 1.` for some Pauli word `pw1 = PauliWord({0: "X", 1: "Y"})` and Pauli
  sentence `ps1 = PauliSentence({pw1: 3.})`.
  [(#5001)](https://github.com/PennyLaneAI/pennylane/pull/5001)

* Upgrade Pauli arithmetic with subtraction. You can now subtract `PauliWord` and `PauliSentence`
  instances, as well as scalars, from each other.
  For example `ps1 - pw1 - 1` for `pw1 = PauliWord({0: "X", 1: "Y"})` and `ps1 = PauliSentence({pw1: 3.})`.
  [(#5003)](https://github.com/PennyLaneAI/pennylane/pull/5003)
  
* A new `pennylane.workflow` module is added. This module now contains `qnode.py`, `execution.py`, `set_shots.py`, `jacobian_products.py`, and the submodule `interfaces`.
  [(#5023)](https://github.com/PennyLaneAI/pennylane/pull/5023)

* Composite operations (eg. those made with `qml.prod` and `qml.sum`) convert `Hamiltonian` and
  `Tensor` operands to `Sum` and `Prod` types, respectively. This helps avoid the mixing of
  incompatible operator types.
  [(#5031)](https://github.com/PennyLaneAI/pennylane/pull/5031)

* Instead of returning a list, `gradient_analysis_and_validation` (renamed to `find_and_validate_gradient_methods`) now returns a dictionary of gradient methods for each parameter index, and no longer mutates the tape.
  [(#5035)](https://github.com/PennyLaneAI/pennylane/pull/5035)

* Raise a more informative error when calling `adjoint_jacobian` with trainable state-prep operations.
  [(#5026)](https://github.com/PennyLaneAI/pennylane/pull/5026)

<h4>Community contributions 🥳</h4>

* The transform `split_non_commuting` now accepts measurements of type `probs`, `sample` and `counts` which accept both wires and observables. 
  [(#4972)](https://github.com/PennyLaneAI/pennylane/pull/4972)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

* Matrix and tensor products between `PauliWord` and `PauliSentence` instances are done using the `@` operator, `*` will be used only for scalar multiplication.
  [(#4989)](https://github.com/PennyLaneAI/pennylane/pull/4989)

<h3>Documentation 📝</h3>

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


<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Abhishek Abhishek,
Astral Cai,
Pablo Antonio Moreno Casares,
Christina Lee,
Isaac De Vlugt,
Korbinian Kottmann,
Matthew Silverman.
