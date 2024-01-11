:orphan:

# Release 0.35.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Update `tests/ops/functions/conftest.py` to ensure all operator types are tested for validity.
  [(#4978)](https://github.com/PennyLaneAI/pennylane/pull/4978)

* A new `pennylane.workflow` module is added. This module now contains `qnode.py`, `execution.py`, `set_shots.py`, `jacobian_products.py`, and the submodule `interfaces`.

* Composite operations (eg. those made with `qml.prod` and `qml.sum`) convert `Hamiltonian` and
  `Tensor` operands to `Sum` and `Prod` types, respectively. This helps avoid the mixing of
  incompatible operator types.
  [(#5031)](https://github.com/PennyLaneAI/pennylane/pull/5031)

<h4>Community contributions 🥳</h4>

* The transform `split_non_commuting` now accepts measurements of type `probs`, `sample` and `counts` which accept both wires and observables. 
  [(#4972)](https://github.com/PennyLaneAI/pennylane/pull/4972)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

* A typo in a code example in the `qml.transforms` API has been fixed.
  [(#5014)](https://github.com/PennyLaneAI/pennylane/pull/5014)

* A typo in the code example for `qml.qchem.dipole_of` has been fixed.
  [(#5036)](https://github.com/PennyLaneAI/pennylane/pull/5036) 

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Abhishek Abhishek,
Pablo Antonio Moreno Casares,
Christina Lee,
Isaac De Vlugt,
Matthew Silverman.
