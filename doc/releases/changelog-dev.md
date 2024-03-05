:orphan:

# Release 0.36.0-dev (development release)

<h3>New features since last release</h3>

* The `QubitDevice` class and children classes support the `dynamic_one_shot` transform provided that they support `MidMeasureMP` operations natively.
  [(#5317)](https://github.com/PennyLaneAI/pennylane/pull/5317)

* Added new `SpectralNormError` class to the new error tracking functionality.
  [(#5154)](https://github.com/PennyLaneAI/pennylane/pull/5154)

* The `dynamic_one_shot` transform is introduced enabling dynamic circuit execution on circuits with shots and devices that support `MidMeasureMP` operations natively.
  [(#5266)](https://github.com/PennyLaneAI/pennylane/pull/5266)

<h3>Improvements 🛠</h3>

* The `molecular_hamiltonian` function calls `PySCF` directly when `method='pyscf'` is selected.
  [(#5118)](https://github.com/PennyLaneAI/pennylane/pull/5118)
  
* All generators in the source code (except those in the `qchem` module) no longer return 
  `Hamiltonian` or `Tensor` instances. Wherever possible, these return `Sum`, `SProd`, and `Prod` instances.
  [(#5253)](https://github.com/PennyLaneAI/pennylane/pull/5253)

* Upgraded `null.qubit` to the new device API. Also, added support for all measurements and various modes of differentiation.
  [(#5211)](https://github.com/PennyLaneAI/pennylane/pull/5211)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* We no longer perform unwanted dtype promotion in the `pauli_rep` of `SProd` instances when using tensorflow.
  [(#5246)](https://github.com/PennyLaneAI/pennylane/pull/5246)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Amintor Dusko
Pietropaolo Frisoni,
Soran Jahangiri,
Korbinian Kottmann,
Vincent Michaud-Rioux,
Matthew Silverman.
