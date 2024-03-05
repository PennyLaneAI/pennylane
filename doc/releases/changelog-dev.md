:orphan:

# Release 0.36.0-dev (development release)

<h3>New features since last release</h3>

* Support for entanglement entropy computation is added. `qml.math.vn_entanglement_entropy` computes the von Neumann entanglement entropy from a density matrix, and a QNode transform `qml.qinfo.vn_entanglement_entropy` is also added.
  [(#5306)](https://github.com/PennyLaneAI/pennylane/pull/5306)

<h3>Improvements 🛠</h3>

* The `molecular_hamiltonian` function calls `PySCF` directly when `method='pyscf'` is selected.
  [(#5118)](https://github.com/PennyLaneAI/pennylane/pull/5118)

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

Astral Cai
Soran Jahangiri,
Korbinian Kottmann,
Matthew Silverman.
