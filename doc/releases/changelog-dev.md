:orphan:

# Release 0.30.0-dev (development release)

<h3>New features since last release</h3>

* The `sample_state` function is added to `devices/qubit` that returns a series of samples based on a given
  state vector and a number of shots.
  [(#3720)](https://github.com/PennyLaneAI/pennylane/pull/3720)

* The `simulate` function added to `devices/qubit` now supports measuring expectation values of large observables such as
  `qml.Hamiltonian`, `qml.SparseHamiltonian`, `qml.Sum`.
  [(#3759)](https://github.com/PennyLaneAI/pennylane/pull/3759)

<h3>Improvements</h3>

* The `qchem.jordan_wigner` function is extended to support more fermionic operator orders.
  [(#3754)](https://github.com/PennyLaneAI/pennylane/pull/3754)
  [(#3751)](https://github.com/PennyLaneAI/pennylane/pull/3751)

* `AdaptiveOptimizer` is updated to use non-default user-defined qnode arguments.
  [(#3765)](https://github.com/PennyLaneAI/pennylane/pull/3765)

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

* A typo was corrected in the documentation for introduction to `inspecting_circuits` and `chemistry`.
[(#3844)](https://github.com/PennyLaneAI/pennylane/pull/3844)

<h3>Bug fixes</h3>

* Fixed a bug where calling `Evolution.generator` with `coeff` being a complex ArrayBox raised an error.
  [(#3796)](https://github.com/PennyLaneAI/pennylane/pull/3796)
* 
<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Utkarsh Azad
Soran Jahangiri
Vincent Michaud-Rioux
Mudit Pandey
Matthew Silverman
Jay Soni
