:orphan:

# Release 0.30.0-dev (development release)

<h3>New features since last release</h3>

* The `sample_state` function is added to `devices/qubit` that returns a series of samples based on a given
  state vector and a number of shots.
  [(#3720)](https://github.com/PennyLaneAI/pennylane/pull/3720)

* Added the needed functions and classes to simulate an ensemble of Rydberg atoms:
  * A new `RydbergHamiltonian` class is added, which contains the Hamiltonian of an ensemble of
    Rydberg atoms.
  * A new `rydberg_interaction` function is added, which returns a `RydbergHamiltonian` containing
    the Hamiltonian of the interaction of all the Rydberg atoms.
  * A new `rydberg_drive` function is added, which returns a `RydbergHamiltonian` containing
    the Hamiltonian of the interaction between a driving laser field and a group of atoms.
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)

* Added `Operation.__truediv__` dunder method to be able to divide operators.
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)

* The `simulate` function added to `devices/qubit` now supports measuring expectation values of large observables such as
  `qml.Hamiltonian`, `qml.SparseHamiltonian`, `qml.Sum`.
  [(#3759)](https://github.com/PennyLaneAI/pennylane/pull/3759)

<h3>Improvements</h3>

* The `qchem.jordan_wigner` function is extended to support more fermionic operator orders.
  [(#3754)](https://github.com/PennyLaneAI/pennylane/pull/3754)
  [(#3751)](https://github.com/PennyLaneAI/pennylane/pull/3751)

* `AdaptiveOptimizer` is updated to use non-default user-defined qnode arguments.
  [(#3765)](https://github.com/PennyLaneAI/pennylane/pull/3765)

* Use `TensorLike` type in `Operator` dunder methods.
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

* Fixed bug where the coefficients where not ordered correctly when summing a `ParametrizedHamiltonian`
  with other operators.
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Utkarsh Azad
Soran Jahangiri
Mudit Pandey
Matthew Silverman
Jay Soni
