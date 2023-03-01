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

* The custom JVP rules in PennyLane now also support non-scalar and mixed-shape tape parameters as
  well as multi-dimensional tape return types, like broadcasted `qml.probs`, for example.
  [(#3766)](https://github.com/PennyLaneAI/pennylane/pull/3766)

* the `qchem.jordan_wigner` function is extended to support more fermionic operator orders.
  [(#3754)](https://github.com/pennylaneai/pennylane/pull/3754)
  [(#3751)](https://github.com/pennylaneai/pennylane/pull/3751)

* `adaptiveoptimizer` is updated to use non-default user-defined qnode arguments.
  [(#3765)](https://github.com/pennylaneai/pennylane/pull/3765)

<h3>breaking changes</h3>

<h3>deprecations</h3>

<h3>documentation</h3>

<h3>bug fixes</h3>

<h3>contributors</h3>

this release contains contributions from (in alphabetical order):

utkarsh azad
soran jahangiri
mudit pandey
matthew silverman
jay soni
david wierichs
