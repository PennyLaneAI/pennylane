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

* `AdaptiveOptimizer` is updated to use non-default user-defined qnode arguments.
  [(#3765)](https://github.com/PennyLaneAI/pennylane/pull/3765)

* When using Jax-jit with gradient transforms the trainable parameters are correctly set (instead of every parameter 
  to be set as trainable), and therefore the derivatives are computed more efficiently.
  [(#3697)](https://github.com/PennyLaneAI/pennylane/pull/3697)

<h3>Breaking changes</h3>

* Trainable parameters for the Jax interface are the parameters being `JVPTracer`, they are defined by setting
  `argnums`.
  [(#3697)](https://github.com/PennyLaneAI/pennylane/pull/3697)

* The keyword argument `argnums` is now used for gradient transform using Jax, instead of `argnum`.
  [(#3697)](https://github.com/PennyLaneAI/pennylane/pull/3697)

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Utkarsh Azad
Soran Jahangiri
Romain Moyard
Mudit Pandey
Matthew Silverman
Jay Soni
David Wierichs