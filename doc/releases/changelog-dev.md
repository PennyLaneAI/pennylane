:orphan:

# Release 0.21.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Bug fixes</h3>

* Fixes a bug in `classical_jacobian` when used with Torch, where the
  Jacobian of the preprocessing was also computed for non-trainable
  parameters.
  [(#2020)](https://github.com/PennyLaneAI/pennylane/pull/2020)

* Fixes a bug in queueing of the `two_qubit_decomposition` method that
  originally led to circuits with >3 two-qubit unitaries failing when passed
  through the `unitary_to_rot` optimization transform.
  [(#2015)](https://github.com/PennyLaneAI/pennylane/pull/2015)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Olivia Di Matteo, David Wierichs
