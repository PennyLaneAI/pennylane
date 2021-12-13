:orphan:

# Release 0.21.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements</h3>

* The method `matrix` of `qml.operations.Tensor` now raises a warning
  whenever the Tensor's observables have partially overlapping 
  wires or its output dimension differs from `2**N` where `N` is the number
  of wires of the Tensor.
  [(#2010)](https://github.com/XanaduAI/pennylane/pull/2010)

<h3>Breaking changes</h3>

<h3>Bug fixes</h3>

* Fixes a bug in queueing of the `two_qubit_decomposition` method that
  originally led to circuits with >3 two-qubit unitaries failing when passed
  through the `unitary_to_rot` optimization transform.
  [(#2015)](https://github.com/PennyLaneAI/pennylane/pull/2015)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Olivia Di Matteo, David Wierichs