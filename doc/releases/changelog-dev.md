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

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

David Wierichs
