:orphan:

# Release 0.21.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements</h3>

* The method `matrix` of `qml.operations.Tensor` now correctly computes
  the matrix of a multi-observable tensor if multiple observables act on
  the same wires but are not sorted within the `Tensor` according to the
  wires they act on. 
  [(#2010)](https://github.com/XanaduAI/pennylane/pull/2010)

  In addition, a `WireError` is raised whenever any pair of observables in the
  Tensor has partial but not full overlap of the wires.
  As an example, the matrix of 
  `qml.operation.Tensor(qml.PauliX(0), qml.Hermitian(A, [0, 1]))`, where `A`
  is some Hermitian `4x4` matrix, is not computed but raises a `WireError`.
  Previously, this computed an `8x8` matrix that does not match the number
  of wires of this `Tensor`.

<h3>Breaking changes</h3>

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

