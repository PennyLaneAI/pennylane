:orphan:

# Release 0.21.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Documentation</h3>

<h3>Contributors</h3>

<h3>Operator class refactor</h3>

The Operator class has undergone a major refactor with the following changes:

* The `matrix` representation has been modified to be a method that accepts a 
  `wire_order` argument and calculate the correct numerical representation 
  with respect to that ordering. The "base matrix", which uses the operator's wire order, 
  is defined in `compute_matrix()`.
  [(#1996)](https://github.com/PennyLaneAI/pennylane/pull/1996)

This release contains contributions from (in alphabetical order):

Maria Schuld