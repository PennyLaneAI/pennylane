:orphan:

# Release 0.21.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Bug fixes</h3>

<h3>Documentation</h3>

<h3>Operator class refactor</h3>

The Operator class has undergone a major refactor with the following changes:

* The `diagonalizing_gates()` representation has been moved to the highest-level 
  `Operator` class and is therefore available to all subclasses. A condition 
  `qml.operation.defines_diagonalizing_gates` has been added, which can be used 
  in tape contexts without queueing.
  [(#1985)](https://github.com/PennyLaneAI/pennylane/pull/1985)

This release contains contributions from (in alphabetical order):

Maria Schuld