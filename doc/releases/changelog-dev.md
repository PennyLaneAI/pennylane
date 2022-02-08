:orphan:

# Release 0.22.0-dev (development release)

<h3>New features since last release</h3>

* Parametric operations now have the `parameter_frequencies`
  method that returns the frequencies with which a parameter
  enters a circuit when using the operation.
  [(#2180)](https://github.com/PennyLaneAI/pennylane/pull/2180)

  The frequencies can be used for circuit analysis, optimization
  via the `RotosolveOptimizer` and differentiation with the
  parameter-shift rule. They assume that the circuit returns
  expectation values or probabilities, for a variance
  measurement the frequencies will differ.

* Continued development of the circuit-cutting compiler:
  
  A method for converting a quantum tape to a directed multigraph that is amenable
  to graph partitioning algorithms for circuit cutting has been added.
  [(#2107)](https://github.com/PennyLaneAI/pennylane/pull/2107)
  
  A method to replace `WireCut` nodes in a directed multigraph with `MeasureNode` 
  and `PrepareNode` placeholders has been added.
  [(#2124)](https://github.com/PennyLaneAI/pennylane/pull/2124)
  
  A method has been added that takes a directed multigraph with `MeasureNode` and
  `PrepareNode` placeholders and fragments into subgraphs and a communication graph.
  [(#2153)](https://github.com/PennyLaneAI/pennylane/pull/2153)

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Bug fixes</h3>

* The operation `OrbitalRotation` previously was wrongfully registered to satisfy
  the four-term parameter shift rule, it now will be decomposed instead when
  using the parameter-shift rule.
  [(#2180)](https://github.com/PennyLaneAI/pennylane/pull/2180)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Anthony Hayes
