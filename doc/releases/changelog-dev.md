:orphan:

# Release 0.22.0-dev (development release)

<h3>New features since last release</h3>

* The general parameter-shift rule is now automatically used by
  `qml.gradients.param_shift`.
  [(#2182)](https://github.com/PennyLaneAI/pennylane/pull/2182)

  If an operation has `parameter_frequencies` defined (also see
  the next feature entry) and calling
  this method does not raise an error, the operation will be
  considered to have an analytic differentiation method registered,
  namely via the parameter-shift rule.
  The frequencies are then used to obtain the shift rule for the
  operation.

  If `parameter_frequencies` raises an error although the operation
  is registered to have an analytic derivative, the `grad_recipe`
  property of an operation will be used instead, if defined.
  If it is not defined, the standard two-term shift rule will
  be used.

  See [Vidal and Theis (2018)](https://arxiv.org/abs/1812.06323)
  and [Wierichs et al. (2021)](https://arxiv.org/abs/2107.12390)
  for additional information.

* Parametric operations now have the `parameter_frequencies`
  method that returns the frequencies with which a parameter
  enters a circuit when using the operation.
  Also see the previous feature.
  [(#2180)](https://github.com/PennyLaneAI/pennylane/pull/2180)

  The frequencies can be used for circuit analysis, optimization
  via the `RotosolveOptimizer` and differentiation with the
  parameter-shift rule. They assume that the circuit returns
  expectation values or probabilities, for a variance
  measurement the frequencies will differ.

  By default, the frequencies will be obtained from the
  `generator` property, if it is defined.

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

* The ``pennylane.numpy`` subpackage is now included in the PennyLane
  API documentation.
  [(#2179)](https://github.com/PennyLaneAI/pennylane/pull/2179)

* Improves the documentation of `RotosolveOptimizer` regarding the
  usage of the passed `substep_optimizer` and its keyword arguments.
  [(#2160)](https://github.com/PennyLaneAI/pennylane/pull/2160)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Anthony Hayes, Josh Izaac, David Wierichs
