:orphan:

# Release 0.28.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

* Added documentation on parameter broadcasting regarding both its usage and technical aspects
  [#34xx](https://github.com/PennyLaneAI/pennylane/pull/34xx)

  The [quickstart guide on circuits](https://docs.pennylane.ai/en/stable/introduction/circuits.html#parameter-broadcasting-in-qnodes)
  as well as the the documentation of
  [QNodes](https://docs.pennylane.ai/en/stable/code/api/pennylane.QNode.html) and
  [Operators](https://docs.pennylane.ai/en/stable/code/api/pennylane.operation.Operator.html)
  now contain introductions and details on parameter broadcasting. The QNode documentation
  mostly contains usage details, the Operator documentation is concerned with implementation
  details and a guide to support broadcasting in custom operators.

<h3>Bug fixes</h3>

* Small fix of `MeasurementProcess.map_wires`, where both the `self.obs` and `self._wires`
  attributes were modified.
  [#3292](https://github.com/PennyLaneAI/pennylane/pull/3292)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

David Wierichs
