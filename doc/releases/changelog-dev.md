:orphan:

# Release 0.28.0-dev (development release)

<h3>New features since last release</h3>

* Added operation `qml.THadamard`, which is the qutrit Hadamard gate. The operation accepts a `subspace`
  keyword argument which determines which variant of the qutrit Hadamard to use.
  [#3340](https://github.com/PennyLaneAI/pennylane/pull/3340)

<h3>Improvements</h3>

* Added `validate_subspace` static method to `qml.Operator` to check the validity of the subspace of certain
  qutrit operations.
  [#3340](https://github.com/PennyLaneAI/pennylane/pull/3340)

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

* Small fix of `MeasurementProcess.map_wires`, where both the `self.obs` and `self._wires`
  attributes were modified.
  [#3292](https://github.com/PennyLaneAI/pennylane/pull/3292)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Mudit Pandey
