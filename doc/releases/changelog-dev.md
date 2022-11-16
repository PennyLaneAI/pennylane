:orphan:

# Release 0.28.0-dev (development release)

<h3>New features since last release</h3>

* Added the `qml.TRX` qutrit operation, which applies an X rotation to a specified subspace.
  ([#2845](https://github.com/PennyLaneAI/pennylane/pull/2845))

<h3>Improvements</h3>

* Updated `pennylane/qnode.py` to support parameter-shift differentiation on qutrit devices.
  ([#2845](https://github.com/PennyLaneAI/pennylane/pull/2845))

* Updated `pennylane/utils.py:sparse_hamiltonian` to include a `level` keyword argument to 
  support Hamiltonians for systems with an arbitrary number of levels per wire (qutrits, etc).
  ([#2845](https://github.com/PennyLaneAI/pennylane/pull/2845))

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

* Small fix of `MeasurementProcess.map_wires`, where both the `self.obs` and `self._wires`
  attributes were modified.
  [#3292](https://github.com/PennyLaneAI/pennylane/pull/3292)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):
