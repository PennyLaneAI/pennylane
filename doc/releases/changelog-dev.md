:orphan:

# Release 0.23.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements</h3>

* The function `qml.ctrl` was given the optional argument `control_values=None`.
  If overridden, `control_values` takes an integer or a list of integers corresponding to
  the binary value that each control value should take. The same change is reflected in
  `ControlledOperation`. Control values of `0` are implemented by `qml.PauliX` applied
  before and after the controlled operation
  [(#2288)](https://github.com/PennyLaneAI/pennylane/pull/2288)
  
<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):
Karim Alaa El-Din