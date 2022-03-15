:orphan:

# Release 0.23.0-dev (development release)

<h3>New features since last release</h3>

* Development of a circuit-cutting compiler extension to circuits with sampling
  measurements has begun:

  - The existing `qcut.tape_to_graph()` method has been extended to convert a
    sample measurement without an observable specified to multiple single-qubit sample
    nodes.
    [(#2313)](https://github.com/PennyLaneAI/pennylane/pull/2313)

<h3>Improvements</h3>

* The parameter-shift Hessian can now be computed for arbitrary
  operations that support the general parameter-shift rule for
  gradients, using `qml.gradients.param_shift_hessian`
  [(#2319)](https://github.com/XanaduAI/pennylane/pull/2319)

  As for `qml.gradients.param_shift`, multiple ways to obtain the
  gradient recipe are supported, in the following order of preference:

  - A custom `grad_recipe`. It is iterated to obtain the shift rule for
    the second-order derivatives in the diagonal entries of the Hessian.

  - Custom `parameter_frequencies`. The second-order shift rule can
    directly be computed using them.

  - An operation's `generator`. Its eigenvalues will be used to obtain
    `parameter_frequencies`, if they are not given explicitly for an operation.

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

Karim Alaa El-Din, Anthony Hayes, David Wierichs
