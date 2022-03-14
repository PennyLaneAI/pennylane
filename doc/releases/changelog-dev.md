:orphan:

# Release 0.23.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements</h3>

* The parameter-shift Hessian can now be computed for arbitrary
  operations that support the general parameter-shift rule for
  gradients, using `qml.gradients.param_shift_hessian`
  [(#23XX)](https://github.com/XanaduAI/pennylane/pull/23XX)

  As for `qml.gradients.param_shift`, multiple ways to obtain the
  gradient recipe are supported, in the following order of preference:

  - A custom `grad_recipe`. It is iterated to obtain the shift rule for
    the second-order derivatives in the diagonal entries of the Hessian.

  - Custom `parameter_frequencies`. The second-order shift rule can
    directly be computed using them.

  - An operation's `generator`. Its eigenvalues will be used to obtain
    `parameter_frequencies`, if they are not given explicitly for an operation.


<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

David Wierichs
