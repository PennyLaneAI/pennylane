:orphan:

# Release 0.28.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements</h3>

* The `coefficients` function and the `visualize` submodule of the `qml.fourier` module
  now allow assigning different degrees for different parameters of the input function.
  [(#3005)](https://github.com/PennyLaneAI/pennylane/pull/3005)

  The arguments `degree` and `filter_threshold` to `qml.fourier.coefficients` previously were
  expected to be integers, and now can be a sequences of integers with one integer per function
  parameter (i.e. `len(degree)==n_inputs`), resulting in a returned array with shape
  `(2*degrees[0]+1,..., 2*degrees[-1]+1)`.
  The functions in `qml.fourier.visualize` accordingly accept such arrays of coefficients.

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

David Wierichs
