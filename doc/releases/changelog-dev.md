:orphan:

# Release 0.20.0-dev (development release)

<h3>New features since last release</h3>
* The `qml.fourier.reconstruct` function is added. It can be used to
  reconstruct one-dimensional Fourier series with a minimal number of calls
  to the original function.
  [(#1864)](https://github.com/PennyLaneAI/pennylane/pull/1864)

  The used reconstruction technique differs for functions with equidistant frequencies
  that are reconstructed using the function value at equidistant sampling points and
  for functions with arbitrary frequencies reconstructed using arbitrary sampling points.

* A thermal relaxation channel is added to the Noisy channels. The channel description can be 
  found on the supplementary information of [Quantum classifier with tailored quantum kernels](https://arxiv.org/abs/1909.02611).
  [(#1766)](https://github.com/PennyLaneAI/pennylane/pull/1766)

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Bug fixes</h3>

* `qml.CSWAP` and `qml.CRot` now define `control_wires`, and `qml.SWAP` 
  returns the default empty wires object.
  [(#1830)](https://github.com/PennyLaneAI/pennylane/pull/1830)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Christina Lee, Alejandro Montanez, David Wierichs
