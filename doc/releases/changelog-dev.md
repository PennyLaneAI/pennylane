:orphan:

# Release 0.20.0-dev (development release)

<h3>New features since last release</h3>

* The `metric_tensor` transform can now be used to compute the full
  tensor, beyond the block diagonal approximation. 
  [(#1725)](https://github.com/PennyLaneAI/pennylane/pull/1725)

  This is performed using Hadamard tests, and requires an additional wire 
  on the device to execute the circuits produced by the transform, 
  as compared to the number of wires required by the original circuit.
  The transform defaults to computing the full tensor, which can
  be controlled by the `approx` keyword argument.
  See the 
  [qml.metric_tensor docstring](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.transforms.metric_tensor.html).
  for more information and usage details.

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
