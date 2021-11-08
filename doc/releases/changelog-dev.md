:orphan:

# Release 0.20.0-dev (development release)

<h3>New features since last release</h3>
* A thermal relaxation channel is added to the Noisy channels. The channel description can be 
  found on the supplementary information of [Quantum classifier with tailored quantum kernels](https://arxiv.org/abs/1909.02611).
  [(#1766)](https://github.com/PennyLaneAI/pennylane/pull/1766)

<h3>Improvements</h3>

* Update the PL test-suite to use the `tf.GradientTape` best-practices. 
This update reduces the total running time of the Python test-suite up to 12%.
[(#1869)](https://github.com/PennyLaneAI/pennylane/pull/1869)

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Bug fixes</h3>

* `qml.CSWAP` and `qml.CRot` now define `control_wires`, and `qml.SWAP` 
  returns the default empty wires object.
  [(#1830)](https://github.com/PennyLaneAI/pennylane/pull/1830)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ali Asadi, Christina Lee, Alejandro Montanez
