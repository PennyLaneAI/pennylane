:orphan:

# Release 0.38.0-dev (development release)

<h3>New features since last release</h3>

* A new method `process_density_matrix` has been added to the `ProbabilityMP` and `DensityMatrixMP`
  classes, allowing for more efficient handling of quantum density matrices, particularly with batch
  processing support. This method simplifies the calculation of probabilities from quantum states
  represented as density matrices.
  [(#5830)](https://github.com/PennyLaneAI/pennylane/pull/5830)

* The `qml.PrepSelPrep` template is added. The template implements a block-encoding of a linear 
  combination of unitaries.
  [(#5756)](https://github.com/PennyLaneAI/pennylane/pull/5756)
  
<h3>Improvements 🛠</h3>

* `qml.UCCSD` now accepts an additional optional argument, `n_repeats`, which defines the number of
  times the UCCSD template is repeated. This can improve the accuracy of the template by reducing
  the Trotter error but would result in deeper circuits.
  [(#5801)](https://github.com/PennyLaneAI/pennylane/pull/5801)

* `QuantumScript.hash` is now cached, leading to performance improvements.
  [(#5919)](https://github.com/PennyLaneAI/pennylane/pull/5919)

* `qml.dynamic_one_shot` now supports circuits using the `"tensorflow"` interface.
  [(#5973)](https://github.com/PennyLaneAI/pennylane/pull/5973)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Yushao Chen,
Christina Lee,
William Maxwell,
Erik Schultheis.
