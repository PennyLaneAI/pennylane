:orphan:

# Release 0.38.0-dev (development release)

<h3>New features since last release</h3>
* A new method `process_density_matrix` has been added to the `ProbabilityMP` and `DensityMatrixMP` classes, allowing for more efficient handling of quantum density matrices, particularly with batch processing support. This method simplifies the calculation of probabilities from quantum states represented as density matrices.
  [(#5830)](https://github.com/PennyLaneAI/pennylane/pull/5830)
  
<h3>Improvements 🛠</h3>

* `QuantumScript.hash` is now cached, leading to performance improvements.
  [(#5919)](https://github.com/PennyLaneAI/pennylane/pull/5919)

* `qml.DefaultQutritMixed` readout error has been added using parameters readout_relaxation_probs and 
  readout_misclassification_probs. These parameters add a `qml.QutritAmplitudeDamping`  and a `qml.TritFlip` channel, respectively,
  after measurement diagonalization. The amplitude damping error represents the potential for
  relaxation errors to occur during a potentially longer measurement. The trit flip error represents misclassification events.
  [(#5842)](https://github.com/PennyLaneAI/pennylane/pull/5842)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Yushao Chen,
Christina Lee,