:orphan:

# Release 0.38.0-dev (development release)

<h3>New features since last release</h3>

* A new method `process_density_matrix` has been added to the `ProbabilityMP` and `DensityMatrixMP` classes, allowing for more efficient handling of quantum density matrices, particularly with batch processing support. This method simplifies the calculation of probabilities from quantum states represented as density matrices.
  [(#5830)](https://github.com/PennyLaneAI/pennylane/pull/5830)

* The `qml.PrepSelPrep` template is added. The template implements a block-encoding of a linear 
  combination of unitaries.
  [(#5756)](https://github.com/PennyLaneAI/pennylane/pull/5756)
  
<h3>Improvements 🛠</h3>

* The `tree-traversal` algorithm implemented in `default.qubit` is refactored
  into an iterative instead of recursive implementation, doing away with
  potential stack overflow for deep circuits. `tree-traversal` is compatible with
  analytic-mode execution (`shots=None`).
  [(#5868)](https://github.com/PennyLaneAI/pennylane/pull/5868)

* Port the fast `apply_operation` implementation of `PauliZ` to `PhaseShift`, `S` and `T`.
  [(#5876)](https://github.com/PennyLaneAI/pennylane/pull/5876)

* `qml.UCCSD` now accepts an additional optional argument, `n_repeats`, which defines the number of
  times the UCCSD template is repeated. This can improve the accuracy of the template by reducing
  the Trotter error but would result in deeper circuits.
  [(#5801)](https://github.com/PennyLaneAI/pennylane/pull/5801)

* `QuantumScript.hash` is now cached, leading to performance improvements.
  [(#5919)](https://github.com/PennyLaneAI/pennylane/pull/5919)

* Observable validation for `default.qubit` is now based on execution mode (analytic vs. finite shots) and measurement type (sample measurement vs. state measurement).
  [(#5890)](https://github.com/PennyLaneAI/pennylane/pull/5890)

<h3>Breaking changes 💔</h3>

* ``qml.transforms.map_batch_transform`` has been removed, since transforms can be applied directly to a batch of tapes.
  See :func:`~.pennylane.transform` for more information.
  [(#5981)](https://github.com/PennyLaneAI/pennylane/pull/5981)

* `QuantumScript.interface` has been removed.
  [(#5980)](https://github.com/PennyLaneAI/pennylane/pull/5980)

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

* Improves the docstring for `QuantumScript.expand` and `qml.tape.tape.expand_tape`.
  [(#5974)](https://github.com/PennyLaneAI/pennylane/pull/5974)

<h3>Bug fixes 🐛</h3>

* `qml.AmplitudeEmbedding` has better support for features using low precision integer data types.
[(#5969)](https://github.com/PennyLaneAI/pennylane/pull/5969)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Ahmed Darwish
Astral Cai,
Yushao Chen,
Christina Lee,
William Maxwell,
Vincent Michaud-Rioux,
Mudit Pandey,
Erik Schultheis.
