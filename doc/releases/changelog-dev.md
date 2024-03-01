:orphan:

# Release 0.36.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>
* Implemented the method `process_counts` in
  * `ExpectationMp` [(#5241)](https://github.com/PennyLaneAI/pennylane/issues/5241)
  * `VarianceMP` [(#5244)](https://github.com/PennyLaneAI/pennylane/issues/5244)
  * `CountsMP` [(#5249)](https://github.com/PennyLaneAI/pennylane/issues/5249)
  
<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* We no longer perform unwanted dtype promotion in the `pauli_rep` of `SProd` instances when using tensorflow.
  [(#5246)](https://github.com/PennyLaneAI/pennylane/pull/5246)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Korbinian Kottmann,
Tarun Kumar Allamsetty