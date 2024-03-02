:orphan:

# Release 0.36.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* We no longer perform unwanted dtype promotion in the `pauli_rep` of `SProd` instances when using tensorflow.
  [(#5246)](https://github.com/PennyLaneAI/pennylane/pull/5246)

* JAX JIT compatibility for `BasisStateProjector` in PennyLane have been enhanced, ensuring it 
  aligns with the non-JIT behavior. Resolved array conversion issues during JIT compilation.
  [(#5102)](https://github.com/PennyLaneAI/pennylane/pull/5102)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Korbinian Kottmann,
Anurav Modak