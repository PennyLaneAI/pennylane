:orphan:

# Release 0.41.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* The higher order primitives in program capture can now accept inputs with abstract shapes.
  [(#6786)](https://github.com/PennyLaneAI/pennylane/pull/6786)

* The `PlxprInterpreter` classes can now handle creating dynamic arrays via `jnp.ones`, `jnp.zeros`,
  `jnp.arange`, and `jnp.full`.
  [#6865)](https://github.com/PennyLaneAI/pennylane/pull/6865)

* The coefficients of observables now have improved differentiability.
  [(#6598)](https://github.com/PennyLaneAI/pennylane/pull/6598)

<h3>Breaking changes 💔</h3>

* The `output_dim` property of `qml.tape.QuantumScript` has been removed. Instead, use method `shape` of `QuantumScript` or `MeasurementProcess` to get the same information.
  [(#6829)](https://github.com/PennyLaneAI/pennylane/pull/6829)

* Removed method `qsvt_legacy` along with its private helper `_qsp_to_qsvt`
  [(#6827)](https://github.com/PennyLaneAI/pennylane/pull/6827)

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

* Updated documentation for vibrational Hamiltonians
  [(#6717)](https://github.com/PennyLaneAI/pennylane/pull/6717)

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Yushao Chen,
Diksha Dhawan,
Christina Lee,
