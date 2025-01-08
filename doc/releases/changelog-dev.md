:orphan:

# Release 0.41.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Devices can now configure whether or not the data is converted to numpy and `jax.pure_callback`
  is used by the new `ExecutionConfig.convert_to_numpy` property. Finite shot executions
  on `default.qubit` can now be jitted end-to-end, even with parameter shift.
  [(#6788)](https://github.com/PennyLaneAI/pennylane/pull/6788)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

* Updated documentation for vibrational Hamiltonians
  [(#6717)](https://github.com/PennyLaneAI/pennylane/pull/6717)

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
Diksha Dhawan