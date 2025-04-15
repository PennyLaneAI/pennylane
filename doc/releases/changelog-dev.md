:orphan:

# Release 0.42.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

* The `Tracker` class has been moved into the `devices` module.
  [(#7281)](https://github.com/PennyLaneAI/pennylane/pull/7281)

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Fixes a bug when special input shapes were used for ``StatePrep`` in ``default.mixed``.
  [(#7280)](https://github.com/PennyLaneAI/pennylane/pull/7280)

* Fixes a bug where the global phase was not being added in the ``QubitUnitary`` decomposition.  
  [(#7244)](https://github.com/PennyLaneAI/pennylane/pull/7244)
  [(#7270)](https://github.com/PennyLaneAI/pennylane/pull/7270)

* Using finite differences with program capture without x64 mode enabled now raises a warning.
  [(#7282)](https://github.com/PennyLaneAI/pennylane/pull/7282)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso-Linaje,
Yushao Chen,
Christina Lee,
