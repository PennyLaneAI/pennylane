:orphan:

# Release 0.39.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Improve unit testing for capturing of nested control flows.
  [(#6111)](https://github.com/PennyLaneAI/pennylane/pull/6111)

* Some custom primitives for the capture project can now be imported via
  `from pennylane.capture.primitives import *`.
  [(#6129)](https://github.com/PennyLaneAI/pennylane/pull/6129)

<h3>Breaking changes 💔</h3>

<<<<<<< HEAD
* `DefaultQubitTF`, `DefaultQubitTorch`, `DefaultQubitJax`, and `DefaultQubitAutograd` are removed.
  Please use `default.qubit` for all interfaces.
=======
* `DefaultQubitTF`, `DefaultQubitTorch`, and `DefaultQubitJax` are removed. Please use `default.qubit` for all interfaces.
[(#6209)](https://github.com/PennyLaneAI/pennylane/pull/6209)
>>>>>>> remove-default-qubit-jax

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Fix Pytree serialization of operators with empty shot vectors:
  [(#6155)](https://github.com/PennyLaneAI/pennylane/pull/6155)

* Fix `qml.PrepSelPrep` template to work with `torch`:
  [(#6191)](https://github.com/PennyLaneAI/pennylane/pull/6191)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Utkarsh Azad
Jack Brown
Christina Lee
William Maxwell
