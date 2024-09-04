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

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Fix Pytree serialization of operators with empty shot vectors:
  [(#6155)](https://github.com/PennyLaneAI/pennylane/pull/6155)

* Fix `qml.PrepSelPrep` template to work with `torch`:
  [(#6191)](https://github.com/PennyLaneAI/pennylane/pull/6191)

* The wires of the ``qml.QSVT`` template take first the ``projector`` wires and then the ``UA`` wires following the expected order of the decomposition.
  [(#6212)](https://github.com/PennyLaneAI/pennylane/pull/6212)

* <h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso
Utkarsh Azad
Jack Brown
Christina Lee
William Maxwell
