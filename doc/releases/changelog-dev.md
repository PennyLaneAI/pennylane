:orphan:

# Release 0.39.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Improve unit testing for capturing of nested control flows.
  [(#6111)](https://github.com/PennyLaneAI/pennylane/pull/6111)

* Some custom primitives for the capture project can now be imported via
  `from pennylane.capture.primitives import *`.
  [(#6129)](https://github.com/PennyLaneAI/pennylane/pull/6129)

* `FermiWord` and `FermiSentence` classes now have methods to compute adjoints.
  [(#6166)](https://github.com/PennyLaneAI/pennylane/pull/6166)

* The `__repr__` methods for `FermiWord` and `FermiSentence` now returns a
  unique representation of the object.
  [(#6167)](https://github.com/PennyLaneAI/pennylane/pull/6167)


<h3>Breaking changes 💔</h3>

* `qml.transforms.hamiltonian_expand` and `qml.transforms.sum_expand` are removed.
  Please use `qml.transforms.split_non_commuting` instead.

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Fix `qml.PrepSelPrep` template to work with `torch`:
  [(#6191)](https://github.com/PennyLaneAI/pennylane/pull/6191)

* The ``qml.QSVT`` template now orders the ``projector`` wires first and the ``UA`` wires second, which is the expected order of the decomposition.
  [(#6212)](https://github.com/PennyLaneAI/pennylane/pull/6212)

* <h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso
Utkarsh Azad
Christina Lee
William Maxwell
