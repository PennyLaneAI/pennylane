:orphan:

# Release 0.39.0-dev (development release)

<h3>New features since last release</h3>

* Functions are added for generating spin Hamiltonians for [Emery](https://arxiv.org/pdf/2309.11786)
  and [Haldane](https://arxiv.org/pdf/2211.13615) models on a lattice.
  [(#6201)](https://github.com/PennyLaneAI/pennylane/pull/6201/)

<h3>Improvements 🛠</h3>

* Improve unit testing for capturing of nested control flows.
  [(#6111)](https://github.com/PennyLaneAI/pennylane/pull/6111)

* Some custom primitives for the capture project can now be imported via
  `from pennylane.capture.primitives import *`.
  [(#6129)](https://github.com/PennyLaneAI/pennylane/pull/6129)

* Improve `qml.Qubitization` decomposition.
  [(#6182)](https://github.com/PennyLaneAI/pennylane/pull/6182)

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

* Now `qml.equal` compares correctly `qml.PrepSelPrep` operators.
  [(#6182)](https://github.com/PennyLaneAI/pennylane/pull/6182)

* The ``qml.QSVT`` template now orders the ``projector`` wires first and the ``UA`` wires second, which is the expected order of the decomposition.
  [(#6212)](https://github.com/PennyLaneAI/pennylane/pull/6212)

* <h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso
Utkarsh Azad
Jack Brown
Diksha Dhawan
Christina Lee
William Maxwell
