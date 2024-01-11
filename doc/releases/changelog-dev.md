:orphan:

# Release 0.35.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Update `tests/ops/functions/conftest.py` to ensure all operator types are tested for validity.
  [(#4978)](https://github.com/PennyLaneAI/pennylane/pull/4978)

* Upgrade Pauli arithmetic with multiplying by scalars `0.5 * PauliWord({0:"X"})`
  [(#4989)](https://github.com/PennyLaneAI/pennylane/pull/4989)

* Upgrade Pauli arithmetic addition `pw1 + pw2 + 1`. You can now intuitively add together 
  ``PauliWord`` and ``PauliSentence`` as well as scalars, which are treated implicitly as identities.
  [(#5001)](https://github.com/PennyLaneAI/pennylane/pull/5001)

* Upgrade Pauli arithmetic with subtraction. You can now subtract `PauliWord` and `PauliSentence`
  instances, as well as scalars, from each other.
  For example `ps1 - pw1 - 1` for `pw1 = PauliWord({0: "X"})` and `ps1 = PauliSentence({pw1: 3.})`.
  [(#5003)](https://github.com/PennyLaneAI/pennylane/pull/5003)
  
<h4>Community contributions 🥳</h4>

* The transform `split_non_commuting` now accepts measurements of type `probs`, `sample` and `counts` which accept both wires and observables. 
  [(#4972)](https://github.com/PennyLaneAI/pennylane/pull/4972)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

* A typo in a code example in the `qml.transforms` API has been fixed.
  [(#5014)](https://github.com/PennyLaneAI/pennylane/pull/5014)

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Abhishek Abhishek,
Isaac De Vlugt,
Korbinian Kottmann,
Matthew Silverman.
