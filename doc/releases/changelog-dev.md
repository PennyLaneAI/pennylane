:orphan:

# Release 0.35.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Update `tests/ops/functions/conftest.py` to ensure all operator types are tested for validity.
  [(#4978)](https://github.com/PennyLaneAI/pennylane/pull/4978)

* Upgrade Pauli arithmetic with multiplying by scalars, addition and subtraction. You can now construct operators like `0.5 * w1 - 1.5 * w2 + 2`.
  [(#4989)](https://github.com/PennyLaneAI/pennylane/pull/4989)
  [(#5001)](https://github.com/PennyLaneAI/pennylane/pull/5001)
  [(#5003)](https://github.com/PennyLaneAI/pennylane/pull/5003)
  [(#5017)](https://github.com/PennyLaneAI/pennylane/pull/5017)
  
<h4>Community contributions 🥳</h4>

* The transform ``split_non_commuting`` now accepts measurements of type `probs`, `sample` and `counts` which accept both wires and observables. [(#4972)](https://github.com/PennyLaneAI/pennylane/pull/4972)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Abhishek Abhishek,
Korbinian Kottmann,
Matthew Silverman.