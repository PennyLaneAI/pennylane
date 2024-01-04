:orphan:

# Release 0.35.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Update `tests/ops/functions/conftest.py` to ensure all operator types are tested for validity.
  [(#4978)](https://github.com/PennyLaneAI/pennylane/pull/4978)
  
<h4>Community contributions 🥳</h4>

* The transform ``split_non_commuting`` now accepts measurements of type `probs`, `sample` and `counts` which accept both wires and observables. [(#4972)](https://github.com/PennyLaneAI/pennylane/pull/4972)

* The `ControlledSequence.compute_decomposition` default now decomposes the `Power` operators, 
  improving compatability with machine learning interfaces. 
  [(#4955)](https://github.com/PennyLaneAI/pennylane/pull/4955)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Abhishek Abhishek,
Matthew Silverman.