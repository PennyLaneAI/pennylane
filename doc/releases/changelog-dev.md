:orphan:

# Release 0.39.0-dev (development release)

<h3>New features since last release</h3>
 
<h3>Improvements 🛠</h3>

<h4>Capturing and representing hybrid programs</h4>

* Differentiation of hybrid programs via `qml.grad` can now be captured into plxpr.
  When evaluating a captured `qml.grad` instruction, it will dispatch to `jax.grad`,
  which differs from the Autograd implementation of `qml.grad` itself.
  Pytree inputs and outputs are supported.
  [(#6120)](https://github.com/PennyLaneAI/pennylane/pull/6120)
  [(#6134)](https://github.com/PennyLaneAI/pennylane/pull/6134)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Fix Pytree serialization of operators with empty shot vectors:
  [(#6155)](https://github.com/PennyLaneAI/pennylane/pull/6155)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
Jack Brown,
David Wierichs,
