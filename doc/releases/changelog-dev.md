:orphan:

# Release 0.32.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Change the `sampler_seed` argument of `qml.gradients.spsa_grad` to `sampler_rng`. One can either provide
  an integer, which will be used to create a PRNG internally. Previously, this lead to the same direction
  being sampled, when `num_directions` is greater than 1. Alternatively, one can provide a NumPy PRNG,
  which allows reproducibly calling `spsa_grad` without getting the same results every time.
  [(4107)](https://github.com/PennyLaneAI/pennylane/issues/4107)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Frederik Wilde.
