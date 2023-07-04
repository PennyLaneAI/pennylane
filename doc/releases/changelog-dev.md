:orphan:

# Release 0.32.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* The `qml.gradients` module no longer mutates operators in-place for any gradient transforms.
  Instead, operators that need to be mutated are copied with new parameters.
  [(#4220)](https://github.com/PennyLaneAI/pennylane/pull/4220)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Raise a warning if control indicators are hidden when calling `qml.draw_mpl`
  [(#4295)](https://github.com/PennyLaneAI/pennylane/pull/4295)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Mudit Pandey
Matthew Silverman
