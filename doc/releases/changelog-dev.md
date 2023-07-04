:orphan:

# Release 0.32.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

* `qml.enable_return` and `qml.disable_return` are deprecated. Please ensure that you are using
  the new return system, as the old return system is deprecated along with these switch functions.
  [(#4316)](https://github.com/PennyLaneAI/pennylane/pull/4316)

* The ``mode`` keyword argument in ``QNode.__init__`` is deprecated, as it was only used in the
  old return system (which is also deprecated).
  [(#4316)](https://github.com/PennyLaneAI/pennylane/pull/4316)

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Raise a warning if control indicators are hidden when calling `qml.draw_mpl`
  [(#4295)](https://github.com/PennyLaneAI/pennylane/pull/4295)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Matthew Silverman
