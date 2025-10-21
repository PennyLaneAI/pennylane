:orphan:

# Release 0.44.0-dev (development release)

<h3>New features since last release</h3>

* A new decomposition has been added for the Controlled :class:`~.SemiAdder`,
  which is efficient and skips controlling all gates in its decomposition.
  [(#8423)](https://github.com/PennyLaneAI/pennylane/pull/8423)

<h3>Improvements 🛠</h3>

* `qml.grad` and `qml.jacobian` now lazily dispatch to catalyst and program
  capture, allowing for `qml.qjit(qml.grad(c))` and `qml.qjit(qml.jacobian(c))` to work.
  [(#8382)](https://github.com/PennyLaneAI/pennylane/pull/8382)

* Both the generic and transform-specific application behavior of a `qml.transforms.core.TransformDispatcher`
  can be overwritten with `TransformDispatcher.generic_register` and `my_transform.register`.
  [(#7797)](https://github.com/PennyLaneAI/pennylane/pull/7797)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

* The experimental xDSL implementation of `diagonalize_measurements` has been updated to fix a bug
  that included the wrong SSA value for final qubit insertion and deallocation at the end of the circuit. A clear error is not also raised when there are observables with overlapping wires.
  [(#8383)](https://github.com/PennyLaneAI/pennylane/pull/8383)

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso
Lillian Frederiksen,
Christina Lee,
