:orphan:

# Release 0.33.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Wires can be provided to the new device API.
  [(#4538)](https://github.com/PennyLaneAI/pennylane/pull/4538)

* A `wires` property has been added to `QNode`.
  [(#4538)](https://github.com/PennyLaneAI/pennylane/pull/4538)

<h3>Breaking changes 💔</h3>

* The old return type and associated functions ``qml.enable_return`` and ``qml.disable_return`` are removed.
 [#4503](https://github.com/PennyLaneAI/pennylane/pull/4503)

* The ``mode`` keyword argument in ``QNode`` is removed. Please use ``grad_on_execution`` instead.
 [#4503](https://github.com/PennyLaneAI/pennylane/pull/4503)

* The method ``tape.unwrap()`` and corresponding ``UnwrapTape`` and ``Unwrap`` classes are removed.
  Instead of ``tape.unwrap()``, use :func:`~.transforms.convert_to_numpy_parameters`.
  [#4535](https://github.com/PennyLaneAI/pennylane/pull/4535)

  
<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Romain Moyard,
Matthew Silverman
