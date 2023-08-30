:orphan:

# Release 0.33.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Tensor-network template `qml.MPS` now supports changing `offset` between subsequent blocks for more flexibility.
 [#4531](https://github.com/PennyLaneAI/pennylane/pull/4531)

<h3>Breaking changes 💔</h3>

* The old return type and associated functions ``qml.enable_return`` and ``qml.disable_return`` are removed.
  [#4503](https://github.com/PennyLaneAI/pennylane/pull/4503)

* The ``mode`` keyword argument in ``QNode`` is removed. Please use ``grad_on_execution`` instead.
  [#4503](https://github.com/PennyLaneAI/pennylane/pull/4503)

* The CV observables ``qml.X`` and ``qml.P`` are removed. Please use ``qml.QuadX`` and ``qml.QuadP`` instead.
  [#4533](https://github.com/PennyLaneAI/pennylane/pull/4533)

* The method ``tape.unwrap()`` and corresponding ``UnwrapTape`` and ``Unwrap`` classes are removed.
  Instead of ``tape.unwrap()``, use :func:`~.transforms.convert_to_numpy_parameters`.
  [#4535](https://github.com/PennyLaneAI/pennylane/pull/4535)

  
<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Utkarsh Azad
Romain Moyard