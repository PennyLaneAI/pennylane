:orphan:

# Release 0.33.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

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

* The ``prep`` keyword argument in ``QuantumScript`` is deprecated and will be removed from `QuantumScript`.
  ``StatePrepBase`` operations should be placed at the beginning of the `ops` list instead.
  [(#4554)](https://github.com/PennyLaneAI/pennylane/pull/4554)

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* `tf.function` no longer breaks `ProbabilityMP.process_state` which is needed by new devices.
  [(#4470)](https://github.com/PennyLaneAI/pennylane/pull/4470)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Romain Moyard,
Matthew Silverman
