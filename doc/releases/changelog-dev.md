:orphan:

# Release 0.33.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

* The old return type and associated functions ``qml.enable_return`` and ``qml.disable_return`` are removed.
 [#4503](https://github.com/PennyLaneAI/pennylane/pull/4503)

* The ``mode`` keyword argument in ``QNode`` is removed. Please use ``grad_on_execution`` instead.
 [#4503](https://github.com/PennyLaneAI/pennylane/pull/4503)

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

* Minor documentation improvements to the new device API. The documentation now correctly states that interface-specific
  parameters are only passed to the device for backpropagation derivatives. 

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Romain Moyard