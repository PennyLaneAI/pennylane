:orphan:

# Release 0.40.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

<h4>Other Improvements</h4>

* Added `qml.devices.qubit_mixed` module for mixed-state qubit device support. This module introduces:

  [(#6379)](https://github.com/PennyLaneAI/pennylane/pull/6379) An `apply_operation` helper function featuring:

  * Two density matrix contraction methods using `einsum` and `tensordot`

  * Optimized handling of special cases including: Diagonal operators, Identity operators, CX (controlled-X), Multi-controlled X gates, Grover operators

* `qml.BasisRotation` template is now JIT compatible.
  [(#6019)](https://github.com/PennyLaneAI/pennylane/pull/6019)

* Expand `ExecutionConfig.gradient_method` to store `TransformDispatcher` type.
  [(#6455)](https://github.com/PennyLaneAI/pennylane/pull/6455)

<h3>Breaking changes 💔</h3>

* The ``qml.workflow.set_shots`` helper func has been removed. We no longer interact with the legacy device interface in our code 
workflow behavior. Instead, shots should be specified on the tape, and the device should pull it's number of shots from the tape.
  [(#6534)](https://github.com/PennyLaneAI/pennylane/pull/6534)

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Yushao Chen,
Andrija Paurevic,
