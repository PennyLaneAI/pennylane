:orphan:

# Release 0.40.0-dev (development release)

<h3>New features since last release</h3>

<h4>New API for Qubit Mixed</h4>

* Added `qml.devices.qubit_mixed` module for mixed-state qubit device support. This module introduces:

  [(#6379)](https://github.com/PennyLaneAI/pennylane/pull/6379) An `apply_operation` helper function featuring:

  * Two density matrix contraction methods using `einsum` and `tensordot`
  * Optimized handling of special cases including: Diagonal operators, Identity operators, CX (controlled-X), Multi-controlled X gates, Grover operators

  [(#6503)](https://github.com/PennyLaneAI/pennylane/pull/6503) A submodule 'initialize_state' featuring:

  * A `density_matrix` function for initializing a density matrix from a state vector

  * A `state_vector` function for initializing a state vector from a density matrix

  * A `mixed_state` function for initializing a mixed state from a state vector

  * A `state_vector_from_mixed` function for initializing a state vector from a mixed state

<h3>Improvements 🛠</h3>

<h4>Other Improvements</h4>

* `qml.BasisRotation` template is now JIT compatible.
  [(#6019)](https://github.com/PennyLaneAI/pennylane/pull/6019)

* Expand `ExecutionConfig.gradient_method` to store `TransformDispatcher` type.
  [(#6455)](https://github.com/PennyLaneAI/pennylane/pull/6455)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Yushao Chen,
Andrija Paurevic,
