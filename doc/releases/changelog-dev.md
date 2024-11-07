:orphan:

# Release 0.40.0-dev (development release)

<h3>New features since last release</h3>

* A `DeviceCapabilities` data class is defined to contain all capabilities of the device's execution interface (i.e. its implementation of `Device.execute`). A TOML file can be used to define the capabilities of a device, and it can be loaded into a `DeviceCapabilities` object.
  [(#6407)](https://github.com/PennyLaneAI/pennylane/pull/6407)

  ```pycon
  >>> from pennylane.devices.capabilities import load_toml_file, parse_toml_document, DeviceCapabilities
  >>> document = load_toml_file("my_device.toml")
  >>> capabilities = parse_toml_document(document)
  >>> isinstance(capabilities, DeviceCapabilities)
  True
  ```

<h3>Improvements 🛠</h3>

<h4>Other Improvements</h4>

* Added `qml.devices.qubit_mixed` module for mixed-state qubit device support. This module introduces:
  - A new API for mixed-state operations
  - An `apply_operation` helper function featuring:
    - Two density matrix contraction methods using `einsum` and `tensordot`
    - Optimized handling of special cases including:
      - Diagonal operators
      - Identity operators 
      - CX (controlled-X)
      - Multi-controlled X gates
      - Grover operators
  [(#6379)](https://github.com/PennyLaneAI/pennylane/pull/6379)

* `qml.BasisRotation` template is now JIT compatible.
  [(#6019)](https://github.com/PennyLaneAI/pennylane/pull/6019)

* Expand `ExecutionConfig.gradient_method` to store `TransformDispatcher` type.
  [(#6455)](https://github.com/PennyLaneAI/pennylane/pull/6455)

<h3>Breaking changes 💔</h3>

* The ``qml.workflow.set_shots`` helper function has been removed. We no longer interact with the legacy device interface
  in our code. Instead, shots should be specified on the tape, and this should be used by the device.
  [(#6534)](https://github.com/PennyLaneAI/pennylane/pull/6534)

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Astral Cai,
Yushao Chen,
Andrija Paurevic
