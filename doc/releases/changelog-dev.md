:orphan:

# Release 0.40.0-dev (development release)

<h3>New features since last release</h3>

* Developers of plugin devices now have the option of providing a TOML-formatted configuration file
  to declare the capabilities of the device. See [Device Capabilities](https://docs.pennylane.ai/en/latest/development/plugins.html#device-capabilities) for details.
  [(#6407)](https://github.com/PennyLaneAI/pennylane/pull/6407)
  [(#6433)](https://github.com/PennyLaneAI/pennylane/pull/6433)

  * An internal module `pennylane.devices.capabilities` is added that defines a new `DeviceCapabilites`
    data class, as well as functions that load and parse the TOML-formatted configuration files.
  * Devices that extends `qml.devices.Device` now has an optional class attribute `capabilities`
    that is an instance of the `DeviceCapabilities` data class, constructed from the configuration
    file if it exists. Otherwise, it is set to `None`.

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

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Shiwen An,
Astral Cai,
Andrija Paurevic,
Justin Pickering,
Pietropaolo Frisoni

