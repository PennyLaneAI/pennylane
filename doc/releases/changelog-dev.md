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

* Added support for the `wire_options` dictionary to customize wire line formatting in `qml.draw_mpl` circuit
  visualizations, allowing global and per-wire customization with options like `color`, `linestyle`, and `linewidth`.
  [(#6486)](https://github.com/PennyLaneAI/pennylane/pull/6486)

<h4>Capturing and representing hybrid programs</h4>

* `jax.vmap` can be captured with `qml.capture.make_plxpr` and is compatible with quantum circuits. 
  [(#6349)](https://github.com/PennyLaneAI/pennylane/pull/6349)

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

* The `QNode.get_best_method`, `QNode.best_method_str` and `QNode.get_gradient_fn` methods have been deprecated. 
  Instead, use the `qml.workflow.get_best_diff_method` function.
  [(#6418)](https://github.com/PennyLaneAI/pennylane/pull/6418)

<h3>Documentation 📝</h3>

* Add a warning message to Gradients and training documentation about ComplexWarnings
  [(#6543)](https://github.com/PennyLaneAI/pennylane/pull/6543)

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Shiwen An
Astral Cai,
Andrija Paurevic,
Justin Pickering
Pietropaolo Frisoni

