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

<h4>New API for Qubit Mixed</h4>

* Added `qml.devices.qubit_mixed` module for mixed-state qubit device support [(#6379)](https://github.com/PennyLaneAI/pennylane/pull/6379). This module introduces an `apply_operation` helper function that features:


  * Two density matrix contraction methods using `einsum` and `tensordot`

  * Optimized handling of special cases including: Diagonal operators, Identity operators, CX (controlled-X), Multi-controlled X gates, Grover operators

* Added submodule 'initialize_state' featuring a `create_initial_state` function for initializing a density matrix from `qml.StatePrep` operations or `qml.QubitDensityMatrix` operations.
  [(#6503)](https://github.com/PennyLaneAI/pennylane/pull/6503) 

<h3>Improvements 🛠</h3>

* Added support for the `wire_options` dictionary to customize wire line formatting in `qml.draw_mpl` circuit
  visualizations, allowing global and per-wire customization with options like `color`, `linestyle`, and `linewidth`.
  [(#6486)](https://github.com/PennyLaneAI/pennylane/pull/6486)

<h4>Capturing and representing hybrid programs</h4>

* `jax.vmap` can be captured with `qml.capture.make_plxpr` and is compatible with quantum circuits. 
  [(#6349)](https://github.com/PennyLaneAI/pennylane/pull/6349)

<h4>Other Improvements</h4>

* `qml.BasisRotation` template is now JIT compatible.
  [(#6019)](https://github.com/PennyLaneAI/pennylane/pull/6019)

* The Jaxpr primitives for `for_loop`, `while_loop` and `cond` now store slices instead of
  numbers of args.
  [(#6521)](https://github.com/PennyLaneAI/pennylane/pull/6521)

* Expand `ExecutionConfig.gradient_method` to store `TransformDispatcher` type.
  [(#6455)](https://github.com/PennyLaneAI/pennylane/pull/6455)

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

* Add a warning message to Gradients and training documentation about ComplexWarnings
  [(#6543)](https://github.com/PennyLaneAI/pennylane/pull/6543)

<h3>Bug fixes 🐛</h3>

* Fixed `Identity.__repr__` to return correct wires list.
  [(#6506)](https://github.com/PennyLaneAI/pennylane/pull/6506)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Shiwen An,
Astral Cai,
Yushao Chen,
Pietropaolo Frisoni,
Andrija Paurevic,
Justin Pickering,
