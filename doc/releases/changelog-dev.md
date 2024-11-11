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

* Shortened the string representation for the `qml.S`, `qml.T`, and `qml.SX` operators.
  [(#6542)](https://github.com/PennyLaneAI/pennylane/pull/6542)

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

<h4>Capturing and representing hybrid programs</h4>

* `jax.vmap` can be captured with `qml.capture.make_plxpr` and is compatible with quantum circuits. 
  [(#6349)](https://github.com/PennyLaneAI/pennylane/pull/6349)

* `qml.capture.PlxprInterpreter` base class has been added for easy transformation and execution of
  pennylane variant jaxpr.
  [(#6141)](https://github.com/PennyLaneAI/pennylane/pull/6141)

<h4>Other Improvements</h4>

* `qml.BasisRotation` template is now JIT compatible.
  [(#6019)](https://github.com/PennyLaneAI/pennylane/pull/6019)

* The Jaxpr primitives for `for_loop`, `while_loop` and `cond` now store slices instead of
  numbers of args.
  [(#6521)](https://github.com/PennyLaneAI/pennylane/pull/6521)

* Expand `ExecutionConfig.gradient_method` to store `TransformDispatcher` type.
  [(#6455)](https://github.com/PennyLaneAI/pennylane/pull/6455)

<h3>Breaking changes 💔</h3>

* Top level access to `Device`, `QubitDevice`, and `QutritDevice` have been removed. Instead, they
  are available as `qml.devices.LegacyDevice`, `qml.devices.QubitDevice`, and `qml.devices.QutritDevice`
  respectively.
  [(#6537)](https://github.com/PennyLaneAI/pennylane/pull/6537)
  
* The `max_expansion` argument for `qml.transforms.clifford_t_decomposition` has been removed.
  [(#6531)](https://github.com/PennyLaneAI/pennylane/pull/6531)

* The `expand_depth` argument for `qml.compile` has been removed.
  [(#6531)](https://github.com/PennyLaneAI/pennylane/pull/6531)
  

* The `qml.shadows.shadow_expval` transform has been removed. Instead, please use the
  `qml.shadow_expval` measurement process.
  [(#6530)](https://github.com/PennyLaneAI/pennylane/pull/6530)
  [(#6561)](https://github.com/PennyLaneAI/pennylane/pull/6561)

<h3>Deprecations 👋</h3>

* The `qml.execute` `gradient_fn` keyword argument has been renamed `diff_method`,
  to better align with the termionology used by the `QNode`.
  `gradient_fn` will be removed in v0.41.
  [(#6549)](https://github.com/PennyLaneAI/pennylane/pull/6549)

<h3>Documentation 📝</h3>
* Add reporting of test warnings as failures.
  [(#6217)](https://github.com/PennyLaneAI/pennylane/pull/6217)

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
Christina Lee,
Andrija Paurevic,
Justin Pickering
