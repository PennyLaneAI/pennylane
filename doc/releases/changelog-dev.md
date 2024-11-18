:orphan:

# Release 0.40.0-dev (development release)

<h3>New features since last release</h3>

* Developers of plugin devices now have the option of providing a TOML-formatted configuration file
  to declare the capabilities of the device. See [Device Capabilities](https://docs.pennylane.ai/en/latest/development/plugins.html#device-capabilities) for details.
  [(#6407)](https://github.com/PennyLaneAI/pennylane/pull/6407)
  [(#6433)](https://github.com/PennyLaneAI/pennylane/pull/6433)

  * An internal module `pennylane.devices.capabilities` is added that defines a new `DeviceCapabilites`
    data class, as well as functions that load and parse the TOML-formatted configuration files.

    ```pycon
      >>> from pennylane.devices.capabilities import DeviceCapabilities
      >>> capabilities = DeviceCapabilities.from_toml_file("my_device.toml")
      >>> isinstance(capabilities, DeviceCapabilities)
      True
    ```

  * Devices that extends `qml.devices.Device` now has an optional class attribute `capabilities`
    that is an instance of the `DeviceCapabilities` data class, constructed from the configuration
    file if it exists. Otherwise, it is set to `None`.

    ```python
    from pennylane.devices import Device
    
    class MyDevice(Device):
    
        config_filepath = "path/to/config.toml"
    
        ...
    ```
    ```pycon
    >>> isinstance(MyDevice.capabilities, DeviceCapabilities)
    True
    ```

* Added a dense implementation of computing the Lie closure in a new function
  `lie_closure_dense` in `pennylane.labs.dla`.
  [(#6371)](https://github.com/PennyLaneAI/pennylane/pull/6371)

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

<h4>Capturing and representing hybrid programs</h4>

* `jax.vmap` can be captured with `qml.capture.make_plxpr` and is compatible with quantum circuits. 
  [(#6349)](https://github.com/PennyLaneAI/pennylane/pull/6349)

* `qml.capture.PlxprInterpreter` base class has been added for easy transformation and execution of
  pennylane variant jaxpr.
  [(#6141)](https://github.com/PennyLaneAI/pennylane/pull/6141)

* An optional method `eval_jaxpr` is added to the device API for native execution of plxpr programs.
[(#6580)](https://github.com/PennyLaneAI/pennylane/pull/6580)

<h4>Other Improvements</h4>

* `qml.BasisRotation` template is now JIT compatible.
  [(#6019)](https://github.com/PennyLaneAI/pennylane/pull/6019)

* The Jaxpr primitives for `for_loop`, `while_loop` and `cond` now store slices instead of
  numbers of args.
  [(#6521)](https://github.com/PennyLaneAI/pennylane/pull/6521)

* Expand `ExecutionConfig.gradient_method` to store `TransformDispatcher` type.
  [(#6455)](https://github.com/PennyLaneAI/pennylane/pull/6455)

<h3>Breaking changes 💔</h3>

* Legacy operator arithmetic has been removed. This includes `qml.ops.Hamiltonian`, `qml.operation.Tensor`,
  `qml.operation.enable_new_opmath`, `qml.operation.disable_new_opmath`, and `qml.operation.convert_to_legacy_H`.
  Note that `qml.Hamiltonian` will continue to dispatch to `qml.ops.LinearCombination`. For more information, 
  check out the [updated operator troubleshooting page](https://docs.pennylane.ai/en/stable/news/new_opmath.html).
  [(#6548)](https://github.com/PennyLaneAI/pennylane/pull/6548)

* The developer-facing `qml.utils` module has been removed. Specifically, the
following 4 sets of functions have been either moved or removed[(#6588)](https://github.com/PennyLaneAI/pennylane/pull/6588):

  * `qml.utils._flatten`, `qml.utils.unflatten` has been moved and renamed to `qml.optimize.qng._flatten_np` and `qml.optimize.qng._unflatten_np` respectively.

  * `qml.utils._inv_dict` and `qml._get_default_args` have been removed.

  * `qml.utils.pauli_eigs` has been moved to `qml.pauli.utils`.

  * `qml.utils.expand_vector` has been moved to `qml.math.expand_vector`.
  
* The `qml.qinfo` module has been removed. Please see the respective functions in the `qml.math` and `qml.measurements`
  modules instead.
  [(#6584)](https://github.com/PennyLaneAI/pennylane/pull/6584)

* Top level access to `Device`, `QubitDevice`, and `QutritDevice` have been removed. Instead, they
  are available as `qml.devices.LegacyDevice`, `qml.devices.QubitDevice`, and `qml.devices.QutritDevice`
  respectively.
  [(#6537)](https://github.com/PennyLaneAI/pennylane/pull/6537)

* The `'ancilla'` argument for `qml.iterative_qpe` has been removed. Instead, use the `'aux_wire'` argument.
  [(#6532)](https://github.com/PennyLaneAI/pennylane/pull/6532)

* The `qml.BasisStatePreparation` template has been removed. Instead, use `qml.BasisState`.
  [(#6528)](https://github.com/PennyLaneAI/pennylane/pull/6528)

* The `qml.workflow.set_shots` helper function has been removed. We no longer interact with the legacy device interface in our code.
  Instead, shots should be specified on the tape, and the device should use these shots.
  [(#6534)](https://github.com/PennyLaneAI/pennylane/pull/6534)

* `QNode.gradient_fn` has been removed. Please use `QNode.diff_method` instead. `QNode.get_gradient_fn` can also be used to
  process the diff method.
  [(#6535)](https://github.com/PennyLaneAI/pennylane/pull/6535)
 
* The `qml.QubitStateVector` template has been removed. Instead, use `qml.StatePrep`.
  [(#6525)](https://github.com/PennyLaneAI/pennylane/pull/6525)

* `qml.broadcast` has been removed. Users should use `for` loops instead.
  [(#6527)](https://github.com/PennyLaneAI/pennylane/pull/6527)

* The `max_expansion` argument for `qml.transforms.clifford_t_decomposition` has been removed.
  [(#6531)](https://github.com/PennyLaneAI/pennylane/pull/6531)
  [(#6571)](https://github.com/PennyLaneAI/pennylane/pull/6571)

* The `expand_depth` argument for `qml.compile` has been removed.
  [(#6531)](https://github.com/PennyLaneAI/pennylane/pull/6531)
  

* The `qml.shadows.shadow_expval` transform has been removed. Instead, please use the
  `qml.shadow_expval` measurement process.
  [(#6530)](https://github.com/PennyLaneAI/pennylane/pull/6530)
  [(#6561)](https://github.com/PennyLaneAI/pennylane/pull/6561)

<h3>Deprecations 👋</h3>

* The `max_expansion` argument in `qml.devices.preprocess.decompose` is deprecated and will be removed in v0.41.
  [(#6400)](https://github.com/PennyLaneAI/pennylane/pull/6400)

* The `decomp_depth` argument in `qml.transforms.set_decomposition` is deprecated and will be removed in v0.41.
  [(#6400)](https://github.com/PennyLaneAI/pennylane/pull/6400)

* The `output_dim` property of `qml.tape.QuantumScript` has been deprecated. 
Instead, use method `shape` of `QuantumScript` or `MeasurementProcess` to get the 
same information.
  [(#6577)](https://github.com/PennyLaneAI/pennylane/pull/6577)

* The `QNode.get_best_method` and `QNode.best_method_str` methods have been deprecated. 
  Instead, use the `qml.workflow.get_best_diff_method` function.
  [(#6418)](https://github.com/PennyLaneAI/pennylane/pull/6418)

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

* `qml.math.get_deep_interface` now works properly for autograd arrays.
  [(#6557)](https://github.com/PennyLaneAI/pennylane/pull/6557)

* Fixed `Identity.__repr__` to return correct wires list.
  [(#6506)](https://github.com/PennyLaneAI/pennylane/pull/6506)


<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Shiwen An,
Astral Cai,
Yushao Chen,
Pietropaolo Frisoni,
Austin Huang,
Korbinian Kottmann,
Christina Lee,
Andrija Paurevic,
Justin Pickering,
