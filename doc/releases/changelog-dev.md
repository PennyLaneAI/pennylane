:orphan:

# Release 0.40.0-dev (development release)

<h3>New features since last release</h3>

* Two new methods: `setup_execution_config` and `preprocess_transforms` are added to the `Device`
  class. Device developers are encouraged to override these two methods separately instead of the
  `preprocess` method. For now, to avoid ambiguity, a device is allowed to override either these
  two methods or `preprocess`, but not both. In the long term, we will slowly phase out the use of
  `preprocess` in favour of these two methods for better separation of concerns.
  [(#6617)](https://github.com/PennyLaneAI/pennylane/pull/6617)

* Developers of plugin devices now have the option of providing a TOML-formatted configuration file
  to declare the capabilities of the device. See [Device Capabilities](https://docs.pennylane.ai/en/latest/development/plugins.html#device-capabilities) for details.

  * An internal module `pennylane.devices.capabilities` is added that defines a new `DeviceCapabilites`
    data class, as well as functions that load and parse the TOML-formatted configuration files.
    [(#6407)](https://github.com/PennyLaneAI/pennylane/pull/6407)

    ```pycon
      >>> from pennylane.devices.capabilities import DeviceCapabilities
      >>> capabilities = DeviceCapabilities.from_toml_file("my_device.toml")
      >>> isinstance(capabilities, DeviceCapabilities)
      True
    ```

  * Devices that extends `qml.devices.Device` now has an optional class attribute `capabilities`
    that is an instance of the `DeviceCapabilities` data class, constructed from the configuration
    file if it exists. Otherwise, it is set to `None`.
    [(#6433)](https://github.com/PennyLaneAI/pennylane/pull/6433)

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

  * Default implementations of `Device.setup_execution_config` and `Device.preprocess_transforms`
    are added to the device API for devices that provides a TOML configuration file and thus have
    a `capabilities` property.
    [(#6632)](https://github.com/PennyLaneAI/pennylane/pull/6632)
    [(#6653)](https://github.com/PennyLaneAI/pennylane/pull/6653)

* Support is added for `if`/`else` statements and `for` and `while` loops in circuits executed with `qml.capture.enabled`, via Autograph.
  Autograph conversion is now used by default in `make_plxpr`, but can be skipped with the keyword arg `autograph=False`.
  [(#6406)](https://github.com/PennyLaneAI/pennylane/pull/6406)
  [(#6413)](https://github.com/PennyLaneAI/pennylane/pull/6413)
  [(#6426)](https://github.com/PennyLaneAI/pennylane/pull/6426)
  [(#6645)](https://github.com/PennyLaneAI/pennylane/pull/6645)

  * New `qml.GQSP` template has been added to perform Generalized Quantum Signal Processing (GQSP).
    The functionality `qml.poly_to_angles` has been also extended to support GQSP.
    [(#6565)](https://github.com/PennyLaneAI/pennylane/pull/6565)


<h4>New `labs` module `dla` for handling dynamical Lie algebras (DLAs)</h4>

* Added a dense implementation of computing the Lie closure in a new function
  `lie_closure_dense` in `pennylane.labs.dla`.
  [(#6371)](https://github.com/PennyLaneAI/pennylane/pull/6371)

* New functionality to calculate angles for QSP and QSVT has been added. This includes the function `qml.poly_to_angles`
  to obtain angles directly and the function `qml.transform_angles` to convert angles from one subroutine to another.
  [(#6483)](https://github.com/PennyLaneAI/pennylane/pull/6483)

* Added a dense implementation of computing the structure constants in a new function
  `structure_constants_dense` in `pennylane.labs.dla`.
  [(#6376)](https://github.com/PennyLaneAI/pennylane/pull/6376)

* Added utility functions for handling dense matrices in the Lie theory context.
  [(#6563)](https://github.com/PennyLaneAI/pennylane/pull/6563)
  [(#6392)](https://github.com/PennyLaneAI/pennylane/pull/6392)

* Added a ``cartan_decomp`` function along with two standard involutions ``even_odd_involution`` and ``concurrence_involution``.
  [(#6392)](https://github.com/PennyLaneAI/pennylane/pull/6392)

* Added `unary_mapping()` function to map `BoseWord` and `BoseSentence` to qubit operators, using unary mapping
  [(#6576)](https://github.com/PennyLaneAI/pennylane/pull/6576); 
added `binary_mapping()` function to map `BoseWord` and `BoseSentence` to qubit operators, using standard-binary mapping.
  [(#6564)](https://github.com/PennyLaneAI/pennylane/pull/6564)


<h4>New API for Qubit Mixed</h4>

* Added `qml.devices.qubit_mixed` module for mixed-state qubit device support [(#6379)](https://github.com/PennyLaneAI/pennylane/pull/6379). This module introduces an `apply_operation` helper function that features:

  * Two density matrix contraction methods using `einsum` and `tensordot`

  * Optimized handling of special cases including: Diagonal operators, Identity operators, CX (controlled-X), Multi-controlled X gates, Grover operators

* Added submodule 'initialize_state' featuring a `create_initial_state` function for initializing a density matrix from `qml.StatePrep` operations or `qml.QubitDensityMatrix` operations.
  [(#6503)](https://github.com/PennyLaneAI/pennylane/pull/6503)
  
* Added support for constructing `BoseWord` and `BoseSentence`, similar to `FermiWord` and `FermiSentence`.
  [(#6518)](https://github.com/PennyLaneAI/pennylane/pull/6518)

* Added method `preprocess` to the `QubitMixed` device class to preprocess the quantum circuit before execution. Necessary non-intrusive interfaces changes to class init method were made along the way to the `QubitMixed` device class to support new API feature.
  [(#6601)](https://github.com/PennyLaneAI/pennylane/pull/6601)

* Added a second class `DefaultMixedNewAPI` to the `qml.devices.qubit_mixed` module, which is to be the replacement of legacy `DefaultMixed` which for now to hold the implementations of `preprocess` and `execute` methods.
  [(#6607)](https://github.com/PennyLaneAI/pennylane/pull/6607)

* Added submodule `devices.qubit_mixed.measure` as a necessary step for the new API, featuring a `measure` function for measuring qubits in mixed-state devices.
  [(#6637)](https://github.com/PennyLaneAI/pennylane/pull/6637)

* Added submodule `devices.qubit_mixed.simulate` as a necessary step for the new API,
featuring a `simulate` function for simulating mixed states in analytic mode.
  [(#6618)](https://github.com/PennyLaneAI/pennylane/pull/6618)

* Added submodule `devices.qubit_mixed.sampling` as a necessary step for the new API, featuring functions `sample_state`, `measure_with_samples` and `sample_probs` for sampling qubits in mixed-state devices.
  [(#6639)](https://github.com/PennyLaneAI/pennylane/pull/6639)

* Added `christiansen_mapping()` function to map `BoseWord` and `BoseSentence` to qubit operators, using christiansen mapping.
  [(#6623)](https://github.com/PennyLaneAI/pennylane/pull/6623)

* The `qml.qchem.factorize` function now supports new methods for double factorization:
  Cholesky decomposition (`cholesky=True`) and compressed double factorization (`compressed=True`).
  [(#6573)](https://github.com/PennyLaneAI/pennylane/pull/6573)
  [(#6611)](https://github.com/PennyLaneAI/pennylane/pull/6611)

* Added `qml.qchem.symmetry_shift` function to perform the
  [block-invariant symmetry shift](https://arxiv.org/pdf/2304.13772) on the electronic integrals.
  [(#6574)](https://github.com/PennyLaneAI/pennylane/pull/6574)

* Added submodule for calculating vibrational Hamiltonians
  * Implemented helper functions for geometry optimization, harmonic analysis,
    and normal-mode localization.
    [(#6453)](https://github.com/PennyLaneAI/pennylane/pull/6453)
    [(#6666)](https://github.com/PennyLaneAI/pennylane/pull/6666)
  * Implemented helper functions for calculating one-mode PES, two-mode PES, and
    three-mode PES.
    [(#6616)](https://github.com/PennyLaneAI/pennylane/pull/6616)
  * Implemented wrapper function for vibrational Hamiltonian calculation and dataclass
    for storing the data.
    [(#6652)](https://github.com/PennyLaneAI/pennylane/pull/6652)

<h3>Improvements 🛠</h3>

* Raises a comprehensive error when using `qml.fourier.qnode_spectrum` with standard numpy
  arguments and `interface="auto"`.
  [(#6622)](https://github.com/PennyLaneAI/pennylane/pull/6622)

* Added support for the `wire_options` dictionary to customize wire line formatting in `qml.draw_mpl` circuit
  visualizations, allowing global and per-wire customization with options like `color`, `linestyle`, and `linewidth`.
  [(#6486)](https://github.com/PennyLaneAI/pennylane/pull/6486)

* Added Pauli String representations for the gates X, Y, Z, S, T, SX, SWAP, ISWAP, ECR, SISWAP.
  [(#6562)](https://github.com/PennyLaneAI/pennylane/pull/6562)
  
* `QNode` and `qml.execute` now forbid certain keyword arguments from being passed positionally.
  [(#6610)](https://github.com/PennyLaneAI/pennylane/pull/6610)

* Shortened the string representation for the `qml.S`, `qml.T`, and `qml.SX` operators.
  [(#6542)](https://github.com/PennyLaneAI/pennylane/pull/6542)

* Added functions and dunder methods to add and multiply Resources objects in series and in parallel.
  [(#6567)](https://github.com/PennyLaneAI/pennylane/pull/6567)

* The `diagonalize_measurements` transform no longer raises an error for unknown observables. Instead,
  they are left undiagonalized, with the expectation that observable validation will catch any undiagonalized
  observables that are also unsupported by the device.
  [(#6653)](https://github.com/PennyLaneAI/pennylane/pull/6653)

<h4>Capturing and representing hybrid programs</h4>

* Execution with capture enabled now follows a new execution pipeline and natively passes the
  captured jaxpr to the device. Since it no longer falls back to the old pipeline, execution
  only works with a reduced feature set.
  [(#6655)](https://github.com/PennyLaneAI/pennylane/pull/6655)
  [(#6596)](https://github.com/PennyLaneAI/pennylane/pull/6596)

* PennyLane transforms can now be captured as primitives with experimental program capture enabled.
  [(#6633)](https://github.com/PennyLaneAI/pennylane/pull/6633)

* `jax.vmap` can be captured with `qml.capture.make_plxpr` and is compatible with quantum circuits.
  [(#6349)](https://github.com/PennyLaneAI/pennylane/pull/6349)
  [(#6422)](https://github.com/PennyLaneAI/pennylane/pull/6422)
  [(#6668)](https://github.com/PennyLaneAI/pennylane/pull/6668)

* `qml.capture.PlxprInterpreter` base class has been added for easy transformation and execution of
  pennylane variant jaxpr.
  [(#6141)](https://github.com/PennyLaneAI/pennylane/pull/6141)

* A `DefaultQubitInterpreter` class has been added to provide plxpr execution using python based tools,
  and the `DefaultQubit.eval_jaxpr` method is now implemented.
  [(#6594)](https://github.com/PennyLaneAI/pennylane/pull/6594)
  [(#6328)](https://github.com/PennyLaneAI/pennylane/pull/6328)

* An optional method `eval_jaxpr` is added to the device API for native execution of plxpr programs.
  [(#6580)](https://github.com/PennyLaneAI/pennylane/pull/6580)

* `qml.capture.qnode_call` has been made private and moved to the `workflow` module.
  [(#6620)](https://github.com/PennyLaneAI/pennylane/pull/6620/)

<h4>Other Improvements</h4>

* Standardize supported interfaces to an internal Enum object. 
  [(#6643)](https://github.com/PennyLaneAI/pennylane/pull/6643)

* Moved all interface handling logic to `interface_utils.py` in the `qml.math` module.
  [(#6649)](https://github.com/PennyLaneAI/pennylane/pull/6649)

* Added PyTree support for measurements in a circuit. 
  [(#6378)](https://github.com/PennyLaneAI/pennylane/pull/6378)

  ```python
  import pennylane as qml

  @qml.qnode(qml.device("default.qubit"))
  def circuit():
      qml.Hadamard(0)
      qml.CNOT([0,1])
      return {"Probabilities": qml.probs(), "State": qml.state()}
  ```
  ```pycon
  >>> circuit()
  {'Probabilities': array([0.5, 0. , 0. , 0.5]), 'State': array([0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.70710678+0.j])}
  ```

* `_cache_transform` transform has been moved to its own file located
  at `qml.workflow._cache_transform.py`.
  [(#6624)](https://github.com/PennyLaneAI/pennylane/pull/6624)

* `qml.BasisRotation` template is now JIT compatible.
  [(#6019)](https://github.com/PennyLaneAI/pennylane/pull/6019)

* The Jaxpr primitives for `for_loop`, `while_loop` and `cond` now store slices instead of
  numbers of args.
  [(#6521)](https://github.com/PennyLaneAI/pennylane/pull/6521)

* Expand `ExecutionConfig.gradient_method` to store `TransformDispatcher` type.
  [(#6455)](https://github.com/PennyLaneAI/pennylane/pull/6455)

* Fix the string representation of `Resources` instances to match the attribute names.
  [(#6581)](https://github.com/PennyLaneAI/pennylane/pull/6581)

<h3>Labs 🧪</h3>

* Added base class `Resources`, `CompressedResourceOp`, `ResourceOperator` for advanced resource estimation.
  [(#6428)](https://github.com/PennyLaneAI/pennylane/pull/6428)

* Added `get_resources()` functionality which allows users to extract resources from a quantum function, tape or
  resource operation. Additionally added some standard gatesets `DefaultGateSet` to track resources with respect to.
  [(#6500)](https://github.com/PennyLaneAI/pennylane/pull/6500)

* Added `ResourceOperator` classes for QFT and all operators in QFT's decomposition.
  [(#6447)](https://github.com/PennyLaneAI/pennylane/pull/6447)

* Added native `ResourceOperator` subclasses for each of the controlled operators.
  [(#6579)](https://github.com/PennyLaneAI/pennylane/pull/6579)

* Added native `ResourceOperator` subclasses for each of the multi qubit operators.
  [(#6538)](https://github.com/PennyLaneAI/pennylane/pull/6538)

<h3>Breaking changes 💔</h3>

* `qml.fourier.qnode_spectrum` no longer automatically converts pure numpy parameters to the
  Autograd framework. As the function uses automatic differentiation for validation, parameters
  from an autodiff framework have to be used.
  [(#6622)](https://github.com/PennyLaneAI/pennylane/pull/6622)

* `qml.math.jax_argnums_to_tape_trainable` is moved and made private to avoid a qnode dependency
  in the math module.
  [(#6609)](https://github.com/PennyLaneAI/pennylane/pull/6609)

* Gradient transforms are now applied after the user's transform program.
  [(#6590)](https://github.com/PennyLaneAI/pennylane/pull/6590)

* Legacy operator arithmetic has been removed. This includes `qml.ops.Hamiltonian`, `qml.operation.Tensor`,
  `qml.operation.enable_new_opmath`, `qml.operation.disable_new_opmath`, and `qml.operation.convert_to_legacy_H`.
  Note that `qml.Hamiltonian` will continue to dispatch to `qml.ops.LinearCombination`. For more information,
  check out the [updated operator troubleshooting page](https://docs.pennylane.ai/en/stable/news/new_opmath.html).
  [(#6548)](https://github.com/PennyLaneAI/pennylane/pull/6548)
  [(#6602)](https://github.com/PennyLaneAI/pennylane/pull/6602)
  [(#6589)](https://github.com/PennyLaneAI/pennylane/pull/6589)

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
  [(#6571)](https://github.com/PennyLaneAI/pennylane/pull/6571)

* The `expand_depth` argument for `qml.compile` has been removed.
  [(#6531)](https://github.com/PennyLaneAI/pennylane/pull/6531)

* The `qml.shadows.shadow_expval` transform has been removed. Instead, please use the
  `qml.shadow_expval` measurement process.
  [(#6530)](https://github.com/PennyLaneAI/pennylane/pull/6530)
  [(#6561)](https://github.com/PennyLaneAI/pennylane/pull/6561)

<h3>Deprecations 👋</h3>

* The `tape` and `qtape` properties of `QNode` have been deprecated.
  Instead, use the `qml.workflow.construct_tape` function.
  [(#6583)](https://github.com/PennyLaneAI/pennylane/pull/6583)
  [(#6650)](https://github.com/PennyLaneAI/pennylane/pull/6650)

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

* Updated the documentation of `TrotterProduct` to include the impact of the operands in the
  Hamiltonian on the strucutre of the created circuit. Included an illustrative example on this.
  [(#6629)](https://github.com/PennyLaneAI/pennylane/pull/6629)

* Add reporting of test warnings as failures.
  [(#6217)](https://github.com/PennyLaneAI/pennylane/pull/6217)

* Add a warning message to Gradients and training documentation about ComplexWarnings.
  [(#6543)](https://github.com/PennyLaneAI/pennylane/pull/6543)

<h3>Bug fixes 🐛</h3>

* Subclasses of `qml.ops.Controlled` no longer bind the primitives of their base operators when program capture
  is enabled.
  [(#6672)](https://github.com/PennyLaneAI/pennylane/pull/6672)

* The `qml.HilbertSchmidt` and `qml.LocalHilbertSchmidt` templates now apply the complex conjugate
  of the unitaries instead of the adjoint, providing the correct result.
  [(#6604)](https://github.com/PennyLaneAI/pennylane/pull/6604)

* `QNode` return behaviour is now consistent for lists and tuples.
  [(#6568)](https://github.com/PennyLaneAI/pennylane/pull/6568)

* `qml.QNode` now accepts arguments with types defined in libraries that are not necessarily
  in the list of supported interfaces, such as the `Graph` class defined in `networkx`.
  [(#6600)](https://github.com/PennyLaneAI/pennylane/pull/6600)

* `qml.math.get_deep_interface` now works properly for autograd arrays.
  [(#6557)](https://github.com/PennyLaneAI/pennylane/pull/6557)

* Fixed `Identity.__repr__` to return correct wires list.
  [(#6506)](https://github.com/PennyLaneAI/pennylane/pull/6506)


<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso,
Shiwen An,
Utkarsh Azad,
Astral Cai,
Yushao Chen,
Diksha Dhawan,
Lasse Dierich,
Lillian Frederiksen,
Pietropaolo Frisoni,
Austin Huang,
Korbinian Kottmann,
Christina Lee,
William Maxwell,
Andrija Paurevic,
Justin Pickering,
Jay Soni,
David Wierichs,
