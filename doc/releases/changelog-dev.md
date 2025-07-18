:orphan:

# Release 0.43.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

<h4>OpenQASM-PennyLane interoperability</h4>

* The :func:`qml.from_qasm3` function can now convert OpenQASM 3.0 circuits that contain
  subroutines, constants, and built-in mathematical functions.
  [(#7651)](https://github.com/PennyLaneAI/pennylane/pull/7651)
  [(#7653)](https://github.com/PennyLaneAI/pennylane/pull/7653)
  [(#7676)](https://github.com/PennyLaneAI/pennylane/pull/7676)
  [(#7679)](https://github.com/PennyLaneAI/pennylane/pull/7679)
  [(#7677)](https://github.com/PennyLaneAI/pennylane/pull/7677)

<h4>Other improvements</h4>

* A new `qml.transforms.resolve_dynamic_wires` transform can allocate concrete wire values for dynamic
  qubit allocation.
  [(#7678)](https://github.com/PennyLaneAI/pennylane/pull/7678)


* The :func:`qml.workflow.set_shots` transform can now be directly applied to a QNode without the need for `functools.partial`, providing a more user-friendly syntax and negating having to import the `functools` package.
  [(#7876)](https://github.com/PennyLaneAI/pennylane/pull/7876)

  ```python
  @qml.set_shots(shots=1000)
  @qml.qnode(dev)
  def circuit():
      qml.H(0)
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> circuit()
  0.002
  ```

* Added a `QuantumParser` class to the `qml.compiler.python_compiler` submodule that automatically loads relevant dialects.
  [(#7888)](https://github.com/PennyLaneAI/pennylane/pull/7888)

* Enforce various modules to follow modular architecture via `tach`.
  [(#7847)](https://github.com/PennyLaneAI/pennylane/pull/7847)

* A compilation pass written with xDSL called `qml.compiler.python_compiler.transforms.MeasurementsFromSamplesPass`
  has been added for the experimental xDSL Python compiler integration. This pass replaces all
  terminal measurements in a program with a single :func:`pennylane.sample` measurement, and adds
  postprocessing instructions to recover the original measurement.
  [(#7620)](https://github.com/PennyLaneAI/pennylane/pull/7620)

* A combine-global-phase pass has been added to the xDSL Python compiler integration.
  Note that the current implementation can only combine all the global phase operations at
  the last global phase operation in the same region. In other words, global phase operations inside a control flow region can't be combined with those in their parent
  region.
  [(#7675)](https://github.com/PennyLaneAI/pennylane/pull/7675)

* The `mbqc` xDSL dialect has been added to the Python compiler, which is used to represent
  measurement-based quantum-computing instructions in the xDSL framework.
  [(#7815)](https://github.com/PennyLaneAI/pennylane/pull/7815)

* The :func:`pennylane.ops.rs_decomposition` method now performs exact decomposition and returns
  complete global phase information when used for decomposing a phase gate to Clifford+T basis.
  [(#7793)](https://github.com/PennyLaneAI/pennylane/pull/7793)

* `default.qubit` will default to the tree-traversal MCM method when `mcm_method="device"`.
  [(#7885)](https://github.com/PennyLaneAI/pennylane/pull/7885)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* Added state of the art resources for the `ResourceSelectPauliRot` template and the
  `ResourceQubitUnitary` templates.
  [(#7786)](https://github.com/PennyLaneAI/pennylane/pull/7786)

<h3>Breaking changes 💔</h3>

* `qml.operation.WiresEnum`, `qml.operation.AllWires`, and `qml.operation.AnyWires` have been removed. Setting `Operator.num_wires = None` (the default)
  should instead indicate that the `Operator` does not need wire validation.
  [(#7911)](https://github.com/PennyLaneAI/pennylane/pull/7911)

* Removed `QNode.get_gradient_fn` method. Instead, use `qml.workflow.get_best_diff_method` to obtain the differentiation method.
  [(#7907)](https://github.com/PennyLaneAI/pennylane/pull/7907)

* Top-level access to ``DeviceError``, ``PennyLaneDeprecationWarning``, ``QuantumFunctionError`` and ``ExperimentalWarning`` has been removed. Please import these objects from the new ``pennylane.exceptions`` module.
  [(#7874)](https://github.com/PennyLaneAI/pennylane/pull/7874)

* `qml.cut_circuit_mc` no longer accepts a `shots` keyword argument. The shots should instead
  be set on the tape itself.
  [(#7882)](https://github.com/PennyLaneAI/pennylane/pull/7882)

<h3>Deprecations 👋</h3>

* `qml.qnn.cost.SquaredErrorLoss` is deprecated and will be removed in version v0.44. Instead, this hybrid workflow can be accomplished 
  with a function like `loss = lambda *args: (circuit(*args) - target)**2`.
  [(#7527)](https://github.com/PennyLaneAI/pennylane/pull/7527)
  
* Access to `add_noise`, `insert` and noise mitigation transforms from the `pennylane.transforms` module is deprecated.
  Instead, these functions should be imported from the `pennylane.noise` module.
  [(#7854)](https://github.com/PennyLaneAI/pennylane/pull/7854)

* The `qml.QNode.add_transform` method is deprecated and will be removed in v0.43.
  Instead, please use `QNode.transform_program.push_back(transform_container=transform_container)`.
  [(#7855)](https://github.com/PennyLaneAI/pennylane/pull/7855)

<h3>Internal changes ⚙️</h3>

* Added a `run_filecheck_qjit` fixture that can be used to run FileCheck on integration tests for the
  `qml.compiler.python_compiler` submodule.
  [(#7888)](https://github.com/PennyLaneAI/pennylane/pull/7888)

* Added a `dialects` submodule to `qml.compiler.python_compiler` which now houses all the xDSL dialects we create.
  Additionally, the `MBQCDialect` and `QuantumDialect` dialects have been renamed to `MBQC` and `Quantum`.
  [(#7897)](https://github.com/PennyLaneAI/pennylane/pull/7897)

* Update minimum supported `pytest` version to `8.4.1`.
  [(#7853)](https://github.com/PennyLaneAI/pennylane/pull/7853)

* `DefaultQubitLegacy` (test suite only) no longer provides a customized classical shadow
  implementation
  [(#7895)](https://github.com/PennyLaneAI/pennylane/pull/7895)

* Make `pennylane.io` a tertiary module.
  [(#7877)](https://github.com/PennyLaneAI/pennylane/pull/7877)

* Seeded tests for the `split_to_single_terms` transformation.
  [(#7851)](https://github.com/PennyLaneAI/pennylane/pull/7851)

* Upgrade `rc_sync.yml` to work with latest `pyproject.toml` changes.
  [(#7808)](https://github.com/PennyLaneAI/pennylane/pull/7808)
  [(#7818)](https://github.com/PennyLaneAI/pennylane/pull/7818)

* `LinearCombination` instances can be created with `_primitive.impl` when
  capture is enabled and tracing is active.
  [(#7893)](https://github.com/PennyLaneAI/pennylane/pull/7893)

<h3>Documentation 📝</h3>

* Updated the code example in the documentation for :func:`~.transforms.split_non_commuting`.
  [(#7892)](https://github.com/PennyLaneAI/pennylane/pull/7892)

<h3>Bug fixes 🐛</h3>

* `get_best_diff_method` now correctly aligns with `execute` and `construct_batch` logic in workflows.
  [(#7898)](https://github.com/PennyLaneAI/pennylane/pull/7898)

* Resolve issues with AutoGraph transforming internal PennyLane library code due to incorrect
  module attribution of wrapper functions.
  [(#7889)](https://github.com/PennyLaneAI/pennylane/pull/7889)

* Calling `QNode.update` no longer acts as if `set_shots` has been applied.
  [(#7881)](https://github.com/PennyLaneAI/pennylane/pull/7881)

* Fixes attributes and types in the quantum dialect.
  This allows for types to be inferred correctly when parsing.
  [(#7825)](https://github.com/PennyLaneAI/pennylane/pull/7825)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Utkarsh Azad,
Joey Carter,
Yushao Chen,
David Ittah,
Erick Ochoa,
Mudit Pandey,
Andrija Paurevic,
Jay Soni,
Jake Zaia
