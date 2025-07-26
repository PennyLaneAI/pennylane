:orphan:

# Release 0.43.0-dev (development release)

<h3>New features since last release</h3>

* Leveraging quantum just-in-time compilation to optimize parameterized hybrid workflows with the momentum
  quantum natural gradient optimizer is now possible with the new :class:`~.MomentumQNGOptimizerQJIT` optimizer.
  [(#7606)](https://github.com/PennyLaneAI/pennylane/pull/7606)

  Similar to the :class:`~.QNGOptimizerQJIT` optimizer, :class:`~.MomentumQNGOptimizerQJIT` offers a
  `qml.qjit`-compatible analogue to the existing :class:`~.MomentumQNGOptimizer` with an Optax-like interface:

  ```python
  import pennylane as qml
  import jax.numpy as jnp

  dev = qml.device("lightning.qubit", wires=2)

  @qml.qnode(dev)
  def circuit(params):
      qml.RX(params[0], wires=0)
      qml.RY(params[1], wires=1)
      return qml.expval(qml.Z(0) + qml.X(1))

  opt = qml.MomentumQNGOptimizerQJIT(stepsize=0.1, momentum=0.2)

  @qml.qjit
  def update_step_qjit(i, args):
      params, state = args
      return opt.step(circuit, params, state)

  @qml.qjit
  def optimization_qjit(params, iters):
      state = opt.init(params)
      args = (params, state)
      params, state = qml.for_loop(iters)(update_step_qjit)(args)
      return params
  ```

  ```pycon
  >>> params = jnp.array([0.1, 0.2])
  >>> iters = 1000
  >>> optimization_qjit(params=params, iters=iters)
  Array([ 3.14159265, -1.57079633], dtype=float64)
  ```

<h3>Improvements 🛠</h3>

* Several templates now have decompositions that can be accessed within the graph-based
  decomposition system (:func:`~.decomposition.enable_graph`), allowing workflows 
  that include these templates to be decomposed in a resource-efficient and performant
  manner.
  [(#7779)](https://github.com/PennyLaneAI/pennylane/pull/7779)
  
  The included templates are:

  * :class:`~.Adder`
    
  * :class:`~.ControlledSequence`
  
  * :class:`~.ModExp`

  * :class:`~.Multiplier`

  * :class:`~.OutAdder`

  * :class:`~.OutMultiplier`

  * :class:`~.OutPoly`

<h4>OpenQASM-PennyLane interoperability</h4>

* The :func:`qml.from_qasm3` function can now convert OpenQASM 3.0 circuits that contain
  subroutines, constants, all remaining stdlib gates, qubit registers, and built-in mathematical functions.
  [(#7651)](https://github.com/PennyLaneAI/pennylane/pull/7651)
  [(#7653)](https://github.com/PennyLaneAI/pennylane/pull/7653)
  [(#7676)](https://github.com/PennyLaneAI/pennylane/pull/7676)
  [(#7679)](https://github.com/PennyLaneAI/pennylane/pull/7679)
  [(#7677)](https://github.com/PennyLaneAI/pennylane/pull/7677)
  [(#7767)](https://github.com/PennyLaneAI/pennylane/pull/7767)
  [(#7690)](https://github.com/PennyLaneAI/pennylane/pull/7690)

<h4>Other improvements</h4>

* The error message raised when using Python compiler transforms with :func:`pennylane.qjit` has been updated
  with suggested fixes.
  [(#7916)](https://github.com/PennyLaneAI/pennylane/pull/7916)

* A new `qml.transforms.resolve_dynamic_wires` transform can allocate concrete wire values for dynamic
  qubit allocation.
  [(#7678)](https://github.com/PennyLaneAI/pennylane/pull/7678)


* The :func:`qml.workflow.set_shots` transform can now be directly applied to a QNode without the need for `functools.partial`, providing a more user-friendly syntax and negating having to import the `functools` package.
  [(#7876)](https://github.com/PennyLaneAI/pennylane/pull/7876)
  [(#7919)](https://github.com/PennyLaneAI/pennylane/pull/7919)

  ```python
  @qml.set_shots(shots=1000)  # or @qml.set_shots(1000)
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

* The `AllocQubitOp` and `DeallocQubitOp` operations have been added to the `Quantum` dialect in the
  Python compiler.
  [(#7915)](https://github.com/PennyLaneAI/pennylane/pull/7915)

* The :func:`pennylane.ops.rs_decomposition` method now performs exact decomposition and returns
  complete global phase information when used for decomposing a phase gate to Clifford+T basis.
  [(#7793)](https://github.com/PennyLaneAI/pennylane/pull/7793)

* `default.qubit` will default to the tree-traversal MCM method when `mcm_method="device"`.
  [(#7885)](https://github.com/PennyLaneAI/pennylane/pull/7885)

<h4>Resource-efficient decompositions 🔎</h4>

* With :func:`~.decomposition.enable_graph()`, dynamically allocated wires are now supported in decomposition rules. This provides a smoother overall experience when decomposing operators in a way that requires auxiliary/work wires.

  [(#7861)](https://github.com/PennyLaneAI/pennylane/pull/7861)
<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* Added state of the art resources for the `ResourceSelectPauliRot` template and the
  `ResourceQubitUnitary` templates.
  [(#7786)](https://github.com/PennyLaneAI/pennylane/pull/7786)

* Added state of the art resources for the `ResourceQFT` and `ResourceAQFT` templates.
  [(#7920)](https://github.com/PennyLaneAI/pennylane/pull/7920)

* The `catalyst` xDSL dialect has been added to the Python compiler, which contains data structures that support core compiler functionality.
  [(#7901)](https://github.com/PennyLaneAI/pennylane/pull/7901)

<h3>Breaking changes 💔</h3>

* Move custom exceptions into `exceptions.py` and add a documentation page for them in the internals.
  [(#7856)](https://github.com/PennyLaneAI/pennylane/pull/7856)

* The boolean functions provided in `qml.operation` are deprecated. See the 
  :doc:`deprecations page </development/deprecations>` for equivalent code to use instead. These 
  include `not_tape`, `has_gen`, `has_grad_method`, `has_multipar`, `has_nopar`, `has_unitary_gen`, 
  `is_measurement`, `defines_diagonalizing_gates`, and `gen_is_multi_term_hamiltonian`.
  [(#7924)](https://github.com/PennyLaneAI/pennylane/pull/7924)

* Removed access for `lie_closure`, `structure_constants` and `center` via `qml.pauli`.
  Top level import and usage is advised. The functions now live in the `liealg` module.

  ```python
  import pennylane.liealg
  from pennylane.liealg import lie_closure, structure_constants, center
  ```

  [(#7928)](https://github.com/PennyLaneAI/pennylane/pull/7928)

* `qml.operation.Observable` and the corresponding `Observable.compare` have been removed, as
  PennyLane now depends on the more general `Operator` interface instead. The
  `Operator.is_hermitian` property can instead be used to check whether or not it is highly likely
  that the operator instance is Hermitian.
  [(#7927)](https://github.com/PennyLaneAI/pennylane/pull/7927)

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

* Providing `num_steps` to `qml.evolve` and `Evolution` is deprecated and will be removed in a future version.
  Instead, use :class:`~.TrotterProduct` for approximate methods, providing the `n` parameter to perform the
  Suzuki-Trotter product approximation of a Hamiltonian with the specified number of Trotter steps.

  As a concrete example, consider the following case:

  ```python
  coeffs = [0.5, -0.6]
  ops = [qml.X(0), qml.X(0) @ qml.Y(1)]
  H_flat = qml.dot(coeffs, ops)
  ```

  Instead of computing the Suzuki-Trotter product approximation as:

  ```pycon
  >>> qml.evolve(H_flat, num_steps=2).decomposition()
  [RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1]),
  RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1])]
  ```

  The same result can be obtained using :class:`~.TrotterProduct` as follows:

  ```pycon
  >>> decomp_ops = qml.adjoint(qml.TrotterProduct(H_flat, time=1.0, n=2)).decomposition()
  >>> [simp_op for op in decomp_ops for simp_op in map(qml.simplify, op.decomposition())]
  [RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1]),
  RX(0.5, wires=[0]),
  PauliRot(-0.6, XY, wires=[0, 1])]
  ```
  [(#7954)](https://github.com/PennyLaneAI/pennylane/pull/7954)

* `MeasurementProcess.expand` is deprecated. The relevant method can be replaced with 
  `qml.tape.QuantumScript(mp.obs.diagonalizing_gates(), [type(mp)(eigvals=mp.obs.eigvals(), wires=mp.obs.wires)])`
  [(#7953)](https://github.com/PennyLaneAI/pennylane/pull/7953)

* `shots=` in `QNode` calls is deprecated and will be removed in v0.44.
  Instead, please use the `qml.workflow.set_shots` transform to set the number of shots for a QNode.
  [(#7906)](https://github.com/PennyLaneAI/pennylane/pull/7906)

* ``QuantumScript.shape`` and ``QuantumScript.numeric_type`` are deprecated and will be removed in version v0.44.
  Instead, the corresponding ``.shape`` or ``.numeric_type`` of the ``MeasurementProcess`` class should be used.
  [(#7950)](https://github.com/PennyLaneAI/pennylane/pull/7950)

* Some unnecessary methods of the `qml.CircuitGraph` class are deprecated and will be removed in version v0.44:
  [(#7904)](https://github.com/PennyLaneAI/pennylane/pull/7904)

    - `print_contents` in favor of `print(obj)`
    - `observables_in_order` in favor of `observables`
    - `operations_in_order` in favor of `operations`
    - `ancestors_in_order` in favor of `ancestors(obj, sort=True)`
    - `descendants_in_order` in favore of `descendants(obj, sort=True)`

* The `QuantumScript.to_openqasm` method is deprecated and will be removed in version v0.44.
  Instead, the `qml.to_openqasm` function should be used.
  [(#7909)](https://github.com/PennyLaneAI/pennylane/pull/7909)

* The `level=None` argument in the :func:`pennylane.workflow.get_transform_program`, :func:`pennylane.workflow.construct_batch`, `qml.draw`, `qml.draw_mpl`, and `qml.specs` transforms is deprecated and will be removed in v0.43.
  Please use `level='device'` instead to apply the noise model at the device level.
  [(#7886)](https://github.com/PennyLaneAI/pennylane/pull/7886)

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

* Update PennyLane's top-level `__init__.py` file imports to improve Python language server support for finding
  PennyLane submodules.
  [(#7959)](https://github.com/PennyLaneAI/pennylane/pull/7959)

* Adds `measurements` as a "core" module in the tach specification.
 [(#7945)](https://github.com/PennyLaneAI/pennylane/pull/7945)

* Improves type hints in the `measurements` module.
  [(#7938)](https://github.com/PennyLaneAI/pennylane/pull/7938)

* Refactored the codebase to adopt modern type hint syntax for Python 3.11+ language features.
  [(#7860)](https://github.com/PennyLaneAI/pennylane/pull/7860)

* Improve the pre-commit hook to add gitleaks.
  [(#7922)](https://github.com/PennyLaneAI/pennylane/pull/7922)

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

* The `TensorLike` type is now compatible with static type checkers.
  [(#7905)](https://github.com/PennyLaneAI/pennylane/pull/7905)

* Update xDSL supported version to `0.46`.
  [(#7923)](https://github.com/PennyLaneAI/pennylane/pull/7923)
  [(#7932)](https://github.com/PennyLaneAI/pennylane/pull/7932)

* Update JAX version used in tests to `0.6.2`
  [(#7925)](https://github.com/PennyLaneAI/pennylane/pull/7925)

* The measurement-plane attribute of the Python compiler `mbqc` dialect now uses the "opaque syntax"
  format when printing in the generic IR format. This enables usage of this attribute when IR needs
  to be passed from the python compiler to Catalyst.
  [(#7957)](https://github.com/PennyLaneAI/pennylane/pull/7957)

<h3>Documentation 📝</h3>

* The docstring of the `is_hermitian` operator property has been updated to better describe its behaviour.
  [(#7946)](https://github.com/PennyLaneAI/pennylane/pull/7946)

* Improved the docstrings of all optimizers for consistency and legibility.
  [(#7891)](https://github.com/PennyLaneAI/pennylane/pull/7891)

* Updated the code example in the documentation for :func:`~.transforms.split_non_commuting`.
  [(#7892)](https://github.com/PennyLaneAI/pennylane/pull/7892)

<h3>Bug fixes 🐛</h3>

* An error is now raised if an `end` statement is found in a measurement conditioned branch in a QASM string being imported into PennyLane.
  [(#7872)](https://github.com/PennyLaneAI/pennylane/pull/7872)

* Fixes issue related to :func:`~.transforms.to_zx` adding the support for
  `Toffoli` and `CCZ` gates conversion into their ZX-graph representation.
  [(#7899)](https://github.com/PennyLaneAI/pennylane/pull/7899)

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

* Fixes `SemiAdder` to work when inputs are defined with a single wire.
  [(#7940)](https://github.com/PennyLaneAI/pennylane/pull/7940)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso,
Utkarsh Azad,
Joey Carter,
Yushao Chen,
Marcus Edwards,
Simone Gasperini,
David Ittah,
Mehrdad Malekmohammadi
Erick Ochoa,
Mudit Pandey,
Andrija Paurevic,
Shuli Shu,
Jay Soni,
Jake Zaia
