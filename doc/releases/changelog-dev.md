
# Release 0.43.0-dev (development release)

<h3>New features since last release</h3>

* The `qml.specs` function now accepts a `compute_depth` keyword argument, which is set to `True` by default.
  This makes the expensive depth computation performed by `qml.specs` optional.
  [(#7998)](https://github.com/PennyLaneAI/pennylane/pull/7998)
  [(#8042)](https://github.com/PennyLaneAI/pennylane/pull/8042)

* New transforms called :func:`~.transforms.match_relative_phase_toffoli` and 
  :func:`~.transforms.match_controlled_iX_gate` have been added to implement passes that make use
  of equivalencies to compile certain patterns to efficient Clifford+T equivalents.
  [(#7748)](https://github.com/PennyLaneAI/pennylane/pull/7748)

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
  [(#7908)](https://github.com/PennyLaneAI/pennylane/pull/7908)
  [(#7385)](https://github.com/PennyLaneAI/pennylane/pull/7385)
  [(#7941)](https://github.com/PennyLaneAI/pennylane/pull/7941)
  
  The included templates are:

  * :class:`~.Adder`

  * :class:`~.ControlledSequence`

  * :class:`~.ModExp`

  * :class:`~.MottonenStatePreparation`

  * :class:`~.MPSPrep`

  * :class:`~.Multiplier`

  * :class:`~.OutAdder`

  * :class:`~.OutMultiplier`

  * :class:`~.OutPoly`

  * :class:`~.PrepSelPrep`

  * :class:`~.ops.Prod`

  * :class:`~.Reflection`

  * :class:`~.Select`

  * :class:`~.StatePrep`

  * :class:`~.TrotterProduct`

  * :class:`~.QROM`

* A new function called :func:`~.math.choi_matrix` is available, which computes the [Choi matrix](https://en.wikipedia.org/wiki/Choi%E2%80%93Jamio%C5%82kowski_isomorphism) of a quantum channel.
  This is a useful tool in quantum information science and to check circuit identities involving non-unitary operations.
  [(#7951)](https://github.com/PennyLaneAI/pennylane/pull/7951)

  ```pycon
  >>> import numpy as np
  >>> Ks = [np.sqrt(0.3) * qml.CNOT((0, 1)), np.sqrt(1-0.3) * qml.X(0)]
  >>> Ks = [qml.matrix(op, wire_order=range(2)) for op in Ks]
  >>> Lambda = qml.math.choi_matrix(Ks)
  >>> np.trace(Lambda), np.trace(Lambda @ Lambda)
  (np.float64(1.0), np.float64(0.58))
  ```

* A new device preprocess transform, `~.devices.preprocess.no_analytic`, is available for hardware devices and hardware-like simulators.
  It validates that all executions are shot-based.
  [(#8037)](https://github.com/PennyLaneAI/pennylane/pull/8037)

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

* PennyLane is now compatible with `quimb` 1.11.2 after a bug affecting `default.tensor` was fixed.
  [(#7931)](https://github.com/PennyLaneAI/pennylane/pull/7931)

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

* The :func:`~.clifford_t_decomposition` transform can now handle circuits with mid-circuit
  measurements including Catalyst's measurements operations. It also now handles `RZ` and `PhaseShift`
  operations where angles are odd multiples of `±pi/4` more efficiently while using `method="gridsynth"`.
  [(#7793)](https://github.com/PennyLaneAI/pennylane/pull/7793)
  [(#7942)](https://github.com/PennyLaneAI/pennylane/pull/7942)

* The default implementation of `Device.setup_execution_config` now choses `"device"` as the default mcm method if it is available as specified by the device TOML file.
  [(#7968)](https://github.com/PennyLaneAI/pennylane/pull/7968)

<h4>Resource-efficient decompositions 🔎</h4>

* With :func:`~.decomposition.enable_graph()`, dynamically allocated wires are now supported in decomposition rules. This provides a smoother overall experience when decomposing operators in a way that requires auxiliary/work wires.

  [(#7861)](https://github.com/PennyLaneAI/pennylane/pull/7861)
<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* Added state of the art resources for the `ResourceSelectPauliRot` template and the
  `ResourceQubitUnitary` templates.
  [(#7786)](https://github.com/PennyLaneAI/pennylane/pull/7786)

* Added state of the art resources for the `ResourceSingleQubitCompare`, `ResourceTwoQubitCompare`,
  `ResourceIntegerComparator` and `ResourceRegisterComparator` templates.
  [(#7857)](https://github.com/PennyLaneAI/pennylane/pull/7857)

* Added state of the art resources for the `ResourceUniformStatePrep`,
  and `ResourceAliasSampling` templates.
  [(#7883)](https://github.com/PennyLaneAI/pennylane/pull/7883)

* Added state of the art resources for the `ResourceQFT` and `ResourceAQFT` templates.
  [(#7920)](https://github.com/PennyLaneAI/pennylane/pull/7920)

* Added an internal `dequeue()` method to the `ResourceOperator` class to simplify the 
  instantiation of resource operators which require resource operators as input.
  [(#7974)](https://github.com/PennyLaneAI/pennylane/pull/7974)

* The `catalyst` xDSL dialect has been added to the Python compiler, which contains data structures that support core compiler functionality.
  [(#7901)](https://github.com/PennyLaneAI/pennylane/pull/7901)

<h3>Breaking changes 💔</h3>

* `ExecutionConfig` and `MCMConfig` from `pennylane.devices` are now frozen dataclasses whose fields should be updated with `dataclass.replace`. 
  [(#7697)](https://github.com/PennyLaneAI/pennylane/pull/7697)

* Functions involving an execution configuration will now default to `None` instead of `pennylane.devices.DefaultExecutionConfig` and have to be handled accordingly. 
  This prevents the potential mutation of a global object. 

  This means that functions like,
  ```python
  ...
    def some_func(..., execution_config = DefaultExecutionConfig):
      ...
  ...
  ```
  should be written as follows,
  ```python
  ...
    def some_func(..., execution_config: ExecutionConfig | None = None):
      if execution_config is None:
          execution_config = ExecutionConfig()
  ...
  ```

  [(#7697)](https://github.com/PennyLaneAI/pennylane/pull/7697)

* The `qml.HilbertSchmidt` and `qml.LocalHilbertSchmidt` templates have been updated and their UI has been remarkably simplified. 
  They now accept an operation or a list of operations as quantum unitaries.
  [(#7933)](https://github.com/PennyLaneAI/pennylane/pull/7933)

  In past versions of PennyLane, these templates required providing the `U` and `V` unitaries as a `qml.tape.QuantumTape` and a quantum function,
  respectively, along with separate parameters and wires.

  With this release, each template has been improved to accept one or more operators as  unitaries. 
  The wires and parameters of the approximate unitary `V` are inferred from the inputs, according to the order provided.

  ```python
  >>> U = qml.Hadamard(0)
  >>> V = qml.RZ(0.1, wires=1)
  >>> qml.HilbertSchmidt(V, U)
  HilbertSchmidt(0.1, wires=[0, 1])
  ```

* Remove support for Python 3.10 and adds support for 3.13.
  [(#7935)](https://github.com/PennyLaneAI/pennylane/pull/7935)

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
  [(#7994)](https://github.com/PennyLaneAI/pennylane/pull/7994)

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

* Specifying the ``work_wire_type`` argument in ``qml.ctrl`` and other controlled operators as ``"clean"`` or 
  ``"dirty"`` is deprecated. Use ``"zeroed"`` to indicate that the work wires are initially in the :math:`|0\rangle`
  state, and ``"borrowed"`` to indicate that the work wires can be in any arbitrary state. In both cases, the
  work wires are restored to their original state upon completing the decomposition.
  [(#7993)](https://github.com/PennyLaneAI/pennylane/pull/7993)

* Providing `num_steps` to :func:`pennylane.evolve`, :func:`pennylane.exp`, :class:`pennylane.ops.Evolution`,
  and :class:`pennylane.ops.Exp` is deprecated and will be removed in a future release. Instead, use
  :class:`~.TrotterProduct` for approximate methods, providing the `n` parameter to perform the Suzuki-Trotter
  product approximation of a Hamiltonian with the specified number of Trotter steps.

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
  [(#7977)](https://github.com/PennyLaneAI/pennylane/pull/7977)

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

* Removed unnecessary execution tests along with accuracy validation in `tests/ops/functions/test_map_wires.py`.
  [(#8032)](https://github.com/PennyLaneAI/pennylane/pull/8032)

* Added a new `all-tests-passed` gatekeeper job to `interface-unit-tests.yml` to ensure all test
  jobs complete successfully before triggering downstream actions. This reduces the need to
  maintain a long list of required checks in GitHub settings. Also added the previously missing
  `capture-jax-tests` job to the list of required test jobs, ensuring this test suite is properly
  enforced in CI.
  [(#7996)](https://github.com/PennyLaneAI/pennylane/pull/7996)

* Equipped `DefaultQubitLegacy` (test suite only) with seeded sampling.
  This allows for reproducible sampling results of legacy classical shadow across CI.
  [(#7903)](https://github.com/PennyLaneAI/pennylane/pull/7903)

* Capture does not block `wires=0` anymore. This allows Catalyst to work with zero-wire devices.
  Note that `wires=None` is still illegal.
  [(#7978)](https://github.com/PennyLaneAI/pennylane/pull/7978)

* Improves readability of `dynamic_one_shot` postprocessing to allow further modification.
  [(#7962)](https://github.com/PennyLaneAI/pennylane/pull/7962)
  [(#8041)](https://github.com/PennyLaneAI/pennylane/pull/8041)

* Update PennyLane's top-level `__init__.py` file imports to improve Python language server support for finding
  PennyLane submodules.
  [(#7959)](https://github.com/PennyLaneAI/pennylane/pull/7959)

* Adds `measurements` as a "core" module in the tach specification.
  [(#7945)](https://github.com/PennyLaneAI/pennylane/pull/7945)

* Improves type hints in the `measurements` module.
  [(#7938)](https://github.com/PennyLaneAI/pennylane/pull/7938)

* Refactored the codebase to adopt modern type hint syntax for Python 3.11+ language features.
  [(#7860)](https://github.com/PennyLaneAI/pennylane/pull/7860)
  [(#7982)](https://github.com/PennyLaneAI/pennylane/pull/7982)

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

* Updated the code examples in the documentation of :func:`~.specs`.
  [(#8003)](https://github.com/PennyLaneAI/pennylane/pull/8003)

* Clarifies the use case for `Operator.pow` and `Operator.adjoint`.
  [(#7999)](https://github.com/PennyLaneAI/pennylane/pull/7999)

* The docstring of the `is_hermitian` operator property has been updated to better describe its behaviour.
  [(#7946)](https://github.com/PennyLaneAI/pennylane/pull/7946)

* Improved the docstrings of all optimizers for consistency and legibility.
  [(#7891)](https://github.com/PennyLaneAI/pennylane/pull/7891)

* Updated the code example in the documentation for :func:`~.transforms.split_non_commuting`.
  [(#7892)](https://github.com/PennyLaneAI/pennylane/pull/7892)

* Fixed :math:`\LaTeX` rendering in the documentation for `qml.TrotterProduct` and `qml.trotterize`.
  [(#8014)](https://github.com/PennyLaneAI/pennylane/pull/8014)

<h3>Bug fixes 🐛</h3>

* Fixes the GPU selection issue in `qml.math` with PyTorch when multiple GPUs are present.
  [(#8008)](https://github.com/PennyLaneAI/pennylane/pull/8008)

* The `~.for_loop` function with capture enabled can now handle over indexing
  into an empty array when `start == stop`.
  [(#8026)](https://github.com/PennyLaneAI/pennylane/pull/8026)

* Plxpr primitives now only return dynamically shaped arrays if their outputs
  actually have dynamic shapes.
  [(#8004)](https://github.com/PennyLaneAI/pennylane/pull/8004)

* Fixes an issue with tree-traversal and non-sequential wire orders.
  [(#7991)](https://github.com/PennyLaneAI/pennylane/pull/7991)

* Fixes a bug in :func:`~.matrix` where an operator's
  constituents were incorrectly queued if its decomposition was requested.
  [(#7975)](https://github.com/PennyLaneAI/pennylane/pull/7975)

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
Ali Asadi,
Utkarsh Azad,
Joey Carter,
Yushao Chen,
Diksha Dhawan,
Marcus Edwards,
Lillian Frederiksen,
Pietropaolo Frisoni,
Simone Gasperini,
David Ittah,
Korbinian Kottmann,
Mehrdad Malekmohammadi
Erick Ochoa,
Mudit Pandey,
Andrija Paurevic,
Alex Preciado,
Shuli Shu,
Jay Soni,
David Wierichs,
Jake Zaia
