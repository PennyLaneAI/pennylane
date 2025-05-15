:orphan:

# Release 0.42.0-dev (development release)

<h3>New features since last release</h3>

* A new function called `qml.to_openqasm` has been added, which allows for converting PennyLane circuits to OpenQASM 2.0 programs.
  [(#7393)](https://github.com/PennyLaneAI/pennylane/pull/7393)

  Consider this simple circuit in PennyLane:
  ```python
  dev = qml.device("default.qubit", wires=2, shots=100)

  @qml.qnode(dev)
  def circuit(theta, phi):
      qml.RX(theta, wires=0)
      qml.CNOT(wires=[0,1])
      qml.RZ(phi, wires=1)
      return qml.sample()
  ```

  This can be easily converted to OpenQASM 2.0 with `qml.to_openqasm`:
  ```pycon
  >>> openqasm_circ = qml.to_openqasm(circuit)(1.2, 0.9)
  >>> print(openqasm_circ)
  OPENQASM 2.0;
  include "qelib1.inc";
  qreg q[2];
  creg c[2];
  rx(1.2) q[0];
  cx q[0],q[1];
  rz(0.9) q[1];
  measure q[0] -> c[0];
  measure q[1] -> c[1];
  ```

* A new template called :class:`~.SelectPauliRot` that applies a sequence of uniformly controlled rotations to a target qubit 
  is now available. This operator appears frequently in unitary decomposition and block encoding techniques. 
  [(#7206)](https://github.com/PennyLaneAI/pennylane/pull/7206)

  ```python
  angles = np.array([1.0, 2.0, 3.0, 4.0])

  wires = qml.registers({"control": 2, "target": 1})
  dev = qml.device("default.qubit", wires=3)

  @qml.qnode(dev)
  def circuit():
      qml.SelectPauliRot(
        angles,
        control_wires=wires["control"],
        target_wire=wires["target"],
        rot_axis="Y")
      return qml.state()
  ```
  
  ```pycon
  >>> print(circuit())
  [0.87758256+0.j 0.47942554+0.j 0.        +0.j 0.        +0.j
   0.        +0.j 0.        +0.j 0.        +0.j 0.        +0.j]
  ```

* The transform `convert_to_mbqc_gateset` is added to the `ftqc` module to convert arbitrary 
  circuits to a limited gate-set that can be translated to the MBQC formalism.
  [(7271)](https://github.com/PennyLaneAI/pennylane/pull/7271)

* The `RotXZX` operation is added to the `ftqc` module to support definition of a universal
  gate-set that can be translated to the MBQC formalism.
  [(7271)](https://github.com/PennyLaneAI/pennylane/pull/7271)

* Two new functions called :func:`~.math.convert_to_su2` and :func:`~.math.convert_to_su4` have been added to `qml.math`, which convert unitary matrices to SU(2) or SU(4), respectively, and optionally a global phase.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

<h4>Resource-efficient Decompositions 🔎</h4>

* New decomposition rules comprising rotation gates and global phases have been added to `QubitUnitary` that 
  can be accessed with the new graph-based decomposition system. The most efficient set of rotations to 
  decompose into will be chosen based on the target gate set.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

  ```python
  from functools import partial
  import numpy as np
  import pennylane as qml
  
  qml.decomposition.enable_graph()
  
  U = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
  
  @partial(qml.transforms.decompose, gate_set={"RX", "RY", "GlobalPhase"})
  @qml.qnode(qml.device("default.qubit"))
  def circuit():
      qml.QubitUnitary(np.array([[1, 1], [1, -1]]) / np.sqrt(2), wires=[0])
      return qml.expval(qml.PauliZ(0))
  ```
  ```pycon
  >>> print(qml.draw(circuit)())
  0: ──RX(0.00)──RY(1.57)──RX(3.14)──GlobalPhase(-1.57)─┤  <Z>
  ```

* Decomposition rules can be marked as not-applicable with :class:`~.decomposition.DecompositionNotApplicable`, allowing for flexibility when creating conditional decomposition 
  rules based on parameters that affects the rule's resources.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

  ```python
  import pennylane as qml
  from pennylane.decomposition import DecompositionNotApplicable
  from pennylane.math.decomposition import zyz_rotation_angles
  
  def _zyz_resource(num_wires):
      if num_wires != 1:
          # This decomposition is only applicable when num_wires is 1
          raise DecompositionNotApplicable
      return {qml.RZ: 2, qml.RY: 1, qml.GlobalPhase: 1}

  @qml.register_resources(_zyz_resource)
  def zyz_decomposition(U, wires, **__):
      phi, theta, omega, phase = zyz_rotation_angles(U, return_global_phase=True)
      qml.RZ(phi, wires=wires[0])
      qml.RY(theta, wires=wires[0])
      qml.RZ(omega, wires=wires[0])
      qml.GlobalPhase(-phase)
  
  qml.add_decomps(QubitUnitary, zyz_decomposition)
  ```
  
  This decomposition will be ignored for `QubitUnitary` on more than one wire.

* The :func:`~.transforms.decompose` transform now supports symbolic operators (e.g., `Adjoint` and `Controlled`) specified as strings in the `gate_set` argument
  when the new graph-based decomposition system is enabled.
  [(#7331)](https://github.com/PennyLaneAI/pennylane/pull/7331)

  ```python
  from functools import partial
  import pennylane as qml
  
  qml.decomposition.enable_graph()
   
  @partial(qml.transforms.decompose, gate_set={"T", "Adjoint(T)", "H", "CNOT"})
  @qml.qnode(qml.device("default.qubit"))
  def circuit():
      qml.Toffoli(wires=[0, 1, 2])
  ```
  ```pycon
  >>> print(qml.draw(circuit)())
  0: ───────────╭●───────────╭●────╭●──T──╭●─┤  
  1: ────╭●─────│─────╭●─────│───T─╰X──T†─╰X─┤  
  2: ──H─╰X──T†─╰X──T─╰X──T†─╰X──T──H────────┤
  ```

<h3>Improvements 🛠</h3>

* PennyLane supports `JAX` version 0.6.0.
  [(#7299)](https://github.com/PennyLaneAI/pennylane/pull/7299)

* PennyLane supports `JAX` version 0.5.3.
  [(#6919)](https://github.com/PennyLaneAI/pennylane/pull/6919)

* Computing the angles for uniformly controlled rotations, used in :class:`~.MottonenStatePreparation`
  and :class:`~.SelectPauliRot`, now takes much less computational effort and memory.
  [(#7377)](https://github.com/PennyLaneAI/pennylane/pull/7377)

* An experimental quantum dialect written in [xDSL](https://xdsl.dev/index) has been introduced.
  This is similar to [Catalyst's MLIR dialects](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/dialects.html#mlir-dialects-in-catalyst), 
  but it is coded in Python instead of C++.
  [(#7357)](https://github.com/PennyLaneAI/pennylane/pull/7357)
  
* The :func:`~.transforms.cancel_inverses` transform no longer changes the order of operations that don't have shared wires, providing a deterministic output.
  [(#7328)](https://github.com/PennyLaneAI/pennylane/pull/7328)

* Alias for Identity (`I`) is now accessible from `qml.ops`.
  [(#7200)](https://github.com/PennyLaneAI/pennylane/pull/7200)

* The `ftqc` module `measure_arbitrary_basis`, `measure_x` and `measure_y` functions
  can now be captured when program capture is enabled.
  [(#7219)](https://github.com/PennyLaneAI/pennylane/pull/7219)
  [(#7368)](https://github.com/PennyLaneAI/pennylane/pull/7368)

* `Operator.num_wires` now defaults to `None` to indicate that the operator can be on
  any number of wires.
  [(#7312)](https://github.com/PennyLaneAI/pennylane/pull/7312)

* Shots can now be overridden for specific `qml.Snapshot` instances via a `shots` keyword argument.
  [(#7326)](https://github.com/PennyLaneAI/pennylane/pull/7326)

  ```python
  dev = qml.device("default.qubit", wires=2, shots=10)

  @qml.qnode(dev)
  def circuit():
      qml.Snapshot("sample", measurement=qml.sample(qml.X(0)), shots=5)
      return qml.sample(qml.X(0))
  ```

  ```pycon
  >>> qml.snapshots(circuit)()
  {'sample': array([-1., -1., -1., -1., -1.]),
   'execution_results': array([ 1., -1., -1., -1., -1.,  1., -1., -1.,  1., -1.])}
  ```

* Two-qubit `QubitUnitary` gates no longer decompose into fundamental rotation gates; it now 
  decomposes into single-qubit `QubitUnitary` gates. This allows the decomposition system to
  further decompose single-qubit unitary gates more flexibly using different rotations.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

* The `gate_set` argument of :func:`~.transforms.decompose` now accepts `"X"`, `"Y"`, `"Z"`, `"H"`, 
  `"I"` as aliases for `"PauliX"`, `"PauliY"`, `"PauliZ"`, `"Hadamard"`, and `"Identity"`. These 
  aliases are also recognized as part of symbolic operators. For example, `"Adjoint(H)"` is now 
  accepted as an alias for `"Adjoint(Hadamard)"`.
  [(#7331)](https://github.com/PennyLaneAI/pennylane/pull/7331)

* PennyLane no longer validates that an operation has at least one wire, as having this check required the abstract
  interface to maintain a list of special implementations.
  [(#7327)](https://github.com/PennyLaneAI/pennylane/pull/7327)

* Two new device-developer transforms have been added to `devices.preprocess`: 
  :func:`~.devices.preprocess.measurements_from_counts` and :func:`~.devices.preprocess.measurements_from_samples`.
  These transforms modify the tape to instead contain a `counts` or `sample` measurement process, 
  deriving the original measurements from the raw counts/samples in post-processing. This allows 
  expanded measurement support for devices that only 
  support counts/samples at execution, like real hardware devices.
  [(#7317)](https://github.com/PennyLaneAI/pennylane/pull/7317)

* Sphinx version was updated to 8.1. Sphinx is upgraded to version 8.1 and uses Python 3.10. References to intersphinx (e.g. `<demos/>` or `<catalyst/>` are updated to remove the :doc: prefix that is incompatible with sphinx 8.1. 
  [(7212)](https://github.com/PennyLaneAI/pennylane/pull/7212)

* Migrated `setup.py` package build and install to `pyproject.toml`
  [(#7375)](https://github.com/PennyLaneAI/pennylane/pull/7375)

* Updated GitHub Actions workflows (`rtd.yml`, `readthedocs.yml`, and `docs.yml`) to use `ubuntu-24.04` runners.
 [(#7396)](https://github.com/PennyLaneAI/pennylane/pull/7396)

* Updated requirements and pyproject files to include the other package.  
  [(#7417)](https://github.com/PennyLaneAI/pennylane/pull/7417)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* A :func:`parity_matrix <pennylane.labs.intermediate_reps.parity_matrix>` function is now available
  in :mod:`pennylane.labs.intermediate_reps <pennylane.labs.intermediate_reps>`.
  It allows computation of the parity matrix of a CNOT circuit; an efficient intermediate representation.
  It is important for CNOT routing algorithms and other quantum compilation routines.
  [(#7229)](https://github.com/PennyLaneAI/pennylane/pull/7229)


<h3>Breaking changes 💔</h3>

* The `return_type` property of `MeasurementProcess` has been removed. Please use `isinstance` for type checking instead.
  [(#7322)](https://github.com/PennyLaneAI/pennylane/pull/7322)

* The `KerasLayer` class in `qml.qnn.keras` has been removed because Keras 2 is no longer actively maintained.
  Please consider using a different machine learning framework, like `PyTorch <demos/tutorial_qnn_module_torch>`__ or `JAX <demos/tutorial_How_to_optimize_QML_model_using_JAX_and_Optax>`__.
  [(#7320)](https://github.com/PennyLaneAI/pennylane/pull/7320)

* The `qml.gradients.hamiltonian_grad` function has been removed because this gradient recipe is no
  longer required with the :doc:`new operator arithmetic system </news/new_opmath>`.
  [(#7302)](https://github.com/PennyLaneAI/pennylane/pull/7302)

* Accessing terms of a tensor product (e.g., `op = X(0) @ X(1)`) via `op.obs` has been removed.
  [(#7324)](https://github.com/PennyLaneAI/pennylane/pull/7324)

* The `mcm_method` keyword argument in `qml.execute` has been removed.
  [(#7301)](https://github.com/PennyLaneAI/pennylane/pull/7301)

* The `inner_transform` and `config` keyword arguments in `qml.execute` have been removed.
  [(#7300)](https://github.com/PennyLaneAI/pennylane/pull/7300)

* `Sum.ops`, `Sum.coeffs`, `Prod.ops` and `Prod.coeffs` have been removed.
  [(#7304)](https://github.com/PennyLaneAI/pennylane/pull/7304)

* Specifying `pipeline=None` with `qml.compile` has been removed.
  [(#7307)](https://github.com/PennyLaneAI/pennylane/pull/7307)

* The `control_wires` argument in `qml.ControlledQubitUnitary` has been removed.
  Furthermore, the `ControlledQubitUnitary` no longer accepts `QubitUnitary` objects as arguments as its `base`.
  [(#7305)](https://github.com/PennyLaneAI/pennylane/pull/7305)

* `qml.tape.TapeError` has been removed.
  [(#7205)](https://github.com/PennyLaneAI/pennylane/pull/7205)

<h3>Deprecations 👋</h3>

Here's a list of deprecations made this release. For a more detailed breakdown of deprecations and alternative code to use instead, Please consult the :doc:`deprecations and removals page </development/deprecations>`.

* `qml.operation.Observable` and the corresponding `Observable.compare` have been deprecated, as
  pennylane now depends on the more general `Operator` interface instead. The
  `Operator.is_hermitian` property can instead be used to check whether or not it is highly likely
  that the operator instance is Hermitian.
  [(#7316)](https://github.com/PennyLaneAI/pennylane/pull/7316)

* The boolean functions provided in `pennylane.operation` are deprecated. See the :doc:`deprecations page </development/deprecations>` 
  for equivalent code to use instead. These include `not_tape`, `has_gen`, `has_grad_method`, `has_multipar`,
  `has_nopar`, `has_unitary_gen`, `is_measurement`, `defines_diagonalizing_gates`, and `gen_is_multi_term_hamiltonian`.
  [(#7319)](https://github.com/PennyLaneAI/pennylane/pull/7319)

* `qml.operation.WiresEnum`, `qml.operation.AllWires`, and `qml.operation.AnyWires` are deprecated. To indicate that
  an operator can act on any number of wires, `Operator.num_wires = None` should be used instead. This is the default
  and does not need to be overwritten unless the operator developer wants to add wire number validation.
  [(#7313)](https://github.com/PennyLaneAI/pennylane/pull/7313)

* The :func:`qml.QNode.get_gradient_fn` method is now deprecated. Instead, use :func:`~.workflow.get_best_diff_method` to obtain the differentiation method.
  [(#7323)](https://github.com/PennyLaneAI/pennylane/pull/7323)

<h3>Internal changes ⚙️</h3>

* Enforce `noise` module to be a tertiary layer module.
  [(#7430)](https://github.com/PennyLaneAI/pennylane/pull/7430)

* Enforce `qaoa` module to be a tertiary layer module.
  [(#7429)](https://github.com/PennyLaneAI/pennylane/pull/7429)

* Enforce `gradients` module to be an auxiliary layer module.
  [(#7416)](https://github.com/PennyLaneAI/pennylane/pull/7416)

* Enforce `optimize` module to be an auxiliary layer module.
  [(#7418)](https://github.com/PennyLaneAI/pennylane/pull/7418)

* A `RuntimeWarning` raised when using versions of JAX > 0.4.28 has been removed.
  [(#7398)](https://github.com/PennyLaneAI/pennylane/pull/7398)

* Wheel releases for PennyLane now follow the `PyPA binary-distribution format <https://packaging.python.org/en/latest/specifications/binary-distribution-format/>_` guidelines more closely.
  [(#7382)](https://github.com/PennyLaneAI/pennylane/pull/7382)

* `null.qubit` can now support an optional `track_resources` argument which allows it to record which gates are executed.
  [(#7226)](https://github.com/PennyLaneAI/pennylane/pull/7226)
  [(#7372)](https://github.com/PennyLaneAI/pennylane/pull/7372)
  [(#7392)](https://github.com/PennyLaneAI/pennylane/pull/7392)

* A new internal module, `qml.concurrency`, is added to support internal use of multiprocess and multithreaded execution of workloads. This also migrates the use of `concurrent.futures` in `default.qubit` to this new design.
  [(#7303)](https://github.com/PennyLaneAI/pennylane/pull/7303)

* Test suites in `tests/transforms/test_defer_measurement.py` use analytic mocker devices to test numeric results.
  [(#7329)](https://github.com/PennyLaneAI/pennylane/pull/7329)

* Introduce module dependency management using `tach`.
  [(#7185)](https://github.com/PennyLaneAI/pennylane/pull/7185)

* Add new `pennylane.exceptions` module for custom errors and warnings.
  [(#7205)](https://github.com/PennyLaneAI/pennylane/pull/7205)

* Clean up `__init__.py` files in `math`, `ops`, `qaoa`, `tape` and `templates` to be explicit in what they import. 
  [(#7200)](https://github.com/PennyLaneAI/pennylane/pull/7200)
  
* The `Tracker` class has been moved into the `devices` module.
  [(#7281)](https://github.com/PennyLaneAI/pennylane/pull/7281)

* Moved functions that calculate rotation angles for unitary decompositions into an internal
  module `qml.math.decomposition`
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

<h3>Documentation 📝</h3>

* Fixed the wrong `theta` to `phi` in :class:`~pennylane.IsingXY`.
 [(#7427)](https://github.com/PennyLaneAI/pennylane/pull/7427)

* In the :doc:`/introduction/compiling_circuits` page, in the "Decomposition in stages" section,
  circuit drawings now render in a way that's easier to read.
  [(#7419)](https://github.com/PennyLaneAI/pennylane/pull/7419)

* The entry in the :doc:`/news/program_capture_sharp_bits` page for using program capture with Catalyst 
  has been updated. Instead of using ``qjit(experimental_capture=True)``, Catalyst is now compatible 
  with the global toggles ``qml.capture.enable()`` and ``qml.capture.disable()`` for enabling and
  disabling program capture.
  [(#7298)](https://github.com/PennyLaneAI/pennylane/pull/7298)

<h3>Bug fixes 🐛</h3>

* Fixed a bug in `to_openfermion` where identity qubit-to-wires mapping was not obeyed.
  [(#7332)](https://github.com/PennyLaneAI/pennylane/pull/7332)

* Fixed a bug in the validation of :class:`~.SelectPauliRot` that prevents parameter broadcasting.
  [(#7377)](https://github.com/PennyLaneAI/pennylane/pull/7377)

* Usage of NumPy in `default.mixed` source code has been converted to `qml.math` to avoid
  unnecessary dependency on NumPy and to fix a bug that caused an error when using `default.mixed` with PyTorch and GPUs.
  [(#7384)](https://github.com/PennyLaneAI/pennylane/pull/7384)

* With program capture enabled (`qml.capture.enable()`), `QSVT` no treats abstract values as metadata.
  [(#7360)](https://github.com/PennyLaneAI/pennylane/pull/7360)

* A fix was made to `default.qubit` to allow for using `qml.Snapshot` with defer-measurements (`mcm_method="deferred"`).
  [(#7335)](https://github.com/PennyLaneAI/pennylane/pull/7335)

* Fixes the repr for empty `Prod` and `Sum` instances to better communicate the existence of an empty instance.
  [(#7346)](https://github.com/PennyLaneAI/pennylane/pull/7346)

* Fixes a bug where circuit execution fails with ``BlockEncode`` initialized with sparse matrices.
  [(#7285)](https://github.com/PennyLaneAI/pennylane/pull/7285)

* Adds an informative error if `qml.cond` is used with an abstract condition with
  jitting on `default.qubit` if capture is enabled.
  [(#7314)](https://github.com/PennyLaneAI/pennylane/pull/7314)

* Fixes a bug where using a ``StatePrep`` operation with `batch_size=1` did not work with ``default.mixed``.
  [(#7280)](https://github.com/PennyLaneAI/pennylane/pull/7280)

* Gradient transforms can now be used in conjunction with batch transforms with all interfaces.
  [(#7287)](https://github.com/PennyLaneAI/pennylane/pull/7287)

* Fixes a bug where the global phase was not being added in the ``QubitUnitary`` decomposition.  
  [(#7244)](https://github.com/PennyLaneAI/pennylane/pull/7244)
  [(#7270)](https://github.com/PennyLaneAI/pennylane/pull/7270)

* Using finite differences with program capture without x64 mode enabled now raises a warning.
  [(#7282)](https://github.com/PennyLaneAI/pennylane/pull/7282)

* When the `mcm_method` is specified to the `"device"`, the `defer_measurements` transform will 
  no longer be applied. Instead, the device will be responsible for all MCM handling.
  [(#7243)](https://github.com/PennyLaneAI/pennylane/pull/7243)

* Fixed coverage of `qml.liealg.CII` and `qml.liealg.AIII`.
  [(#7291)](https://github.com/PennyLaneAI/pennylane/pull/7291)

* Fixed a bug where the phase is used as the wire label for a `qml.GlobalPhase` when capture is enabled.
  [(#7211)](https://github.com/PennyLaneAI/pennylane/pull/7211)

* Fixed a bug that caused `CountsMP.process_counts` to return results in the computational basis, even if
  an observable was specified.
  [(#7342)](https://github.com/PennyLaneAI/pennylane/pull/7342)

* Fixed a bug that caused `SamplesMP.process_counts` used with an observable to return a list of eigenvalues 
  for each individual operation in the observable, instead of the overall result.
  [(#7342)](https://github.com/PennyLaneAI/pennylane/pull/7342)

* Fixed a bug where `two_qubit_decomposition` provides an incorrect decomposition for some special matrices.
  [(#7340)](https://github.com/PennyLaneAI/pennylane/pull/7340)

* Fixes a bug where the powers of `qml.ISWAP` and `qml.SISWAP` were decomposed incorrectly.
  [(#7361)](https://github.com/PennyLaneAI/pennylane/pull/7361)

* Returning `MeasurementValue`s from the `ftqc` module's parametric mid-circuit measurements
  (`measure_arbitrary_basis`, `measure_x` and `measure_y`) no longer raises an error in circuits 
  using `diagonalize_mcms`.
  [(#7387)](https://github.com/PennyLaneAI/pennylane/pull/7387)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso-Linaje,
Astral Cai,
Yushao Chen,
Lillian Frederiksen,
Pietropaolo Frisoni,
Simone Gasperini,
Korbinian Kottmann,
Christina Lee,
Anton Naim Ibrahim,
Lee J. O'Riordan,
Mudit Pandey,
Andrija Paurevic,
Kalman Szenes,
David Wierichs,
Jake Zaia
