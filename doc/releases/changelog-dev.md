:orphan:

# Release 0.32.0-dev (development release)

<h3>New features since last release</h3>

<h4>Encode matrices using a linear combination of unitaries ⛓️️</h4>

* A new operation called `qml.Select` is available. It applies specific input operations depending on the
  state of the designated control qubits.
  [(#4431)](https://github.com/PennyLaneAI/pennylane/pull/4431)

  ```pycon
  >>> dev = qml.device('default.qubit',wires=4)
  >>> ops = [qml.PauliX(wires=2),qml.PauliX(wires=3),qml.PauliY(wires=2),qml.SWAP([2,3])]
  >>> @qml.qnode(dev)
  >>> def circuit():
  >>>     qml.Select(ops,control_wires=[0,1])
  >>>     return qml.state()
  ...
  >>> print(qml.draw(circuit,expansion_strategy='device')())
  0: ─╭○─╭○─╭●─╭●────┤  State
  1: ─├○─├●─├○─├●────┤  State
  2: ─╰X─│──╰Y─├SWAP─┤  State
  3: ────╰X────╰SWAP─┤  State
  ```

* `qml.pauli_decompose` is now exponentially faster and differentiable.
  [(#4395)](https://github.com/PennyLaneAI/pennylane/pull/4395)
  [(#4479)](https://github.com/PennyLaneAI/pennylane/pull/4479)

<h4>Reset and reuse qubits after mid-circuit measurements ♻️</h4>

* `qml.measure` now includes a boolean keyword argument `reset` to reset a wire to the
  $|0\rangle$ computational basis state after measurement.
  [(#4402)](https://github.com/PennyLaneAI/pennylane/pull/4402/)

  TODO: CODE EXAMPLE

<h4>Monitor PennyLane's inner workings with Logging 📃</h4>

* Python-native logging can now be enabled with `qml.logging.enable_logging()`.
  [(#4383)](https://github.com/PennyLaneAI/pennylane/pull/4383)

  TODO: CODE EXAMPLE

* Provide users access to the logging configuration file path and improve the logging configuration structure.
  [(#4377)](https://github.com/PennyLaneAI/pennylane/pull/4377)

<h4>More input states for quantum chemistry calculations ⚛️</h4>

* Functions are available to obtain a state vector from PySCF solver objects.
  [(#4427)](https://github.com/PennyLaneAI/pennylane/pull/4427)
  [(#4433)](https://github.com/PennyLaneAI/pennylane/pull/4433)

  The `qml.qchem.import_state` function can be used to import a PySCF solver object and return the
  corresponding state vector.

  ```pycon
  >>> from pyscf import gto, scf, ci
  >>> mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0,0,0.71)]], basis='sto6g')
  >>> myhf = scf.UHF(mol).run()
  >>> myci = ci.UCISD(myhf).run()
  >>> wf_cisd = qml.qchem.import_state(myci, tol=1e-1)
  >>> print(wf_cisd)
  [ 0.        +0.j  0.        +0.j  0.        +0.j  0.1066467 +0.j
    1.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
    2.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
   -0.99429698+0.j  0.        +0.j  0.        +0.j  0.        +0.j]
  ```

  The currently supported objects are RCISD, UCISD, RCCSD, and UCCSD which correspond to 
  restricted (R) and unrestricted (U) configuration interaction (CI) and coupled cluster (CC) 
  calculations with single and double (SD) excitations.

<h3>Improvements 🛠</h3>

<h4>Making operators immutable and PyTrees</h4>

* Any class inheriting from `Operator` is now automatically registered as a pytree with JAX.
  This unlocks the ability to jit functions of `Operator`.
  [(#4458)](https://github.com/PennyLaneAI/pennylane/pull/4458/)

  ```pycon
  >>> op = qml.adjoint(qml.RX(1.0, wires=0))
  >>> jax.jit(qml.matrix)(op)
  Array([[0.87758255-0.j        , 0.        +0.47942555j],
       [0.        +0.47942555j, 0.87758255-0.j        ]],      dtype=complex64, weak_type=True)
  >>> jax.tree_util.tree_map(lambda x: x+1, op)
  Adjoint(RX(2.0, wires=[0]))
  ```

* All `Operator` objects now define `Operator._flatten` and `Operator._unflatten` methods that separate
  trainable from untrainable components. These methods will be used in serialization and pytree registration.
  Custom operations may need an update to ensure compatibility with new PennyLane features.
  [(#4483)](https://github.com/PennyLaneAI/pennylane/pull/4483)
  [(#4314)](https://github.com/PennyLaneAI/pennylane/pull/4314)

* The `QuantumScript` class now has a `bind_new_parameters` method that allows creation of
  new `QuantumScript` objects with the provided parameters.
  [(#4345)](https://github.com/PennyLaneAI/pennylane/pull/4345)

* The `qml.gradients` module no longer mutates operators in-place for any gradient transforms.
  Instead, operators that need to be mutated are copied with new parameters.
  [(#4220)](https://github.com/PennyLaneAI/pennylane/pull/4220)

* PennyLane no longer directly relies on `Operator.__eq__`.
  [(#4398)](https://github.com/PennyLaneAI/pennylane/pull/4398)

* `qml.equal` no longer raises errors when operators or measurements of different types are compared.
  Instead, it returns `False`.
  [(#4315)](https://github.com/PennyLaneAI/pennylane/pull/4315)

<h4>Transforms</h4>

* Transform programs are now integrated with the QNode.
  [(#4404)](https://github.com/PennyLaneAI/pennylane/pull/4404)

  ```python
  def null_postprocessing(results: qml.typing.ResultBatch) -> qml.typing.Result:
      return results[0]

  @qml.transforms.core.transform
  def scale_shots(tape: qml.tape.QuantumTape, shot_scaling) -> (Tuple[qml.tape.QuantumTape], Callable):
      new_shots = tape.shots.total_shots * shot_scaling
      new_tape = qml.tape.QuantumScript(tape.operations, tape.measurements, shots=new_shots)
      return (new_tape, ), null_postprocessing

  dev = qml.devices.experimental.DefaultQubit2()

  @partial(scale_shots, shot_scaling=2)
  @qml.qnode(dev, interface=None)
  def circuit():
      return qml.sample(wires=0)
  ```

  ```pycon
  >>> circuit(shots=1)
  array([False, False])
  ```

* Transform Programs, `qml.transforms.core.TransformProgram`, can now be called on a batch of circuits
  and return a new batch of circuits and a single post processing function.
  [(#4364)](https://github.com/PennyLaneAI/pennylane/pull/4364)

* `TransformDispatcher` now allows registration of custom `QNode` transforms.
  [(#4466)](https://github.com/PennyLaneAI/pennylane/pull/4466)

* QNode transforms in `qml.qinfo` now support custom wire labels.
  [#4331](https://github.com/PennyLaneAI/pennylane/pull/4331)

* `qml.transforms.adjoint_metric_tensor` now uses the simulation tools in `qml.devices.qubit` instead of
  private methods of `qml.devices.DefaultQubit`.
  [(#4456)](https://github.com/PennyLaneAI/pennylane/pull/4456)

* Auxiliary wires and device wires are now treated the same way in `transforms.metric_tensor`
  as in `gradients.hadamard_grad`. All valid wire input formats for `aux_wire` are supported.
  [(#4328)](https://github.com/PennyLaneAI/pennylane/pull/4328)

<h4>Next-generation device API</h4>

* The experimental device interface has been integrated with the QNode for JAX, JAX-JIT, TensorFlow and PyTorch.
  [(#4323)](https://github.com/PennyLaneAI/pennylane/pull/4323)
  [(#4352)](https://github.com/PennyLaneAI/pennylane/pull/4352)
  [(#4392)](https://github.com/PennyLaneAI/pennylane/pull/4392)
  [(#4393)](https://github.com/PennyLaneAI/pennylane/pull/4393)

* The experimental `DefaultQubit2` device now supports computing VJPs and JVPs using the adjoint method.
  [(#4374)](https://github.com/PennyLaneAI/pennylane/pull/4374)

* New functions called `adjoint_jvp` and `adjoint_vjp` that compute the JVP and VJP of a tape using the adjoint method 
  have been added to `qml.devices.qubit.adjoint_jacobian` 
  [(#4358)](https://github.com/PennyLaneAI/pennylane/pull/4358)

* `DefaultQubit2` now accepts a `max_workers` argument which controls multiprocessing.
  A `ProcessPoolExecutor` executes tapes asynchronously
  using a pool of at most `max_workers` processes. If `max_workers` is `None`
  or not given, only the current process executes tapes. If you experience any
  issue, say using JAX, TensorFlow, Torch, try setting `max_workers` to `None`.
  [(#4319)](https://github.com/PennyLaneAI/pennylane/pull/4319)
  [(#4425)](https://github.com/PennyLaneAI/pennylane/pull/4425)

* `qml.devices.experimental.Device` now accepts a shots keyword argument and has a `shots`
  property. This property is only used to set defaults for a workflow, and does not directly
  influence the number of shots used in executions or derivatives.
  [(#4388)](https://github.com/PennyLaneAI/pennylane/pull/4388)

* `expand_fn()` for `DefaultQubit2` has been updated to decompose `StatePrep` operations present in the middle of a circuit.
  [(#4444)](https://github.com/PennyLaneAI/pennylane/pull/4444)

* If no seed is specified on initialization with `DefaultQubit2`, the local random number generator will be
  seeded from NumPy's global random number generator.
  [(#4394)](https://github.com/PennyLaneAI/pennylane/pull/4394)

<h4>Improvements to machine learning library interfaces</h4>

* `pennylane/interfaces` has been refactored. The `execute_fn` passed to the machine learning framework boundaries 
  is now responsible for converting parameters to NumPy. The gradients module can now handle TensorFlow parameters,
  but gradient tapes now retain the original `dtype` instead of converting to `float64`.  This may cause instability 
  with finite-difference differentiation and `float32` parameters. The machine learning boundary functions are now uncoupled from their legacy counterparts.
  [(#4415)](https://github.com/PennyLaneAI/pennylane/pull/4415)

* `qml.interfaces.set_shots` now accepts a `Shots` object as well as `int`'s and tuples of `int`'s.
  [(#4388)](https://github.com/PennyLaneAI/pennylane/pull/4388)

* Readability improvements and stylistic changes have been made to `pennylane/interfaces/jax_jit_tuple.py`
  [(#4379)](https://github.com/PennyLaneAI/pennylane/pull/4379/)

<h4>Pulses</h4>

* A `HardwareHamiltonian` can now be summed with `int` or `float`.
  A sequence of `HardwareHamiltonian`s can now be summed via the builtin `sum`.
  [(#4343)](https://github.com/PennyLaneAI/pennylane/pull/4343)

* `transmon_drive` has been updated in accordance with [1904.06560](https://arxiv.org/abs/1904.06560). In particular, the functional form has been changed from $\Omega(t)(\cos(\omega_d t + \phi) X - \sin(\omega_d t + \phi) Y)$ to $\Omega(t) \sin(\omega_d t + \phi) Y$.
  [(#4418)](https://github.com/PennyLaneAI/pennylane/pull/4418/)
  [(#4465)](https://github.com/PennyLaneAI/pennylane/pull/4465/)
  [(#4478)](https://github.com/PennyLaneAI/pennylane/pull/4478/)

<h4>Other improvements</h4>

* The `qchem` module has been upgraded to use the fermionic operators of the `fermi` module.
  [#4336](https://github.com/PennyLaneAI/pennylane/pull/4336)

* `draw_mpl` now accepts `style='pennylane'` to draw PennyLane-style circuit diagrams, and `style.use` in `matplotlib.pyplot` accepts `qml.drawer.plot` to create PennyLane-style plots. If the font Quicksand Bold isn't available, an available default font is used instead. 
  [(#3950)](https://github.com/PennyLaneAI/pennylane/pull/3950)

* The calculation of `PauliWord` and `PauliSentence` sparse matrices are orders of magnitude faster.
  [(#4272)](https://github.com/PennyLaneAI/pennylane/pull/4272)
  [($4411)](https://github.com/PennyLaneAI/pennylane/pull/4411)

* A function called `qml.math.fidelity_statevector` that computes the fidelity between two state vectors has been added.
  [(#4322)](https://github.com/PennyLaneAI/pennylane/pull/4322)

* `qml.ctrl(qml.PauliX)` returns a `CNOT`, `Toffoli`, or `MultiControlledX` operation instead of `Controlled(PauliX)`.
  [(#4339)](https://github.com/PennyLaneAI/pennylane/pull/4339)

* When given a callable, `qml.ctrl` now does its custom pre-processing on all queued operators from the callable.
  [(#4370)](https://github.com/PennyLaneAI/pennylane/pull/4370)

* The `qchem` functions `primitive_norm` and `contracted_norm` have been modified to be compatible with
  higher versions of SciPy. The private function `_fac2` for computing double factorials has also been added.
  [#4321](https://github.com/PennyLaneAI/pennylane/pull/4321)

* `tape_expand` now uses `Operator.decomposition` instead of `Operator.expand` in order to make
  more performant choices.
  [(#4355)](https://github.com/PennyLaneAI/pennylane/pull/4355)

* CI now runs tests with TensorFlow 2.13.0
  [(#4472)](https://github.com/PennyLaneAI/pennylane/pull/4472)

* All tests in CI and pre-commit hooks now enable linting.
  [(#4335)](https://github.com/PennyLaneAI/pennylane/pull/4335)

* The default label for a `StatePrep` operator is now `|Ψ⟩`.
  [(#4340)](https://github.com/PennyLaneAI/pennylane/pull/4340)

* `Device.default_expand_fn()` has been updated to decompose `StatePrep` operations present in the middle of a provided circuit.
  [(#4437)](https://github.com/PennyLaneAI/pennylane/pull/4437)

<h3>Breaking changes 💔</h3>

* Applying gradient transforms to broadcasted/batched tapes has been deactivated until it is consistently
  supported for QNodes as well.
  [(#4480)](https://github.com/PennyLaneAI/pennylane/pull/4480)

* Gradient transforms no longer implicitly cast `float32` parameters to `float64`. Finite difference differentiation
  with `float32` parameters may no longer give accurate results.
  [(#4415)](https://github.com/PennyLaneAI/pennylane/pull/4415)

* Support for Python 3.8 has been dropped.
  [(#4453)](https://github.com/PennyLaneAI/pennylane/pull/4453)

* `MeasurementValue`'s signature has been updated to accept a list of `MidMeasureMP`'s rather than a list of
  their IDs.
  [(#4446)](https://github.com/PennyLaneAI/pennylane/pull/4446)

* `Operator.expand` now uses the output of `Operator.decomposition` instead of what it queues.
  [(#4355)](https://github.com/PennyLaneAI/pennylane/pull/4355)

* The `do_queue` keyword argument in `qml.operation.Operator` has been removed. Instead of
  setting `do_queue=False`, use the `qml.QueuingManager.stop_recording()` context instead.
  [(#4317)](https://github.com/PennyLaneAI/pennylane/pull/4317)

* The `grouping_type` and `grouping_method` keyword arguments have been removed from `qchem.molecular_hamiltonian`.

* `zyz_decomposition` and `xyx_decomposition` have been removed. Use `one_qubit_decomposition` instead.

* `LieAlgebraOptimizer` has been removed. Use `RiemannianGradientOptimizer` instead.

* `Operation.base_name` has been removed.

* `QuantumScript.name` has been removed.

* `qml.math.reduced_dm` has been removed. Use `qml.math.reduce_dm` or `qml.math.reduce_statevector` instead.

* The ``qml.specs`` dictionary no longer supports direct key access to certain keys. Instead,
  these quantities can be accessed as fields of the new ``Resources`` object saved under
  ``specs_dict["resources"]``:

  - `num_operations` is no longer supported, use `specs_dict["resources"].num_gates`
  - `num_used_wires` is no longer supported, use `specs_dict["resources"].num_wires`
  - `gate_types` is no longer supported, use `specs_dict["resources"].gate_types`
  - `gate_sizes` is no longer supported, use `specs_dict["resources"].gate_sizes`
  - `depth` is no longer supported, use `specs_dict["resources"].depth`

* `qml.math.purity`, `qml.math.vn_entropy`, `qml.math.mutual_info`, `qml.math.fidelity`,
  `qml.math.relative_entropy`, and `qml.math.max_entropy` no longer support state vectors as
  input.
  [(#4322)](https://github.com/PennyLaneAI/pennylane/pull/4322)

* The Pauli-X-term in `transmon_drive` has been removed in accordance with [1904.06560](https://arxiv.org/abs/1904.06560)
  [(#4418)](https://github.com/PennyLaneAI/pennylane/pull/4418/)

* The gradients module no longer needs shot information passed to it explicitly, as the shots are on the tapes.
  [(#4448)](https://github.com/PennyLaneAI/pennylane/pull/4448)

* `StatePrep` is renamed to `StatePrepBase` and `QubitStateVector` is renamed to `StatePrep`.
  `qml.operation.StatePrep` and `qml.QubitStateVector` will still be accessible for the time being.
  [(#4450)](https://github.com/PennyLaneAI/pennylane/pull/4450)

<h3>Deprecations 👋</h3>

* `qml.qchem.jordan_wigner` has been deprecated. Use `qml.jordan_wigner` instead.
  List input to define the fermionic operator has also been deprecated; the fermionic
  operators in the `qml.fermi` module should be used instead.
  [(#4332)](https://github.com/PennyLaneAI/pennylane/pull/4332)

* The `qml.RandomLayers.compute_decomposition` keyword argument `ratio_imprimitive` will be changed to `ratio_imprim` to
  match the call signature of the operation.
  [(#4314)](https://github.com/PennyLaneAI/pennylane/pull/4314)

* The CV observables `qml.X` and `qml.P` have been deprecated. Use `qml.QuadX`
  and `qml.QuadP` instead.
  [(#4330)](https://github.com/PennyLaneAI/pennylane/pull/4330)

* The method `tape.unwrap()` and corresponding `UnwrapTape` and `Unwrap` classes
  have been deprecated. Use `convert_to_numpy_parameters` instead.
  [(#4344)](https://github.com/PennyLaneAI/pennylane/pull/4344)

* `qml.enable_return` and `qml.disable_return` have been deprecated. Please avoid calling
  `disable_return`, as the old return system has been deprecated along with these switch functions.
  [(#4316)](https://github.com/PennyLaneAI/pennylane/pull/4316)

* The `mode` keyword argument in `QNode` has been deprecated, as it was only used in the
  old return system (which has also been deprecated). Please use `grad_on_execution` instead.
  [(#4316)](https://github.com/PennyLaneAI/pennylane/pull/4316)

* The `QuantumScript.set_parameters` method and the `QuantumScript.data` setter have
  been deprecated. Please use `QuantumScript.bind_new_parameters` instead.
  [(#4346)](https://github.com/PennyLaneAI/pennylane/pull/4346)

* The `__eq__` and `__hash__` dunder methods of `Operator` and `MeasurementProcess` will now raise
  warnings to reflect upcoming changes to operator and measurement process equality and hashing.
  [(#4144)](https://github.com/PennyLaneAI/pennylane/pull/4144)
  [(#4454)](https://github.com/PennyLaneAI/pennylane/pull/4454)

* The `sampler_seed` argument of `qml.gradients.spsa_grad` has been deprecated, along with a bug
  fix of the seed-setting behaviour.
  Instead, the `sampler_rng` argument should be set, either to an integer value, which will be used
  to create a PRNG internally or to a NumPy pseudo-random number generator created via
  `np.random.default_rng(seed)`.
  [(4165)](https://github.com/PennyLaneAI/pennylane/pull/4165)

<h3>Documentation 📝</h3>

* The `qml.pulse.transmon_interaction` and `qml.pulse.transmon_drive` documentation has been updated.
  [#4327](https://github.com/PennyLaneAI/pennylane/pull/4327)

* `qml.ApproxTimeEvolution.compute_decomposition()` now has a code example.
  [(#4354)](https://github.com/PennyLaneAI/pennylane/pull/4354)

* The documentation for `qml.devices.experimental.Device` has been improved to clarify
  some aspects of its use.
  [(#4391)](https://github.com/PennyLaneAI/pennylane/pull/4391)

* `qml.import_state` is now accounted for in `doc/introduction/chemistry.rst`, adding the documentation for the function.
  [(#4461)](https://github.com/PennyLaneAI/pennylane/pull/4461)

* Input types and sources for external wavefunctions and operators for `qml.import_state` 
  and `qml.import_operator` are clarified. [(#4476)](https://github.com/PennyLaneAI/pennylane/pull/4476)

<h3>Bug fixes 🐛</h3>

* `_copy_and_shift_params` no longer casts or converts integral types, it just relies on `+` and `*`'s casting rules in this case.
  [(#4477)](https://github.com/PennyLaneAI/pennylane/pull/4477)

* `qml.Projector` is pickle-able again.
  [(#4452)](https://github.com/PennyLaneAI/pennylane/pull/4452)

* Sparse matrix calculations of `SProd`s containing a `Tensor` are now allowed. When using
  `Tensor.sparse_matrix()`, it is recommended to use the `wire_order` keyword argument over `wires`. 
  [(#4424)](https://github.com/PennyLaneAI/pennylane/pull/4424)
  
* `op.adjoint` has been replaced with `qml.adjoint` in `QNSPSAOptimizer`.
  [(#4421)](https://github.com/PennyLaneAI/pennylane/pull/4421)

* `jax.ad` (deprecated) has been replaced by `jax.interpreters.ad`.
  [(#4403)](https://github.com/PennyLaneAI/pennylane/pull/4403)

* `metric_tensor` stops accidentally catching errors that stem from
  flawed wires assignments in the original circuit, leading to recursion errors.
  [(#4328)](https://github.com/PennyLaneAI/pennylane/pull/4328)

* A warning is now raised if control indicators are hidden when calling `qml.draw_mpl`
  [(#4295)](https://github.com/PennyLaneAI/pennylane/pull/4295)

* `qml.qinfo.purity` now produces correct results with custom wire labels.
  [(#4331)](https://github.com/PennyLaneAI/pennylane/pull/4331)

* `default.qutrit` now supports all qutrit operations used with `qml.adjoint`.
  [(#4348)](https://github.com/PennyLaneAI/pennylane/pull/4348)

* The observable data of `qml.GellMann` now includes its index, allowing correct comparison
  between instances of `qml.GellMann`, as well as Hamiltonians and Tensors
  containing `qml.GellMann`.
  [(#4366)](https://github.com/PennyLaneAI/pennylane/pull/4366)

* `qml.transforms.merge_amplitude_embedding` now works correctly when the `AmplitudeEmbedding`s
  have a batch dimension.
  [(#4353)](https://github.com/PennyLaneAI/pennylane/pull/4353)

* The `jordan_wigner` function has been modified to work with Hamiltonians built with an active space.
  [(#4372)](https://github.com/PennyLaneAI/pennylane/pull/4372)

* When a `style` option is not provided, `qml.draw_mpl` uses the current style set from
  `qml.drawer.use_style` instead of `black_white`.
  [(#4357)](https://github.com/PennyLaneAI/pennylane/pull/4357)

* `qml.devices.qubit.preprocess.validate_and_expand_adjoint` no longer sets the
  trainable parameters of the expanded tape.
  [(#4365)](https://github.com/PennyLaneAI/pennylane/pull/4365)

* `qml.default_expand_fn` now selectively expands operations or measurements allowing more 
  operations to be executed in circuits when measuring non-qwc Hamiltonians.
  [(#4401)](https://github.com/PennyLaneAI/pennylane/pull/4401)

* `qml.ControlledQubitUnitary` no longer reports `has_decomposition` as `True` when it does
  not really have a decomposition.
  [(#4407)](https://github.com/PennyLaneAI/pennylane/pull/4407)

* `qml.transforms.split_non_commuting` now correctly works on tapes containing both `expval`
  and `var` measurements.
  [(#4426)](https://github.com/PennyLaneAI/pennylane/pull/4426)

* Subtracting a `Prod` from another operator now works as expected.
  [(#4441)](https://github.com/PennyLaneAI/pennylane/pull/4441)

* The `sampler_seed` argument of `qml.gradients.spsa_grad` has been changed to `sampler_rng`. One can either provide
  an integer, which will be used to create a PRNG internally. Previously, this lead to the same direction
  being sampled, when `num_directions` is greater than 1. Alternatively, one can provide a NumPy PRNG,
  which allows reproducibly calling `spsa_grad` without getting the same results every time.
  [(4165)](https://github.com/PennyLaneAI/pennylane/pull/4165)
  [(4482)](https://github.com/PennyLaneAI/pennylane/pull/4482)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Utkarsh Azad,
Thomas Bromley,
Isaac De Vlugt,
Amintor Dusko,
Stepan Fomichev,
Lillian M. A. Frederiksen,
Soran Jahangiri,
Edward Jiang,
Korbinian Kottmann,
Ivana Kurečić,
Christina Lee,
Vincent Michaud-Rioux,
Romain Moyard,
Lee James O'Riordan,
Mudit Pandey,
Borja Requena,
Matthew Silverman,
Jay Soni,
David Wierichs,
Frederik Wilde
