:orphan:

# Release 0.32.0-dev (development release)

<h3>New features since last release</h3>

* `qml.measure` now includes a boolean keyword argument `reset` to reset a wire to the
  $|0\rangle$ computational basis state after measurement.
  [(#4402)](https://github.com/PennyLaneAI/pennylane/pull/4402/)

* Python-native logging can now be enabled with `qml.logging.enable_logging()`.
  [(#4383)](https://github.com/PennyLaneAI/pennylane/pull/4383)

* `DefaultQubit2` accepts a `max_workers` argument which controls multiprocessing.
  A `ProcessPoolExecutor` executes tapes asynchronously
  using a pool of at most `max_workers` processes. If `max_workers` is `None`
  or not given, only the current process executes tapes. If you experience any
  issue, say using JAX, TensorFlow, Torch, try setting `max_workers` to `None`.
  [(#4319)](https://github.com/PennyLaneAI/pennylane/pull/4319)
  [(#4425)](https://github.com/PennyLaneAI/pennylane/pull/4425)

* Transform Programs are now integrated with the `QNode`.
  [(#4404)](https://github.com/PennyLaneAI/pennylane/pull/4404)

```
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

>>> circuit(shots=1)
array([False, False])


* A new `qml.Select` operation is available. It applies specific input operations depending on the
  state of the designated control qubits
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

* Functions are available to obtain a state vector from `PySCF` solver objects.
  [(#4427)](https://github.com/PennyLaneAI/pennylane/pull/4427)
  [(#4433)](https://github.com/PennyLaneAI/pennylane/pull/4433)

  The `qml.qchem.import_state` function can be used to import a `PySCF` solver object and return the
  corresponding state vector.

  ```pycon
  >>> from pyscf import gto, scf, ci
  >>> mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0,0,0.71)]], basis='sto6g')
  >>> myhf = scf.UHF(mol).run()
  >>> myci = ci.UCISD(myhf).run()
  >>> wf_cisd = qml.qchem.import_state(myci, tol=1e-1)
  >>> print(wf_cisd)
  [ 0.        +0.j  0.        +0.j  0.        +0.j  0.1066467 +0.j
    0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
    0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
   -0.99429698+0.j  0.        +0.j  0.        +0.j  0.        +0.j]
  ```

  The currently supported objects are RCISD, UCISD, RCCSD, and UCCSD which correspond to 
  restricted (R) and unrestricted (U) configuration interaction (CI) and coupled cluster (CC) 
  calculations with single and double (SD) excitations.

<h3>Improvements 🛠</h3>

* Wires can now be reused after making a mid-circuit measurement on them.
  [(#4402)](https://github.com/PennyLaneAI/pennylane/pull/4402/)

* Transform Programs, `qml.transforms.core.TransformProgram`, can now be called on a batch of circuits
  and return a new batch of circuits and a single post processing function.
  [(#4364)](https://github.com/PennyLaneAI/pennylane/pull/4364)

* `HardwareHamiltonian`s can now be summed with `int` or `float`.
  A sequence of `HardwareHamiltonian`s can now be summed via the builtin `sum`.
  [(#4343)](https://github.com/PennyLaneAI/pennylane/pull/4343)

* All `Operator` objects now define `Operator._flatten` and `Operator._unflatten` methods that separate
  trainable from untrainable components. These methods will be used in serialization and pytree registration.
  Custom operations may need an update to ensure compatibility with new PennyLane features.
  [(#4314)](https://github.com/PennyLaneAI/pennylane/pull/4314)

* Treat auxiliary wires and device wires in the same way in `transforms.metric_tensor`
  as in `gradients.hadamard_grad`. Support all valid wire input formats for `aux_wire`.
  [(#4328)](https://github.com/PennyLaneAI/pennylane/pull/4328)

* `qml.equal` no longer raises errors when operators or measurements of different types are compared.
  Instead, it returns `False`.
  [(#4315)](https://github.com/PennyLaneAI/pennylane/pull/4315)

* The `qml.gradients` module no longer mutates operators in-place for any gradient transforms.
  Instead, operators that need to be mutated are copied with new parameters.
  [(#4220)](https://github.com/PennyLaneAI/pennylane/pull/4220)

* The calculation of `PauliWord` and `PauliSentence` sparse matrices are orders of magnitude faster.
  [(#4272)](https://github.com/PennyLaneAI/pennylane/pull/4272)
  [($4411)](https://github.com/PennyLaneAI/pennylane/pull/4411)

* Enable linting of all tests in CI and the pre-commit hook.
  [(#4335)](https://github.com/PennyLaneAI/pennylane/pull/4335)

* Added a function `qml.math.fidelity_statevector` that computes the fidelity between two state vectors.
  [(#4322)](https://github.com/PennyLaneAI/pennylane/pull/4322)

* The `qchem` module is upgraded to use the fermionic operators of the `fermi` module.
  [#4336](https://github.com/PennyLaneAI/pennylane/pull/4336)

* QNode transforms in `qml.qinfo` now support custom wire labels.
  [#4331](https://github.com/PennyLaneAI/pennylane/pull/4331)

* The `qchem` functions `primitive_norm` and `contracted_norm` are modified to be compatible with
  higher versions of scipy. The private function `_fac2` for computing double factorials is added.
  [#4321](https://github.com/PennyLaneAI/pennylane/pull/4321)

* The default label for a `StatePrep` operator is now `|Ψ⟩`.
  [(#4340)](https://github.com/PennyLaneAI/pennylane/pull/4340)

* The experimental device interface is integrated with the `QNode` for jax, jax-jit, tensorflow and torch.
  [(#4323)](https://github.com/PennyLaneAI/pennylane/pull/4323)
  [(#4352)](https://github.com/PennyLaneAI/pennylane/pull/4352)
  [(#4392)](https://github.com/PennyLaneAI/pennylane/pull/4392)
  [(#4393)](https://github.com/PennyLaneAI/pennylane/pull/4393)

* `tape_expand` now uses `Operator.decomposition` instead of `Operator.expand` in order to make
  more performant choices.
  [(#4355)](https://github.com/PennyLaneAI/pennylane/pull/4355)

* The `QuantumScript` class now has a `bind_new_parameters` method that allows creation of
  new `QuantumScript` objects with the provided parameters.
  [(#4345)](https://github.com/PennyLaneAI/pennylane/pull/4345)

* `qml.ctrl(qml.PauliX)` returns a `CNOT`, `Toffoli` or `MultiControlledX` instead of a `Controlled(PauliX)`.
  [(#4339)](https://github.com/PennyLaneAI/pennylane/pull/4339)

* Added functions `adjoint_jvp` and `adjoint_vjp` to `qml.devices.qubit.adjoint_jacobian` that computes
  the JVP and VJP of a tape using the adjoint method.
  [(#4358)](https://github.com/PennyLaneAI/pennylane/pull/4358)

* Readability improvements and stylistic changes to `pennylane/interfaces/jax_jit_tuple.py`
  [(#4379)](https://github.com/PennyLaneAI/pennylane/pull/4379/)

* When given a callable, `qml.ctrl` now does its custom pre-processing on all queued operators from the callable.
  [(#4370)](https://github.com/PennyLaneAI/pennylane/pull/4370)

* `qml.pauli_decompose` is now differentiable and works with any non-Hermitian and non-square matrices.
  [(#4395)](https://github.com/PennyLaneAI/pennylane/pull/4395)

* `qml.interfaces.set_shots` accepts `Shots` object as well as `int`'s and tuples of `int`'s.
  [(#4388)](https://github.com/PennyLaneAI/pennylane/pull/4388)

* `pennylane.devices.experimental.Device` now accepts a shots keyword argument and has a `shots`
  property. This property is merely used to set defaults for a workflow, and does not directly
  influence the number of shots used in executions or derivatives.
  [(#4388)](https://github.com/PennyLaneAI/pennylane/pull/4388)

* PennyLane no longer directly relies on `Operator.__eq__`.
  [(#4398)](https://github.com/PennyLaneAI/pennylane/pull/4398)

* If no seed is specified on initialization with `DefaultQubit2`, the local random number generator will be
  seeded from on the NumPy's global random number generator.
  [(#4394)](https://github.com/PennyLaneAI/pennylane/pull/4394)

* The experimental `DefaultQubit2` device now supports computing VJPs and JVPs using the adjoint method.
  [(#4374)](https://github.com/PennyLaneAI/pennylane/pull/4374)
  
* Provide users access to the logging configuration file path and improve the logging configuration structure.
  [(#4377)](https://github.com/PennyLaneAI/pennylane/pull/4377)

* Refactoring of `pennylane/interfaces`.  The `execute_fn` passed to the machine learning framework boundaries 
  is now responsible for converting parameters to numpy. The gradients module can now handle tensorflow parameters,
  but gradient tapes now retain the original dtype instead of converting to float64.  This may cause instability 
  with finite diff and float32 parameters. The ml boundary functions are now uncoupled from their legacy
  counterparts.
  [(#4415)](https://github.com/PennyLaneAI/pennylane/pull/4415)

* `qml.transforms.adjoint_metric_tensor` now uses the simulation tools in `pennylane.devices.qubit` instead of
  private methods of `pennylane.devices.DefaultQubit`.
  [(#4456)](https://github.com/PennyLaneAI/pennylane/pull/4456)

* Updated `Device.default_expand_fn()` to decompose `StatePrep` operations present in the middle of a provided circuit.
  [(#4437)](https://github.com/PennyLaneAI/pennylane/pull/4437)

* Updated `expand_fn()` for `DefaultQubit2` to decompose `StatePrep` operations present in the middle of a circuit.
  [(#4444)](https://github.com/PennyLaneAI/pennylane/pull/4444)

* `transmon_drive` is updated in accordance with [1904.06560](https://arxiv.org/abs/1904.06560). In particular, the functional form has been changed from $\Omega(t)(\cos(\omega_d t + \phi) X - \sin(\omega_d t + \phi) Y)$ to $\Omega(t) \sin(\omega_d t + \phi) Y$.
  [(#4418)](https://github.com/PennyLaneAI/pennylane/pull/4418/)
  [(#4465)](https://github.com/PennyLaneAI/pennylane/pull/4465/)
  [(#4478)](https://github.com/PennyLaneAI/pennylane/pull/4478/)

<h3>Breaking changes 💔</h3>

* Gradient transforms no longer implicitly cast `float32` parameters to `float64`. Finite diff
  with float32 parameters may no longer give accurate results.
  [(#4415)](https://github.com/PennyLaneAI/pennylane/pull/4415)

* Support for Python 3.8 is dropped.
  [(#4453)](https://github.com/PennyLaneAI/pennylane/pull/4453)

* `MeasurementValue`'s signature has been updated to accept a list of `MidMeasureMP`'s rather than a list of
  their IDs.
  [(#4446)](https://github.com/PennyLaneAI/pennylane/pull/4446)

* `Operator.expand` now uses the output of `Operator.decomposition` instead of what it queues.
  [(#4355)](https://github.com/PennyLaneAI/pennylane/pull/4355)

* The `do_queue` keyword argument in `qml.operation.Operator` has been removed. Instead of
  setting `do_queue=False`, use the `qml.QueuingManager.stop_recording()` context.
  [(#4317)](https://github.com/PennyLaneAI/pennylane/pull/4317)

* The `grouping_type` and `grouping_method` keyword arguments are removed from `qchem.molecular_hamiltonian`.

* `zyz_decomposition` and `xyx_decomposition` are removed. Use `one_qubit_decomposition` instead.

* `LieAlgebraOptimizer` has been removed. Use `RiemannianGradientOptimizer` instead.

* `Operation.base_name` has been removed.

* `QuantumScript.name` has been removed.

* `qml.math.reduced_dm` has been removed. Use `qml.math.reduce_dm` or `qml.math.reduce_statevector` instead.

* The ``qml.specs`` dictionary longer supports direct key access to certain keys. Instead
  these quantities can be accessed as fields of the new ``Resources`` object saved under
  ``specs_dict["resources"]``:

  - ``num_operations`` is no longer supported, use ``specs_dict["resources"].num_gates``
  - ``num_used_wires`` is no longer supported, use ``specs_dict["resources"].num_wires``
  - ``gate_types`` is no longer supported, use ``specs_dict["resources"].gate_types``
  - ``gate_sizes`` is no longer supported, use ``specs_dict["resources"].gate_sizes``
  - ``depth`` is no longer supported, use ``specs_dict["resources"].depth``

* `qml.math.purity`, `qml.math.vn_entropy`, `qml.math.mutual_info`, `qml.math.fidelity`,
  `qml.math.relative_entropy`, and `qml.math.max_entropy` no longer support state vectors as
  input.
  [(#4322)](https://github.com/PennyLaneAI/pennylane/pull/4322)

* The Pauli-X-term in `transmon_drive` has been removed in accordance with [1904.06560](https://arxiv.org/abs/1904.06560)
  [(#4418)](https://github.com/PennyLaneAI/pennylane/pull/4418/)

* The gradients module no longer needs shot information passed to it explicitly, as the shots are on the tapes.
  [(#4448)](https://github.com/PennyLaneAI/pennylane/pull/4448)

<h3>Deprecations 👋</h3>

* ``qml.qchem.jordan_wigner`` is deprecated, use ``qml.jordan_wigner`` instead.
  List input to define the fermionic operator is also deprecated; the fermionic
  operators in the ``qml.fermi`` module should be used instead.
  [(#4332)](https://github.com/PennyLaneAI/pennylane/pull/4332)

* The `qml.RandomLayers.compute_decomposition` keyword argument `ratio_imprimitive` will be changed to `ratio_imprim` to
  match the call signature of the operation.
  [(#4314)](https://github.com/PennyLaneAI/pennylane/pull/4314)

* The CV observables ``qml.X`` and ``qml.P`` have been deprecated. Use ``qml.QuadX``
  and ``qml.QuadP`` instead.
  [(#4330)](https://github.com/PennyLaneAI/pennylane/pull/4330)

* The method ``tape.unwrap()`` and corresponding ``UnwrapTape`` and ``Unwrap`` classes
  are deprecated. Use ``convert_to_numpy_parameters`` instead.
  [(#4344)](https://github.com/PennyLaneAI/pennylane/pull/4344)

* `qml.enable_return` and `qml.disable_return` are deprecated. Please avoid calling
  `disable_return`, as the old return system is deprecated along with these switch functions.
  [(#4316)](https://github.com/PennyLaneAI/pennylane/pull/4316)

* The `mode` keyword argument in `QNode` is deprecated, as it was only used in the
  old return system (which is also deprecated). Please use `grad_on_execution` instead.
  [(#4316)](https://github.com/PennyLaneAI/pennylane/pull/4316)

* The `QuantumScript.set_parameters` method and the `QuantumScript.data` setter has
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

* The documentation for `pennylane.devices.experimental.Device` is improved to clarify
  some aspects of its use.
  [(#4391)](https://github.com/PennyLaneAI/pennylane/pull/4391)

* `qml.import_state` is now accounted for in `doc/introduction/chemistry.rst`, adding the documentation for the function.
  [(#4461)](https://github.com/PennyLaneAI/pennylane/pull/4461)

* Input types and sources for external wavefunctions and operators for `qml.import_state` 
  and `qml.import_operator` are clarified. [(#4476)](https://github.com/PennyLaneAI/pennylane/pull/4476)

<h3>Bug fixes 🐛</h3>

* `_copy_and_shift_params` does not cast or convert integral types, just relying on `+` and `*`'s casting rules in this case.
  [(#4477)](https://github.com/PennyLaneAI/pennylane/pull/4477)

* `qml.Projector` is pickle-able again.
  [(#4452)](https://github.com/PennyLaneAI/pennylane/pull/4452)

* Allow sparse matrix calculation of `SProd`s containing a `Tensor`. When using
  `Tensor.sparse_matrix()`, it is recommended to use the `wire_order` keyword argument over `wires`. 
  [(#4424)](https://github.com/PennyLaneAI/pennylane/pull/4424)
  
* Replace `op.adjoint` with `qml.adjoint` in `QNSPSAOptimizer`.
  [(#4421)](https://github.com/PennyLaneAI/pennylane/pull/4421)

* Replace deprecated `jax.ad` by `jax.interpreters.ad`.
  [(#4403)](https://github.com/PennyLaneAI/pennylane/pull/4403)

* Stop `metric_tensor` from accidentally catching errors that stem from
  flawed wires assignments in the original circuit, leading to recursion errors.
  [(#4328)](https://github.com/PennyLaneAI/pennylane/pull/4328)

* Raise a warning if control indicators are hidden when calling `qml.draw_mpl`
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

* The `jordan_wigner` function is modified to work with Hamiltonians built with an active space.
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

* Change the `sampler_seed` argument of `qml.gradients.spsa_grad` to `sampler_rng`. One can either provide
  an integer, which will be used to create a PRNG internally. Previously, this lead to the same direction
  being sampled, when `num_directions` is greater than 1. Alternatively, one can provide a NumPy PRNG,
  which allows reproducibly calling `spsa_grad` without getting the same results every time.
  [(4165)](https://github.com/PennyLaneAI/pennylane/pull/4165)


<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Utkarsh Azad,
Isaac De Vlugt,
Amintor Dusko,
Stepan Fomichev,
Lillian M. A. Frederiksen,
Soran Jahangiri,
Edward Jiang,
Korbinian Kottmann
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
