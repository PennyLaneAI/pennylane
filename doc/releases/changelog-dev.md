:orphan:

# Release 0.32.0-dev (development release)

<h3>New features since last release</h3>

* `DefaultQubit2` accepts a `max_workers` argument which controls multiprocessing. 
  A `ProcessPoolExecutor` executes tapes asynchronously
  using a pool of at most `max_workers` processes. If `max_workers` is `None`
  or not given, only the current process executes tapes. If you experience any
  issue, say using JAX, TensorFlow, Torch, try setting `max_workers` to `None`.
  [(#4319)](https://github.com/PennyLaneAI/pennylane/pull/4319)

<h3>Improvements 🛠</h3>

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

* `PauliWord` sparse matrices are much faster, which directly improves `PauliSentence`.
  [(#4272)](https://github.com/PennyLaneAI/pennylane/pull/4272)

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

* PennyLane no longer directly relies on `Operator.__eq__`.
  [(#4398)](https://github.com/PennyLaneAI/pennylane/pull/4398)

* If no seed is specified on initialization with `DefaultQubit2`, the local random number generator will be
  seeded from on the NumPy's global random number generator.
  [(#4394)](https://github.com/PennyLaneAI/pennylane/pull/4394)

* The experimental `DefaultQubit2` device now supports computing VJPs and JVPs using the adjoint method.
  [(#4374)](https://github.com/PennyLaneAI/pennylane/pull/4374)

<h3>Breaking changes 💔</h3>

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

* `Operator.__eq__` and `Operator.__hash__` will now raise warnings to reflect upcoming
  changes to operator equality and hashing.
  [(#4144)](https://github.com/PennyLaneAI/pennylane/pull/4144)

<h3>Documentation 📝</h3>

* The `qml.pulse.transmon_interaction` and `qml.pulse.transmon_drive` documentation has been updated.
  [#4327](https://github.com/PennyLaneAI/pennylane/pull/4327)

* `qml.ApproxTimeEvolution.compute_decomposition()` now has a code example.
  [(#4354)](https://github.com/PennyLaneAI/pennylane/pull/4354)

* The documentation for `pennylane.devices.experimental.Device` is improved to clarify
  some aspects of its use.
  [(#4391)](https://github.com/PennyLaneAI/pennylane/pull/4391)

<h3>Bug fixes 🐛</h3>

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

* `qml.ControlledQubitUnitary` no longer reports `has_decomposition` as `True` when it does
  not really have a decomposition.
  [(#4407)](https://github.com/PennyLaneAI/pennylane/pull/4407)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Isaac De Vlugt,
Lillian M. A. Frederiksen,
Soran Jahangiri,
Edward Jiang,
Christina Lee,
Vincent Michaud-Rioux,
Romain Moyard,
Mudit Pandey,
Borja Requena,
Matthew Silverman,
David Wierichs,
