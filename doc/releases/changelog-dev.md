# Release 0.45.0 (development release)

<h3>New features since last release</h3>

* Added a convenience function :func:`~.math.ceil_log2` that computes the ceiling of the base-2
  logarithm of its input and casts the result to an ``int``. It is equivalent to 
  ``int(np.ceil(np.log2(n)))``.
  [(#8972)](https://github.com/PennyLaneAI/pennylane/pull/8972)

* Added a ``qml.gate_sets`` that contains pre-defined gate sets such as ``qml.gate_sets.CLIFFORD_T_PLUS_RZ``
  that can be plugged into the ``gate_set`` argument of the :func:`~pennylane.transforms.decompose` transform.
  [(#8915)](https://github.com/PennyLaneAI/pennylane/pull/8915)

* Added a `qml.decomposition.local_decomps` context
  manager that allows one to add decomposition rules to an operator, only taking effect within the context.
  [(#8955)](https://github.com/PennyLaneAI/pennylane/pull/8955)

* Added a `qml.workflow.get_compile_pipeline(qnode, level)(*args, **kwargs)` function to extract the 
  compile pipeline of a given QNode at a specific level.
  [(#8979)](https://github.com/PennyLaneAI/pennylane/pull/8979)
  [(#8987)](https://github.com/PennyLaneAI/pennylane/pull/8987)

<h3>Improvements 🛠</h3>

* `qml.vjp` can now be captured into plxpr.
  [(#8736)](https://github.com/PennyLaneAI/pennylane/pull/8736)

* :func:`~.matrix` can now also be applied to a sequence of operators.
  [(#8861)](https://github.com/PennyLaneAI/pennylane/pull/8861)

* The ``qml.estimator.Resources`` class now has a nice string representation in Jupyter Notebooks.
  [(#8880)](https://github.com/PennyLaneAI/pennylane/pull/8880)

* A function for setting up transform inputs, including setting default values and basic validation,
  can now be provided to `qml.transform` via `setup_inputs`.
  [(#8732)](https://github.com/PennyLaneAI/pennylane/pull/8732)

* Circuits containing `GlobalPhase` are now trainable without removing the `GlobalPhase`.
  [(#8950)](https://github.com/PennyLaneAI/pennylane/pull/8950)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* Removed all of the resource estimation functionality from the `labs.resource_estimation`
  module. Users can now directly access a more stable version of this functionality using the 
  `estimator` module. All experimental development of resource estimation
  will be added to `.labs.estimator_beta`
  [(#8868)](https://github.com/PennyLaneAI/pennylane/pull/8868)

* The integration test for computing perturbation error of a compressed double-factorized (CDF)
  Hamiltonian in `labs.trotter_error` is upgraded to use a more realistic molecular geometry and
  a more reliable reference error.
  [(#8790)](https://github.com/PennyLaneAI/pennylane/pull/8790)

<h3>Breaking changes 💔</h3>

* Dropped support for NumPy 1.x following its end-of-life. NumPy 2.0 or higher is now required.
  [(#8914)](https://github.com/PennyLaneAI/pennylane/pull/8914)
  [(#8954)](https://github.com/PennyLaneAI/pennylane/pull/8954)
  
* ``compute_qfunc_decomposition`` and ``has_qfunc_decomposition`` have been removed from  :class:`~.Operator`
  and all subclasses that implemented them. The graph decomposition system should be used when capture is enabled.
  [(#8922)](https://github.com/PennyLaneAI/pennylane/pull/8922)

* The :func:`pennylane.devices.preprocess.mid_circuit_measurements` transform is removed. Instead,
  the device should determine which mcm method to use, and explicitly include :func:`~pennylane.transforms.dynamic_one_shot`
  or :func:`~pennylane.transforms.defer_measurements` in its preprocess transforms if necessary. See
  :func:`DefaultQubit.setup_execution_config <pennylane.devices.DefaultQubit.setup_execution_config>` and 
  :func:`DefaultQubit.preprocess_transforms <pennylane.devices.DefaultQubit.preprocess_transforms>` for an example.
  [(#8926)](https://github.com/PennyLaneAI/pennylane/pull/8926)

* The ``custom_decomps`` keyword argument to ``qml.device`` has been removed in 0.45. Instead, 
  with ``qml.decomposition.enable_graph()``, new decomposition rules can be defined as
  quantum functions with registered resources. See :mod:`pennylane.decomposition` for more details.
  [(#8928)](https://github.com/PennyLaneAI/pennylane/pull/8928)

  As an example, consider the case of running the following circuit on a device that does not natively support ``CNOT`` gates:

  .. code-block:: python3


    def circuit():
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.X(1))

  Instead of defining the ``CNOT`` decomposition as:

  .. code-block:: python3

    def custom_cnot(wires):
      return [
          qml.Hadamard(wires=wires[1]),
          qml.CZ(wires=[wires[0], wires[1]]),
          qml.Hadamard(wires=wires[1])
      ]

    dev = qml.device('default.qubit', wires=2, custom_decomps={"CNOT" : custom_cnot})
    qnode = qml.QNode(circuit, dev)
    print(qml.draw(qnode, level="device")())

  The same result would now be obtained using:

  .. code-block:: python3

    @qml.decomposition.register_resources({
        qml.H: 2,
        qml.CZ: 1
    })
    def _custom_cnot_decomposition(wires, **_):
      qml.Hadamard(wires=wires[1])
      qml.CZ(wires=[wires[0], wires[1]])
      qml.Hadamard(wires=wires[1])

    qml.decomposition.add_decomps(qml.CNOT, _custom_cnot_decomposition)

    qml.decomposition.enable_graph()

    @qml.transforms.decompose(gate_set={qml.CZ, qml.H})
    def circuit():
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.X(1))

    dev = qml.device('default.qubit', wires=2)
    qnode = qml.QNode(circuit, dev)

  >>> print(qml.draw(qnode, level="device")())
  0: ────╭●────┤
  1: ──H─╰Z──H─┤  <X>

* The `pennylane.operation.Operator.is_hermitian` property has been removed and replaced 
  with `pennylane.operation.Operator.is_verified_hermitian` as it better reflects the functionality of this property.
  Alternatively, consider using the `pennylane.is_hermitian` function instead as it provides a more reliable check for hermiticity.
  Please be aware that it comes with a higher computational cost.
  [(#8919)](https://github.com/PennyLaneAI/pennylane/pull/8919)

* Passing a function to the `gate_set` argument in the `pennylane.transforms.decompose` transform
  is removed. The `gate_set` argument expects a static iterable of operator type and/or operator names,
  and the function should be passed to the `stopping_condition` argument instead.
  [(#8919)](https://github.com/PennyLaneAI/pennylane/pull/8919)

* `argnum` has been renamed `argnums` in `qml.grad`, `qml.jacobian`, `qml.jvp`, and `qml.vjp`
  to better match Catalyst and JAX.
  [(#8919)](https://github.com/PennyLaneAI/pennylane/pull/8919)

* Access to the following functions and classes from the `~pennylane.resources` module has 
  been removed. Instead, these functions must be imported from the `~pennylane.estimator` module.
  [(#8919)](https://github.com/PennyLaneAI/pennylane/pull/8919)

    - `qml.estimator.estimate_shots` in favor of `qml.resources.estimate_shots`
    - `qml.estimator.estimate_error` in favor of `qml.resources.estimate_error`
    - `qml.estimator.FirstQuantization` in favor of `qml.resources.FirstQuantization`
    - `qml.estimator.DoubleFactorization` in favor of `qml.resources.DoubleFactorization`

<h3>Deprecations 👋</h3>

* :func:`~pennylane.tape.qscript.expand` and the related functions :func:`~pennylane.tape.expand_tape`, :func:`~pennylane.tape.expand_tape_state_prep`, and :func:`~pennylane.tape.create_expand_trainable_multipar` 
  have been deprecated and will be removed in v0.46. Instead, please use the :func:`qml.transforms.decompose <.transforms.decompose>` 
  function for decomposing circuits.
  [(#8943)](https://github.com/PennyLaneAI/pennylane/pull/8943)

* Providing a value of ``None`` to ``aux_wire`` of ``qml.gradients.hadamard_grad`` in reversed or standard mode has been
  deprecated and will no longer be supported in 0.46. An ``aux_wire`` will no longer be automatically assigned.
  [(#8905)](https://github.com/PennyLaneAI/pennylane/pull/8905)

* The ``transform_program`` property of ``QNode`` has been renamed to ``compile_pipeline``.
  The deprecated access through ``transform_program`` will be removed in PennyLane v0.46.
  [(#8906)](https://github.com/PennyLaneAI/pennylane/pull/8906)

* Providing a value of ``None`` to ``aux_wire`` of ``qml.gradients.hadamard_grad`` with ``mode="reversed"`` or ``mode="standard"`` has been
  deprecated and will no longer be supported in 0.46. An ``aux_wire`` will no longer be automatically assigned.
  [(#8905)](https://github.com/PennyLaneAI/pennylane/pull/8905)

* The ``qml.transforms.create_expand_fn`` has been deprecated and will be removed in v0.46.
  Instead, please use the :func:`qml.transforms.decompose <.transforms.decompose>` function for decomposing circuits.
  [(#8941)](https://github.com/PennyLaneAI/pennylane/pull/8941)
  [(#8977)](https://github.com/PennyLaneAI/pennylane/pull/8977)

* The ``transform_program`` property of ``QNode`` has been renamed to ``compile_pipeline``.
  The deprecated access through ``transform_program`` will be removed in PennyLane v0.46.
  [(#8906)](https://github.com/PennyLaneAI/pennylane/pull/8906)
  [(#8945)](https://github.com/PennyLaneAI/pennylane/pull/8945)

<h3>Internal changes ⚙️</h3>

* Seeded a test `tests/measurements/test_classical_shadow.py::TestClassicalShadow::test_return_distribution` to fix stochastic failures by adding a `seed` parameter to the circuit helper functions and the test method.
  [(#xxxx)](https://github.com/PennyLaneAI/pennylane/pull/xxxx)

* Standardized the tolerances of several stochastic tests to use a 3-sigma rule based on theoretical variance and number of shots, reducing spurious failures. This includes `TestHamiltonianSamples::test_multi_wires`, `TestSampling::test_complex_hamiltonian`, and `TestBroadcastingPRNG::test_nonsample_measure`.
  Bumped `rng_salt` to `v0.45.0`.
  [(#8959)](https://github.com/PennyLaneAI/pennylane/pull/8959)
  [(#8958)](https://github.com/PennyLaneAI/pennylane/pull/8958)
  [(#8938)](https://github.com/PennyLaneAI/pennylane/pull/8938)
  [(#8908)](https://github.com/PennyLaneAI/pennylane/pull/8908)
  [(#8963)](https://github.com/PennyLaneAI/pennylane/pull/8963)

* Updated test helper `get_device` to correctly seed lightning devices.
  [(#8942)](https://github.com/PennyLaneAI/pennylane/pull/8942)

* Updated internal dependencies `autoray` (to 0.8.4), `tach` (to 0.33).
  [(#8911)](https://github.com/PennyLaneAI/pennylane/pull/8911)

* Relaxed the `torch` dependency from `==2.9.0` to `~=2.9.0` to allow for compatible patch updates.
  [(#8911)](https://github.com/PennyLaneAI/pennylane/pull/8911)

* Internal calls to the `decompose` transform have been updated to provide a `target_gates` argument so that
  they are compatible with the new graph-based decomposition system.
  [(#8939)](https://github.com/PennyLaneAI/pennylane/pull/8939)

* Added a `qml.decomposition.toggle_graph_ctx` context manager to temporarily enable or disable graph-based
  decompositions in a thread-safe way. The fixtures `"enable_graph_decomposition"`, `"disable_graph_decomposition"`,
  and `"enable_and_disable_graph_decomp"` have been updated to use this method so that they are thread-safe.
  [(#8966)](https://github.com/PennyLaneAI/pennylane/pull/8966)

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Improves the error messages when the inputs and outputs to a `qml.for_loop` function do not match.
  [(#8984)](https://github.com/PennyLaneAI/pennylane/pull/8984)

* Fixes a bug that `qml.QubitDensityMatrix` was applied in `default.mixed` device using `qml.math.partial_trace` incorrectly.
  This would cause wrong results as described in [this issue](https://github.com/PennyLaneAI/pennylane/pull/8932).
  [(#8933)](https://github.com/PennyLaneAI/pennylane/pull/8933)

* Fixes an issue when binding a transform when the first positional arg
  is a `Sequence`, but not a `Sequence` of tapes.
  [(#8920)](https://github.com/PennyLaneAI/pennylane/pull/8920)

* Fixes a bug with `qml.estimator.templates.QSVT` which allows users to instantiate the class without
  providing wires. This is now consistent with the standard in the estimator module.
  [(#8949)](https://github.com/PennyLaneAI/pennylane/pull/8949)

* Fixes a bug where decomposition raises an error for `Pow` operators when the exponent is batched.
  [(#8969)](https://github.com/PennyLaneAI/pennylane/pull/8969)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Astral Cai,
Yushao Chen,
Marcus Edwards,
Christina Lee,
Andrija Paurevic,
Omkar Sarkar,
Jay Soni,
David Wierichs,
