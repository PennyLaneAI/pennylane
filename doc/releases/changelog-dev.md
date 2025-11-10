# Release 0.44.0-dev (development release)

<h3>New features since last release</h3>

* A new decomposition has been added for the Controlled :class:`~.SemiAdder`,
  which is efficient and skips controlling all gates in its decomposition.
  [(#8423)](https://github.com/PennyLaneAI/pennylane/pull/8423)

* Added a :meth:`~pennylane.devices.DeviceCapabilities.gate_set` method to :class:`~pennylane.devices.DeviceCapabilities`
  that produces a set of gate names to be used as the target gate set in decompositions.
  [(#8522)](https://github.com/PennyLaneAI/pennylane/pull/8522)

* Added a :func:`~pennylane.measurements.pauli_measure` that takes a Pauli product measurement.
  [(#8461)](https://github.com/PennyLaneAI/pennylane/pull/8461)

<h3>Improvements 🛠</h3>

* The `~.BasisRotation` graph decomposition was re-written in a qjit friendly way with PennyLane control flow.
  [(#8560)](https://github.com/PennyLaneAI/pennylane/pull/8560)

* The new graph based decompositions system enabled via :func:`~.decomposition.enable_graph` now supports the following
  additional templates.
  [(#8520)](https://github.com/PennyLaneAI/pennylane/pull/8520)
  [(#8515)](https://github.com/PennyLaneAI/pennylane/pull/8515)
  [(#8516)](https://github.com/PennyLaneAI/pennylane/pull/8516)
  [(#8555)](https://github.com/PennyLaneAI/pennylane/pull/8555)
  [(#8558)](https://github.com/PennyLaneAI/pennylane/pull/8558)
  [(#8538)](https://github.com/PennyLaneAI/pennylane/pull/8538)  
  [(#8534)](https://github.com/PennyLaneAI/pennylane/pull/8534)
  [(#8582)](https://github.com/PennyLaneAI/pennylane/pull/8582)
  [(#8543)](https://github.com/PennyLaneAI/pennylane/pull/8543)
  [(#8554)](https://github.com/PennyLaneAI/pennylane/pull/8554)
  
  - :class:`~.QSVT`
  - :class:`~.AmplitudeEmbedding`
  - :class:`~.AllSinglesDoubles`
  - :class:`~.SimplifiedTwoDesign`
  - :class:`~.GateFabric`
  - :class:`~.AngleEmbedding`
  - :class:`~.IQPEmbedding`
  - :class:`~.kUpCCGSD`
  - :class:`~.QAOAEmbedding`
  - :class:`~.BasicEntanglerLayers`

* A new `qml.compiler.python_compiler.utils` submodule has been added, containing general-purpose utilities for
  working with xDSL. This includes a function that extracts the concrete value of scalar, constant SSA values.
  [(#8514)](https://github.com/PennyLaneAI/pennylane/pull/8514)

* Added a keyword argument ``recursive`` to ``qml.transforms.cancel_inverses`` that enables
  recursive cancellation of nested pairs of mutually inverse gates. This makes the transform
  more powerful, because it can cancel larger blocks of inverse gates without having to scan
  the circuit from scratch. By default, the recursive cancellation is enabled (``recursive=True``).
  To obtain previous behaviour, disable it by setting ``recursive=False``.
  [(#8483)](https://github.com/PennyLaneAI/pennylane/pull/8483)

* `qml.grad` and `qml.jacobian` now lazily dispatch to catalyst and program
  capture, allowing for `qml.qjit(qml.grad(c))` and `qml.qjit(qml.jacobian(c))` to work.
  [(#8382)](https://github.com/PennyLaneAI/pennylane/pull/8382)

* Both the generic and transform-specific application behavior of a `qml.transforms.core.TransformDispatcher`
  can be overwritten with `TransformDispatcher.generic_register` and `my_transform.register`.
  [(#7797)](https://github.com/PennyLaneAI/pennylane/pull/7797)

* With capture enabled, measurements can now be performed on Operator instances passed as closure
  variables from outside the workflow scope.
  [(#8504)](https://github.com/PennyLaneAI/pennylane/pull/8504)

* Users can now estimate the resources for quantum circuits that contain or decompose into
  any of the following symbolic operators: :class:`~.ChangeOpBasis`, :class:`~.Prod`,
  :class:`~.Controlled`, :class:`~.ControlledOp`, :class:`~.Pow`, and :class:`~.Adjoint`.
  [(#8464)](https://github.com/PennyLaneAI/pennylane/pull/8464)

* Wires can be specified via `range` with program capture and autograph.
  [(#8500)](https://github.com/PennyLaneAI/pennylane/pull/8500)

* The :func:`~pennylane.transforms.decompose` transform no longer raises an error if both `gate_set` and
  `stopping_condition` are provided, or if `gate_set` is a dictionary, when the new graph-based decomposition
  system is disabled.
  [(#8532)](https://github.com/PennyLaneAI/pennylane/pull/8532)

* A new decomposition has been added to :class:`pennylane.Toffoli`. This decomposition uses one
  work wire and :class:`pennylane.TemporaryAND` operators to reduce the resources needed.
  [(#8549)](https://github.com/PennyLaneAI/pennylane/pull/8549)

* Scipy sparse matrices can now be passed directly to :func:`~pennylane.pauli_decompose` without
  manual conversion to dense arrays.
  [(#8612)](https://github.com/PennyLaneAI/pennylane/pull/8612)

<h3>Breaking changes 💔</h3>

* Providing ``num_steps`` to :func:`pennylane.evolve`, :func:`pennylane.exp`, :class:`pennylane.ops.Evolution`,
  and :class:`pennylane.ops.Exp` has been disallowed. Instead, use :class:`~.TrotterProduct` for approximate
  methods, providing the ``n`` parameter to perform the Suzuki-Trotter product approximation of a Hamiltonian
  with the specified number of Trotter steps.
  [(#8474)](https://github.com/PennyLaneAI/pennylane/pull/8474)

  As a concrete example, consider the following case:

  .. code-block:: python

    coeffs = [0.5, -0.6]
    ops = [qml.X(0), qml.X(0) @ qml.Y(1)]
    H_flat = qml.dot(coeffs, ops)

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

* The value ``None`` has been removed as a valid argument to the ``level`` parameter in the
  :func:`pennylane.workflow.get_transform_program`, :func:`pennylane.workflow.construct_batch`,
  :func:`pennylane.draw`, :func:`pennylane.draw_mpl`, and :func:`pennylane.specs` transforms.
  Please use ``level='device'`` instead to apply the transform at the device level.
  [(#8477)](https://github.com/PennyLaneAI/pennylane/pull/8477)

* Access to ``add_noise``, ``insert`` and noise mitigation transforms from the ``pennylane.transforms`` module is deprecated.
  Instead, these functions should be imported from the ``pennylane.noise`` module.
  [(#8477)](https://github.com/PennyLaneAI/pennylane/pull/8477)

* ``qml.qnn.cost.SquaredErrorLoss`` has been removed. Instead, this hybrid workflow can be accomplished
  with a function like ``loss = lambda *args: (circuit(*args) - target)**2``.
  [(#8477)](https://github.com/PennyLaneAI/pennylane/pull/8477)

* Some unnecessary methods of the ``qml.CircuitGraph`` class have been removed:
  [(#8477)](https://github.com/PennyLaneAI/pennylane/pull/8477)

  - ``print_contents`` in favor of ``print(obj)``
  - ``observables_in_order`` in favor of ``observables``
  - ``operations_in_order`` in favor of ``operations``
  - ``ancestors_in_order(obj)`` in favor of ``ancestors(obj, sort=True)``
  - ``descendants_in_order(obj)`` in favor of ``descendants(obj, sort=True)``

* ``pennylane.devices.DefaultExecutionConfig`` has been removed. Instead, use
  ``qml.devices.ExecutionConfig()`` to create a default execution configuration.
  [(#8470)](https://github.com/PennyLaneAI/pennylane/pull/8470)

* Specifying the ``work_wire_type`` argument in ``qml.ctrl`` and other controlled operators as ``"clean"`` or
  ``"dirty"`` is disallowed. Use ``"zeroed"`` to indicate that the work wires are initially in the :math:`|0\rangle`
  state, and ``"borrowed"`` to indicate that the work wires can be in any arbitrary state. In both cases, the
  work wires are assumed to be restored to their original state upon completing the decomposition.
  [(#8470)](https://github.com/PennyLaneAI/pennylane/pull/8470)

* `QuantumScript.shape` and `QuantumScript.numeric_type` are removed. The corresponding `MeasurementProcess`
  methods should be used instead.
  [(#8468)](https://github.com/PennyLaneAI/pennylane/pull/8468)

* `MeasurementProcess.expand` is removed.
  `qml.tape.QuantumScript(mp.obs.diagonalizing_gates(), [type(mp)(eigvals=mp.obs.eigvals(), wires=mp.obs.wires)])`
  can be used instead.
  [(#8468)](https://github.com/PennyLaneAI/pennylane/pull/8468)

* The `qml.QNode.add_transform` method is removed.
  Instead, please use `QNode.transform_program.push_back(transform_container=transform_container)`.
  [(#8468)](https://github.com/PennyLaneAI/pennylane/pull/8468)

<h3>Deprecations 👋</h3>

* `qml.measure`, `qml.measurements.MidMeasureMP`, `qml.measurements.MeasurementValue`,
  and `qml.measurements.get_mcm_predicates` are now located in `qml.ops.mid_measure`.
  `MidMeasureMP` is now renamed to `MidMeasure`.
  `qml.measurements.find_post_processed_mcms` is now `qml.devices.qubit.simulate._find_post_processed_mcms`,
  and is being made private, as it is an utility for tree-traversal.
  [(#8466)](https://github.com/PennyLaneAI/pennylane/pull/8466)

* The ``pennylane.operation.Operator.is_hermitian`` property has been deprecated and renamed
  to ``pennylane.operation.Operator.is_verified_hermitian`` as it better reflects the functionality of this property.
  The deprecated access through ``is_hermitian`` will be removed in PennyLane v0.45.
  Alternatively, consider using the ``pennylane.is_hermitian`` function instead as it provides a more reliable check for hermiticity.
  Please be aware that it comes with a higher computational cost.
  [(#8494)](https://github.com/PennyLaneAI/pennylane/pull/8494)

* Access to the follow functions and classes from the ``pennylane.resources`` module are deprecated. Instead, these functions must be imported from the ``pennylane.estimator`` module.
  [(#8484)](https://github.com/PennyLaneAI/pennylane/pull/8484)

    - ``qml.estimator.estimate_shots`` in favor of ``qml.resources.estimate_shots``
    - ``qml.estimator.estimate_error`` in favor of ``qml.resources.estimate_error``
    - ``qml.estimator.FirstQuantization`` in favor of ``qml.resources.FirstQuantization``
    - ``qml.estimator.DoubleFactorization`` in favor of ``qml.resources.DoubleFactorization``

* ``argnum`` has been renamed ``argnums`` for ``qml.grad``, ``qml.jacobian``, ``qml.jvp`` and ``qml.vjp``.
  [(#8496)](https://github.com/PennyLaneAI/pennylane/pull/8496)
  [(#8481)](https://github.com/PennyLaneAI/pennylane/pull/8481)

* The :func:`pennylane.devices.preprocess.mid_circuit_measurements` transform is deprecated. Instead,
  the device should determine which mcm method to use, and explicitly include :func:`~pennylane.transforms.dynamic_one_shot`
  or :func:`~pennylane.transforms.defer_measurements` in its preprocess transforms if necessary.
  [(#8467)](https://github.com/PennyLaneAI/pennylane/pull/8467)

<h3>Internal changes ⚙️</h3>

* The `grad` and `jacobian` primitives now store the function under `fn`. There is also now a single `jacobian_p`
  primitive for use in program capture.
  [(#8357)](https://github.com/PennyLaneAI/pennylane/pull/8357)

* Fix all NumPy 1.X `DeprecationWarnings` in our source code.
  [(#8497)](https://github.com/PennyLaneAI/pennylane/pull/8497)

* Update versions for `pylint`, `isort` and `black` in `format.yml`
  [(#8506)](https://github.com/PennyLaneAI/pennylane/pull/8506)

* Reclassifies `registers` as a tertiary module for use with tach.
  [(#8513)](https://github.com/PennyLaneAI/pennylane/pull/8513)

* A new `split_non_commuting_pass` compiler pass has been added to the xDSL transforms. This pass
  splits quantum functions that measure observables on the same wires into multiple function executions,
  where each execution measures observables on different wires (using the "wires" grouping strategy).
  The original function is replaced with calls to these generated functions, and the results are combined
  appropriately.
  [(#8531)](https://github.com/PennyLaneAI/pennylane/pull/8531)

* The experimental xDSL implementation of `diagonalize_measurements` has been updated to fix a bug
  that included the wrong SSA value for final qubit insertion and deallocation at the end of the
  circuit. A clear error is now also raised when there are observables with overlapping wires.
  [(#8383)](https://github.com/PennyLaneAI/pennylane/pull/8383)

* Add an `outline_state_evolution_pass` pass to the MBQC xDSL transform, which moves all
  quantum gate operations to a private callable.
  [(#8367)](https://github.com/PennyLaneAI/pennylane/pull/8367)

* The experimental xDSL implementation of `measurements_from_samples_pass` has been updated to support `shots` defined by an `arith.constant` operation.
  [(#8460)](https://github.com/PennyLaneAI/pennylane/pull/8460)

* The :class:`~pennylane.devices.LegacyDeviceFacade` is slightly refactored to implement `setup_execution_config` and `preprocess_transforms`
  separately as opposed to implementing a single `preprocess` method. Additionally, the `mid_circuit_measurements` transform has been removed
  from the preprocess transform program. Instead, the best mcm method is chosen in `setup_execution_config`. By default, the ``_capabilities``
  dictionary is queried for the ``"supports_mid_measure"`` property. If the underlying device defines a TOML file, the ``supported_mcm_methods``
  field in the TOML file is used as the source of truth.
  [(#8469)](https://github.com/PennyLaneAI/pennylane/pull/8469)
  [(#8486)](https://github.com/PennyLaneAI/pennylane/pull/8486)
  [(#8495)](https://github.com/PennyLaneAI/pennylane/pull/8495)

* The various private functions of the :class:`~pennylane.estimator.FirstQuantization` class have
  been modified to avoid using `numpy.matrix` as this function is deprecated.
  [(#8523)](https://github.com/PennyLaneAI/pennylane/pull/8523)

* The `ftqc` module now includes dummy transforms for several Catalyst/MLIR passes (`to-ppr`, `commute-ppr`, `merge-ppr-ppm`,
  `decompose-clifford-ppr`, `decompose-non-clifford-ppr`, `ppr-to-ppm`, `ppr-to-mbqc` and `reduce-t-depth`), to allow them to
  be captured as primitives in PLxPR and mapped to the MLIR passes in Catalyst. This enables using the passes with the unified
  compiler and program capture.
  [(#8519)](https://github.com/PennyLaneAI/pennylane/pull/8519)
  [(#8544)](https://github.com/PennyLaneAI/pennylane/pull/8544)

* The decompositions for several templates have been updated to use
  :class:`~.ops.op_math.ChangeOpBasis`, which makes their decompositions more resource efficient
  by eliminating unnecessary controlled operations. The templates include :class:`~.PhaseAdder`,
  :class:`~.TemporaryAND`, :class:`~.QSVT`, and :class:`~.SelectPauliRot`.
  [(#8490)](https://github.com/PennyLaneAI/pennylane/pull/8490)
  [(#8577)](https://github.com/PennyLaneAI/pennylane/pull/8577)

* The constant to convert the length unit Bohr to Angstrom in ``qml.qchem`` is updated to use scipy
  constants.
  [(#8537)](https://github.com/PennyLaneAI/pennylane/pull/8537)

* Solovay-Kitaev decomposition using the :func:`~.clifford_t_decompostion` transform
  with ``method="sk"`` or directly via :func:`~.ops.sk_decomposition` now raises a more
  informative ``RuntimeError`` when used with JAX-JIT or :func:`~.qjit`.
  [(#8489)](https://github.com/PennyLaneAI/pennylane/pull/8489)

* The `is_xdsl_pass` function has been added to the `pennylane.compiler.python_compiler.pass_api` module.
  This function checks if a pass name corresponds to an xDSL implemented pass.
  [(#8572)](https://github.com/PennyLaneAI/pennylane/pull/8572)

* The :func:`~pennylane.compiler.python_compiler.Compiler.run` method now accepts a string as input,
  which is parsed and transformed with xDSL.
  [(#8587)](https://github.com/PennyLaneAI/pennylane/pull/8587)

<h3>Documentation 📝</h3>

* Added a "Unified Compiler Cookbook" RST file, along with tutorials, to ``qml.compiler.python_compiler`,
  which provides a quickstart guide for getting started with xDSL and its integration with PennyLane and
  Catalyst.
  [(#8571)](https://github.com/PennyLaneAI/pennylane/pull/8571)

* The documentation of ``qml.transforms.rz_phase_gradient`` has been updated with respect to the
  sign convention of phase gradient states, how it prepares the phase gradient state in the code
  example, and the verification of the code example result.

* The code example in the documentation for ``qml.decomposition.register_resources`` has been
  updated to adhere to renamed keyword arguments and default behaviour of ``max_work_wires``.
  [(#8536)](https://github.com/PennyLaneAI/pennylane/pull/8536)

* The docstring for ``qml.device`` has been updated to include a section on custom decompositions,
  and a warning about the removal of the ``custom_decomps`` kwarg in v0.45. Additionally, the page
  :doc:`Building a plugin <../development/plugins>` now includes instructions on using
  the :func:`~pennylane.devices.preprocess.decompose` transform for device-level decompositions.
  The documentation for :doc:`Compiling circuits <../introduction/compiling_circuits>` has also been
  updated with a warning message about ``custom_decomps`` future removal.
  [(#8492)](https://github.com/PennyLaneAI/pennylane/pull/8492)
  [(#8564)](https://github.com/PennyLaneAI/pennylane/pull/8564)

A warning message has been added to :doc:`Building a plugin <../development/plugins>`
  docstring for ``qml.device`` has been updated to include a section on custom decompositions,
  and a warning about the removal of the ``custom_decomps`` kwarg in v0.44. Additionally, the page
  :doc:`Building a plugin <../development/plugins>` now includes instructions on using
  the :func:`~pennylane.devices.preprocess.decompose` transform for device-level decompositions.
  [(#8492)](https://github.com/PennyLaneAI/pennylane/pull/8492)

* Improves documentation in the transforms module and adds documentation testing for it.
  [(#8557)](https://github.com/PennyLaneAI/pennylane/pull/8557)

<h3>Bug fixes 🐛</h3>

* Add an exception to the warning for unsolved operators within the graph-based decomposition
  system if the unsolved operators are :class:`.allocation.Allocate` or :class:`.allocation.Deallocate`.
  [(#8553)](https://github.com/PennyLaneAI/pennylane/pull/8553)

* Fixes a bug in `clifford_t_decomposition` with `method="gridsynth"` and qjit, where using cached decomposition with the same parameter causes an error.
  [(#8535)](https://github.com/PennyLaneAI/pennylane/pull/8535)

* Fixes a bug in :class:`~.SemiAdder` where the results were incorrect when more ``work_wires`` than required were passed.
 [(#8423)](https://github.com/PennyLaneAI/pennylane/pull/8423)

* Fixes a bug in ``QubitUnitaryOp.__init__`` in the unified compiler module that prevented an
  instance from being constructed.
  [(#8456)](https://github.com/PennyLaneAI/pennylane/pull/8456)

* Fixes a bug where the deferred measurement method is used silently even if ``mcm_method="one-shot"`` is explicitly requested,
  when a device that extends the ``LegacyDevice`` does not declare support for mid-circuit measurements.
  [(#8486)](https://github.com/PennyLaneAI/pennylane/pull/8486)

* Fixes a bug where a `KeyError` is raised when querying the decomposition rule for an operator in the gate set from a :class:`~pennylane.decomposition.DecompGraphSolution`.
  [(#8526)](https://github.com/PennyLaneAI/pennylane/pull/8526)

* Fixes a bug where mid-circuit measurements were generating incomplete QASM.
  [(#8556)](https://github.com/PennyLaneAI/pennylane/pull/8556)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):


Guillermo Alonso,
Utkarsh Azad,
Astral Cai,
Marcus Edwards,
Lillian Frederiksen,
Soran Jahangiri,
Jacob Kitchen,
Christina Lee,
Joseph Lee,
Gabriela Sanchez Diaz,
Mudit Pandey,
Shuli Shu,
Jay Soni,
nate stemen,
David Wierichs,
Hongsheng Zheng
