:orphan:

# Release 0.41.0-dev (development release)

<h3>New features since last release</h3>

* `qml.defer_measurements` can now be used with program capture enabled. Programs transformed by
  `qml.defer_measurements` can be executed on `default.qubit`.
  [(#6838)](https://github.com/PennyLaneAI/pennylane/pull/6838)
  [(#6937)](https://github.com/PennyLaneAI/pennylane/pull/6937)

  Using `qml.defer_measurements` with program capture enables many new features, including:
  * Significantly richer variety of classical processing on mid-circuit measurement values.
  * Using mid-circuit measurement values as gate parameters.

  Functions such as the following can now be captured:

  ```python
  import jax.numpy as jnp

  qml.capture.enable()

  def f(x):
      m0 = qml.measure(0)
      m1 = qml.measure(0)
      a = jnp.sin(0.5 * jnp.pi * m0)
      phi = a - (m1 + 1) ** 4

      qml.s_prod(x, qml.RZ(phi, 0))

      return qml.expval(qml.Z(0))
  ```

* Added class `qml.capture.transforms.UnitaryToRotInterpreter` that decomposes `qml.QubitUnitary` operators 
  following the same API as `qml.transforms.unitary_to_rot` when experimental program capture is enabled.
  [(#6916)](https://github.com/PennyLaneAI/pennylane/pull/6916)

<h3>Improvements 🛠</h3>

* `Controlled` operators now have a full implementation of `sparse_matrix` that supports `wire_order` configuration.
  [(#6994)](https://github.com/PennyLaneAI/pennylane/pull/6994)

* The `qml.measurements.NullMeasurement` measurement process is added to allow for profiling problems
  without the overheads associated with performing measurements.
  [(#6989)](https://github.com/PennyLaneAI/pennylane/pull/6989)

* `pauli_rep` property is now accessible for `Adjoint` operator when there is a Pauli representation.
  [(#6871)](https://github.com/PennyLaneAI/pennylane/pull/6871)

* `qml.SWAP` now has sparse representation.
  [(#6965)](https://github.com/PennyLaneAI/pennylane/pull/6965)

* `qml.QubitUnitary` now accepts sparse CSR matrices (from `scipy.sparse`). This allows efficient representation of large unitaries with mostly zero entries. Note that sparse unitaries are still in early development and may not support all features of their dense counterparts.
  [(#6889)](https://github.com/PennyLaneAI/pennylane/pull/6889)

  ```pycon
  >>> import numpy as np
  >>> import pennylane as qml
  >>> import scipy as sp
  >>> U_dense = np.eye(4)  # 2-wire identity
  >>> U_sparse = sp.sparse.csr_matrix(U_dense)
  >>> op = qml.QubitUnitary(U_sparse, wires=[0, 1])
  >>> print(op.matrix())
  <Compressed Sparse Row sparse matrix of dtype 'float64'
          with 4 stored elements and shape (4, 4)>
    Coords        Values
    (0, 0)        1.0
    (1, 1)        1.0
    (2, 2)        1.0
    (3, 3)        1.0
  >>> op.matrix().toarray()
  array([[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]])
  ```

* Add a decomposition for multi-controlled global phases into a one-less-controlled phase shift.
  [(#6936)](https://github.com/PennyLaneAI/pennylane/pull/6936)
 
* `qml.StatePrep` now accepts sparse state vectors. Users can create `StatePrep` using `scipy.sparse.csr_matrix`. Note that non-zero `pad_with` is forbidden.
  [(#6863)](https://github.com/PennyLaneAI/pennylane/pull/6863)

  ```pycon
  >>> import scipy as sp
  >>> init_state = sp.sparse.csr_matrix([0, 0, 1, 0])
  >>> qsv_op = qml.StatePrep(init_state, wires=[1, 2])
  >>> wire_order = [0, 1, 2]
  >>> ket = qsv_op.state_vector(wire_order=wire_order)
  >>> print(ket)
  <Compressed Sparse Row sparse matrix of dtype 'float64'
         with 1 stored elements and shape (1, 8)>
    Coords        Values
    (0, 2)        1.0
  ```

* A `RuntimeWarning` is now raised by `qml.QNode` and `qml.execute` if executing JAX workflows and the installed version of JAX
  is greater than `0.4.28`.
  [(#6864)](https://github.com/PennyLaneAI/pennylane/pull/6864)

* Added the `qml.workflow.construct_execution_config(qnode)(*args,**kwargs)` helper function.
  Users can now construct the execution configuration from a particular `QNode` instance.
  [(#6901)](https://github.com/PennyLaneAI/pennylane/pull/6901)

  ```python
  @qml.qnode(qml.device("default.qubit", wires=1))
  def circuit(x):
      qml.RX(x, 0)
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> config = qml.workflow.construct_execution_config(circuit)(1)
  >>> pprint.pprint(config)
  ExecutionConfig(grad_on_execution=False,
                  use_device_gradient=True,
                  use_device_jacobian_product=False,
                  gradient_method='backprop',
                  gradient_keyword_arguments={},
                  device_options={'max_workers': None,
                                  'prng_key': None,
                                  'rng': Generator(PCG64) at 0x15F6BB680},
                  interface=<Interface.NUMPY: 'numpy'>,
                  derivative_order=1,
                  mcm_config=MCMConfig(mcm_method=None, postselect_mode=None),
                  convert_to_numpy=True)
  ```

* `QNode` objects now have an `update` method that allows for re-configuring settings like `diff_method`, `mcm_method`, and more. This allows for easier on-the-fly adjustments to workflows. Any arguments not specified will retain their original value.
  [(#6803)](https://github.com/PennyLaneAI/pennylane/pull/6803)

  After constructing a `QNode`,

  ```python
  import pennylane as qml

  @qml.qnode(device=qml.device("default.qubit"))
  def circuit():
    qml.H(0)
    qml.CNOT([0,1])
    return qml.probs()
  ```

  its settings can be modified with `update`, which returns a new `QNode` object. Here is an example
  of updating a QNode's `diff_method`:

  ```pycon
  >>> print(circuit.diff_method)
  best
  >>> new_circuit = circuit.update(diff_method="parameter-shift")
  >>> print(new_circuit.diff_method)
  'parameter-shift'
  ```

* Devices can now configure whether or not ML framework data is sent to them
  via an `ExecutionConfig.convert_to_numpy` parameter. End-to-end jitting on
  `default.qubit` is used if the user specified a `jax.random.PRNGKey` as a seed.
  [(#6899)](https://github.com/PennyLaneAI/pennylane/pull/6899)
  [(#6788)](https://github.com/PennyLaneAI/pennylane/pull/6788)
  [(#6869)](https://github.com/PennyLaneAI/pennylane/pull/6869)

* The coefficients of observables now have improved differentiability.
  [(#6598)](https://github.com/PennyLaneAI/pennylane/pull/6598)

* An empty basis set in `qml.compile` is now recognized as valid, resulting in decomposition of all operators that can be decomposed.
   [(#6821)](https://github.com/PennyLaneAI/pennylane/pull/6821)

* An informative error is raised when a `QNode` with `diff_method=None` is differentiated.
  [(#6770)](https://github.com/PennyLaneAI/pennylane/pull/6770)

* `qml.ops.sk_decomposition` has been improved to produce less gates for certain edge cases. This greatly impacts
  the performance of `qml.clifford_t_decomposition`, which should now give less extraneous `qml.T` gates.
  [(#6855)](https://github.com/PennyLaneAI/pennylane/pull/6855)

* `qml.gradients.finite_diff_jvp` has been added to compute the jvp of an arbitrary numeric
  function.
  [(#6853)](https://github.com/PennyLaneAI/pennylane/pull/6853)

* With program capture enabled, `QNode`'s can now be differentiated with `diff_method="finite-diff"`.
  [(#6853)](https://github.com/PennyLaneAI/pennylane/pull/6853)

* The requested `diff_method` is now validated when program capture is enabled.
  [(#6852)](https://github.com/PennyLaneAI/pennylane/pull/6852)

* The `qml.clifford_t_decomposition` has been improved to use less gates when decomposing `qml.PhaseShift`.
  [(#6842)](https://github.com/PennyLaneAI/pennylane/pull/6842)

* A `ParametrizedMidMeasure` class is added to represent a mid-circuit measurement in an arbitrary
  measurement basis in the XY, YZ or ZX plane. 
  [(#6938)](https://github.com/PennyLaneAI/pennylane/pull/6938)

* A `diagonalize_mcms` transform is added that diagonalizes any `ParametrizedMidMeasure`, for devices 
  that only natively support mid-circuit measurements in the computational basis.
  [(#6938)](https://github.com/PennyLaneAI/pennylane/pull/6938)
  
* `null.qubit` can now execute jaxpr.
  [(#6924)](https://github.com/PennyLaneAI/pennylane/pull/6924)

<h4>Capturing and representing hybrid programs</h4>

* `qml.QNode` can now cache plxpr. When executing a `QNode` for the first time, its plxpr representation will
  be cached based on the abstract evaluation of the arguments. Later executions that have arguments with the
  same shapes and data types will be able to use this cached plxpr instead of capturing the program again.
  [(#6923)](https://github.com/PennyLaneAI/pennylane/pull/6923)

* `qml.QNode` now accepts a `static_argnums` argument. This argument can be used to indicate any arguments that
  should be considered static when capturing the quantum program.
  [(#6923)](https://github.com/PennyLaneAI/pennylane/pull/6923)

* Implemented a `compute_plxpr_decomposition` method in the `qml.operation.Operator` class to apply dynamic decompositions
  with program capture enabled.
  [(#6859)](https://github.com/PennyLaneAI/pennylane/pull/6859)

  * Autograph can now be used with custom operations defined outside of the pennylane namespace.
  [(#6931)](https://github.com/PennyLaneAI/pennylane/pull/6931)

  * Add a `qml.capture.pause()` context manager for pausing program capture in an error-safe way.
  [(#6911)](https://github.com/PennyLaneAI/pennylane/pull/6911)

* Python control flow (`if/else`, `for`, `while`) is now supported when program capture is enabled by setting 
  `autograph=True` at the QNode level. 
  [(#6837)](https://github.com/PennyLaneAI/pennylane/pull/6837)

  ```python
  qml.capture.enable()

  dev = qml.device("default.qubit", wires=[0, 1, 2])

  @qml.qnode(dev, autograph=True)
  def circuit(num_loops: int):
      for i in range(num_loops):
          if i % 2 == 0:
              qml.H(i)
          else:
              qml.RX(1,i)
      return qml.state()
  ```

  ```pycon
  >>> print(qml.draw(circuit)(num_loops=3))
  0: ──H────────┤  State
  1: ──RX(1.00)─┤  State
  2: ──H────────┤  State
  >>> circuit(3)
  Array([0.43879125+0.j        , 0.43879125+0.j        ,
         0.        -0.23971277j, 0.        -0.23971277j,
         0.43879125+0.j        , 0.43879125+0.j        ,
         0.        -0.23971277j, 0.        -0.23971277j], dtype=complex64)
  ```

* The higher order primitives in program capture can now accept inputs with abstract shapes.
  [(#6786)](https://github.com/PennyLaneAI/pennylane/pull/6786)

* The `PlxprInterpreter` classes can now handle creating dynamic arrays via `jnp.ones`, `jnp.zeros`,
  `jnp.arange`, and `jnp.full`.
  [#6865)](https://github.com/PennyLaneAI/pennylane/pull/6865)

* The qnode primitive now stores the `ExecutionConfig` instead of `qnode_kwargs`.
  [(#6991)](https://github.com/PennyLaneAI/pennylane/pull/6991)

* `Device.eval_jaxpr` now accepts an `execution_config` keyword argument.
  [(#6991)](https://github.com/PennyLaneAI/pennylane/pull/6991)

* The adjoint jvp of a jaxpr can be computed using default.qubit tooling.
  [(#6875)](https://github.com/PennyLaneAI/pennylane/pull/6875)

<h3>Breaking changes 💔</h3>

* `MultiControlledX` no longer accepts strings as control values.
  [(#6835)](https://github.com/PennyLaneAI/pennylane/pull/6835)

* The input argument `control_wires` of `MultiControlledX` has been removed.
  [(#6832)](https://github.com/PennyLaneAI/pennylane/pull/6832)
  [(#6862)](https://github.com/PennyLaneAI/pennylane/pull/6862)

* `qml.execute` now has a collection of keyword-only arguments.
  [(#6598)](https://github.com/PennyLaneAI/pennylane/pull/6598)

* The ``decomp_depth`` argument in :func:`~pennylane.transforms.set_decomposition` has been removed.
  [(#6824)](https://github.com/PennyLaneAI/pennylane/pull/6824)

* The ``max_expansion`` argument in :func:`~pennylane.devices.preprocess.decompose` has been removed.
  [(#6824)](https://github.com/PennyLaneAI/pennylane/pull/6824)

* The ``tape`` and ``qtape`` properties of ``QNode`` have been removed.
  Instead, use the ``qml.workflow.construct_tape`` function.
  [(#6825)](https://github.com/PennyLaneAI/pennylane/pull/6825)

* The ``gradient_fn`` keyword argument to ``qml.execute`` has been removed. Instead, it has been replaced with ``diff_method``.
  [(#6830)](https://github.com/PennyLaneAI/pennylane/pull/6830)
  
* The ``QNode.get_best_method`` and ``QNode.best_method_str`` methods have been removed.
  Instead, use the ``qml.workflow.get_best_diff_method`` function.
  [(#6823)](https://github.com/PennyLaneAI/pennylane/pull/6823)

* The `output_dim` property of `qml.tape.QuantumScript` has been removed. Instead, use method `shape` of `QuantumScript` or `MeasurementProcess` to get the same information.
  [(#6829)](https://github.com/PennyLaneAI/pennylane/pull/6829)

* Removed method `qsvt_legacy` along with its private helper `_qsp_to_qsvt`
  [(#6827)](https://github.com/PennyLaneAI/pennylane/pull/6827)

<h3>Deprecations 👋</h3>

* The ``ControlledQubitUnitary`` will stop accepting `QubitUnitary` objects as arguments as its ``base``. Instead, use ``qml.ctrl`` to construct a controlled `QubitUnitary`.
  A folllow-on PR fixed accidental double-queuing when using `qml.ctrl` with `QubitUnitary`.
  [(#6840)](https://github.com/PennyLaneAI/pennylane/pull/6840)
  [(#6926)](https://github.com/PennyLaneAI/pennylane/pull/6926)

* The `control_wires` argument in `qml.ControlledQubitUnitary` has been deprecated.
  Instead, use the `wires` argument as the second positional argument.
  [(#6839)](https://github.com/PennyLaneAI/pennylane/pull/6839)

* The `mcm_method` keyword in `qml.execute` has been deprecated.
  Instead, use the ``mcm_method`` and ``postselect_mode`` arguments.
  [(#6807)](https://github.com/PennyLaneAI/pennylane/pull/6807)

* Specifying gradient keyword arguments as any additional keyword argument to the qnode is deprecated
  and will be removed in v0.42.  The gradient keyword arguments should be passed to the new
  keyword argument `gradient_kwargs` via an explicit dictionary. This change will improve qnode argument
  validation.
  [(#6828)](https://github.com/PennyLaneAI/pennylane/pull/6828)

* The `qml.gradients.hamiltonian_grad` function has been deprecated.
  This gradient recipe is not required with the new operator arithmetic system.
  [(#6849)](https://github.com/PennyLaneAI/pennylane/pull/6849)

* The ``inner_transform_program`` and ``config`` keyword arguments in ``qml.execute`` have been deprecated.
  If more detailed control over the execution is required, use ``qml.workflow.run`` with these arguments instead.
  [(#6822)](https://github.com/PennyLaneAI/pennylane/pull/6822)
  [(#6879)](https://github.com/PennyLaneAI/pennylane/pull/6879)

* The property `MeasurementProcess.return_type` has been deprecated.
  If observable type checking is needed, please use direct `isinstance`; if other text information is needed, please use class name, or another internal temporary private member `_shortname`.
  [(#6841)](https://github.com/PennyLaneAI/pennylane/pull/6841)
  [(#6906)](https://github.com/PennyLaneAI/pennylane/pull/6906)
  [(#6910)](https://github.com/PennyLaneAI/pennylane/pull/6910)

<h3>Internal changes ⚙️</h3>

* Minor changes to `DQInterpreter` for speedups with program capture execution.
  [(#6984)](https://github.com/PennyLaneAI/pennylane/pull/6984)

* Globally silences `no-member` pylint issues from jax.
  [(#6987)](https://github.com/PennyLaneAI/pennylane/pull/6987)

* Fix `pylint=3.3.4` errors in source code.
  [(#6980)](https://github.com/PennyLaneAI/pennylane/pull/6980)
  [(#6988)](https://github.com/PennyLaneAI/pennylane/pull/6988)

* Remove `QNode.get_gradient_fn` from source code.
  [(#6898)](https://github.com/PennyLaneAI/pennylane/pull/6898)
  
* The source code has been updated use black 25.1.0.
  [(#6897)](https://github.com/PennyLaneAI/pennylane/pull/6897)

* Improved the `InterfaceEnum` object to prevent direct comparisons to `str` objects.
  [(#6877)](https://github.com/PennyLaneAI/pennylane/pull/6877)

* Added a `QmlPrimitive` class that inherits `jax.core.Primitive` to a new `qml.capture.custom_primitives` module.
  This class contains a `prim_type` property so that we can differentiate between different sets of PennyLane primitives.
  Consequently, `QmlPrimitive` is now used to define all PennyLane primitives.
  [(#6847)](https://github.com/PennyLaneAI/pennylane/pull/6847)

* The `RiemannianGradientOptimizer` has been updated to take advantage of newer features.
  [(#6882)](https://github.com/PennyLaneAI/pennylane/pull/6882)

* Use `keep_intermediate=True` flag to keep Catalyst's IR when testing.
  Also use a different way of testing to see if something was compiled.
  [(#6990)](https://github.com/PennyLaneAI/pennylane/pull/6990)

<h3>Documentation 📝</h3>

* The code example in the docstring for `qml.PauliSentence` now properly copy-pastes.
  [(#6949)](https://github.com/PennyLaneAI/pennylane/pull/6949)

* The docstrings for `qml.unary_mapping`, `qml.binary_mapping`, `qml.christiansen_mapping`,
  `qml.qchem.localize_normal_modes`, and `qml.qchem.VibrationalPES` have been updated to include better
  code examples.
  [(#6717)](https://github.com/PennyLaneAI/pennylane/pull/6717)

* The docstrings for `qml.qchem.localize_normal_modes` and `qml.qchem.VibrationalPES` have been updated to include
  examples that can be copied.
  [(#6834)](https://github.com/PennyLaneAI/pennylane/pull/6834)

* Fixed a typo in the code example for `qml.labs.dla.lie_closure_dense`.
  [(#6858)](https://github.com/PennyLaneAI/pennylane/pull/6858)

* The code example in the docstring for `qml.BasisRotation` was corrected by including `wire_order` in the 
  call to `qml.matrix`.
  [(#6891)](https://github.com/PennyLaneAI/pennylane/pull/6891)

* The docstring of `qml.noise.meas_eq` has been updated to make its functionality clearer.
  [(#6920)](https://github.com/PennyLaneAI/pennylane/pull/6920)

<h3>Bug fixes 🐛</h3>

* `qml.capture.PlxprInterpreter` now flattens pytree arguments before evaluation.
  [(#6975)](https://github.com/PennyLaneAI/pennylane/pull/6975)

* `qml.GlobalPhase.sparse_matrix` now correctly returns a sparse matrix of the same shape as `matrix`.
  [(#6940)](https://github.com/PennyLaneAI/pennylane/pull/6940)

* `qml.expval` no longer silently casts to a real number when observable coefficients are imaginary.
  [(#6939)](https://github.com/PennyLaneAI/pennylane/pull/6939)

* Fixed `qml.wires.Wires` initialization to disallow `Wires` objects as wires labels.
  Now, `Wires` is idempotent, e.g. `Wires([Wires([0]), Wires([1])])==Wires([0, 1])`.
  [(#6933)](https://github.com/PennyLaneAI/pennylane/pull/6933)

* `qml.capture.PlxprInterpreter` now correctly handles propagation of constants when interpreting higher-order primitives
  [(#6913)](https://github.com/PennyLaneAI/pennylane/pull/6913)

* `qml.capture.PlxprInterpreter` now uses `Primitive.get_bind_params` to resolve primitive calling signatures before binding
  primitives.
  [(#6913)](https://github.com/PennyLaneAI/pennylane/pull/6913)

* The interface is now detected from the data in the circuit, not the arguments to the `QNode`. This allows
  interface data to be strictly passed as closure variables and still be detected.
  [(#6892)](https://github.com/PennyLaneAI/pennylane/pull/6892)

* `BasisState` now casts its input to integers.
  [(#6844)](https://github.com/PennyLaneAI/pennylane/pull/6844)

* The `workflow.contstruct_batch` and `workflow.construct_tape` functions now correctly reflect the `mcm_method`
  passed to the `QNode`, instead of assuming the method is always `deferred`.
  [(#6903)](https://github.com/PennyLaneAI/pennylane/pull/6903)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Utkarsh Azad,
Henry Chang,
Yushao Chen,
Isaac De Vlugt,
Diksha Dhawan,
Lillian M.A. Frederiksen,
Pietropaolo Frisoni,
Marcus Gisslén,
Christina Lee,
Mudit Pandey,
Andrija Paurevic,
David Wierichs
