:orphan:

# Release 0.41.0-dev (development release)

<h3>New features since last release</h3>

* Added method `qml.math.sqrt_matrix_sparse` to compute the square root of a sparse Hermitian matrix.
  [(#6976)](https://github.com/PennyLaneAI/pennylane/pull/6976)

* Added a class `qml.capture.transforms.MergeRotationsInterpreter` that merges rotation operators
  following the same API as `qml.transforms.optimization.merge_rotations` when experimental program capture is enabled.
  [(#6957)](https://github.com/PennyLaneAI/pennylane/pull/6957)

* `qml.defer_measurements` can now be used with program capture enabled. Programs transformed by
  `qml.defer_measurements` can be executed on `default.qubit`.
  [(#6838)](https://github.com/PennyLaneAI/pennylane/pull/6838)
  [(#6937)](https://github.com/PennyLaneAI/pennylane/pull/6937)
  [(#6961)](https://github.com/PennyLaneAI/pennylane/pull/6961)

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
  [(#6977)](https://github.com/PennyLaneAI/pennylane/pull/6977)

* Created a new ``qml.liealg`` module for Lie algebra functionality.
  [(#6935)](https://github.com/PennyLaneAI/pennylane/pull/6935)

* ``qml.lie_closure`` now accepts and outputs matrix inputs using the ``matrix`` keyword.
  Also added ``qml.pauli.trace_inner_product`` that can handle batches of dense matrices.
  [(#6811)](https://github.com/PennyLaneAI/pennylane/pull/6811)

* ``qml.structure_constants`` now accepts and outputs matrix inputs using the ``matrix`` keyword.
  [(#6861)](https://github.com/PennyLaneAI/pennylane/pull/6861)

<h3>Improvements 🛠</h3>
  
* The decompositions of `qml.SX`, `qml.X` and `qml.Y` use `qml.GlobalPhase` instead of `qml.PhaseShift`.
  [(#7073)](https://github.com/PennyLaneAI/pennylane/pull/7073)  

* The `reference.qubit` device now enforces `sum(probs)==1` in `sample_state`.
  [(#7076)](https://github.com/PennyLaneAI/pennylane/pull/7076)

* The `default.mixed` device now adheres to the newer device API introduced in
  [v0.33](https://docs.pennylane.ai/en/stable/development/release_notes.html#release-0-33-0).
  This means that `default.mixed` now supports not having to specify the number of wires,
  more predictable behaviour with interfaces, support for `qml.Snapshot`, and more.
  [(#6684)](https://github.com/PennyLaneAI/pennylane/pull/6684)

* `qml.BlockEncode` now accepts sparse input and outputs sparse matrices.
  [(#6963)](https://github.com/PennyLaneAI/pennylane/pull/6963)

* `Operator.sparse_matrix` now supports `format` parameter to specify the returned scipy sparse matrix format,
  with the default being `'csr'`
  [(#6995)](https://github.com/PennyLaneAI/pennylane/pull/6995)

* Dispatch the linear algebra methods of `scipy` backend to `scipy.sparse.linalg` explicitly. Now `qml.math` can correctly
  handle sparse matrices.
  [(#6947)](https://github.com/PennyLaneAI/pennylane/pull/6947)

* Added a class `qml.capture.transforms.MergeAmplitudeEmbedding` that merges `qml.AmplitudeEmbedding` operators
  following the same API as `qml.transforms.merge_amplitude_embedding` when experimental program capture is enabled.
  [(#6925)](https://github.com/PennyLaneAI/pennylane/pull/6925)
  
* `default.qubit` now supports the sparse matrices to be applied to the state vector. Specifically, `QubitUnitary` initialized with a sparse matrix can now be applied to the state vector in the `default.qubit` device.
  [(#6883)](https://github.com/PennyLaneAI/pennylane/pull/6883)

* `merge_rotations` now correctly simplifies merged `qml.Rot` operators whose angles yield the identity operator.
  [(#7011)](https://github.com/PennyLaneAI/pennylane/pull/7011)
  
* Bump `rng_salt` to `v0.40.0`.
  [(#6854)](https://github.com/PennyLaneAI/pennylane/pull/6854)

* `qml.gradients.hadamard_grad` can now differentiate anything with a generator, and can accept circuits with non-commuting measurements.
[(#6928)](https://github.com/PennyLaneAI/pennylane/pull/6928)

* `Controlled` operators now have a full implementation of `sparse_matrix` that supports `wire_order` configuration.
  [(#6994)](https://github.com/PennyLaneAI/pennylane/pull/6994)

* The `qml.measurements.NullMeasurement` measurement process is added to allow for profiling problems
  without the overheads associated with performing measurements.
  [(#6989)](https://github.com/PennyLaneAI/pennylane/pull/6989)

* `pauli_rep` property is now accessible for `Adjoint` operator when there is a Pauli representation.
  [(#6871)](https://github.com/PennyLaneAI/pennylane/pull/6871)

* `qml.SWAP` now has sparse representation.
  [(#6965)](https://github.com/PennyLaneAI/pennylane/pull/6965)

* A `Lattice` class and a `generate_lattice` method is added to the `qml.ftqc` module. The `generate_lattice` method is to generate 1D, 2D, 3D grid graphs with the given geometric parameters.
  [(#6958)](https://github.com/PennyLaneAI/pennylane/pull/6958)

* `qml.QubitUnitary` now accepts sparse CSR matrices (from `scipy.sparse`). This allows efficient representation of large unitaries with mostly zero entries. Note that sparse unitaries are still in early development and may not support all features of their dense counterparts.
  [(#6889)](https://github.com/PennyLaneAI/pennylane/pull/6889)
  [(#6986)](https://github.com/PennyLaneAI/pennylane/pull/6986)

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

* The template `MPSPrep` now has a gate decomposition. This enables its use with any device.
  The `right_canonicalize_mps` function has also been added to transform an MPS into its right-canonical form.
  [(#6896)](https://github.com/PennyLaneAI/pennylane/pull/6896)

* The `qml.clifford_t_decomposition` has been improved to use less gates when decomposing `qml.PhaseShift`.
  [(#6842)](https://github.com/PennyLaneAI/pennylane/pull/6842)

* `qml.qchem.taper` now handles wire ordering for the tapered observables more robustly.
  [(#6954)](https://github.com/PennyLaneAI/pennylane/pull/6954)

* A `ParametrizedMidMeasure` class is added to represent a mid-circuit measurement in an arbitrary
  measurement basis in the XY, YZ or ZX plane. Subclasses `XMidMeasureMP` and `YMidMeasureMP` represent
  X-basis and Y-basis measurements. These classes are part of the experimental `ftqc` module.
  [(#6938)](https://github.com/PennyLaneAI/pennylane/pull/6938)
  [(#6953)](https://github.com/PennyLaneAI/pennylane/pull/6953)

* A `diagonalize_mcms` transform is added that diagonalizes any `ParametrizedMidMeasure`, for devices
  that only natively support mid-circuit measurements in the computational basis.
  [(#6938)](https://github.com/PennyLaneAI/pennylane/pull/6938)
  [(#7037)](https://github.com/PennyLaneAI/pennylane/pull/7037)

* Measurement functions `measure_x`, `measure_y` and `measure_arbitrary_basis` are added in the experimental `ftqc` module. These functions
  apply a mid-circuit measurement and return a `MeasurementValue`. They are analogous to `qml.measure` for
  the computational basis, but instead measure in the X-basis, Y-basis, or an arbitrary basis, respectively.
  Function `qml.ftqc.measure_z` is also added as an alias for `qml.measure`.
  [(#6953)](https://github.com/PennyLaneAI/pennylane/pull/6953)

* The function `cond_measure` is added to the experimental `ftqc` module to apply a mid-circuit 
  measurement with a measurement basis conditional on the function input.
  [(#7037)](https://github.com/PennyLaneAI/pennylane/pull/7037)
  
* `null.qubit` can now execute jaxpr.
  [(#6924)](https://github.com/PennyLaneAI/pennylane/pull/6924)

* A new class, `qml.ftqc.QubitGraph`, is now available for representing a qubit memory-addressing
  model for mappings between logical and physical qubits. This representation allows for nesting of
  lower-level qubits with arbitrary depth to allow easy insertion of arbitrarily many levels of
  abstractions between logical qubits and physical qubits.
  [(#6962)](https://github.com/PennyLaneAI/pennylane/pull/6962)

<h4>Capturing and representing hybrid programs</h4>

* Traditional tape transforms in PennyLane can be automatically converted to work with program capture enabled.
  [(#6922)](https://github.com/PennyLaneAI/pennylane/pull/6922)

  As an example, here is a custom tape transform, working with capture enabled, that shifts every `qml.RX` gate to the end of the circuit:

  ```python
  qml.capture.enable()

  @qml.transform
  def shift_rx_to_end(tape):
      """Transform that moves all RX gates to the end of the operations list."""
      new_ops, rxs = [], []

      for op in tape.operations:
          if isinstance(op, qml.RX):
              rxs.append(op)
          else:
                new_ops.append(op)

      operations = new_ops + rxs
      new_tape = tape.copy(operations=operations)
      return [new_tape], lambda res: res[0]
  ```

  A requirement for tape transforms to be compatible with program capture is to further decorate QNodes with the experimental
  `qml.capture.expand_plxpr_transforms` decorator.

  ```python
  @qml.capture.expand_plxpr_transforms
  @shift_rx_to_end
  @qml.qnode(qml.device("default.qubit", wires=1))
  def circuit():
      qml.RX(0.1, wires=0)
      qml.H(wires=0)
      return qml.state()
  ```

  ```pycon
  >>> print(qml.draw(circuit)())
  0: ──H──RX(0.10)─┤  State
  ```

  There are some exceptions to getting tape transforms to work with capture enabled:
  * Transforms that return multiple tapes cannot be converted.
  * Transforms that return non-trivial post-processing functions cannot be converted.
  * Transforms will fail to execute if the transformed quantum function or QNode contains:
    * `qml.cond` with dynamic parameters as predicates.
    * `qml.for_loop` with dynamic parameters for ``start``, ``stop``, or ``step``.
    * `qml.while_loop`.

* `Device.jaxpr_jvp` has been added to the device API to allow the definition of device derivatives
  when using program capture to jaxpr.
  [(#7019)](https://github.com/PennyLaneAI/pennylane/pull/7019)

* Device-provided derivatives are integrated into the program capture pipeline.
  `diff_method="adjoint"` can now be used with `default.qubit` when capture is enabled.
  [(#7019)](https://github.com/PennyLaneAI/pennylane/pull/7019)

* The `qml.transforms.single_qubit_fusion` quantum transform can now be applied with program capture enabled.
  [(#6945)](https://github.com/PennyLaneAI/pennylane/pull/6945)
  [(#7020)](https://github.com/PennyLaneAI/pennylane/pull/7020)

* Added class `qml.capture.transforms.CommuteControlledInterpreter` that moves commuting gates past control
  and target qubits of controlled operations when experimental program capture is enabled.
  It follows the same API as `qml.transforms.commute_controlled`.
  [(#6946)](https://github.com/PennyLaneAI/pennylane/pull/6946)

* `qml.QNode` can now cache plxpr. When executing a `QNode` for the first time, its plxpr representation will
  be cached based on the abstract evaluation of the arguments. Later executions that have arguments with the
  same shapes and data types will be able to use this cached plxpr instead of capturing the program again.
  [(#6923)](https://github.com/PennyLaneAI/pennylane/pull/6923)

* `qml.QNode` now accepts a `static_argnums` argument. This argument can be used to indicate any arguments that
  should be considered static when capturing the quantum program.
  [(#6923)](https://github.com/PennyLaneAI/pennylane/pull/6923)

* A new, experimental `Operator` method called `compute_qfunc_decomposition` has been added to represent decompositions with structure (e.g., control flow).
  This method is only used when capture is enabled with `qml.capture.enable()`.
  [(#6859)](https://github.com/PennyLaneAI/pennylane/pull/6859)
  [(#6881)](https://github.com/PennyLaneAI/pennylane/pull/6881)
  [(#7022)](https://github.com/PennyLaneAI/pennylane/pull/7022)
  [(#6917)](https://github.com/PennyLaneAI/pennylane/pull/6917)
  [(#7081)](https://github.com/PennyLaneAI/pennylane/pull/7081)

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

* A template class, `qml.ftqc.GraphStatePrep`, is added for the Graph state construction.
  [(#6985)](https://github.com/PennyLaneAI/pennylane/pull/6985)

* `qml.cond` can return arrays with dynamic shapes.
  [(#6888)](https://github.com/PennyLaneAI/pennylane/pull/6888/)
  [(#7080)](https://github.com/PennyLaneAI/pennylane/pull/7080)

* The qnode primitive now stores the `ExecutionConfig` instead of `qnode_kwargs`.
  [(#6991)](https://github.com/PennyLaneAI/pennylane/pull/6991)

* `Device.eval_jaxpr` now accepts an `execution_config` keyword argument.
  [(#6991)](https://github.com/PennyLaneAI/pennylane/pull/6991)

* The adjoint jvp of a jaxpr can be computed using default.qubit tooling.
  [(#6875)](https://github.com/PennyLaneAI/pennylane/pull/6875)

* A new `qml.capture.eval_jaxpr` function has been implemented. This is a variant of `jax.core.eval_jaxpr` that can handle the creation
  of arrays with dynamic shapes.
  [(#7052)](https://github.com/PennyLaneAI/pennylane/pull/7052)

* Add a `qml.capture.register_custom_staging_rule` for handling higher-order primitives
  that return new dynamically shaped arrays.
  [(#7086)](https://github.com/PennyLaneAI/pennylane/pull/7086)

* Execution interpreters and `qml.capture.eval_jaxpr` can now handle jax `pjit` primitives when dynamic shapes are being used.
  [(#7078)](https://github.com/PennyLaneAI/pennylane/pull/7078)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* ``pennylane.labs.dla.lie_closure_dense`` is removed and integrated into ``qml.lie_closure`` using the new ``dense`` keyword.
  [(#6811)](https://github.com/PennyLaneAI/pennylane/pull/6811)

* ``pennylane.labs.dla.structure_constants_dense`` is removed and integrated into ``qml.structure_constants`` using the new ``matrix`` keyword.
  [(#6861)](https://github.com/PennyLaneAI/pennylane/pull/6861)

* ``ResourceOperator.resource_params`` is changed to a property.
  [(#6973)](https://github.com/PennyLaneAI/pennylane/pull/6973)

<h3>Breaking changes 💔</h3>

* `num_diagonalizing_gates` is no longer accessible in `qml.specs` or `QuantumScript.specs`. The calculation of
  this quantity is extremely expensive, and the definition is ambiguous for non-commuting observables.
  [(#7047)](https://github.com/PennyLaneAI/pennylane/pull/7047)

* `qml.gradients.gradient_transform.choose_trainable_params` has been renamed to `choose_trainable_param_indices`
  to better reflect what it actually does.
  [(#6928)](https://github.com/PennyLaneAI/pennylane/pull/6928)

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

* Specifying `pipeline=None` with `qml.compile` is now deprecated. A sequence of
  transforms should always be specified.
  [(#7004)](https://github.com/PennyLaneAI/pennylane/pull/7004)

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

* Pauli module level imports of ``lie_closure``, ``structure_constants`` and ``center`` are deprecated, as functionality is moved to new ``liealg`` module.
  [(#6935)](https://github.com/PennyLaneAI/pennylane/pull/6935)

<h3>Internal changes ⚙️</h3>

* The test for `qml.math.quantum._denman_beavers_iterations` has been improved such that tested random matrices are guaranteed positive.
  [(#7071)](https://github.com/PennyLaneAI/pennylane/pull/7071)

* Replace `matrix_power` dispatch for `scipy` interface with an in-place implementation.
  [(#7055)](https://github.com/PennyLaneAI/pennylane/pull/7055)

* Add support to `CollectOpsandMeas` for handling `qnode` primitives.
  [(#6922)](https://github.com/PennyLaneAI/pennylane/pull/6922)

* Change some `scipy` imports from submodules to whole module to reduce memory footprint of importing pennylane.
  [(#7040)](https://github.com/PennyLaneAI/pennylane/pull/7040)

* Add `NotImplementedError`s for `grad` and `jacobian` in `CollectOpsandMeas`.
  [(#7041)](https://github.com/PennyLaneAI/pennylane/pull/7041)

* Quantum transform interpreters now perform argument validation and will no longer
  check if the equation in the `jaxpr` is a transform primitive.
  [(#7023)](https://github.com/PennyLaneAI/pennylane/pull/7023)

* `qml.for_loop` and `qml.while_loop` have been moved from the `compiler` module
  to a new `control_flow` module.
  [(#7017)](https://github.com/PennyLaneAI/pennylane/pull/7017)

* `qml.capture.run_autograph` is now idempotent.
  This means `run_autograph(fn) = run_autograph(run_autograph(fn))`.
  [(#7001)](https://github.com/PennyLaneAI/pennylane/pull/7001)

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

* The docstring for `qml.prod` has been updated to explain that the order of the output may seem reversed but it is correct.
  [(#7083)](https://github.com/PennyLaneAI/pennylane/pull/7083)

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

* Modulo operator calls on MCMs now correctly offload to the autoray-backed `qml.math.mod` dispatch.
  [(#7085)](https://github.com/PennyLaneAI/pennylane/pull/7085)

* Dynamic one-shot workloads are now faster for `null.qubit`.
  Removed a redundant `functools.lru_cache` call that was capturing all `SampleMP` objects in a workload.
  [(#7077)](https://github.com/PennyLaneAI/pennylane/pull/7077)

* `qml.transforms.single_qubit_fusion` and `qml.transforms.cancel_inverses` now correctly handle mid-circuit measurements
  when experimental program capture is enabled.
  [(#7020)](https://github.com/PennyLaneAI/pennylane/pull/7020)

* `qml.math.get_interface` now correctly extracts the `"scipy"` interface if provided a list/array
  of sparse matrices.
  [(#7015)](https://github.com/PennyLaneAI/pennylane/pull/7015)

* `qml.ops.Controlled.has_sparse_matrix` now provides the correct information
  by checking if the target operator has a sparse or dense matrix defined.
  [(#7025)](https://github.com/PennyLaneAI/pennylane/pull/7025)

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

* The `poly_to_angles` function has been improved to correctly work with different interfaces and
  no longer manipulate the input angles tensor internally.
  [(#6979)](https://github.com/PennyLaneAI/pennylane/pull/6979)

* The `QROM` template is upgraded to decompose more efficiently when `work_wires` are not used.
  [#6967)](https://github.com/PennyLaneAI/pennylane/pull/6967)

* Applying mid-circuit measurements inside `qml.cond` is not supported, and previously resulted in 
  unclear error messages or incorrect results. It is now explicitly not allowed, and raises an error when 
  calling the function returned by `qml.cond`.
  [(#7027)](https://github.com/PennyLaneAI/pennylane/pull/7027)  
  [(#7051)](https://github.com/PennyLaneAI/pennylane/pull/7051)

* `qml.qchem.givens_decomposition` no longer raises a `RuntimeWarning` when the input is a zero matrix.
  [#7053)](https://github.com/PennyLaneAI/pennylane/pull/7053)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso,
Daniela Angulo,
Utkarsh Azad,
Joey Carter,
Henry Chang,
Yushao Chen,
Isaac De Vlugt,
Diksha Dhawan,
Lillian M.A. Frederiksen,
Pietropaolo Frisoni,
Marcus Gisslén,
Korbinian Kottmann,
Christina Lee,
Joseph Lee,
Lee J. O'Riordan,
Mudit Pandey,
Andrija Paurevic,
Shuli Shu,
David Wierichs
