:orphan:

# Release 0.41.0 (current release)

<h3>New features since last release</h3>

* Added additional Hadamard gradient modes and `"reversed"`, `"direct"`, and `"reversed-direct"` modes are now available for use with the hadamard gradient.
  [(#7046)](https://github.com/PennyLaneAI/pennylane/pull/7046)

<h4>Resource-efficient decompositions 🔎</h4>

A new, experimental graph-based decomposition system is now available in PennyLane under the `qml.decomposition` 
module. 
[(#6950)](https://github.com/PennyLaneAI/pennylane/pull/6950)
[(#6952)](https://github.com/PennyLaneAI/pennylane/pull/6952)
[(#7045)](https://github.com/PennyLaneAI/pennylane/pull/7045)
[(#7058)](https://github.com/PennyLaneAI/pennylane/pull/7058)
[(#7064)](https://github.com/PennyLaneAI/pennylane/pull/7064)
[(#6951)](https://github.com/PennyLaneAI/pennylane/pull/6951)
[(#7223)](https://github.com/PennyLaneAI/pennylane/pull/7223)

PennyLane's new decomposition system offers a graph-based alternative to the current system, which provides 
better resource efficiency and versatility by traversing an internal graph structure that is weighted 
by the resources (e.g., gate counts) required to decompose down to a given set of gates. 

This new system is experimental and is disabled by default, but it can be enabled by adding `qml.decompositions.enable_graph()` 
to the top of your program. Conversely, `qml.decompositions.disable_graph` disables the new system from 
being active.

With `qml.decompositions.enable_graph()`, the following new features are available:

* Operators in PennyLane can now accommodate multiple decompositions, which can be queried with the 
  new `qml.list_decomps` function:

  ```pycon
  >>> import pennylane as qml
  >>> qml.decomposition.enable_graph()
  >>> qml.list_decomps(qml.CRX)
  [<pennylane.decomposition.decomposition_rule.DecompositionRule at 0x136da9de0>,
    <pennylane.decomposition.decomposition_rule.DecompositionRule at 0x136da9db0>,
    <pennylane.decomposition.decomposition_rule.DecompositionRule at 0x136da9f00>]
  >>> print(qml.draw(qml.list_decomps(qml.CRX)[0])(0.5, wires=[0, 1]))
  0: ───────────╭●────────────╭●─┤
  1: ──RX(0.25)─╰Z──RX(-0.25)─╰Z─┤
  ```

  When an operator within a circuit needs to be decomposed (e.g., when `qml.transforms.decompose` is 
  present), the chosen decomposition rule is that which leads to the most resource efficient set of 
  gates (i.e., the least amount of gates produced).

* New decomposition rules can be globally added to operators in PennyLane with the new `qml.add_decomps` 
  function. Creating a valid decomposition rule requires:

  * Defining quantum function that represents the decomposition.
  * Adding resource requirements (gate counts) to the above quantum function by decorating it with the 
    new `qml.register_resources` function, which requires a dictionary mapping operator types present 
    in the quantum function to their number of occurrences.

  ```python
  qml.decomposition.enable_graph()

  @qml.register_resources({qml.H: 2, qml.CZ: 1})
  def my_cnot(wires):
      qml.H(wires=wires[1])
      qml.CZ(wires=wires)
      qml.H(wires=wires[1])

  qml.add_decomps(qml.CNOT, my_cnot)
  ```

  This newly added rule for `qml.CNOT` can be verified as being available to use:

  ```pycon
  >>> my_new_rule = qml.list_decomps(qml.CNOT)[-1]
  >>> print(my_new_rule)
  @qml.register_resources({qml.H: 2, qml.CZ: 1})
  def my_cnot(wires):
      qml.H(wires=wires[1])
      qml.CZ(wires=wires)
      qml.H(wires=wires[1])
  ```

  Operators with dynamic resource requirements must be declared in a resource estimate using the new
  `qml.resource_rep` function. For each operator class, the set of parameters that affects the type 
  of gates and their number of occurrences in its decompositions is given by the `resource_keys` attribute.

  ```pycon
  >>> qml.MultiRZ.resource_keys
  {'num_wires'}
  ```

  The output of `resource_keys` indicates that custom decompositions for the operator should be registered 
  to a resource function (as opposed to a static dictionary) that accepts those exact arguments and 
  returns a dictionary. Consider this dummy example of a ficticious decomposition rule comprising three 
  `qml.MultiRZ` gates:

  ```python
  qml.decomposition.enable_graph()

  def resource_fn(num_wires):
      return {
          qml.resource_rep(qml.MultiRZ, num_wires=num_wires - 1): 1,
          qml.resource_rep(qml.MultiRZ, num_wires=3): 2
      }
  
  @qml.register_resources(resource_fn)
  def my_decomp(theta, wires):
      qml.MultiRZ(theta, wires=wires[:3])
      qml.MultiRZ(theta, wires=wires[1:])
      qml.MultiRZ(theta, wires=wires[:3])
  ```

  More information for defining complex decomposition rules can be found in the documentation for `qml.register_resources`.

* The `qml.transforms.decompose` transform works when the new decompositions system is enabled, and 
  offers the ability to inject new decomposition rules via two new keyword arguments:

  * `fixed_decomps`: decomposition rules provided to this keyword argument are guaranteed to be used 
    by the new system, bypassing all other decomposition rules that may exist for the relevant operators.
  * `alt_decomps`: decomposition rules provided to this keyword argument are alternative decomposition 
    rules that the new system may choose if they're the most resource efficient.
  [(#6966)](https://github.com/PennyLaneAI/pennylane/pull/6966)
  [(#7149)](https://github.com/PennyLaneAI/pennylane/pull/7149)
  [(#7184)](https://github.com/PennyLaneAI/pennylane/pull/7184)

  Each keyword argument must be assigned a dictionary that maps operator types to decomposition rules.
  Here is an example of both keyword arguments in use:

  ```python
  qml.decomposition.enable_graph()

  @qml.register_resources({qml.CNOT: 2, qml.RX: 1})
  def my_isingxx(phi, wires, **__):
      qml.CNOT(wires=wires)
      qml.RX(phi, wires=[wires[0]])
      qml.CNOT(wires=wires)

  @qml.register_resources({qml.H: 2, qml.CZ: 1})
  def my_cnot(wires, **__):
      qml.H(wires=wires[1])
      qml.CZ(wires=wires)
      qml.H(wires=wires[1])

  @partial(
      qml.transforms.decompose,
      gate_set={"RX", "RZ", "CZ", "GlobalPhase"},
      alt_decomps={qml.CNOT: my_cnot},
      fixed_decomps={qml.IsingXX: my_isingxx},
  )
  @qml.qnode(qml.device("default.qubit"))
  def circuit():
      qml.CNOT(wires=[0, 1])
      qml.IsingXX(0.5, wires=[0, 1])
      return qml.state()
  ```

  ```pycon
  >>> circuit()
  array([ 9.68912422e-01+2.66934210e-16j, -1.57009246e-16+3.14018492e-16j,
        8.83177008e-17-2.94392336e-17j,  5.44955495e-18-2.47403959e-01j])
  ```

  More details about using `fixed_decomps` and `alt_decomps` can be found in the usage details section
  in the `qml.transforms.decompose` documentation.

<h4>Capturing and Representing Hybrid Programs 📥</h4>

* Transformations can now be directly applied to a `QNode` with program capture enabled without having
  to use the `@qml.capture.expand_plxpr_transforms` decorator.
  [(#7199)](https://github.com/PennyLaneAI/pennylane/pull/7199)

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

* The sizes of dynamically shaped arrays can now be updated in a `while_loop` and `for_loop`
  when capture is enabled.
  [(#7084)](https://github.com/PennyLaneAI/pennylane/pull/7084)
  [(#7098)](https://github.com/PennyLaneAI/pennylane/pull/7098/)

* `qml.cond` can return arrays with dynamic shapes.
  [(#6888)](https://github.com/PennyLaneAI/pennylane/pull/6888/)
  [(#7080)](https://github.com/PennyLaneAI/pennylane/pull/7080)

* `cond`, `adjoint`, `ctrl`, and the `QNode` can now handle accepting dynamically
  shaped arrays with the abstract shape matching another argument.
  [(#7059)](https://github.com/PennyLaneAI/pennylane/pull/7059)

* A new `qml.capture.eval_jaxpr` function has been implemented. This is a variant of `jax.core.eval_jaxpr` that can handle the creation
  of arrays with dynamic shapes.
  [(#7052)](https://github.com/PennyLaneAI/pennylane/pull/7052)

* The `qml.transforms.single_qubit_fusion` quantum transform can now be applied with program capture enabled.
  [(#6945)](https://github.com/PennyLaneAI/pennylane/pull/6945)
  [(#7020)](https://github.com/PennyLaneAI/pennylane/pull/7020)

* The higher order primitives in program capture can now accept inputs with abstract shapes.
  [(#6786)](https://github.com/PennyLaneAI/pennylane/pull/6786)

* Execution interpreters and `qml.capture.eval_jaxpr` can now handle jax `pjit` primitives when dynamic shapes are being used.
  [(#7078)](https://github.com/PennyLaneAI/pennylane/pull/7078)
  [(#7117)](https://github.com/PennyLaneAI/pennylane/pull/7117)

* The `PlxprInterpreter` classes can now handle creating dynamic arrays via `jnp.ones`, `jnp.zeros`,
  `jnp.arange`, and `jnp.full`.
  [#6865)](https://github.com/PennyLaneAI/pennylane/pull/6865)

* Added class `qml.capture.transforms.CommuteControlledInterpreter` that moves commuting gates past control
  and target qubits of controlled operations when experimental program capture is enabled.
  It follows the same API as `qml.transforms.commute_controlled`.
  [(#6946)](https://github.com/PennyLaneAI/pennylane/pull/6946)
  [(#7247)](https://github.com/PennyLaneAI/pennylane/pull/7247)

* `qml.QNode` can now cache plxpr. When executing a `QNode` for the first time, its plxpr representation will
  be cached based on the abstract evaluation of the arguments. Later executions that have arguments with the
  same shapes and data types will be able to use this cached plxpr instead of capturing the program again.
  [(#6923)](https://github.com/PennyLaneAI/pennylane/pull/6923)

* `qml.QNode` now accepts a `static_argnums` argument. This argument can be used to indicate any arguments that
  should be considered static when capturing the quantum program.
  [(#6923)](https://github.com/PennyLaneAI/pennylane/pull/6923)

* Autograph can now be used with custom operations defined outside of the pennylane namespace.
  [(#6931)](https://github.com/PennyLaneAI/pennylane/pull/6931)

* Device preprocessing is now being performed in the execution pipeline for program capture.
  [(#7057)](https://github.com/PennyLaneAI/pennylane/pull/7057)
  [(#7089)](https://github.com/PennyLaneAI/pennylane/pull/7089)
  [(#7131)](https://github.com/PennyLaneAI/pennylane/pull/7131)
  [(#7135)](https://github.com/PennyLaneAI/pennylane/pull/7135)

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

* Added a class `qml.capture.transforms.MergeAmplitudeEmbedding` that merges `qml.AmplitudeEmbedding` operators
  following the same API as `qml.transforms.merge_amplitude_embedding` when experimental program capture is enabled.
  [(#6925)](https://github.com/PennyLaneAI/pennylane/pull/6925)

* With program capture enabled, `QNode`'s can now be differentiated with `diff_method="finite-diff"`.
  [(#6853)](https://github.com/PennyLaneAI/pennylane/pull/6853)

* Device-provided derivatives are integrated into the program capture pipeline.
  `diff_method="adjoint"` can now be used with `default.qubit` when capture is enabled.
  [(#6875)](https://github.com/PennyLaneAI/pennylane/pull/6875)
  [(#7019)](https://github.com/PennyLaneAI/pennylane/pull/7019)

<h4>End-to-end Sparse Execution 🌌</h4>

* Added method `qml.math.sqrt_matrix_sparse` to compute the square root of a sparse Hermitian matrix.
  [(#6976)](https://github.com/PennyLaneAI/pennylane/pull/6976)

* `qml.BlockEncode` now accepts sparse input and outputs sparse matrices.
  [(#6963)](https://github.com/PennyLaneAI/pennylane/pull/6963)
  [(#7140)](https://github.com/PennyLaneAI/pennylane/pull/7140)

* `Operator.sparse_matrix` now supports `format` parameter to specify the returned scipy sparse matrix format,
  with the default being `'csr'`
  [(#6995)](https://github.com/PennyLaneAI/pennylane/pull/6995)

* Dispatch the linear algebra methods of `scipy` backend to `scipy.sparse.linalg` explicitly. Now `qml.math` can correctly
  handle sparse matrices.
  [(#6947)](https://github.com/PennyLaneAI/pennylane/pull/6947)

* `default.qubit` now supports the sparse matrices to be applied to the state vector. Specifically, `QubitUnitary` initialized with a sparse matrix can now be applied to the state vector in the `default.qubit` device.
  [(#6883)](https://github.com/PennyLaneAI/pennylane/pull/6883)
  [(#7139)](https://github.com/PennyLaneAI/pennylane/pull/7139)
  [(#7191)](https://github.com/PennyLaneAI/pennylane/pull/7191)

* `Controlled` operators now have a full implementation of `sparse_matrix` that supports `wire_order` configuration.
  [(#6994)](https://github.com/PennyLaneAI/pennylane/pull/6994)

* `qml.SWAP` now has sparse representation.
  [(#6965)](https://github.com/PennyLaneAI/pennylane/pull/6965)

* `qml.QubitUnitary` now accepts sparse CSR matrices (from `scipy.sparse`). This allows efficient representation of large unitaries with mostly zero entries. Note that sparse unitaries are still in early development and may not support all features of their dense counterparts.
  [(#6889)](https://github.com/PennyLaneAI/pennylane/pull/6889)
  [(#6986)](https://github.com/PennyLaneAI/pennylane/pull/6986)
  [(#7143)](https://github.com/PennyLaneAI/pennylane/pull/7143)

  ```pycon
  >>> import numpy as np
  >>> import pennylane as qml
  >>> import scipy as sp
  >>> U_dense = np.eye(4)  # 2-wire identity
  >>> U_sparse = sp.sparse.csr_matrix(U_dense)
  >>> op = qml.QubitUnitary(U_sparse, wires=[0, 1])
  >>> print(op.sparse_matrix())
  <Compressed Sparse Row sparse matrix of dtype 'float64'
          with 4 stored elements and shape (4, 4)>
    Coords        Values
    (0, 0)        1.0
    (1, 1)        1.0
    (2, 2)        1.0
    (3, 3)        1.0
  >>> op.sparse_matrix().toarray()
  array([[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]])
  ```

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

<h4>QROM State Preparation 📖</h4>

* Added template `qml.QROMStatePreparation` that prepares arbitrary states using `qml.QROM`.
  [(#6974)](https://github.com/PennyLaneAI/pennylane/pull/6974)

<h4>Dynamical Lie Algebras 🕓</h4>

* Created a new `qml.liealg` module for Lie algebra functionality.

  `qml.liealg.cartan_decomp` allows to perform Cartan decompositions `g = k + m` using _involution_ functions that return a boolean value.
  A variety of typically encountered involution functions are included in the module, in particular the following:

  ```
  even_odd_involution
  concurrence_involution
  A
  AI
  AII
  AIII
  BD
  BDI
  DIII
  C
  CI
  CII
  ```

  ```pycon
  >>> g = qml.lie_closure([X(0) @ X(1), Y(0), Y(1)])
  >>> k, m = qml.liealg.cartan_decomp(g, qml.liealg.even_odd_involution)
  >>> g, k, m
  ([X(0) @ X(1), Y(0), Y(1), Z(0) @ X(1), X(0) @ Z(1), Z(0) @ Z(1)],
   [Y(0), Y(1)],
   [X(0) @ X(1), Z(0) @ X(1), X(0) @ Z(1), Z(0) @ Z(1)])
  ```

  The vertical subspace `k` and `m` fulfil the commutation relations `[k, m] ⊆ m`, `[k, k] ⊆ k` and `[m, m] ⊆ k` that make them a proper Cartan decomposition. These can be checked using the function `qml.liealg.check_cartan_decomp`.

  ```pycon
  >>> qml.liealg.check_cartan_decomp(k, m) # check Cartan commutation relations
  True
  ```

  `qml.liealg.horizontal_cartan_subalgebra` computes a horizontal Cartan subalgebra `a` of `m`.

  ```pycon
  >>> newg, k, mtilde, a, new_adj = qml.liealg.horizontal_cartan_subalgebra(k, m)
  ```

  `newg` is ordered such that the elements are `newg = k + mtilde + a`, where `mtilde` is the remainder of `m` without `a`. A Cartan subalgebra is an Abelian subalgebra of `m`, and we can confirm that indeed all elements in `a` are mutually commuting via `qml.liealg.check_abelian`.

  ```pycon
  >>> qml.liealg.check_abelian(a)
  True
  ```

  The following functions have also been added:
  * `qml.liealg.check_commutation_relation(A, B, C)` checks if all commutators between `A` and `B`
  map to a subspace of `C`, i.e. `[A, B] ⊆ C`.

  * `qml.liealg.adjvec_to_op` and `qml.liealg.op_to_adjvec` allow transforming operators within a Lie algebra to their adjoint vector representations and back.

  * `qml.liealg.change_basis_ad_rep` allows the transformation of an adjoint representation tensor according to a basis transformation on the underlying Lie algebra, without re-computing the representation.

  [(#6935)](https://github.com/PennyLaneAI/pennylane/pull/6935)
  [(#7026)](https://github.com/PennyLaneAI/pennylane/pull/7026)
  [(#7054)](https://github.com/PennyLaneAI/pennylane/pull/7054)
  [(#7129)](https://github.com/PennyLaneAI/pennylane/pull/7129)

* ``qml.lie_closure`` now accepts and outputs matrix inputs using the ``matrix`` keyword.
  Also added ``qml.pauli.trace_inner_product`` that can handle batches of dense matrices.
  [(#6811)](https://github.com/PennyLaneAI/pennylane/pull/6811)

* Added new `MultiControlledX` gate decompositions utilizing conditionally clean work wires, improving 
   circuit depth and efficiency.
   [(#7028)](https://github.com/PennyLaneAI/pennylane/pull/7028)
   * Implemented `_decompose_mcx_with_two_workers` and `_decompose_mcx_with_one_worker_kg24`.
   * Introduced `work_wire_type: Literal["clean", "dirty"]` to `decompose_mcx`.
   * Updated `decompose_mcx` to select decomposition strategy based on available work wires.

* Added class ``qml.FromBloq`` that takes Qualtran bloqs and translates them into equivalent PennyLane operators. For example, we can now import Bloqs and use them in a way similar to how we use PennyLane templates:
  ```python
  >>> from qualtran.bloqs.basic_gates import CNOT
  
  >>> dev = qml.device("default.qubit") # Execute on device
  >>> @qml.qnode(dev)
  ... def circuit():
  ...    qml.FromBloq(CNOT(), wires=[0, 1])
  ...    return qml.state()
  >>> circuit()
  array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
  ```
  [(#7148)](https://github.com/PennyLaneAI/pennylane/pull/7148)

* ``qml.structure_constants`` now accepts and outputs matrix inputs using the ``matrix`` keyword.
  [(#6861)](https://github.com/PennyLaneAI/pennylane/pull/6861)

<h4>Qualtran Integration 🔗</h4>

<h3>Improvements 🛠</h3>

  
<h4>QNode improvements</h4>

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

* The qnode primitive now stores the `ExecutionConfig` instead of `qnode_kwargs`.
  [(#6991)](https://github.com/PennyLaneAI/pennylane/pull/6991)

<h4>Decompositions</h4>

* The decomposition of a single qubit `qml.QubitUnitary` now includes the global phase.
  [(#7143)](https://github.com/PennyLaneAI/pennylane/pull/7143)
  
* The decompositions of `qml.SX`, `qml.X` and `qml.Y` use `qml.GlobalPhase` instead of `qml.PhaseShift`.
  [(#7073)](https://github.com/PennyLaneAI/pennylane/pull/7073)  

* Add a decomposition for multi-controlled global phases into a one-less-controlled phase shift.
  [(#6936)](https://github.com/PennyLaneAI/pennylane/pull/6936)

* `qml.ops.sk_decomposition` has been improved to produce less gates for certain edge cases. This greatly impacts
  the performance of `qml.clifford_t_decomposition`, which should now give less extraneous `qml.T` gates.
  [(#6855)](https://github.com/PennyLaneAI/pennylane/pull/6855)

* The template `MPSPrep` now has a gate decomposition. This enables its use with any device.
  The `right_canonicalize_mps` function has also been added to transform an MPS into its right-canonical form.
  [(#6896)](https://github.com/PennyLaneAI/pennylane/pull/6896)

* The `qml.clifford_t_decomposition` has been improved to use less gates when decomposing `qml.PhaseShift`.
  [(#6842)](https://github.com/PennyLaneAI/pennylane/pull/6842)

* An empty basis set in `qml.compile` is now recognized as valid, resulting in decomposition of all operators that can be decomposed.
  [(#6821)](https://github.com/PennyLaneAI/pennylane/pull/6821)

* The `assert_valid` method now validates that an operator's decomposition does not contain 
  the operator itself, instead of checking that it does not contain any operators of the same class as the operator.
  [(#7099)](https://github.com/PennyLaneAI/pennylane/pull/7099)

<h4>Better drawing functionality</h4>

* `qml.draw_mpl` can now split deep circuits over multiple figures via a `max_length` keyword argument.
   [(#7128)](https://github.com/PennyLaneAI/pennylane/pull/7128)

* `qml.draw` and `qml.draw_mpl` can now reuse lines for different classical wires, saving whitespace without
  changing the represented circuit.
  [(#7163)](https://github.com/PennyLaneAI/pennylane/pull/7163)

* `PrepSelPrep` now has a concise representation when drawn with `qml.draw` or `qml.draw_mpl`.
  [(#7164)](https://github.com/PennyLaneAI/pennylane/pull/7164)

<h4>Gradients and differentiability</h4>

* `qml.gradients.hadamard_grad` can now differentiate anything with a generator, and can accept circuits with non-commuting measurements.
  [(#6928)](https://github.com/PennyLaneAI/pennylane/pull/6928)

* The coefficients of observables now have improved differentiability.
  [(#6598)](https://github.com/PennyLaneAI/pennylane/pull/6598)

* An informative error is raised when a `QNode` with `diff_method=None` is differentiated.
  [(#6770)](https://github.com/PennyLaneAI/pennylane/pull/6770)

* `qml.gradients.finite_diff_jvp` has been added to compute the jvp of an arbitrary numeric
  function.
  [(#6853)](https://github.com/PennyLaneAI/pennylane/pull/6853)

<h4>Device improvements</h4>

* Devices can now configure whether or not ML framework data is sent to them
  via an `ExecutionConfig.convert_to_numpy` parameter. End-to-end jitting on
  `default.qubit` is used if the user specified a `jax.random.PRNGKey` as a seed.
  [(#6899)](https://github.com/PennyLaneAI/pennylane/pull/6899)
  [(#6788)](https://github.com/PennyLaneAI/pennylane/pull/6788)
  [(#6869)](https://github.com/PennyLaneAI/pennylane/pull/6869)

* The `reference.qubit` device now enforces `sum(probs)==1` in `sample_state`.
  [(#7076)](https://github.com/PennyLaneAI/pennylane/pull/7076)

* The `default.mixed` device now adheres to the newer device API introduced in
  [v0.33](https://docs.pennylane.ai/en/stable/development/release_notes.html#release-0-33-0).
  This means that `default.mixed` now supports not having to specify the number of wires,
  more predictable behaviour with interfaces, support for `qml.Snapshot`, and more.
  [(#6684)](https://github.com/PennyLaneAI/pennylane/pull/6684)

* `null.qubit` can now execute jaxpr.
  [(#6924)](https://github.com/PennyLaneAI/pennylane/pull/6924)

<h4>Experimental FTQC module</h4>

* A template class, `qml.ftqc.GraphStatePrep`, is added for the Graph state construction.
  [(#6985)](https://github.com/PennyLaneAI/pennylane/pull/6985)
  [(#7092)](https://github.com/PennyLaneAI/pennylane/pull/7092)

* A new utility module `qml.ftqc.utils` is provided, with support for functionality such as dynamic qubit recycling.
  [(#7075)](https://github.com/PennyLaneAI/pennylane/pull/7075/)

* A new class, `qml.ftqc.QubitGraph`, is now available for representing a qubit memory-addressing
  model for mappings between logical and physical qubits. This representation allows for nesting of
  lower-level qubits with arbitrary depth to allow easy insertion of arbitrarily many levels of
  abstractions between logical qubits and physical qubits.
  [(#6962)](https://github.com/PennyLaneAI/pennylane/pull/6962)

* A `Lattice` class and a `generate_lattice` method is added to the `qml.ftqc` module. The `generate_lattice` method is to generate 1D, 2D, 3D grid graphs with the given geometric parameters.
  [(#6958)](https://github.com/PennyLaneAI/pennylane/pull/6958)

* Measurement functions `measure_x`, `measure_y` and `measure_arbitrary_basis` are added in the experimental `ftqc` module. These functions
  apply a mid-circuit measurement and return a `MeasurementValue`. They are analogous to `qml.measure` for
  the computational basis, but instead measure in the X-basis, Y-basis, or an arbitrary basis, respectively.
  Function `qml.ftqc.measure_z` is also added as an alias for `qml.measure`.
  [(#6953)](https://github.com/PennyLaneAI/pennylane/pull/6953)

* The function `cond_measure` is added to the experimental `ftqc` module to apply a mid-circuit 
  measurement with a measurement basis conditional on the function input.
  [(#7037)](https://github.com/PennyLaneAI/pennylane/pull/7037)

* A `ParametrizedMidMeasure` class is added to represent a mid-circuit measurement in an arbitrary
  measurement basis in the XY, YZ or ZX plane. Subclasses `XMidMeasureMP` and `YMidMeasureMP` represent
  X-basis and Y-basis measurements. These classes are part of the experimental `ftqc` module.
  [(#6938)](https://github.com/PennyLaneAI/pennylane/pull/6938)
  [(#6953)](https://github.com/PennyLaneAI/pennylane/pull/6953)

* A `diagonalize_mcms` transform is added that diagonalizes any `ParametrizedMidMeasure`, for devices
  that only natively support mid-circuit measurements in the computational basis.
  [(#6938)](https://github.com/PennyLaneAI/pennylane/pull/6938)
  [(#7037)](https://github.com/PennyLaneAI/pennylane/pull/7037)

<h4>Other improvements</h4>

* The `gates`, `qubits` and `lamb` attributes of `DoubleFactorization` and `FirstQuantization` have
  dedicated documentation.
  [(#7173)](https://github.com/PennyLaneAI/pennylane/pull/7173)

* The qchem functions that accept a string input have been updated to consistently work with both
  lower-case and upper-case inputs.
  [(#7186)](https://github.com/PennyLaneAI/pennylane/pull/7186)

* `PSWAP.matrix()` and `PSWAP.eigvals()` now support parameter broadcasting.
  [(#7179)](https://github.com/PennyLaneAI/pennylane/pull/7179)
  [(#7228)](https://github.com/PennyLaneAI/pennylane/pull/7228)

* `Device.eval_jaxpr` now accepts an `execution_config` keyword argument.
  [(#6991)](https://github.com/PennyLaneAI/pennylane/pull/6991)

* Add a `qml.capture.pause()` context manager for pausing program capture in an error-safe way.
  [(#6911)](https://github.com/PennyLaneAI/pennylane/pull/6911)

* The requested `diff_method` is now validated when program capture is enabled.
  [(#6852)](https://github.com/PennyLaneAI/pennylane/pull/6852)

* Add a `qml.capture.register_custom_staging_rule` for handling higher-order primitives
  that return new dynamically shaped arrays.
  [(#7086)](https://github.com/PennyLaneAI/pennylane/pull/7086)

* A new, experimental `Operator` method called `compute_qfunc_decomposition` has been added to represent decompositions with structure (e.g., control flow).
  This method is only used when capture is enabled with `qml.capture.enable()`.
  [(#6859)](https://github.com/PennyLaneAI/pennylane/pull/6859)
  [(#6881)](https://github.com/PennyLaneAI/pennylane/pull/6881)
  [(#7022)](https://github.com/PennyLaneAI/pennylane/pull/7022)
  [(#6917)](https://github.com/PennyLaneAI/pennylane/pull/6917)
  [(#7081)](https://github.com/PennyLaneAI/pennylane/pull/7081)

* Improves support when specifying wires as type `jax.numpy.ndarray` if program capture is enabled.
  [(#7108)](https://github.com/PennyLaneAI/pennylane/pull/7108)

* `merge_rotations` now correctly simplifies merged `qml.Rot` operators whose angles yield the identity operator.
  [(#7011)](https://github.com/PennyLaneAI/pennylane/pull/7011)

* The `qml.measurements.NullMeasurement` measurement process is added to allow for profiling problems
  without the overheads associated with performing measurements.
  [(#6989)](https://github.com/PennyLaneAI/pennylane/pull/6989)

* `pauli_rep` property is now accessible for `Adjoint` operator when there is a Pauli representation.
  [(#6871)](https://github.com/PennyLaneAI/pennylane/pull/6871)

* `qml.pauli.PauliVSpace` is now iterable.
  [(#7054)](https://github.com/PennyLaneAI/pennylane/pull/7054)

* `qml.qchem.taper` now handles wire ordering for the tapered observables more robustly.
  [(#6954)](https://github.com/PennyLaneAI/pennylane/pull/6954)

* A `RuntimeWarning` is now raised by `qml.QNode` and `qml.execute` if executing JAX workflows and the installed version of JAX
  is greater than `0.4.28`.
  [(#6864)](https://github.com/PennyLaneAI/pennylane/pull/6864)

* Bump `rng_salt` to `v0.40.0`.
  [(#6854)](https://github.com/PennyLaneAI/pennylane/pull/6854)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* ``pennylane.labs.dla.lie_closure_dense`` is removed and integrated into ``qml.lie_closure`` using the new ``dense`` keyword.
  [(#6811)](https://github.com/PennyLaneAI/pennylane/pull/6811)

* ``pennylane.labs.dla.structure_constants_dense`` is removed and integrated into ``qml.structure_constants`` using the new ``matrix`` keyword.
  [(#6861)](https://github.com/PennyLaneAI/pennylane/pull/6861)

* ``ResourceOperator.resource_params`` is changed to a property.
  [(#6973)](https://github.com/PennyLaneAI/pennylane/pull/6973)

* Added ResourceOperator implementations for the ``ModExp``, ``PhaseAdder``, ``Multiplier``, ``ControlledSequence``, ``AmplitudeAmplification``, ``QROM``, ``SuperPosition``, ``MottonenStatePreparation``, ``StatePrep``, ``BasisState`` templates.
  [(#6638)](https://github.com/PennyLaneAI/pennylane/pull/6638)

* `pennylane.labs.khaneja_glaser_involution` is removed.
  `pennylane.labs.check_commutation` is moved to `qml.liealg.check_commutation_relation`.
  `pennylane.labs.check_cartan_decomp` is moved to `qml.liealg.check_cartan_decomp`.
  All involution functions are moved to `qml.liealg`.
  `pennylane.labs.adjvec_to_op` is moved to `qml.liealg.adjvec_to_op`.
  `pennylane.labs.op_to_adjvec` is moved to `qml.liealg.op_to_adjvec`.
  `pennylane.labs.change_basis_ad_rep` is moved to `qml.liealg.change_basis_ad_rep`.
  `pennylane.labs.cartan_subalgebra` is moved to `qml.liealg.horizontal_cartan_subalgebra`.
  [(#7026)](https://github.com/PennyLaneAI/pennylane/pull/7026)
  [(#7054)](https://github.com/PennyLaneAI/pennylane/pull/7054)

* Adding `HOState` and `VibronicHO` classes for representing harmonic oscillator states.
  [(#7035)](https://github.com/PennyLaneAI/pennylane/pull/7035)

* Adding base classes for Trotter error estimation on Realspace Hamiltonians: ``RealspaceOperator``, ``RealspaceSum``, ``RealspaceCoeffs``, and ``RealspaceMatrix``
  [(#7034)](https://github.com/PennyLaneAI/pennylane/pull/7034)

* Adding functions for Trotter error estimation and Hamiltonian fragment generation: ``trotter_error``, ``perturbation_error``, ``vibrational_fragments``, ``vibronic_fragments``, and ``generic_fragments``.
  [(#7036)](https://github.com/PennyLaneAI/pennylane/pull/7036)

  As an example we compute the peruturbation error of a vibrational Hamiltonian.
  First we generate random harmonic frequences and Taylor coefficients to iniitialize the vibrational Hamiltonian.

  ```pycon
  >>> from pennylane.labs.trotter_error import HOState, vibrational_fragments, perturbation_error
  >>> import numpy as np
  >>> n_modes = 2
  >>> r_state = np.random.RandomState(42)
  >>> freqs = r_state.random(n_modes)
  >>> taylor_coeffs = [
  >>>     np.array(0),
  >>>     r_state.random(size=(n_modes, )),
  >>>     r_state.random(size=(n_modes, n_modes)),
  >>>     r_state.random(size=(n_modes, n_modes, n_modes))
  >>> ]
  ```
    
  We call ``vibrational_fragments`` to get the harmonic and anharmonic fragments of the vibrational Hamiltonian.
  ```pycon
  >>> frags = vibrational_fragments(n_modes, freqs, taylor_coeffs)
  ```

  We build state vectors in the harmonic oscilator basis with the ``HOState`` class. 

  ```pycon
  >>> gridpoints = 5
  >>> state1 = HOState(n_modes, gridpoints, {(0, 0): 1})
  >>> state2 = HOState(n_modes, gridpoints, {(1, 1): 1})
  ```

  Finally, we compute the error by calling ``perturbation_error``.

  ```pycon
  >>> perturbation_error(frags, [state1, state2])
  [(-0.9189251160920879+0j), (-4.797716682426851+0j)]
  ```

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

* The `KerasLayer` in `qml.qnn.keras` is deprecated because Keras 2 is no longer actively maintained.  Please consider using a different machine learning framework instead of `TensorFlow/Keras 2`.
  [(#7097)](https://github.com/PennyLaneAI/pennylane/pull/7097)

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

* Add an informative error message for users if they try to `autograph` a function that has a `lambda` loop condition in `qml.while_loop`.
  [(#7178)](https://github.com/PennyLaneAI/pennylane/pull/7178)

* Clean up logic in `qml.drawer.tape_text`
  [(#7133)](https://github.com/PennyLaneAI/pennylane/pull/7133)

* Add intermediate caching to `null.qubit` zero value generation to improve memory consumption for larger workloads.
  [(#7155)](https://github.com/PennyLaneAI/pennylane/pull/7155)

* All use of `ABC` for intermediate variables will be renamed to preserve the label for the Python abstract base class `abc.ABC`.
  [(#7156)](https://github.com/PennyLaneAI/pennylane/pull/7156)

* The error message when device wires are not specified when program capture is enabled is more clear.
  [(#7130)](https://github.com/PennyLaneAI/pennylane/pull/7130)

* Clean up logic in `_capture_qnode.py`.
  [(#7115)](https://github.com/PennyLaneAI/pennylane/pull/7115)

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

* The :doc:`Compiling Circuits page <../introduction/compiling_circuits>` has been updated to include information
  on using the new experimental decompositions system.
  [(#7066)](https://github.com/PennyLaneAI/pennylane/pull/7066)

* The docstring for `qml.transforms.decompose` now recommends the `qml.clifford_t_decomposition` 
  transform when decomposing to the Clifford + T gate set.
  [(#7177)](https://github.com/PennyLaneAI/pennylane/pull/7177)

* Typos were fixed in the docstring for `qml.QubitUnitary`.
  [(#7187)](https://github.com/PennyLaneAI/pennylane/pull/7187)

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

* The docstring for `qml.devices.default_tensor.DefaultTensor` has been updated to clarify differentiation support.
  [(#7150)](https://github.com/PennyLaneAI/pennylane/pull/7150)

* The docstring for `QuantumScripts` has been updated to remove outdated references to `set_parameters`.
  [(#7174)](https://github.com/PennyLaneAI/pennylane/pull/7174)

<h3>Bug fixes 🐛</h3>

* PennyLane is now compatible with `pyzx 0.9`.
  [(#7188)](https://github.com/PennyLaneAI/pennylane/pull/7188)

* Fix a bug when `qml.matrix` is applied on a sparse operator, which caused the output to have unnecessary epsilon inaccuracy.
  [(#7147)](https://github.com/PennyLaneAI/pennylane/pull/7147)
  [(#7182)](https://github.com/PennyLaneAI/pennylane/pull/7182)


* Revert [(#6933)](https://github.com/PennyLaneAI/pennylane/pull/6933) to remove non-negligible performance impact due to wire flattening.
  [(#7136)](https://github.com/PennyLaneAI/pennylane/pull/7136)

* Fixes a bug that caused the output of `qml.fourier.qnode_spectrum()` to
  differ depending if equivalent gate generators are defined using
  different PennyLane operators. This was resolved by updating
  `qml.operation.gen_is_multi_term_hamiltonian` to work with more complicated generators.
  [(#7121)])(https://github.com/PennyLaneAI/pennylane/pull/7121)

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

* Comparing an adjoint of an `Observable` with another `Operation` using `qml.equal` no longer incorrectly 
  skips the check ensuring that the operator types match.
  [(#7107)](https://github.com/PennyLaneAI/pennylane/pull/7107)

* Downloading specific attributes of datasets in the `'other'` category via `qml.data.load` no longer fails.
  [(#7144)](https://github.com/PennyLaneAI/pennylane/pull/7144)

* Minor docstring upgrades for `qml.labs.trotter_error`.
  [(#7190)](https://github.com/PennyLaneAI/pennylane/pull/7190)

* Function `qml.labs.trotter_error.vibronic_fragments` now returns `RealspaceMatrix` objects with the correct number of electronic states.
  [(#7251)](https://github.com/PennyLaneAI/pennylane/pull/7251)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso,
Daniela Angulo,
Ali Asadi,
Utkarsh Azad,
Astral Cai,
Joey Carter,
Henry Chang,
Yushao Chen,
Isaac De Vlugt,
Diksha Dhawan,
Lillian M.A. Frederiksen,
Pietropaolo Frisoni,
Marcus Gisslén,
Diego Guala,
Austin Huang,
Soran Jahangiri,
Korbinian Kottmann,
Christina Lee,
Joseph Lee,
Dantong Li,
William Maxwell,
Anton Naim Ibrahim,
Lee J. O'Riordan,
Mudit Pandey,
Vyom Patel,
Andrija Paurevic,
Justin Pickering,
Shuli Shu,
David Wierichs
