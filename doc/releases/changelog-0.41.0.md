
# Release 0.41.0

<h3>New features since last release</h3>

<h4>Resource-efficient Decompositions 🔎</h4>

A new, experimental graph-based decomposition system is now available in PennyLane, offering more resource-efficiency and versatility.
[(#6950)](https://github.com/PennyLaneAI/pennylane/pull/6950)
[(#6951)](https://github.com/PennyLaneAI/pennylane/pull/6951)
[(#6952)](https://github.com/PennyLaneAI/pennylane/pull/6952)
[(#7028)](https://github.com/PennyLaneAI/pennylane/pull/7028)
[(#7045)](https://github.com/PennyLaneAI/pennylane/pull/7045)
[(#7058)](https://github.com/PennyLaneAI/pennylane/pull/7058)
[(#7064)](https://github.com/PennyLaneAI/pennylane/pull/7064)
[(#7149)](https://github.com/PennyLaneAI/pennylane/pull/7149)
[(#7184)](https://github.com/PennyLaneAI/pennylane/pull/7184)
[(#7223)](https://github.com/PennyLaneAI/pennylane/pull/7223)
[(#7263)](https://github.com/PennyLaneAI/pennylane/pull/7263)

PennyLane's new experimental graph decomposition system offers a resource-efficient and versatile alternative 
to the current system. This is done by traversing an internal graph structure that is weighted by the 
resources (e.g., gate counts) required to decompose down to a given set of gates. 

The graph-based system is experimental and is disabled by default, but it can be enabled by adding 
:func:`qml.decomposition.enable_graph() <pennylane.decomposition.enable_graph>` to the top of your 
program. Conversely, :func:`qml.decomposition.disable_graph() <pennylane.decomposition.disable_graph>` 
disables the new system from being active.

With :func:`qml.decomposition.enable_graph() <pennylane.decomposition.enable_graph>`, the following 
new features are available:

* Operators in PennyLane can now accommodate multiple decompositions, which can be queried with the 
  new :func:`qml.list_decomps <pennylane.list_decomps>` function:

  ```pycon
  >>> import pennylane as qp
  >>> qml.decomposition.enable_graph()
  >>> qml.list_decomps(qml.CRX)
  [<pennylane.decomposition.decomposition_rule.DecompositionRule at 0x136da9de0>,
    <pennylane.decomposition.decomposition_rule.DecompositionRule at 0x136da9db0>,
    <pennylane.decomposition.decomposition_rule.DecompositionRule at 0x136da9f00>]
  >>> decomp_rule = qml.list_decomps(qml.CRX)[0]
  >>> print(decomp_rule, "\n")
  @register_resources(_crx_to_rx_cz_resources)
  def _crx_to_rx_cz(phi: TensorLike, wires: WiresLike, **__):
      qml.RX(phi / 2, wires=wires[1])
      qml.CZ(wires=wires)
      qml.RX(-phi / 2, wires=wires[1])
      qml.CZ(wires=wires) 

  >>> print(qml.draw(decomp_rule)(0.5, wires=[0, 1]))
  0: ───────────╭●────────────╭●─┤
  1: ──RX(0.25)─╰Z──RX(-0.25)─╰Z─┤
  ```

  When an operator within a circuit needs to be decomposed (e.g., when 
  :func:`qml.transforms.decompose <pennylane.transforms.decompose>` is present), the chosen
decomposition rule is that which is most resource efficient (results in the smallest gate count).

* New decomposition rules can be globally added to operators in PennyLane with the new 
  :func:`qml.add_decomps <pennylane.add_decomps>` function. Creating a valid decomposition rule requires:

  * Defining a quantum function that represents the decomposition.
  * Adding resource requirements (gate counts) to the above quantum function by decorating it with the 
    new :func:`qml.register_resources <pennylane.register_resources>` function, which requires a dictionary 
    mapping operator types present in the quantum function to their number of occurrences.

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
  :func:`qml.resource_rep <pennylane.resource_rep>` function. For each operator class, the set of parameters 
  that affects the type of gates and their number of occurrences in its decompositions is given by the 
  `resource_keys` attribute.

  ```pycon
  >>> qml.MultiRZ.resource_keys
  {'num_wires'}
  ```

  The output of `resource_keys` indicates that custom decompositions for the operator should be registered 
  to a resource function (as opposed to a static dictionary) that accepts those exact arguments and 
  returns a dictionary. Consider this dummy example of a fictitious decomposition rule comprising three 
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

  More information for defining complex decomposition rules can be found in the documentation for 
  :func:`qml.register_resources <pennylane.register_resources>`.

* The :func:`qml.transforms.decompose <pennylane.transforms.decompose>` transform works when the new 
  decompositions system is enabled and offers the ability to inject new decomposition rules for operators
  in QNodes.
  [(#6966)](https://github.com/PennyLaneAI/pennylane/pull/6966)

  With the graph-based system enabled, the :func:`qml.transforms.decompose <pennylane.transforms.decompose>` 
  transform offers the ability to inject new decomposition rules via two new keyword arguments:

  * `fixed_decomps`: decomposition rules provided to this keyword argument will be used 
    by the new system, bypassing all other decomposition rules that may exist for the relevant operators.
  * `alt_decomps`: decomposition rules provided to this keyword argument are alternative decomposition 
    rules that the new system may choose if they're the most resource efficient.

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

  More details about what `fixed_decomps` and `alt_decomps` do can be found in the usage details section
  in the :func:`qml.transforms.decompose <pennylane.transforms.decompose>` documentation.

<h4>Capturing and Representing Hybrid Programs 📥</h4>

Quantum operations, classical processing, structured control flow, and dynamicism can be efficiently 
expressed with *program capture* (enabled with :func:`qml.capture.enable <pennylane.capture.enable>`).

In the last several releases of PennyLane, we have been working on a new and experimental feature called 
*program capture*, which allows for compactly expressing details about hybrid workflows while 
also providing a smoother integration with just-in-time compilation frameworks like 
[Catalyst](https://docs.pennylane.ai/projects/catalyst/en/stable/index.html) (via the :func:`~.pennylane.qjit` 
decorator) and JAX-jit. 

Internally, program capture is supported by representing hybrid programs via a new intermediate representation 
(IR) called ``plxpr``, rather than a quantum tape. The ``plxpr`` IR is an adaptation of JAX's ``jaxpr`` 
IR.

There are some quirks and restrictions to be aware of, which are outlined in the :doc:`/news/program_capture_sharp_bits` 
page. But with this release, many of the core features of PennyLane—and more!—are available with program 
capture enabled by adding :func:`qml.capture.enable() <pennylane.capture.enable>` to the top of your program:

* QNodes can now contain mid-circuit measurements (MCMs) and classical processing on MCMs with `mcm_method = "deferred"` 
  when program capture is enabled.
  [(#6838)](https://github.com/PennyLaneAI/pennylane/pull/6838)
  [(#6937)](https://github.com/PennyLaneAI/pennylane/pull/6937)
  [(#6961)](https://github.com/PennyLaneAI/pennylane/pull/6961)

  With `mcm_method = "deferred"`, workflows with mid-circuit measurements can be executed with program
  capture enabled. Additionally, program capture unlocks the ability to classically process MCM values
  and use MCM values as gate parameters.

  ```python
  import jax 
  import jax.numpy as jnp
  jax.config.update("jax_enable_x64", True)

  qml.capture.enable()

  @qml.qnode(qml.device("default.qubit", wires=3), mcm_method="deferred")
  def f(x):
      m0 = qml.measure(0)
      m1 = qml.measure(0)

      # classical processing on m0 and m1
      a = jnp.sin(0.5 * jnp.pi * m0)
      phi = a - (m1 + 1) ** 4

      qml.s_prod(x, qml.RX(phi, 0))

      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> f(0.1)
  Array(0.00540302, dtype=float64)
  ```

  Note that with capture enabled, automatic qubit management is not supported on devices; the number 
  of wires given to the device must coincide with how many MCMs are present in the circuit, since 
  deferred measurements add one auxiliary qubit per MCM.

* Quantum circuits can now be differentiated on `default.qubit` and `lightning.qubit` with 
  `diff_method="finite-diff"`, `"adjoint"`, and `"backprop"` when program capture is enabled.
  [(#6853)](https://github.com/PennyLaneAI/pennylane/pull/6853)
  [(#6875)](https://github.com/PennyLaneAI/pennylane/pull/6875)
  [(#7019)](https://github.com/PennyLaneAI/pennylane/pull/7019)

  ```python
  qml.capture.enable()

  @qml.qnode(qml.device("default.qubit", wires=3), diff_method="adjoint")
  def f(phi):
      qml.RX(phi, 0)
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> qml.grad(f)(jnp.array(0.1))
  Array(-0.09983342, dtype=float64, weak_type=True)
  ```

* QNode arguments can now be assigned as static. In turn, PennyLane can now determine when plxpr 
  needs to be reconstructed based on dynamic and static arguments, providing efficiency for repeated
  circuit executions.
  [(#6923)](https://github.com/PennyLaneAI/pennylane/pull/6923)

  Specifying static arguments can be done at the QNode level with the `static_argnums` keyword argument.
  Its values (integers) indicate which arguments are to be treated as static. By default, all QNode
  arguments are dynamic. 
  
  Consider the following example, where the first argument is indicated to be static:

  ```python
  qml.capture.enable()

  @qml.qnode(qml.device("default.qubit", wires=1), static_argnums=0)
  def f(x, y):
      print("I constructed plxpr")
      qml.RX(x, 0)
      qml.RY(y, 0)
      return qml.expval(qml.Z(0))
  ```

  When the value of `x` changes, PennyLane must (re)construct the plxpr representation of the program
  for the program to execute. In this example, the act of (re)constructing plxpr is indicated
  by the `print` statement executing. However, if the value of `y` changes (a dynamic argument), PennyLane
  does not need to reconstruct the plxpr representation of the program:

  ```pycon
  >>> f(0.1, 0.2)
  I constructed plxpr
  0.97517043
  >>> f(0.1, 0.3)
  0.9505638
  >>> f(0.2, 0.3)
  I constructed plxpr
  0.93629336
  ```

* All PennyLane transforms that return one device execution are compatible with program capture, including 
  those without a plxpr-native implementation.
  [(#6916)](https://github.com/PennyLaneAI/pennylane/pull/6916)
  [(#6922)](https://github.com/PennyLaneAI/pennylane/pull/6922)
  [(#6925)](https://github.com/PennyLaneAI/pennylane/pull/6925)
  [(#6945)](https://github.com/PennyLaneAI/pennylane/pull/6945)
  [(#6946)](https://github.com/PennyLaneAI/pennylane/pull/6946)
  [(#6957)](https://github.com/PennyLaneAI/pennylane/pull/6957)
  [(#6977)](https://github.com/PennyLaneAI/pennylane/pull/6977)
  [(#7020)](https://github.com/PennyLaneAI/pennylane/pull/7020)
  [(#7199)](https://github.com/PennyLaneAI/pennylane/pull/7199)
  [(#7247)](https://github.com/PennyLaneAI/pennylane/pull/7247)
  
  The following transforms now have native support for program capture (i.e., they directly transform 
  ``plxpr``) and can be used as you would normally use a transform in PennyLane:

  * :func:`merge_rotations <pennylane.transforms.merge_rotations>`
  * :func:`single_qubit_fusion <pennylane.transforms.single_qubit_fusion>`
  * :func:`unitary_to_rot <pennylane.transforms.unitary_to_rot>`
  * :func:`merge_amplitude_embedding <pennylane.transforms.merge_amplitude_embedding>`
  * :func:`commute_controlled <pennylane.transforms.commute_controlled>`
  * :func:`decompose <pennylane.transforms.decompose>`
  * :func:`map_wires <pennylane.map_wires>`
  * :func:`cancel_inverses <pennylane.transforms.cancel_inverses>`

  Other transforms without a plxpr-native implementation are also supported by internally converting 
  the tape implementation. As an example, here is a custom tape transform that shifts every `qml.RX` 
  gate to the end of the circuit:

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

  @shift_rx_to_end
  @qml.qnode(qml.device("default.qubit", wires=1))
  def circuit1():
      qml.RX(0.1, wires=0)
      qml.H(wires=0)
      return qml.state()

  @qml.qnode(qml.device("default.qubit", wires=1))
  def circuit2():
      qml.H(wires=0)
      qml.RX(0.1, wires=0)
      return qml.state()
  ```

  ```pycon
  >>> circuit1() == circuit2()
  Array([ True,  True], dtype=bool)
  ```

  There are some exceptions to getting transforms without a plxpr-native implementation to work with 
  capture enabled:
  * Transforms that return multiple device executions cannot be converted.
  * Transforms that return non-trivial post-processing functions cannot be converted.
  * Transforms will fail to execute if the transformed quantum function or QNode contains:
    * `qml.cond` with dynamic parameters as predicates.
    * `qml.for_loop` with dynamic parameters for ``start``, ``stop``, or ``step``.
    * `qml.while_loop`.

* Python control flow (`if/else`, `for`, `while`) is now supported when program capture is enabled.
  [(#6837)](https://github.com/PennyLaneAI/pennylane/pull/6837)
  [(#6931)](https://github.com/PennyLaneAI/pennylane/pull/6931)

  One of the strengths of program capture is preserving a program's structure, eliminating the need 
  to unroll control flow operations. In previous releases, this could only be accomplished by using 
  :func:`qml.for_loop <pennylane.for_loop>`, :func:`qml.cond <pennylane.cond>`, and :func:`qml.while_loop <pennylane.while_loop>`. With this release, 
  standard Python control flow operations are now supported with program capture:

  ```python
  qml.capture.enable()
  dev = qml.device("default.qubit", wires=[0, 1, 2])

  @qml.qnode(dev)
  def circuit(num_loops: int):
      for i in range(num_loops):
          if i % 2 == 0:
              qml.H(i)
          else:
              qml.RX(1, i)
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

  This is enabled internally by [diastatic-malt](https://github.com/PennyLaneAI/diastatic-malt), which 
  is our home-brewed implementation of Autograph. 

  Support for this can be disabled by setting `autograph=False` at the QNode level (by default, `autograph=True`).

<h4>End-to-end Sparse Execution 🌌</h4>

* Sparse data structures in compressed-sparse-row (csr) format are now supported end-to-end
  in PennyLane, resulting in faster execution for large sparse objects.
  [(#6883)](https://github.com/PennyLaneAI/pennylane/pull/6883) 
  [(#7139)](https://github.com/PennyLaneAI/pennylane/pull/7139) 
  [(#7191)](https://github.com/PennyLaneAI/pennylane/pull/7191)

  Sparse-array input and execution are now supported on `default.qubit` and `lightning.qubit`
  with a variety of templates, preserving sparsity throughout the entire simulation.
  
  Specifically, the following templates now support sparse data structures:

  * :class:`qml.StatePrep <pennylane.StatePrep>`
    [(#6863)](https://github.com/PennyLaneAI/pennylane/pull/6863)
  * :class:`qml.QubitUnitary <pennylane.QubitUnitary>` 
    [(#6889)](https://github.com/PennyLaneAI/pennylane/pull/6889)
    [(#6986)](https://github.com/PennyLaneAI/pennylane/pull/6986)
    [(#7143)](https://github.com/PennyLaneAI/pennylane/pull/7143)
  * :class:`qml.BlockEncode <pennylane.BlockEncode>`
    [(#6963)](https://github.com/PennyLaneAI/pennylane/pull/6963)
    [(#7140)](https://github.com/PennyLaneAI/pennylane/pull/7140)
  * :class:`qml.SWAP <pennylane.SWAP>`
    [(#6965)](https://github.com/PennyLaneAI/pennylane/pull/6965)
  * :func:`Controlled <pennylane.ops.op_math.Controlled>` operations
    [(#6994)](https://github.com/PennyLaneAI/pennylane/pull/6994)
  
  ```python
  import scipy
  import numpy as np

  sparse_state = scipy.sparse.csr_array([0, 1, 0, 0])
  mat = np.kron(np.identity(2**12), qml.X.compute_matrix())
  sparse_mat = scipy.sparse.csr_array(mat)
  sparse_x = scipy.sparse.csr_array(qml.X.compute_matrix())

  dev = qml.device("default.qubit")

  @qml.qnode(dev)
  def circuit():
      qml.StatePrep(sparse_state, wires=range(2))

      for i in range(10):
          qml.H(i)
          qml.CNOT(wires=[i, i + 1])

      qml.QubitUnitary(sparse_mat, wires=range(13))
      qml.ctrl(qml.QubitUnitary(sparse_x, wires=0), control=1)
      return qml.state()
    ```

    ```pycon
    >>> circuit()
    array([ 0.     +0.j,  0.03125+0.j,  0.     +0.j, ..., -0.03125+0.j,
            0.     +0.j,  0.     +0.j])
    ```

* Operators that have a :func:`sparse_matrix <pennylane.Operator.sparse_matrix>` method can now 
  choose a sparse-matrix format and the order of their wires. Available
  `scipy` [sparse-matrix formats](https://docs.scipy.org/doc/scipy/tutorial/sparse.html#:~:text=the%20scipy.sparse,disadvantages) include `'bsr'`, `'csr'`, `'csc'`, `'coo'`, `'dia'`, `'dok'`, and `'lil'`.
  [(#6995)](https://github.com/PennyLaneAI/pennylane/pull/6995)

  ```pycon
  >>> op = qml.CNOT([0,1])
  >>> type(op.sparse_matrix(format='dok'))
  scipy.sparse._dok.dok_matrix
  >>> type(op.sparse_matrix(format='csc'))
  scipy.sparse._csc.csc_matrix
  >>> print(op.sparse_matrix(wire_order=[1,0]))
  (0, 0)	1.0
  (1, 3)	1.0
  (2, 2)	1.0
  (3, 1)	1.0
  >>> print(op.sparse_matrix(wire_order=[0,1]))
  (0, 0)	1
  (1, 1)	1
  (2, 3)	1
  (3, 2)	1
  ```

* Sparse functionality is now available in `qml.math`:

  * `qml.math.sqrt_matrix_sparse` is available to compute the square root of a sparse Hermitian matrix.
    [(#6976)](https://github.com/PennyLaneAI/pennylane/pull/6976)

  * Most `qml.math` functions can now correctly handle sparse matrices as input by dispatching to  `scipy.sparse.linalg` internally. 
    [(#6947)](https://github.com/PennyLaneAI/pennylane/pull/6947)

<h4>QROM State Preparation 📖</h4>

* A new state-of-the-art state preparation technique based on QROM is now available with the
  :class:`qml.QROMStatePreparation <pennylane.QROMStatePreparation>` template.
  [(#6974)](https://github.com/PennyLaneAI/pennylane/pull/6974)  
  
  Using :class:`qml.QROMStatePreparation <pennylane.QROMStatePreparation>` is analogous to using other state preparation techniques in PennyLane. 
    
  ```python
  state_vector = np.array([0.5, -0.5, 0.5, 0.5])

  dev = qml.device("default.qubit")
  wires = qml.registers({"work_wires": 1, "prec_wires": 3, "state_wires": 2})

  @qml.qnode(dev)
  def circuit():
      qml.QROMStatePreparation(
          state_vector, wires["state_wires"], wires["prec_wires"], wires["work_wires"]
      )
      return qml.state()
  ```

  ```pycon
  >>> print(circuit()[:4].real)
  [ 0.5 -0.5  0.5  0.5]
  ```

<h4>Dynamical Lie Algebras 🕓</h4>

The new :mod:`qml.liealg <pennylane.liealg>` module provides a variety of Lie algebra functionality:

* Compute the dynamical Lie algebra from a set of generators with :func:`qml.lie_closure <pennylane.lie_closure>`.  
  This function accepts and outputs matrices when `matrix=True`.

  ```python
  import pennylane as qp
  from pennylane import X, Y, Z, I
  n = 2
  gens = [qml.X(0), qml.X(0) @ qml.X(1), qml.Y(1)]
  dla = qml.lie_closure(gens)
  ```
  ```pycon
  >>> dla
  [X(0), X(0) @ X(1), Y(1), X(0) @ Z(1)]
  ```

* Compute the structure constants that make up the adjoint representation of a Lie algebra
  using :func:`qml.structure_constants <pennylane.structure_constants>`.  
  This function accepts and outputs matrices when `matrix=True`.

  ```pycon
  >>> adjoint_rep = qml.structure_constants(dla)
  >>> adjoint_rep.shape
  (4, 4, 4)
  ```

* The center of a Lie algebra, which is the collection of operators that commute with all other operators in the DLA,
  can be found with :func:`qml.center <pennylane.center>`.

  ```pycon
  >>> qml.center(dla)
  [X(0)]
  ```

* Cartan decompositions, `g = k + m`, can be performed with :func:`qml.liealg.cartan_decomp <pennylane.liealg.cartan_decomp>`.  
  These use _involution_ functions that return a boolean value.
  A variety of typically encountered involution functions are included in the module,
  such as `even_odd_involution, concurrence_involution, A, AI, AII, AIII, BD, BDI, DIII, C, CI, CII`.
  ```python
  from pennylane.liealg import concurrence_involution
  k, m = qml.liealg.cartan_decomp(dla, concurrence_involution)
  ```
  ```pycon
  >>> k, m
  ([Y(1)], [X(0), X(0) @ X(1), X(0) @ Z(1)])
  ```
  The vertical subspace `k` and `m` fulfill the commutation relations `[k, m] ⊆ m`, `[k, k] ⊆ k` and `[m, m] ⊆ k` that make them a proper Cartan decomposition.
  These can be verified using the function :func:`qml.liealg.check_cartan_decomp <pennylane.liealg.check_cartan_decomp>`.
  ```pycon
  >>> qml.liealg.check_cartan_decomp(k, m)
  True
  ```

* The horizontal Cartan subalgebra `a` of `m` can be computed with
  :func:`qml.liealg.horizontal_cartan_subalgebra <pennylane.liealg.horizontal_cartan_subalgebra>`.

  ```python
  from pennylane.liealg import horizontal_cartan_subalgebra
  newg, k, mtilde, a, new_adj = horizontal_cartan_subalgebra(k, m, return_adjvec=True)
  ```
  ```pycon
  >>> newg.shape, k.shape, mtilde.shape, a.shape, new_adj.shape
  ((4, 4), (1, 4), (1, 4), (2, 4), (4, 4, 4))
  ```
  `newg` is ordered such that the elements are `newg = k + mtilde + a`, where `mtilde` is the remainder of `m` without `a`. A Cartan subalgebra is an Abelian subalgebra of `m`,
  and we can confirm that all elements in `a` are mutually commuting via `qml.liealg.check_abelian`.
  ```pycon
  >>> qml.liealg.check_abelian(a)
  True
  ```

* The following functions have also been added:
  * `qml.liealg.check_commutation_relation(A, B, C)` checks if all commutators between `A` and `B`
  map to a subspace of `C`, i.e. `[A, B] ⊆ C`.
  * `qml.liealg.adjvec_to_op` and `qml.liealg.op_to_adjvec` allow transforming operators within a Lie algebra to and from their adjoint vector representations.
  * `qml.liealg.change_basis_ad_rep` allows the transformation of an adjoint representation tensor according to a basis transformation on the underlying Lie algebra, without re-computing the representation.
  * `qml.pauli.trace_inner_product` can handle batches of dense matrices.

[(#6811)](https://github.com/PennyLaneAI/pennylane/pull/6811)
[(#6861)](https://github.com/PennyLaneAI/pennylane/pull/6861)
[(#6935)](https://github.com/PennyLaneAI/pennylane/pull/6935)
[(#7026)](https://github.com/PennyLaneAI/pennylane/pull/7026)
[(#7054)](https://github.com/PennyLaneAI/pennylane/pull/7054)
[(#7129)](https://github.com/PennyLaneAI/pennylane/pull/7129)

<h4>Qualtran Integration 🔗</h4>

* It's now possible to use [Qualtran](https://qualtran.readthedocs.io/en/latest/) bloqs in PennyLane
  with the new :func:`qml.FromBloq <pennylane.FromBloq>` class. 
  [(#7148)](https://github.com/PennyLaneAI/pennylane/pull/7148)
  
  :func:`qml.FromBloq <pennylane.FromBloq>` translates [Qualtran bloqs](https://qualtran.readthedocs.io/en/latest/bloqs/index.html#bloqs-library) 
  into equivalent PennyLane operators. It requires two inputs:
  * `bloq`: an initialized Qualtran Bloq
  * `wires`: the wires the operator acts on
  
  The following example applies a PennyLane Operator and Qualtran Bloq in the same circuit:

  ```python
  >>> from qualtran.bloqs.basic_gates import CNOT
  
  >>> dev = qml.device("default.qubit")
  >>> @qml.qnode(dev)
  ... def circuit():
  ...    qml.X(wires=0)
  ...    qml.FromBloq(CNOT(), wires=[0, 1])
  ...    return qml.state()
  >>> circuit()
  array([0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j])
  ```

* A new function called :func:`qml.bloq_registers <pennylane.bloq_registers>` is available to help
  determine the required wires for complex Qualtran ``Bloqs`` with multiple registers.
  [(#7148)](https://github.com/PennyLaneAI/pennylane/pull/7148)
  
  Given a Qualtran Bloq, this function returns a dictionary of register names and wires.

  ```pycon
  >>> from qualtran.bloqs.phase_estimation import RectangularWindowState, TextbookQPE
  >>> from qualtran.bloqs.basic_gates import ZPowGate
  >>> textbook_qpe = TextbookQPE(ZPowGate(exponent=2 * 0.234), RectangularWindowState(3))
  >>> registers = qml.bloq_registers(textbook_qpe)
  >>> registers
  {'q': Wires([0]), 'qpe_reg': Wires([1, 2, 3])}
  ```

  In the following example, we use these registers to measure the correct qubits in quantum phase
  estimation:

  ```python
  dev = qml.device("default.qubit")
  @qml.qnode(dev)
  def circuit():
      qml.FromBloq(textbook_qpe, wires=range(textbook_qpe.signature.n_qubits()))
      return qml.probs(wires=registers['qpe_reg'])
  ```

  ```pycon
  >>> print(qml.draw(circuit)())
  0: ─╭FromBloq─┤       
  1: ─├FromBloq─┤ ╭Probs
  2: ─├FromBloq─┤ ├Probs
  3: ─╰FromBloq─┤ ╰Probs
  ```

<h4>Hadamard Gradient Variants and Improvements 🌈</h4>

* Variants of the Hadamard gradient outlined in [arXiv:2408.05406](https://arxiv.org/pdf/2408.05406) 
  have been added as new differentiation methods: `diff_method="reversed-hadamard"`, `"direct-hamadard"`, 
  and `"reversed-direct-hadamard`".
  [(#7046)](https://github.com/PennyLaneAI/pennylane/pull/7046)
  [(#7198)](https://github.com/PennyLaneAI/pennylane/pull/7198)\

  The three variants of the Hadamard gradient added in this release offer tradeoffs that could be advantageous
  in certain cases:

  * `"reversed-hadamard"`: the observable being measured and the generators of the unitary operations 
    in the circuit are reversed; the generators are now the observables, and the Pauli decomposition 
    of the observables are now gates in the circuit.
  * `"direct-hadamard"`: the additional auxiliary qubit needed in the standard Hadamard gradient is 
    exchanged for additional circuit executions.
  * `"reversed-direct-hadamard"`: a combination of the "direct" and "reversed" modes, where the role 
    of the observable and the generators of the unitary operations in the circuit swap, and the additional 
    auxiliary qubit is exchanged for additional circuit executions.

  Using them in your code is just like any other differentiation method in PennyLane:
  
  ```python
  import pennylane as qp

  dev = qml.device("default.qubit")

  @qml.qnode(dev, diff_method="reversed-hadamard")
  def circuit(x):
      qml.RX(x, 0)
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> qml.grad(circuit)(qml.numpy.array(0.5))
  np.float64(-0.47942553860420284)
  ```

  More information on how these three new gradient methods work can be found in [arXiv:2408.05406](https://arxiv.org/pdf/2408.05406).

* The Hadamard gradient method and its variants can now differentiate any operator with a generator 
  defined, and can accept circuits with non-commuting measurements.
  [(#6928)](https://github.com/PennyLaneAI/pennylane/pull/6928)

<h3>Improvements 🛠</h3>

<h4>QNode execution configuration</h4>

* QNodes now have an `update` method that allows for re-configuring settings like `diff_method`, `mcm_method`, 
  and more. This allows for easier on-the-fly adjustments to workflows.
  [(#6803)](https://github.com/PennyLaneAI/pennylane/pull/6803)

  After constructing a QNode,

  ```python
  import pennylane as qp

  @qml.qnode(device=qml.device("default.qubit"))
  def circuit():
      qml.H(0)
      qml.CNOT([0,1])
      return qml.probs()
  ```

  its settings can be modified with `update`, which returns a new `QNode` object (note: any arguments 
  not specified in `update` will retain their original value). Here is an example of updating a QNode's 
  `diff_method`:

  ```pycon
  >>> print(circuit.diff_method)
  best
  >>> new_circuit = circuit.update(diff_method="parameter-shift")
  >>> print(new_circuit.diff_method)
  'parameter-shift'
  ```

* A new helper function called `qml.workflow.construct_execution_config(qnode)(*args,**kwargs)` is now
  available, which allows users to construct an execution configuration from a given QNode instance.
  [(#6901)](https://github.com/PennyLaneAI/pennylane/pull/6901)

  ```python
  @qml.qnode(qml.device("default.qubit", wires=1))
  def circuit(x):
      qml.RX(x, 0)
      return qml.expval(qml.Z(0))
  ```

  ```pycon
  >>> config = qml.workflow.construct_execution_config(circuit)(1)
  >>> print(config)
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

* QNodes now store their `ExecutionConfig` instead of `qnode_kwargs`.
  [(#6991)](https://github.com/PennyLaneAI/pennylane/pull/6991)

<h4>Decompositions</h4>

* The `qml.QROM` template has been upgraded to decompose more efficiently when `work_wires` are not 
  used.
  [#6967)](https://github.com/PennyLaneAI/pennylane/pull/6967)

* The decomposition of a single-qubit `qml.QubitUnitary` now includes the global phase.
  [(#7143)](https://github.com/PennyLaneAI/pennylane/pull/7143)
  
* The decompositions of `qml.SX`, `qml.X` and `qml.Y` use `qml.GlobalPhase` instead of `qml.PhaseShift`.
  [(#7073)](https://github.com/PennyLaneAI/pennylane/pull/7073)  

* A new decomposition for multi-controlled global phases that uses one less controlled phase shift
  has been added.
  [(#6936)](https://github.com/PennyLaneAI/pennylane/pull/6936)

* `qml.ops.sk_decomposition` has been improved to produce less gates for certain edge cases. This greatly 
  impacts the performance of `qml.clifford_t_decomposition`, which should now give less extraneous `qml.T` 
  gates.
  [(#6855)](https://github.com/PennyLaneAI/pennylane/pull/6855)

* `qml.MPSPrep` now has a gate decomposition. This enables its use with any device. Additionally, the 
  `right_canonicalize_mps` function has also been added to transform an MPS into its right-canonical 
  form.
  [(#6896)](https://github.com/PennyLaneAI/pennylane/pull/6896)

* The `qml.clifford_t_decomposition` has been improved to use less gates when decomposing `qml.PhaseShift`.
  [(#6842)](https://github.com/PennyLaneAI/pennylane/pull/6842)

* An empty basis set in `qml.compile` is now recognized as valid, resulting in decomposition of all 
  operators that can be decomposed.
  [(#6821)](https://github.com/PennyLaneAI/pennylane/pull/6821)

* The `assert_valid` method now validates that an operator's decomposition does not contain the operator 
  itself, instead of checking that it does not contain any operators of the same class as the operator.
  [(#7099)](https://github.com/PennyLaneAI/pennylane/pull/7099)

<h4>Improved drawing</h4>

* `qml.draw_mpl` can now split deep circuits over multiple figures via a `max_length` keyword argument.
  [(#7128)](https://github.com/PennyLaneAI/pennylane/pull/7128)

* `qml.draw` now re-displays wire labels at the start of each partitioned chunk when using the `max_length` keyword argument.
  [(#7250)](https://github.com/PennyLaneAI/pennylane/pull/7250)

* `qml.draw` and `qml.draw_mpl` can now reuse lines for different classical wires, saving whitespace 
  without changing the represented circuit.
  [(#7163)](https://github.com/PennyLaneAI/pennylane/pull/7163)

* `qml.PrepSelPrep` now has a concise representation when drawn with `qml.draw` or `qml.draw_mpl`.
  [(#7164)](https://github.com/PennyLaneAI/pennylane/pull/7164)

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

* A template class, `qml.ftqc.GraphStatePrep`, has been added for the Graph state construction.
  [(#6985)](https://github.com/PennyLaneAI/pennylane/pull/6985)
  [(#7092)](https://github.com/PennyLaneAI/pennylane/pull/7092)

* A new utility module `qml.ftqc.utils` is provided, with support for functionality such as dynamic 
  qubit recycling.
  [(#7075)](https://github.com/PennyLaneAI/pennylane/pull/7075/)

* A new class, `qml.ftqc.QubitGraph`, is now available for representing a qubit memory-addressing
  model for mappings between logical and physical qubits. This representation allows for nesting of
  lower-level qubits with arbitrary depth to allow easy insertion of arbitrarily many levels of
  abstractions between logical qubits and physical qubits.
  [(#6962)](https://github.com/PennyLaneAI/pennylane/pull/6962)

* A `Lattice` class and a `generate_lattice` method has been added to the `qml.ftqc` module. The 
  `generate_lattice` method is to generate 1D, 2D, 3D grid graphs with the given geometric parameters.
  [(#6958)](https://github.com/PennyLaneAI/pennylane/pull/6958)

* Measurement functions `measure_x`, `measure_y` and `measure_arbitrary_basis` are added in the 
  experimental `ftqc` module. These functions apply a mid-circuit measurement and return a `MeasurementValue`. 
  They are analogous to `qml.measure` for the computational basis, but instead measure in the X-basis, 
  Y-basis, or an arbitrary basis, respectively. Function `qml.ftqc.measure_z` is also added as an alias 
  for `qml.measure`.
  [(#6953)](https://github.com/PennyLaneAI/pennylane/pull/6953)

* The function `cond_measure` has been added to the experimental `ftqc` module to apply a mid-circuit 
  measurement with a measurement basis conditional on the function input.
  [(#7037)](https://github.com/PennyLaneAI/pennylane/pull/7037)

* A `ParametrizedMidMeasure` class has been added to represent a mid-circuit measurement in an arbitrary
  measurement basis in the XY, YZ or ZX plane. Subclasses `XMidMeasureMP` and `YMidMeasureMP` represent
  X-basis and Y-basis measurements. These classes are part of the experimental `ftqc` module.
  [(#6938)](https://github.com/PennyLaneAI/pennylane/pull/6938)
  [(#6953)](https://github.com/PennyLaneAI/pennylane/pull/6953)

* A `diagonalize_mcms` transform has been added that diagonalizes any `ParametrizedMidMeasure`, for devices
  that only natively support mid-circuit measurements in the computational basis.
  [(#6938)](https://github.com/PennyLaneAI/pennylane/pull/6938)
  [(#7037)](https://github.com/PennyLaneAI/pennylane/pull/7037)

<h4>Program capture and dynamic shapes</h4>

* Workflows that create dynamically shaped arrays via `jnp.ones`, `jnp.zeros`, `jnp.arange`, and `jnp.full`
  can now be captured.
  [#6865)](https://github.com/PennyLaneAI/pennylane/pull/6865)

* Workflows wherein the sizes of dynamically shaped arrays are updated in a `qml.while_loop` or `qml.for_loop` 
  are now capturable.
  [(#7084)](https://github.com/PennyLaneAI/pennylane/pull/7084)
  [(#7098)](https://github.com/PennyLaneAI/pennylane/pull/7098/)

* Workflows containing `qml.cond` instances that return arrays with dynamic shapes can now be captured.
  [(#6888)](https://github.com/PennyLaneAI/pennylane/pull/6888/)
  [(#7080)](https://github.com/PennyLaneAI/pennylane/pull/7080)

* A `qml.capture.pause()` context manager has been added for pausing program capture in an error-safe 
  way.
  [(#6911)](https://github.com/PennyLaneAI/pennylane/pull/6911)

* The requested `diff_method` is now validated when program capture is enabled.
  [(#6852)](https://github.com/PennyLaneAI/pennylane/pull/6852)

* A `qml.capture.register_custom_staging_rule` has been added for handling higher-order primitives
  that return new dynamically shaped arrays.
  [(#7086)](https://github.com/PennyLaneAI/pennylane/pull/7086)

* Support has been improved for when wires are specified as `jax.numpy.ndarray` if program capture is 
  enabled.
  [(#7108)](https://github.com/PennyLaneAI/pennylane/pull/7108)

* `qml.cond`, `qml.adjoint`, `qml.ctrl`, and QNodes can now handle accepting dynamically shaped arrays 
  with the abstract shape matching another argument.
  [(#7059)](https://github.com/PennyLaneAI/pennylane/pull/7059)

* A new `qml.capture.eval_jaxpr` function has been implemented. This is a variant of `jax.core.eval_jaxpr` 
  that can handle the creation of arrays with dynamic shapes.
  [(#7052)](https://github.com/PennyLaneAI/pennylane/pull/7052)

* A new, experimental `Operator` method called `compute_qfunc_decomposition` has been added to represent 
  decompositions with structure (e.g., control flow). This method is only used when capture is enabled 
  with `qml.capture.enable()`.
  [(#6859)](https://github.com/PennyLaneAI/pennylane/pull/6859)
  [(#6881)](https://github.com/PennyLaneAI/pennylane/pull/6881)
  [(#7022)](https://github.com/PennyLaneAI/pennylane/pull/7022)
  [(#6917)](https://github.com/PennyLaneAI/pennylane/pull/6917)
  [(#7081)](https://github.com/PennyLaneAI/pennylane/pull/7081)

* The higher order primitives in program capture can now accept inputs with abstract shapes.
  [(#6786)](https://github.com/PennyLaneAI/pennylane/pull/6786)

* Execution interpreters and `qml.capture.eval_jaxpr` can now handle jax `pjit` primitives when dynamic 
  shapes are being used.
  [(#7078)](https://github.com/PennyLaneAI/pennylane/pull/7078)
  [(#7117)](https://github.com/PennyLaneAI/pennylane/pull/7117)

* Device preprocessing is now being performed in the execution pipeline for program capture.
  [(#7057)](https://github.com/PennyLaneAI/pennylane/pull/7057)
  [(#7089)](https://github.com/PennyLaneAI/pennylane/pull/7089)
  [(#7131)](https://github.com/PennyLaneAI/pennylane/pull/7131)
  [(#7135)](https://github.com/PennyLaneAI/pennylane/pull/7135)

<h4>Other improvements</h4>

* The `poly_to_angles` function has been improved to correctly work with different interfaces and no 
  longer manipulate the input angles tensor internally.
  [(#6979)](https://github.com/PennyLaneAI/pennylane/pull/6979)

* Dynamic one-shot workloads are now faster for `null.qubit` by removing a redundant `functools.lru_cache` 
  call that was capturing all `SampleMP` objects in a workload.
  [(#7077)](https://github.com/PennyLaneAI/pennylane/pull/7077)

* The coefficients of observables now have improved differentiability.
  [(#6598)](https://github.com/PennyLaneAI/pennylane/pull/6598)

* An informative error is now raised when a QNode with `diff_method=None` is differentiated.
  [(#6770)](https://github.com/PennyLaneAI/pennylane/pull/6770)

* `qml.gradients.finite_diff_jvp` has been added to compute the jvp of an arbitrary numeric function.
  [(#6853)](https://github.com/PennyLaneAI/pennylane/pull/6853)

* The qchem functions that accept a string input have been updated to consistently work with both
  lower-case and upper-case inputs.
  [(#7186)](https://github.com/PennyLaneAI/pennylane/pull/7186)

* `PSWAP.matrix()` and `PSWAP.eigvals()` now support parameter broadcasting.
  [(#7179)](https://github.com/PennyLaneAI/pennylane/pull/7179)
  [(#7228)](https://github.com/PennyLaneAI/pennylane/pull/7228)

* `Device.eval_jaxpr` now accepts an `execution_config` keyword argument.
  [(#6991)](https://github.com/PennyLaneAI/pennylane/pull/6991)

* `merge_rotations` now correctly simplifies merged `qml.Rot` operators whose angles yield the identity operator.
  [(#7011)](https://github.com/PennyLaneAI/pennylane/pull/7011)

* The `qml.measurements.NullMeasurement` measurement process has been added to allow for profiling problems
  without the overheads associated with performing measurements.
  [(#6989)](https://github.com/PennyLaneAI/pennylane/pull/6989)

* The `pauli_rep` property is now accessible for `Adjoint` operators when there is a Pauli representation.
  [(#6871)](https://github.com/PennyLaneAI/pennylane/pull/6871)

* `qml.pauli.PauliVSpace` is now iterable.
  [(#7054)](https://github.com/PennyLaneAI/pennylane/pull/7054)

* `qml.qchem.taper` now handles wire ordering for the tapered observables more robustly.
  [(#6954)](https://github.com/PennyLaneAI/pennylane/pull/6954)

* A `RuntimeWarning` is now raised by `qml.QNode` and `qml.execute` if executing JAX workflows and the 
  installed version of JAX is greater than `0.4.28`.
  [(#6864)](https://github.com/PennyLaneAI/pennylane/pull/6864)

* The `rng_salt` version has been bumped to `v0.40.0`.
  [(#6854)](https://github.com/PennyLaneAI/pennylane/pull/6854)

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

* `pennylane.labs.dla.lie_closure_dense` has been removed and integrated into `qml.lie_closure` 
  using the new `dense` keyword.
  [(#6811)](https://github.com/PennyLaneAI/pennylane/pull/6811)

* `pennylane.labs.dla.structure_constants_dense` has been removed and integrated into `qml.structure_constants` 
  using the new `matrix` keyword.
  [(#6861)](https://github.com/PennyLaneAI/pennylane/pull/6861)

* `ResourceOperator.resource_params` has been changed to a property.
  [(#6973)](https://github.com/PennyLaneAI/pennylane/pull/6973)

* `ResourceOperator` implementations for the `ModExp`, `PhaseAdder`, `Multiplier`, `ControlledSequence`, 
  `AmplitudeAmplification`, `QROM`, `SuperPosition`, `MottonenStatePreparation`, `StatePrep`, `BasisState` 
  templates have been added.
  [(#6638)](https://github.com/PennyLaneAI/pennylane/pull/6638)

* `pennylane.labs.khaneja_glaser_involution` has been removed,
  `pennylane.labs.check_commutation` has moved to `qml.liealg.check_commutation_relation`.
  `pennylane.labs.check_cartan_decomp` has moved to `qml.liealg.check_cartan_decomp`.
  All involution functions have been moved to `qml.liealg`.
  `pennylane.labs.adjvec_to_op` has moved to `qml.liealg.adjvec_to_op`.
  `pennylane.labs.op_to_adjvec` has moved to `qml.liealg.op_to_adjvec`.
  `pennylane.labs.change_basis_ad_rep` has moved to `qml.liealg.change_basis_ad_rep`.
  `pennylane.labs.cartan_subalgebra` has moved to `qml.liealg.horizontal_cartan_subalgebra`.
  [(#7026)](https://github.com/PennyLaneAI/pennylane/pull/7026)
  [(#7054)](https://github.com/PennyLaneAI/pennylane/pull/7054)

* New classes called `HOState` and `VibronicHO` have been added for representing harmonic oscillator 
  states.
  [(#7035)](https://github.com/PennyLaneAI/pennylane/pull/7035)

* Base classes for Trotter error estimation on Realspace Hamiltonians have been added: `RealspaceOperator`, 
  `RealspaceSum`, `RealspaceCoeffs`, and `RealspaceMatrix`
  [(#7034)](https://github.com/PennyLaneAI/pennylane/pull/7034)

* Functions for Trotter error estimation and Hamiltonian fragment generation have been added: `trotter_error`,
  `perturbation_error`, `vibrational_fragments`, `vibronic_fragments`, and `generic_fragments`.
  [(#7036)](https://github.com/PennyLaneAI/pennylane/pull/7036)

  As an example we compute the perturbation error of a vibrational Hamiltonian.
  First we generate random harmonic frequencies and Taylor coefficients to initialize the vibrational Hamiltonian.

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
    
  We call `vibrational_fragments` to get the harmonic and anharmonic fragments of the vibrational Hamiltonian.
  ```pycon
  >>> frags = vibrational_fragments(n_modes, freqs, taylor_coeffs)
  ```

  We build state vectors in the harmonic oscillator basis with the `HOState` class. 

  ```pycon
  >>> gridpoints = 5
  >>> state1 = HOState(n_modes, gridpoints, {(0, 0): 1})
  >>> state2 = HOState(n_modes, gridpoints, {(1, 1): 1})
  ```

  Finally, we compute the error by calling `perturbation_error`.

  ```pycon
  >>> perturbation_error(frags, [state1, state2])
  [(-0.9189251160920879+0j), (-4.797716682426851+0j)]
  ```

* Function `qml.labs.trotter_error.vibronic_fragments` now returns `RealspaceMatrix` objects with the correct number of electronic states.
  [(#7251)](https://github.com/PennyLaneAI/pennylane/pull/7251)

<h3>Breaking changes 💔</h3>

* Executing `qml.specs` is now much more efficient with the removal of accessing `num_diagonalizing_gates`. 
  The calculation of this quantity is extremely expensive, and the definition is ambiguous for non-commuting 
  observables.
  [(#7047)](https://github.com/PennyLaneAI/pennylane/pull/7047)

* `qml.gradients.gradient_transform.choose_trainable_params` has been renamed to `choose_trainable_param_indices`
  to better reflect what it actually does.
  [(#6928)](https://github.com/PennyLaneAI/pennylane/pull/6928)

* `qml.MultiControlledX` no longer accepts strings as control values.
  [(#6835)](https://github.com/PennyLaneAI/pennylane/pull/6835)

* The `control_wires` argument in `qml.MultiControlledX` has been removed.
  [(#6832)](https://github.com/PennyLaneAI/pennylane/pull/6832)
  [(#6862)](https://github.com/PennyLaneAI/pennylane/pull/6862)

* `qml.execute` now has a collection of keyword-only arguments.
  [(#6598)](https://github.com/PennyLaneAI/pennylane/pull/6598)

* The `decomp_depth` argument in `qml.transforms.set_decomposition` has been removed.
  [(#6824)](https://github.com/PennyLaneAI/pennylane/pull/6824)

* The `max_expansion` argument in `qml.devices.preprocess.decompose` has been removed.
  [(#6824)](https://github.com/PennyLaneAI/pennylane/pull/6824)

* The `tape` and `qtape` properties of `QNode` have been removed. Instead, use the `qml.workflow.construct_tape` 
  function.
  [(#6825)](https://github.com/PennyLaneAI/pennylane/pull/6825)

* The `gradient_fn` keyword argument in `qml.execute` has been removed. Instead, it has been replaced 
  with `diff_method`.
  [(#6830)](https://github.com/PennyLaneAI/pennylane/pull/6830)
  
* The `QNode.get_best_method` and `QNode.best_method_str` methods have been removed. Instead, use the 
  `qml.workflow.get_best_diff_method` function.
  [(#6823)](https://github.com/PennyLaneAI/pennylane/pull/6823)

* The `output_dim` property of `qml.tape.QuantumScript` has been removed. Instead, use method `shape` 
  of `QuantumScript` or `MeasurementProcess` to get the same information.
  [(#6829)](https://github.com/PennyLaneAI/pennylane/pull/6829)

* The `qsvt_legacy` method, along with its private helper `_qsp_to_qsvt`, has been removed.
  [(#6827)](https://github.com/PennyLaneAI/pennylane/pull/6827)

<h3>Deprecations 👋</h3>

* The `qml.qnn.KerasLayer` class has been deprecated because Keras 2 is no longer actively maintained. 
  Please consider using a different machine learning framework instead of `TensorFlow/Keras 2`, like
  Pytorch or JAX.
  [(#7097)](https://github.com/PennyLaneAI/pennylane/pull/7097)

* Specifying `pipeline=None` with `qml.compile` is now deprecated. A sequence of transforms should now 
  always be specified.
  [(#7004)](https://github.com/PennyLaneAI/pennylane/pull/7004)

* `qml.ControlledQubitUnitary` will stop accepting `qml.QubitUnitary` objects as arguments as its `base`. 
  Instead, use `qml.ctrl` to construct a controlled `qml.QubitUnitary`. A follow-up PR fixed accidental 
  double-queuing when using `qml.ctrl` with `QubitUnitary`.
  [(#6840)](https://github.com/PennyLaneAI/pennylane/pull/6840)
  [(#6926)](https://github.com/PennyLaneAI/pennylane/pull/6926)

* The `control_wires` argument in `qml.ControlledQubitUnitary` has been deprecated. Instead, use the 
`wires` argument as the second positional argument.
  [(#6839)](https://github.com/PennyLaneAI/pennylane/pull/6839)

* The `mcm_method` keyword in `qml.execute` has been deprecated. Instead, use the `mcm_method` and 
  `postselect_mode` arguments.
  [(#6807)](https://github.com/PennyLaneAI/pennylane/pull/6807)

* Specifying gradient keyword arguments as any additional keyword argument to the qnode is deprecated
  and will be removed in v0.42. The gradient keyword arguments should be passed to the new keyword argument `gradient_kwargs` 
  via an explicit dictionary. This change will improve QNode argument validation.
  [(#6828)](https://github.com/PennyLaneAI/pennylane/pull/6828)

* The `qml.gradients.hamiltonian_grad` function has been deprecated. This gradient recipe is not required 
  with the new operator arithmetic system.
  [(#6849)](https://github.com/PennyLaneAI/pennylane/pull/6849)

* The `inner_transform_program` and `config` keyword arguments in `qml.execute` have been deprecated.
  If more detailed control over the execution is required, use `qml.workflow.run` with these arguments instead.
  [(#6822)](https://github.com/PennyLaneAI/pennylane/pull/6822)
  [(#6879)](https://github.com/PennyLaneAI/pennylane/pull/6879)

* The property `MeasurementProcess.return_type` has been deprecated. If observable type checking is 
  needed, please use direct `isinstance`; if other text information is needed, please use class name, 
  or another internal temporary private member `_shortname`.
  [(#6841)](https://github.com/PennyLaneAI/pennylane/pull/6841)
  [(#6906)](https://github.com/PennyLaneAI/pennylane/pull/6906)
  [(#6910)](https://github.com/PennyLaneAI/pennylane/pull/6910)

* Pauli module level imports of `lie_closure`, `structure_constants`, and `center` are deprecated, as 
  functionality has moved to the new `liealg` module.
  [(#6935)](https://github.com/PennyLaneAI/pennylane/pull/6935)

<h3>Internal changes ⚙️</h3>

* An informative error message has been added if Autograph is employed on a function that has a `lambda` 
  loop condition in `qml.while_loop`.
  [(#7178)](https://github.com/PennyLaneAI/pennylane/pull/7178)

* Logic in `qml.drawer.tape_text` has been cleaned.
  [(#7133)](https://github.com/PennyLaneAI/pennylane/pull/7133)

* An intermediate caching to `null.qubit` zero-value generation has been added to improve memory consumption 
  for larger workloads.
  [(#7155)](https://github.com/PennyLaneAI/pennylane/pull/7155)

* All use of `ABC` for intermediate variables has been renamed to preserve the label for the Python 
  abstract base class `abc.ABC`.
  [(#7156)](https://github.com/PennyLaneAI/pennylane/pull/7156)

* The error message when device wires are not specified when program capture is enabled is more clear.
  [(#7130)](https://github.com/PennyLaneAI/pennylane/pull/7130)

* Logic in `_capture_qnode.py` has been cleaned.
  [(#7115)](https://github.com/PennyLaneAI/pennylane/pull/7115)

* The test for `qml.math.quantum._denman_beavers_iterations` has been improved such that tested random 
  matrices are guaranteed positive.
  [(#7071)](https://github.com/PennyLaneAI/pennylane/pull/7071)

* The `matrix_power` dispatch for the `scipy` interface has been replaced with an in-place implementation.
  [(#7055)](https://github.com/PennyLaneAI/pennylane/pull/7055)

* Support has been added to `CollectOpsandMeas` for handling QNode primitives.
  [(#6922)](https://github.com/PennyLaneAI/pennylane/pull/6922)

* Change some `scipy` imports from submodules to whole module to reduce memory footprint of importing pennylane.
  [(#7040)](https://github.com/PennyLaneAI/pennylane/pull/7040)

* `NotImplementedError`s have been added for `grad` and `jacobian` in `CollectOpsandMeas`.
  [(#7041)](https://github.com/PennyLaneAI/pennylane/pull/7041)

* Quantum transform interpreters now perform argument validation and will no longer check if the equation 
  in the `jaxpr` is a transform primitive.
  [(#7023)](https://github.com/PennyLaneAI/pennylane/pull/7023)

* `qml.for_loop` and `qml.while_loop` have been moved from the `compiler` module to a new `control_flow` 
  module.
  [(#7017)](https://github.com/PennyLaneAI/pennylane/pull/7017)

* `qml.capture.run_autograph` is now idempotent. This means `run_autograph(fn) = run_autograph(run_autograph(fn))`.
  [(#7001)](https://github.com/PennyLaneAI/pennylane/pull/7001)

* Minor changes to `DQInterpreter` have been made for speedups with program capture execution.
  [(#6984)](https://github.com/PennyLaneAI/pennylane/pull/6984)

* `no-member` pylint issues from JAX are now globally silenced
  [(#6987)](https://github.com/PennyLaneAI/pennylane/pull/6987)

* `pylint=3.3.4` errors in our source code have been fixed.
  [(#6980)](https://github.com/PennyLaneAI/pennylane/pull/6980)
  [(#6988)](https://github.com/PennyLaneAI/pennylane/pull/6988)

* `QNode.get_gradient_fn` has been removed from the source code.
  [(#6898)](https://github.com/PennyLaneAI/pennylane/pull/6898)
  
* The source code has been updated use `black==25.1.0`.
  [(#6897)](https://github.com/PennyLaneAI/pennylane/pull/6897)

* The `InterfaceEnum` object has been improved to prevent direct comparisons to `str` objects.
  [(#6877)](https://github.com/PennyLaneAI/pennylane/pull/6877)

* A `QmlPrimitive` class has been added that inherits `jax.core.Primitive` to a new `qml.capture.custom_primitives` 
  module. This class contains a `prim_type` property so that we can differentiate between different 
  sets of PennyLane primitives. Consequently, `QmlPrimitive` is now used to define all PennyLane primitives.
  [(#6847)](https://github.com/PennyLaneAI/pennylane/pull/6847)

* The `RiemannianGradientOptimizer` has been updated to take advantage of newer features.
  [(#6882)](https://github.com/PennyLaneAI/pennylane/pull/6882)

* The `keep_intermediate=True` flag is now used to keep Catalyst's IR when testing.
  [(#6990)](https://github.com/PennyLaneAI/pennylane/pull/6990)

<h3>Documentation 📝</h3>

* A page on sharp bits and debugging tips has been added for PennyLane program capture: 
  :doc:`/news/program_capture_sharp_bits`. This page is recommended to consult any time errors occur 
  when `qml.capture.enable()` is present.
  [(#7062)](https://github.com/PennyLaneAI/pennylane/pull/7062)

* The :doc:`Compiling Circuits page <../introduction/compiling_circuits>` has been updated to include 
  information on using the new experimental decompositions system.
  [(#7066)](https://github.com/PennyLaneAI/pennylane/pull/7066)

* The docstring for `qml.transforms.decompose` now recommends the `qml.clifford_t_decomposition` 
  transform when decomposing to the Clifford + T gate set.
  [(#7177)](https://github.com/PennyLaneAI/pennylane/pull/7177)

* Typos were fixed in the docstring for `qml.QubitUnitary`.
  [(#7187)](https://github.com/PennyLaneAI/pennylane/pull/7187)

* The docstring for `qml.prod` has been updated to explain that the order of the output may seem reversed,
  but it is correct.
  [(#7083)](https://github.com/PennyLaneAI/pennylane/pull/7083)

* The docstring for `qml.labs.trotter_error` has been updated.
  [(#7190)](https://github.com/PennyLaneAI/pennylane/pull/7190)

* The `gates`, `qubits` and `lamb` attributes of `DoubleFactorization` and `FirstQuantization` have
  dedicated documentation.
  [(#7173)](https://github.com/PennyLaneAI/pennylane/pull/7173)

* The code example in the docstring for `qml.PauliSentence` now properly copy-pastes.
  [(#6949)](https://github.com/PennyLaneAI/pennylane/pull/6949)

* The docstrings for `qml.unary_mapping`, `qml.binary_mapping`, `qml.christiansen_mapping`,
  `qml.qchem.localize_normal_modes`, and `qml.qchem.VibrationalPES` have been updated to include better
  code examples.
  [(#6717)](https://github.com/PennyLaneAI/pennylane/pull/6717)

* The docstrings for `qml.qchem.localize_normal_modes` and `qml.qchem.VibrationalPES` have been updated 
  to include examples that can be copied.
  [(#6834)](https://github.com/PennyLaneAI/pennylane/pull/6834)

* A typo has been fixed in the code example for `qml.labs.dla.lie_closure_dense`.
  [(#6858)](https://github.com/PennyLaneAI/pennylane/pull/6858)

* The code example in the docstring for `qml.BasisRotation` was corrected by including `wire_order` 
  in the call to `qml.matrix`.
  [(#6891)](https://github.com/PennyLaneAI/pennylane/pull/6891)

* The docstring of `qml.noise.meas_eq` has been updated to make its functionality clearer.
  [(#6920)](https://github.com/PennyLaneAI/pennylane/pull/6920)

* The docstring for `qml.devices.default_tensor.DefaultTensor` has been updated to clarify differentiation 
  support.
  [(#7150)](https://github.com/PennyLaneAI/pennylane/pull/7150)

* The docstring for `QuantumScripts` has been updated to remove outdated references to `set_parameters`.
  [(#7174)](https://github.com/PennyLaneAI/pennylane/pull/7174)

* The documentation for `qml.device` has been updated to include `null.qubit` in the list of readily available devices.
  [(#7233)](https://github.com/PennyLaneAI/pennylane/pull/7233)

<h3>Bug fixes 🐛</h3>

* Using transforms with program capture enabled now works correctly with functions that accept pytree arguments.
  [(#7233)](https://github.com/PennyLaneAI/pennylane/pull/7233)

* `qml.math.requires_grad` no longer returns `True` for JAX inputs other than `jax.interpreters.ad.JVPTracer` instances.
  [(#7233)](https://github.com/PennyLaneAI/pennylane/pull/7233)

* PennyLane is now compatible with `pyzx 0.9`.
  [(#7188)](https://github.com/PennyLaneAI/pennylane/pull/7188)

* Fixed a bug when `qml.matrix` is applied on a sparse operator, which caused the output to have unnecessary 
  epsilon inaccuracy.
  [(#7147)](https://github.com/PennyLaneAI/pennylane/pull/7147)
  [(#7182)](https://github.com/PennyLaneAI/pennylane/pull/7182)

* Reverted [(#6933)](https://github.com/PennyLaneAI/pennylane/pull/6933) to remove non-negligible performance 
  impact due to wire flattening.
  [(#7136)](https://github.com/PennyLaneAI/pennylane/pull/7136)

* Fixed a bug that caused the output of `qml.fourier.qnode_spectrum()` to differ depending if equivalent 
  gate generators are defined using different PennyLane operators. This was resolved by updating
  `qml.operation.gen_is_multi_term_hamiltonian` to work with more complicated generators.
  [(#7121)](https://github.com/PennyLaneAI/pennylane/pull/7121)

* Modulo operator calls on MCMs now correctly offload to the autoray-backed `qml.math.mod` dispatch.
  [(#7085)](https://github.com/PennyLaneAI/pennylane/pull/7085)

* `qml.transforms.single_qubit_fusion` and `qml.transforms.cancel_inverses` now correctly handle mid-circuit 
  measurements when program capture is enabled.
  [(#7020)](https://github.com/PennyLaneAI/pennylane/pull/7020)

* `qml.math.get_interface` now correctly extracts the `"scipy"` interface if provided a list/array of 
  sparse matrices.
  [(#7015)](https://github.com/PennyLaneAI/pennylane/pull/7015)

* `qml.ops.Controlled.has_sparse_matrix` now provides the correct information by checking if the target 
  operator has a sparse or dense matrix defined.
  [(#7025)](https://github.com/PennyLaneAI/pennylane/pull/7025)

* `qml.capture.PlxprInterpreter` now flattens pytree arguments before evaluation.
  [(#6975)](https://github.com/PennyLaneAI/pennylane/pull/6975)

* `qml.GlobalPhase.sparse_matrix` now correctly returns a sparse matrix of the same shape as `matrix`.
  [(#6940)](https://github.com/PennyLaneAI/pennylane/pull/6940)

* `qml.expval` no longer silently casts to a real number when observable coefficients are imaginary.
  [(#6939)](https://github.com/PennyLaneAI/pennylane/pull/6939)

* Fixed `qml.wires.Wires` initialization to disallow `Wires` objects as wires labels. Now, `Wires` is 
  idempotent, e.g. `Wires([Wires([0]), Wires([1])])==Wires([0, 1])`.
  [(#6933)](https://github.com/PennyLaneAI/pennylane/pull/6933)

* `qml.capture.PlxprInterpreter` now correctly handles propagation of constants when interpreting higher-order 
  primitives.
  [(#6913)](https://github.com/PennyLaneAI/pennylane/pull/6913)

* `qml.capture.PlxprInterpreter` now uses `Primitive.get_bind_params` to resolve primitive calling signatures 
  before binding primitives.
  [(#6913)](https://github.com/PennyLaneAI/pennylane/pull/6913)

* The interface is now detected from the data in the circuit, not the arguments to the QNode. This allows
  interface data to be strictly passed as closure variables and still be detected.
  [(#6892)](https://github.com/PennyLaneAI/pennylane/pull/6892)

* `qml.BasisState` now casts its input to integers.
  [(#6844)](https://github.com/PennyLaneAI/pennylane/pull/6844)

* The `workflow.contstruct_batch` and `workflow.construct_tape` functions now correctly reflect the 
  `mcm_method` passed to the `QNode`, instead of assuming the method is always `deferred`.
  [(#6903)](https://github.com/PennyLaneAI/pennylane/pull/6903)

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
Alex Preciado,
Shuli Shu,
David Wierichs
