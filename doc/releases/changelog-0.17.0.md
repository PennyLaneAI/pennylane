
# Release 0.17.0

<h3>New features since the last release</h3>

<h4>Circuit optimization</h4>

* PennyLane can now perform quantum circuit optimization using the
  top-level transform `qp.compile`. The `compile` transform allows you
  to chain together sequences of tape and quantum function transforms
  into custom circuit optimization pipelines.
  [(#1475)](https://github.com/PennyLaneAI/pennylane/pull/1475)

  For example, take the following decorated quantum function:

  ```python
  dev = qp.device('default.qubit', wires=[0, 1, 2])

  @qp.qnode(dev)
  @qp.compile()
  def qfunc(x, y, z):
      qp.Hadamard(wires=0)
      qp.Hadamard(wires=1)
      qp.Hadamard(wires=2)
      qp.RZ(z, wires=2)
      qp.CNOT(wires=[2, 1])
      qp.RX(z, wires=0)
      qp.CNOT(wires=[1, 0])
      qp.RX(x, wires=0)
      qp.CNOT(wires=[1, 0])
      qp.RZ(-z, wires=2)
      qp.RX(y, wires=2)
      qp.PauliY(wires=2)
      qp.CZ(wires=[1, 2])
      return qp.expval(qp.PauliZ(wires=0))
  ```

  The default behaviour of `qp.compile` is to apply a sequence of three
  transforms: `commute_controlled`, `cancel_inverses`, and then `merge_rotations`.

  ```pycon
  >>> print(qp.draw(qfunc)(0.2, 0.3, 0.4))
   0: ──H───RX(0.6)──────────────────┤ ⟨Z⟩
   1: ──H──╭X────────────────────╭C──┤
   2: ──H──╰C────────RX(0.3)──Y──╰Z──┤
  ```

  The `qp.compile` transform is flexible and accepts a custom pipeline
  of tape and quantum function transforms (you can even write your own!).
  For example, if we wanted to only push single-qubit gates through
  controlled gates and cancel adjacent inverses, we could do:

  ```python
  from pennylane.transforms import commute_controlled, cancel_inverses
  pipeline = [commute_controlled, cancel_inverses]

  @qp.qnode(dev)
  @qp.compile(pipeline=pipeline)
  def qfunc(x, y, z):
      qp.Hadamard(wires=0)
      qp.Hadamard(wires=1)
      qp.Hadamard(wires=2)
      qp.RZ(z, wires=2)
      qp.CNOT(wires=[2, 1])
      qp.RX(z, wires=0)
      qp.CNOT(wires=[1, 0])
      qp.RX(x, wires=0)
      qp.CNOT(wires=[1, 0])
      qp.RZ(-z, wires=2)
      qp.RX(y, wires=2)
      qp.PauliY(wires=2)
      qp.CZ(wires=[1, 2])
      return qp.expval(qp.PauliZ(wires=0))
  ```

  ```pycon
  >>> print(qp.draw(qfunc)(0.2, 0.3, 0.4))
   0: ──H───RX(0.4)──RX(0.2)────────────────────────────┤ ⟨Z⟩
   1: ──H──╭X───────────────────────────────────────╭C──┤
   2: ──H──╰C────────RZ(0.4)──RZ(-0.4)──RX(0.3)──Y──╰Z──┤
  ```

  The following compilation transforms have been added and are also available
  to use, either independently, or within a `qp.compile` pipeline:

  * `commute_controlled`: push commuting single-qubit gates through controlled operations.
    [(#1464)](https://github.com/PennyLaneAI/pennylane/pull/1464)

  * `cancel_inverses`: removes adjacent pairs of operations that cancel out.
    [(#1455)](https://github.com/PennyLaneAI/pennylane/pull/1455)

  * `merge_rotations`: combines adjacent rotation gates of
    the same type into a single gate, including controlled rotations.
    [(#1455)](https://github.com/PennyLaneAI/pennylane/pull/1455)

  * `single_qubit_fusion`: acts on all sequences of
    single-qubit operations in a quantum function, and converts each
    sequence to a single `Rot` gate.
    [(#1458)](https://github.com/PennyLaneAI/pennylane/pull/1458)

  For more details on `qp.compile` and the available compilation transforms, see
  [the compilation documentation](https://pennylane.readthedocs.io/en/stable/code/qml_transforms.html#transforms-for-circuit-compilation).

<h4>QNodes are even more powerful</h4>

* Computational basis samples directly from the underlying device can
  now be returned directly from QNodes via `qp.sample()`.
  [(#1441)](https://github.com/PennyLaneAI/pennylane/pull/1441)

  ```python
  dev = qp.device("default.qubit", wires=3, shots=5)

  @qp.qnode(dev)
  def circuit_1():
      qp.Hadamard(wires=0)
      qp.Hadamard(wires=1)
      return qp.sample()

  @qp.qnode(dev)
  def circuit_2():
      qp.Hadamard(wires=0)
      qp.Hadamard(wires=1)
      return qp.sample(wires=[0,2])    # no observable provided and wires specified
  ```

  ```pycon
  >>> print(circuit_1())
  [[1, 0, 0],
   [1, 1, 0],
   [1, 0, 0],
   [0, 0, 0],
   [0, 1, 0]]

  >>> print(circuit_2())
  [[1, 0],
   [1, 0],
   [1, 0],
   [0, 0],
   [0, 0]]

  >>> print(qp.draw(circuit_2)())
   0: ──H──╭┤ Sample[basis]
   1: ──H──│┤
   2: ─────╰┤ Sample[basis]
  ```

* The new `qp.apply` function can be used to add operations that might have
  already been instantiated elsewhere to the QNode and other queuing contexts:
  [(#1433)](https://github.com/PennyLaneAI/pennylane/pull/1433)

  ```python
  op = qp.RX(0.4, wires=0)
  dev = qp.device("default.qubit", wires=2)

  @qp.qnode(dev)
  def circuit(x):
      qp.RY(x, wires=0)
      qp.apply(op)
      return qp.expval(qp.PauliZ(0))
  ```

  ```pycon
  >>> print(qp.draw(circuit)(0.6))
  0: ──RY(0.6)──RX(0.4)──┤ ⟨Z⟩
  ```

  Previously instantiated measurements can also be applied to QNodes.

<h4>Device Resource Tracker</h4>

* The new Device Tracker capabilities allows for flexible and versatile tracking of executions,
  even inside parameter-shift gradients. This functionality will improve the ease of monitoring
  large batches and remote jobs.
  [(#1355)](https://github.com/PennyLaneAI/pennylane/pull/1355)

  ```python
  dev = qp.device('default.qubit', wires=1, shots=100)

  @qp.qnode(dev, diff_method="parameter-shift")
  def circuit(x):
      qp.RX(x, wires=0)
      return qp.expval(qp.PauliZ(0))

  x = np.array(0.1)

  with qp.Tracker(circuit.device) as tracker:
      qp.grad(circuit)(x)
  ```

  ```pycon
  >>> tracker.totals
  {'executions': 3, 'shots': 300, 'batches': 1, 'batch_len': 2}
  >>> tracker.history
  {'executions': [1, 1, 1],
   'shots': [100, 100, 100],
   'batches': [1],
   'batch_len': [2]}
  >>> tracker.latest
  {'batches': 1, 'batch_len': 2}
  ```

  Users can also provide a custom function to the `callback` keyword that gets called each time
  the information is updated.  This functionality allows users to monitor remote jobs or large
  parameter-shift batches.

  ```pycon
  >>> def shots_info(totals, history, latest):
  ...     print("Total shots: ", totals['shots'])
  >>> with qp.Tracker(circuit.device, callback=shots_info) as tracker:
  ...     qp.grad(circuit)(0.1)
  Total shots:  100
  Total shots:  200
  Total shots:  300
  Total shots:  300
  ```

<h4>Containerization support</h4>

* Docker support for building PennyLane with support for all interfaces (TensorFlow,
  Torch, and Jax), as well as device plugins and QChem, for GPUs and CPUs, has been added.
  [(#1391)](https://github.com/PennyLaneAI/pennylane/pull/1391)

  The build process using Docker and `make` requires that the repository source
  code is cloned or downloaded from GitHub. Visit the the detailed description
  for an [extended list of
  options](https://pennylane.readthedocs.io/en/stable/development/guide/installation.html#installation).

<h4>Improved Hamiltonian simulations</h4>

* Added a sparse Hamiltonian observable and the functionality to support computing its expectation
  value with `default.qubit`. [(#1398)](https://github.com/PennyLaneAI/pennylane/pull/1398)

  For example, the following QNode returns the expectation value of a sparse Hamiltonian:

  ```python
  dev = qp.device("default.qubit", wires=2)

  @qp.qnode(dev, diff_method="parameter-shift")
  def circuit(param, H):
      qp.PauliX(0)
      qp.SingleExcitation(param, wires=[0, 1])
      return qp.expval(qp.SparseHamiltonian(H, [0, 1]))
  ```

  We can execute this QNode, passing in a sparse identity matrix:

  ```pycon
  >>> print(circuit([0.5], scipy.sparse.eye(4).tocoo()))
  0.9999999999999999
  ```

  The expectation value of the sparse Hamiltonian is computed directly, which leads to executions
  that are faster by orders of magnitude. Note that "parameter-shift" is the only differentiation
  method that is currently supported when the observable is a sparse Hamiltonian.

* VQE problems can now be intuitively set up by passing the Hamiltonian
  as an observable. [(#1474)](https://github.com/PennyLaneAI/pennylane/pull/1474)

  ``` python
  dev = qp.device("default.qubit", wires=2)
  H = qp.Hamiltonian([1., 2., 3.],  [qp.PauliZ(0), qp.PauliY(0), qp.PauliZ(1)])
  w = qp.init.strong_ent_layers_uniform(1, 2, seed=1967)

  @qp.qnode(dev)
  def circuit(w):
      qp.templates.StronglyEntanglingLayers(w, wires=range(2))
      return qp.expval(H)
  ```

  ```pycon
  >>> print(circuit(w))
  -1.5133943637878295
  >>> print(qp.grad(circuit)(w))
  [[[-8.32667268e-17  1.39122955e+00 -9.12462052e-02]
  [ 1.02348685e-16 -7.77143238e-01 -1.74708049e-01]]]
  ```

  Note that other measurement types like `var(H)` or `sample(H)`, as well
  as multiple expectations like `expval(H1), expval(H2)` are not supported.

* Added functionality to compute the sparse matrix representation of a `qp.Hamiltonian` object.
  [(#1394)](https://github.com/PennyLaneAI/pennylane/pull/1394)

<h4>New gradients module</h4>

* A new gradients module `qp.gradients` has been added, which provides
  differentiable quantum gradient transforms.
  [(#1476)](https://github.com/PennyLaneAI/pennylane/pull/1476)
  [(#1479)](https://github.com/PennyLaneAI/pennylane/pull/1479)
  [(#1486)](https://github.com/PennyLaneAI/pennylane/pull/1486)

  Available quantum gradient transforms include:

  - `qp.gradients.finite_diff`
  - `qp.gradients.param_shift`
  - `qp.gradients.param_shift_cv`

  For example,

  ```pycon
  >>> params = np.array([0.3,0.4,0.5], requires_grad=True)
  >>> with qp.tape.JacobianTape() as tape:
  ...     qp.RX(params[0], wires=0)
  ...     qp.RY(params[1], wires=0)
  ...     qp.RX(params[2], wires=0)
  ...     qp.expval(qp.PauliZ(0))
  ...     qp.var(qp.PauliZ(0))
  >>> tape.trainable_params = {0, 1, 2}
  >>> gradient_tapes, fn = qp.gradients.finite_diff(tape)
  >>> res = dev.batch_execute(gradient_tapes)
  >>> fn(res)
  array([[-0.69688381, -0.32648317, -0.68120105],
         [ 0.8788057 ,  0.41171179,  0.85902895]])
  ```

<h4>Even more new operations and templates</h4>

* Grover Diffusion Operator template added.
  [(#1442)](https://github.com/PennyLaneAI/pennylane/pull/1442)

  For example, if we have an oracle that marks the "all ones" state with a
  negative sign:

  ```python
  n_wires = 3
  wires = list(range(n_wires))

  def oracle():
      qp.Hadamard(wires[-1])
      qp.Toffoli(wires=wires)
      qp.Hadamard(wires[-1])
  ```

  We can perform [Grover's Search Algorithm](https://en.wikipedia.org/wiki/Grover%27s_algorithm):

  ```python
  dev = qp.device('default.qubit', wires=wires)

  @qp.qnode(dev)
  def GroverSearch(num_iterations=1):
      for wire in wires:
          qp.Hadamard(wire)

      for _ in range(num_iterations):
          oracle()
          qp.templates.GroverOperator(wires=wires)

      return qp.probs(wires)
  ```

  We can see this circuit yields the marked state with high probability:

  ```pycon
  >>> GroverSearch(num_iterations=1)
  tensor([0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125, 0.03125,
          0.78125], requires_grad=True)
  >>> GroverSearch(num_iterations=2)
  tensor([0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
      0.0078125, 0.9453125], requires_grad=True)
  ```

* A decomposition has been added to `QubitUnitary` that makes the
  single-qubit case fully differentiable in all interfaces. Furthermore,
  a quantum function transform, `unitary_to_rot()`, has been added to decompose all
  single-qubit instances of `QubitUnitary` in a quantum circuit.
  [(#1427)](https://github.com/PennyLaneAI/pennylane/pull/1427)

  Instances of `QubitUnitary` may now be decomposed directly to `Rot`
  operations, or `RZ` operations if the input matrix is diagonal. For
  example, let

  ```python
  >>> U = np.array([
      [-0.28829348-0.78829734j,  0.30364367+0.45085995j],
      [ 0.53396245-0.10177564j,  0.76279558-0.35024096j]
  ])
  ```

  Then, we can compute the decomposition as:

  ```pycon
  >>> qp.QubitUnitary.decomposition(U, wires=0)
  [Rot(-0.24209530281458358, 1.1493817777199102, 1.733058145303424, wires=[0])]
  ```

  We can also apply the transform directly to a quantum function, and compute the
  gradients of parameters used to construct the unitary matrices.

  ```python
  def qfunc_with_qubit_unitary(angles):
      z, x = angles[0], angles[1]

      Z_mat = np.array([[np.exp(-1j * z / 2), 0.0], [0.0, np.exp(1j * z / 2)]])

      c = np.cos(x / 2)
      s = np.sin(x / 2) * 1j
      X_mat = np.array([[c, -s], [-s, c]])

      qp.Hadamard(wires="a")
      qp.QubitUnitary(Z_mat, wires="a")
      qp.QubitUnitary(X_mat, wires="b")
      qp.CNOT(wires=["b", "a"])
      return qp.expval(qp.PauliX(wires="a"))
  ```

  ```pycon
  >>> dev = qp.device("default.qubit", wires=["a", "b"])
  >>> transformed_qfunc = qp.transforms.unitary_to_rot(qfunc_with_qubit_unitary)
  >>> transformed_qnode = qp.QNode(transformed_qfunc, dev)
  >>> input = np.array([0.3, 0.4], requires_grad=True)
  >>> transformed_qnode(input)
  tensor(0.95533649, requires_grad=True)
  >>> qp.grad(transformed_qnode)(input)
  array([-0.29552021,  0.        ])
  ```

* Ising YY gate functionality added.
  [(#1358)](https://github.com/PennyLaneAI/pennylane/pull/1358)

<h3>Improvements</h3>

* The tape does not verify any more that all Observables have owners in the annotated queue.
  [(#1505)](https://github.com/PennyLaneAI/pennylane/pull/1505)

  This allows manipulation of Observables inside a tape context. An example is
  `expval(Tensor(qp.PauliX(0), qp.Identity(1)).prune())` which makes the expval an owner
  of the pruned tensor and its constituent observables, but leaves the original tensor in
  the queue without an owner.

* The `step` and `step_and_cost` methods of `QNGOptimizer` now accept a custom `grad_fn`
  keyword argument to use for gradient computations.
  [(#1487)](https://github.com/PennyLaneAI/pennylane/pull/1487)

* The precision used by `default.qubit.jax` now matches the float precision
  indicated by
  ```python
  from jax.config import config
  config.read('jax_enable_x64')
  ```
  where `True` means `float64`/`complex128` and `False` means `float32`/`complex64`.
  [(#1485)](https://github.com/PennyLaneAI/pennylane/pull/1485)

* The `./pennylane/ops/qubit.py` file is broken up into a folder of six separate files.
  [(#1467)](https://github.com/PennyLaneAI/pennylane/pull/1467)

* Changed to using commas as the separator of wires in the string
  representation of `qp.Hamiltonian` objects for multi-qubit terms.
  [(#1465)](https://github.com/PennyLaneAI/pennylane/pull/1465)

* Changed to using `np.object_` instead of `np.object` as per the NumPy
  deprecations starting version 1.20.
  [(#1466)](https://github.com/PennyLaneAI/pennylane/pull/1466)

* Change the order of the covariance matrix and the vector of means internally
  in `default.gaussian`. [(#1331)](https://github.com/PennyLaneAI/pennylane/pull/1331)

* Added the `id` attribute to templates.
  [(#1438)](https://github.com/PennyLaneAI/pennylane/pull/1438)

* The `qp.math` module, for framework-agnostic tensor manipulation,
  has two new functions available:
  [(#1490)](https://github.com/PennyLaneAI/pennylane/pull/1490)

  - `qp.math.get_trainable_indices(sequence_of_tensors)`: returns the indices corresponding to
    trainable tensors in the input sequence.

  - `qp.math.unwrap(sequence_of_tensors)`: unwraps a sequence of tensor-like objects to NumPy
    arrays.

  In addition, the behaviour of `qp.math.requires_grad` has been improved in order to
  correctly determine trainability during Autograd and JAX backwards passes.

* A new tape method, `tape.unwrap()` is added. This method is a context manager; inside the
  context, the tape's parameters are unwrapped to NumPy arrays and floats, and the trainable
  parameter indices are set.
  [(#1491)](https://github.com/PennyLaneAI/pennylane/pull/1491)

  These changes are temporary, and reverted on exiting the context.

  ```pycon
  >>> with tf.GradientTape():
  ...     with qp.tape.QuantumTape() as tape:
  ...         qp.RX(tf.Variable(0.1), wires=0)
  ...         qp.RY(tf.constant(0.2), wires=0)
  ...         qp.RZ(tf.Variable(0.3), wires=0)
  ...     with tape.unwrap():
  ...         print("Trainable params:", tape.trainable_params)
  ...         print("Unwrapped params:", tape.get_parameters())
  Trainable params: {0, 2}
  Unwrapped params: [0.1, 0.3]
  >>> print("Original parameters:", tape.get_parameters())
  Original parameters: [<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.1>,
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.3>]
  ```

  In addition, `qp.tape.Unwrap` is a context manager that unwraps multiple tapes:

  ```pycon
  >>> with qp.tape.Unwrap(tape1, tape2):
  ```

<h3>Breaking changes</h3>

* Removed the deprecated tape methods `get_resources` and `get_depth` as they are
  superseded by the `specs` tape attribute.
  [(#1522)](https://github.com/PennyLaneAI/pennylane/pull/1522)

* Specifying `shots=None` with `qp.sample` was previously deprecated.
  From this release onwards, setting `shots=None` when sampling will
  raise an error.
  [(#1522)](https://github.com/PennyLaneAI/pennylane/pull/1522)

* The existing `pennylane.collections.apply` function is no longer accessible
  via `qp.apply`, and needs to be imported directly from the `collections`
  package.
  [(#1358)](https://github.com/PennyLaneAI/pennylane/pull/1358)

<h3>Bug fixes</h3>

* Fixes a bug in `qp.adjoint` and `qp.ctrl`
  where the adjoint of operations outside of a `QNode` or a `QuantumTape` could
  not be obtained.
  [(#1532)](https://github.com/PennyLaneAI/pennylane/pull/1532)

* Fixes a bug in `GradientDescentOptimizer` and `NesterovMomentumOptimizer`
  where a cost function with one trainable parameter and non-trainable
  parameters raised an error.
  [(#1495)](https://github.com/PennyLaneAI/pennylane/pull/1495)

* Fixed an example in the documentation's
  [introduction to numpy gradients](https://pennylane.readthedocs.io/en/stable/introduction/interfaces/numpy.html), where
  the wires were a non-differentiable argument to the QNode.
  [(#1499)](https://github.com/PennyLaneAI/pennylane/pull/1499)

* Fixed a bug where the adjoint of `qp.QFT` when using the `qp.adjoint` function
  was not correctly computed.
  [(#1451)](https://github.com/PennyLaneAI/pennylane/pull/1451)

* Fixed the differentiability of the operation`IsingYY` for Autograd, Jax and Tensorflow.
  [(#1425)](https://github.com/PennyLaneAI/pennylane/pull/1425)

* Fixed a bug in the `torch` interface that prevented gradients from being
  computed on a GPU. [(#1426)](https://github.com/PennyLaneAI/pennylane/pull/1426)

* Quantum function transforms now preserve the format of the measurement
  results, so that a single measurement returns a single value rather than
  an array with a single element. [(#1434)](https://github.com/PennyLaneAI/pennylane/pull/1434)

* Fixed a bug in the parameter-shift Hessian implementation, which resulted
  in the incorrect Hessian being returned for a cost function
  that performed post-processing on a vector-valued QNode.
  [(#1436)](https://github.com/PennyLaneAI/pennylane/pull/1436)

* Fixed a bug in the initialization of `QubitUnitary` where the size of
  the matrix was not checked against the number of wires.
  [(#1439)](https://github.com/PennyLaneAI/pennylane/pull/1439)

<h3>Documentation</h3>

* Improved Contribution Guide and Pull Requests Guide.
  [(#1461)](https://github.com/PennyLaneAI/pennylane/pull/1461)

* Examples have been added to clarify use of the continuous-variable
  `FockStateVector` operation in the multi-mode case.
  [(#1472)](https://github.com/PennyLaneAI/pennylane/pull/1472)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Olivia Di Matteo, Anthony Hayes, Theodor Isacsson, Josh
Izaac, Soran Jahangiri, Nathan Killoran, Arshpreet Singh Khangura, Leonhard
Kunczik, Christina Lee, Romain Moyard, Lee James O'Riordan, Ashish Panigrahi,
Nahum Sá, Maria Schuld, Jay Soni, Antal Száva, David Wierichs.
