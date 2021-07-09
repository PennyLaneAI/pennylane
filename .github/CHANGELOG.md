# Release 0.17.0-dev (development release)

<h3>New features since last release</h3>

* A decomposition has been added to ``QubitUnitary`` that makes the
  single-qubit case fully differentiable in all interfaces. Furthermore,
  a quantum function transform, ``unitary_to_rot()``, has been added to decompose all
  single-qubit instances of ``QubitUnitary`` in a quantum circuit.
  [(#1427)](https://github.com/PennyLaneAI/pennylane/pull/1427)

  Instances of ``QubitUnitary`` may now be decomposed directly to ``Rot``
  operations, or ``RZ`` operations if the input matrix is diagonal. For
  example, let

  ```python
  >>> U = np.array([
      [-0.28829348-0.78829734j,  0.30364367+0.45085995j],
      [ 0.53396245-0.10177564j,  0.76279558-0.35024096j]
  ])
  ```

  Then, we can compute the decomposition as:

  ```pycon
  >>> qml.QubitUnitary.decomposition(U, wires=0)
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

      qml.Hadamard(wires="a")
      qml.QubitUnitary(Z_mat, wires="a")
      qml.QubitUnitary(X_mat, wires="b")
      qml.CNOT(wires=["b", "a"])
      return qml.expval(qml.PauliX(wires="a"))
  ```

  ```pycon
  >>> dev = qml.device("default.qubit", wires=["a", "b"])
  >>> transformed_qfunc = qml.transforms.unitary_to_rot(qfunc_with_qubit_unitary)
  >>> transformed_qnode = qml.QNode(transformed_qfunc, dev)
  >>> input = np.array([0.3, 0.4], requires_grad=True)
  >>> transformed_qnode(input)
  tensor(0.95533649, requires_grad=True)
  >>> qml.grad(transformed_qnode)(input)
  array([-0.29552021,  0.        ])
  ```

* The new ``qml.apply`` function can be used to add operations that might have
  already been instantiated elsewhere to the QNode and other queuing contexts:
  [(#1433)](https://github.com/PennyLaneAI/pennylane/pull/1433)

  ```python
  op = qml.RX(0.4, wires=0)
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def circuit(x):
      qml.RY(x, wires=0)
      qml.apply(op)
      return qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> print(qml.draw(circuit)(0.6))
  0: ──RY(0.6)──RX(0.4)──┤ ⟨Z⟩
  ```

  Previously instantiated measurements can also be applied to QNodes.

* Ising YY gate functionality added.
  [(#1358)](https://github.com/PennyLaneAI/pennylane/pull/1358)

<h3>Improvements</h3>

* Added the `id` attribute to templates, which was missing from 
  PR [(#1377)](https://github.com/PennyLaneAI/pennylane/pull/1377).
  [(#1438)](https://github.com/PennyLaneAI/pennylane/pull/1438)
  
<h3>Breaking changes</h3>

* The existing `pennylane.collections.apply` function is no longer accessible
  via `qml.apply`, and needs to be imported directly from the ``collections``
  package.
  [(#1358)](https://github.com/PennyLaneAI/pennylane/pull/1358)

<h3>Bug fixes</h3>

* Fixes a bug where the adjoint of `qml.QFT` when using the `qml.adjoint` function
  was not correctly computed.
  [(#1451)](https://github.com/PennyLaneAI/pennylane/pull/1451)

* Fixes the differentiability of the operation`IsingYY` for Autograd, Jax and Tensorflow.
  [(#1425)](https://github.com/PennyLaneAI/pennylane/pull/1425)
  
* Fixed a bug in the `torch` interface that prevented gradients from being
  computed on a GPU. [(#1426)](https://github.com/PennyLaneAI/pennylane/pull/1426)

* Quantum function transforms now preserve the format of the measurement
  results, so that a single measurement returns a single value rather than
  an array with a single element. [(#1434)](https://github.com/PennyLaneAI/pennylane/pull/1434)

* Fixed a bug in the parameter-shift Hessian implementation, which resulted
  in the incorrect Hessian being returned for a cost function
  that performed post-processing on a vector-valued QNode.
  [(#)](https://github.com/PennyLaneAI/pennylane/pull/)

* Fixed a bug in the initialization of `QubitUnitary` where the size of
  the matrix was not checked against the number of wires.
  [(#1439)](https://github.com/PennyLaneAI/pennylane/pull/1439)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Olivia Di Matteo, Josh Izaac, Romain Moyard, Ashish Panigrahi, Maria Schuld.


# Release 0.16.0 (current release)

<h3>New features since last release</h3>

* Added a sparse Hamiltonian observable and the functionality to support computing its expectation
  value. [(#1398)](https://github.com/PennyLaneAI/pennylane/pull/1398)

  For example, the following QNode returns the expectation value of a sparse Hamiltonian:

  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev, diff_method="parameter-shift")
  def circuit(param, H):
      qml.PauliX(0)
      qml.SingleExcitation(param, wires=[0, 1])
      return qml.expval(qml.SparseHamiltonian(H, [0, 1]))
  ```
  
  We can execute this QNode, passing in a sparse identity matrix:

  ```pycon
  >>> print(circuit([0.5], scipy.sparse.eye(4).tocoo()))
  0.9999999999999999
  ```

  The expectation value of the sparse Hamiltonian is computed directly, which leads to executions
  that are faster by orders of magnitude. Note that "parameter-shift" is the only differentiation
  method that is currently supported when the observable is a sparse Hamiltonian.

* Added functionality to compute the sparse matrix representation of a `qml.Hamiltonian` object.
  [(#1394)](https://github.com/PennyLaneAI/pennylane/pull/1394)

<h4>First class support for quantum kernels</h4>

* The new `qml.kernels` module provides basic functionalities for [working with quantum
  kernels](https://pennylane.readthedocs.io/en/stable/code/qml_kernels.html) as
  well as post-processing methods to mitigate sampling errors and device noise:
  [(#1102)](https://github.com/PennyLaneAI/pennylane/pull/1102)

  ```python

  num_wires = 6
  wires = range(num_wires)
  dev = qml.device('default.qubit', wires=wires)

  @qml.qnode(dev)
  def kernel_circuit(x1, x2):
      qml.templates.AngleEmbedding(x1, wires=wires)
      qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=wires)
      return qml.probs(wires)

  kernel = lambda x1, x2: kernel_circuit(x1, x2)[0]
  X_train = np.random.random((10, 6))
  X_test = np.random.random((5, 6))

  # Create symmetric square kernel matrix (for training)
  K = qml.kernels.square_kernel_matrix(X_train, kernel)

  # Compute kernel between test and training data.
  K_test = qml.kernels.kernel_matrix(X_train, X_test, kernel)
  K1 = qml.kernels.mitigate_depolarizing_noise(K, num_wires, method='single')
  ```

<h4>Extract the fourier representation of quantum circuits</h4>

* PennyLane now has a `fourier` module, which hosts a [growing library
  of methods](https://pennylane.readthedocs.io/en/stable/code/qml_fourier.html)
  that help with investigating the Fourier representation of functions
  implemented by quantum circuits. The Fourier representation can be used
  to examine and characterize the expressivity of the quantum circuit.
  [(#1160)](https://github.com/PennyLaneAI/pennylane/pull/1160)
  [(#1378)](https://github.com/PennyLaneAI/pennylane/pull/1378)

  For example, one can plot distributions over Fourier series coefficients like
  this one:

  <img src="https://pennylane.readthedocs.io/en/latest/_static/fourier.png" width=70%/>

<h4>Seamless support for working with the Pauli group</h4>

* Added functionality for constructing and manipulating the Pauli group
  [(#1181)](https://github.com/PennyLaneAI/pennylane/pull/1181).

  The function `qml.grouping.pauli_group` provides a generator to
  easily loop over the group, or construct and store it in its entirety.
  For example, we can construct the single-qubit Pauli group like so:

  ```pycon
  >>> from pennylane.grouping import pauli_group
  >>> pauli_group_1_qubit = list(pauli_group(1))
  >>> pauli_group_1_qubit
  [Identity(wires=[0]), PauliZ(wires=[0]), PauliX(wires=[0]), PauliY(wires=[0])]
  ```

  We can multiply together its members at the level of Pauli words
  using the `pauli_mult` and `pauli_multi_with_phase` functions.
  This can be done on arbitrarily-labeled wires as well, by defining a wire map.

  ```pycon
  >>> from pennylane.grouping import pauli_group, pauli_mult
  >>> wire_map = {'a' : 0, 'b' : 1, 'c' : 2}
  >>> pg = list(pauli_group(3, wire_map=wire_map))
  >>> pg[3]
  PauliZ(wires=['b']) @ PauliZ(wires=['c'])
  >>> pg[55]
  PauliY(wires=['a']) @ PauliY(wires=['b']) @ PauliZ(wires=['c'])
  >>> pauli_mult(pg[3], pg[55], wire_map=wire_map)
  PauliY(wires=['a']) @ PauliX(wires=['b'])
  ```

  Functions for conversion of Pauli observables to strings (and back),
  are included.

  ```pycon
  >>> from pennylane.grouping import pauli_word_to_string, string_to_pauli_word
  >>> pauli_word_to_string(pg[55], wire_map=wire_map)
  'YYZ'
  >>> string_to_pauli_word('ZXY', wire_map=wire_map)
  PauliZ(wires=['a']) @ PauliX(wires=['b']) @ PauliY(wires=['c'])
  ```

  Calculation of the matrix representation for arbitrary Paulis and wire maps is now
  also supported.

  ```pycon
  >>> from pennylane.grouping import pauli_word_to_matrix
  >>> wire_map = {'a' : 0, 'b' : 1}
  >>> pauli_word = qml.PauliZ('b')  # corresponds to Pauli 'IZ'
  >>> pauli_word_to_matrix(pauli_word, wire_map=wire_map)
  array([[ 1.,  0.,  0.,  0.],
         [ 0., -1.,  0., -0.],
         [ 0.,  0.,  1.,  0.],
         [ 0., -0.,  0., -1.]])
  ```

<h4>New transforms</h4>

* The `qml.specs` QNode transform creates a function that returns specifications or
  details about the QNode, including depth, number of gates, and number of
  gradient executions required.
  [(#1245)](https://github.com/PennyLaneAI/pennylane/pull/1245)

  For example:

  ```python
  dev = qml.device('default.qubit', wires=4)

  @qml.qnode(dev, diff_method='parameter-shift')
  def circuit(x, y):
      qml.RX(x[0], wires=0)
      qml.Toffoli(wires=(0, 1, 2))
      qml.CRY(x[1], wires=(0, 1))
      qml.Rot(x[2], x[3], y, wires=0)
      return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))
  ```

  We can now use the `qml.specs` transform to generate a function that returns
  details and resource information:

  ```pycon
  >>> x = np.array([0.05, 0.1, 0.2, 0.3], requires_grad=True)
  >>> y = np.array(0.4, requires_grad=False)
  >>> specs_func = qml.specs(circuit)
  >>> specs_func(x, y)
  {'gate_sizes': defaultdict(int, {1: 2, 3: 1, 2: 1}),
   'gate_types': defaultdict(int, {'RX': 1, 'Toffoli': 1, 'CRY': 1, 'Rot': 1}),
   'num_operations': 4,
   'num_observables': 2,
   'num_diagonalizing_gates': 1,
   'num_used_wires': 3,
   'depth': 4,
   'num_trainable_params': 4,
   'num_parameter_shift_executions': 11,
   'num_device_wires': 4,
   'device_name': 'default.qubit',
   'diff_method': 'parameter-shift'}
  ```

  The tape methods `get_resources` and `get_depth` are superseded by `specs` and will be
  deprecated after one release cycle.

* Adds a decorator `@qml.qfunc_transform` to easily create a transformation
  that modifies the behaviour of a quantum function.
  [(#1315)](https://github.com/PennyLaneAI/pennylane/pull/1315)

  For example, consider the following transform, which scales the parameter of
  all `RX` gates by :math:`x \rightarrow \sin(a) \sqrt{x}`, and the parameters
  of all `RY` gates by :math:`y \rightarrow \cos(a * b) y`:

  ```python
  @qml.qfunc_transform
  def my_transform(tape, a, b):
      for op in tape.operations + tape.measurements:
          if op.name == "RX":
              x = op.parameters[0]
              qml.RX(qml.math.sin(a) * qml.math.sqrt(x), wires=op.wires)
          elif op.name == "RY":
              y = op.parameters[0]
              qml.RX(qml.math.cos(a * b) * y, wires=op.wires)
          else:
              op.queue()
  ```

  We can now apply this transform to any quantum function:

  ```python
  dev = qml.device("default.qubit", wires=2)

  def ansatz(x):
      qml.Hadamard(wires=0)
      qml.RX(x[0], wires=0)
      qml.RY(x[1], wires=1)
      qml.CNOT(wires=[0, 1])

  @qml.qnode(dev)
  def circuit(params, transform_weights):
      qml.RX(0.1, wires=0)

      # apply the transform to the ansatz
      my_transform(*transform_weights)(ansatz)(params)

      return qml.expval(qml.PauliZ(1))
  ```

  We can print this QNode to show that the qfunc transform is taking place:

  ```pycon
  >>> x = np.array([0.5, 0.3], requires_grad=True)
  >>> transform_weights = np.array([0.1, 0.6], requires_grad=True)
  >>> print(qml.draw(circuit)(x, transform_weights))
   0: ──RX(0.1)────H──RX(0.0706)──╭C──┤
   1: ──RX(0.299)─────────────────╰X──┤ ⟨Z⟩
  ```

  Evaluating the QNode, as well as the derivative, with respect to the gate
  parameter *and* the transform weights:

  ```pycon
  >>> circuit(x, transform_weights)
  tensor(0.00672829, requires_grad=True)
  >>> qml.grad(circuit)(x, transform_weights)
  (array([ 0.00671711, -0.00207359]), array([6.69695008e-02, 3.73694364e-06]))
  ```

* Adds a `hamiltonian_expand` tape transform. This takes a tape ending in
  `qml.expval(H)`, where `H` is a Hamiltonian, and maps it to a collection
  of tapes which can be executed and passed into a post-processing function yielding
  the expectation value.
  [(#1142)](https://github.com/PennyLaneAI/pennylane/pull/1142)

  Example use:

  ```python
  H = qml.PauliZ(0) + 3 * qml.PauliZ(0) @ qml.PauliX(1)

  with qml.tape.QuantumTape() as tape:
      qml.Hadamard(wires=1)
      qml.expval(H)

  tapes, fn = qml.transforms.hamiltonian_expand(tape)
  ```

  We can now evaluate the transformed tapes, and apply the post-processing
  function:

  ```pycon
  >>> dev = qml.device("default.qubit", wires=3)
  >>> res = dev.batch_execute(tapes)
  >>> fn(res)
  3.999999999999999
  ```

* The `quantum_monte_carlo` transform has been added, allowing an input circuit to be transformed
  into the full quantum Monte Carlo algorithm.
  [(#1316)](https://github.com/PennyLaneAI/pennylane/pull/1316)

  Suppose we want to measure the expectation value of the sine squared function according to
  a standard normal distribution. We can calculate the expectation value analytically as
  `0.432332`, but we can also estimate using the quantum Monte Carlo algorithm. The first step is to
  discretize the problem:

  ```python
  from scipy.stats import norm

  m = 5
  M = 2 ** m

  xmax = np.pi  # bound to region [-pi, pi]
  xs = np.linspace(-xmax, xmax, M)

  probs = np.array([norm().pdf(x) for x in xs])
  probs /= np.sum(probs)

  func = lambda i: np.sin(xs[i]) ** 2
  r_rotations = np.array([2 * np.arcsin(np.sqrt(func(i))) for i in range(M)])
  ```

  The `quantum_monte_carlo` transform can then be used:

  ```python
  from pennylane.templates.state_preparations.mottonen import (
      _uniform_rotation_dagger as r_unitary,
  )

  n = 6
  N = 2 ** n

  a_wires = range(m)
  wires = range(m + 1)
  target_wire = m
  estimation_wires = range(m + 1, n + m + 1)

  dev = qml.device("default.qubit", wires=(n + m + 1))

  def fn():
      qml.templates.MottonenStatePreparation(np.sqrt(probs), wires=a_wires)
      r_unitary(qml.RY, r_rotations, control_wires=a_wires[::-1], target_wire=target_wire)

  @qml.qnode(dev)
  def qmc():
      qml.quantum_monte_carlo(fn, wires, target_wire, estimation_wires)()
      return qml.probs(estimation_wires)

  phase_estimated = np.argmax(qmc()[:int(N / 2)]) / N
  ```

  The estimated value can be retrieved using:

  ```pycon
  >>> (1 - np.cos(np.pi * phase_estimated)) / 2
  0.42663476277231915
  ```

  The resources required to perform the quantum Monte Carlo algorithm can also be inspected using
  the `specs` transform.

<h4>Extended QAOA module</h4>

* Functionality to support solving the maximum-weighted cycle problem has been added to the `qaoa`
  module.
  [(#1207)](https://github.com/PennyLaneAI/pennylane/pull/1207)
  [(#1209)](https://github.com/PennyLaneAI/pennylane/pull/1209)
  [(#1251)](https://github.com/PennyLaneAI/pennylane/pull/1251)
  [(#1213)](https://github.com/PennyLaneAI/pennylane/pull/1213)
  [(#1220)](https://github.com/PennyLaneAI/pennylane/pull/1220)
  [(#1214)](https://github.com/PennyLaneAI/pennylane/pull/1214)
  [(#1283)](https://github.com/PennyLaneAI/pennylane/pull/1283)
  [(#1297)](https://github.com/PennyLaneAI/pennylane/pull/1297)
  [(#1396)](https://github.com/PennyLaneAI/pennylane/pull/1396)
  [(#1403)](https://github.com/PennyLaneAI/pennylane/pull/1403)

  The `max_weight_cycle` function returns the appropriate cost and mixer Hamiltonians:

  ```pycon
  >>> a = np.random.random((3, 3))
  >>> np.fill_diagonal(a, 0)
  >>> g = nx.DiGraph(a)  # create a random directed graph
  >>> cost, mixer, mapping = qml.qaoa.max_weight_cycle(g)
  >>> print(cost)
    (-0.9775906842165344) [Z2]
  + (-0.9027248603361988) [Z3]
  + (-0.8722207409852838) [Z0]
  + (-0.6426184210832898) [Z5]
  + (-0.2832594164291379) [Z1]
  + (-0.0778133996933755) [Z4]
  >>> print(mapping)
  {0: (0, 1), 1: (0, 2), 2: (1, 0), 3: (1, 2), 4: (2, 0), 5: (2, 1)}
  ```
  Additional functionality can be found in the
  [qml.qaoa.cycle](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.qaoa.cycle.html)
  module.


<h4>Extended operations and templates</h4>

* Added functionality to compute the sparse matrix representation of a `qml.Hamiltonian` object.
  [(#1394)](https://github.com/PennyLaneAI/pennylane/pull/1394)

  ```python
  coeffs = [1, -0.45]
  obs = [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliY(0) @ qml.PauliZ(1)]
  H = qml.Hamiltonian(coeffs, obs)
  H_sparse = qml.utils.sparse_hamiltonian(H)
  ```

  The resulting matrix is a sparse matrix in scipy coordinate list (COO) format:

  ```python
  >>> H_sparse
  <4x4 sparse matrix of type '<class 'numpy.complex128'>'
      with 8 stored elements in COOrdinate format>
  ```

  The sparse matrix can be converted to an array as:

  ```python
  >>> H_sparse.toarray()
  array([[ 1.+0.j  ,  0.+0.j  ,  0.+0.45j,  0.+0.j  ],
         [ 0.+0.j  , -1.+0.j  ,  0.+0.j  ,  0.-0.45j],
         [ 0.-0.45j,  0.+0.j  , -1.+0.j  ,  0.+0.j  ],
         [ 0.+0.j  ,  0.+0.45j,  0.+0.j  ,  1.+0.j  ]])
  ```

* Adds the new template `AllSinglesDoubles` to prepare quantum states of molecules
  using the `SingleExcitation` and `DoubleExcitation` operations.
  The new template reduces significantly the number of operations
  and the depth of the quantum circuit with respect to the traditional UCCSD
  unitary.
  [(#1383)](https://github.com/PennyLaneAI/pennylane/pull/1383)

  For example, consider the case of two particles and four qubits.
  First, we define the Hartree-Fock initial state and generate all
  possible single and double excitations.

  ```python
  import pennylane as qml
  from pennylane import numpy as np

  electrons = 2
  qubits = 4

  hf_state = qml.qchem.hf_state(electrons, qubits)
  singles, doubles = qml.qchem.excitations(electrons, qubits)
  ```

  Now we can use the template ``AllSinglesDoubles`` to define the
  quantum circuit,

  ```python
  from pennylane.templates import AllSinglesDoubles

  wires = range(qubits)

  dev = qml.device('default.qubit', wires=wires)

  @qml.qnode(dev)
  def circuit(weights, hf_state, singles, doubles):
      AllSinglesDoubles(weights, wires, hf_state, singles, doubles)
      return qml.expval(qml.PauliZ(0))

  params = np.random.normal(0, np.pi, len(singles) + len(doubles))
  ```
  and execute it:
  ```pycon
  >>> circuit(params, hf_state, singles=singles, doubles=doubles)
  tensor(-0.73772194, requires_grad=True)
  ```

* Adds `QubitCarry` and `QubitSum` operations for basic arithmetic.
  [(#1169)](https://github.com/PennyLaneAI/pennylane/pull/1169)

  The following example adds two 1-bit numbers, returning a 2-bit answer:

  ```python
  dev = qml.device('default.qubit', wires = 4)
  a = 0
  b = 1

  @qml.qnode(dev)
  def circuit():
      qml.BasisState(np.array([a, b]), wires=[1, 2])
      qml.QubitCarry(wires=[0, 1, 2, 3])
      qml.CNOT(wires=[1, 2])
      qml.QubitSum(wires=[0, 1, 2])
      return qml.probs(wires=[3, 2])

  probs = circuit()
  bitstrings = tuple(itertools.product([0, 1], repeat = 2))
  indx = np.argwhere(probs == 1).flatten()[0]
  output = bitstrings[indx]
  ```

  ```pycon
  >>> print(output)
  (0, 1)
  ```

* Added the `qml.Projector` observable, which is available on all devices
  inheriting from the `QubitDevice` class.
  [(#1356)](https://github.com/PennyLaneAI/pennylane/pull/1356)
  [(#1368)](https://github.com/PennyLaneAI/pennylane/pull/1368)

  Using `qml.Projector`, we can define the basis state projectors to use when
  computing expectation values. Let us take for example a circuit that prepares
  Bell states:

  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def circuit(basis_state):
      qml.Hadamard(wires=[0])
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.Projector(basis_state, wires=[0, 1]))
  ```

  We can then specify the `|00>` basis state to construct the `|00><00|`
  projector and compute the expectation value:

  ```pycon
  >>> basis_state = [0, 0]
  >>> circuit(basis_state)
  tensor(0.5, requires_grad=True)
  ```

  As expected, we get similar results when specifying the `|11>` basis state:

  ```pycon
  >>> basis_state = [1, 1]
  >>> circuit(basis_state)
  tensor(0.5, requires_grad=True)
  ```

* The following new operations have been added:

  - The IsingXX gate `qml.IsingXX` [(#1194)](https://github.com/PennyLaneAI/pennylane/pull/1194)
  - The IsingZZ gate `qml.IsingZZ` [(#1199)](https://github.com/PennyLaneAI/pennylane/pull/1199)
  - The ISWAP gate `qml.ISWAP` [(#1298)](https://github.com/PennyLaneAI/pennylane/pull/1298)
  - The reset error noise channel `qml.ResetError` [(#1321)](https://github.com/PennyLaneAI/pennylane/pull/1321)


<h3>Improvements</h3>

* The ``argnum`` keyword argument can now be specified for a QNode to define a
  subset of trainable parameters used to estimate the Jacobian.
  [(#1371)](https://github.com/PennyLaneAI/pennylane/pull/1371)

  For example, consider two trainable parameters and a quantum function:

  ```python
  dev = qml.device("default.qubit", wires=2)

  x = np.array(0.543, requires_grad=True)
  y = np.array(-0.654, requires_grad=True)

  def circuit(x,y):
      qml.RX(x, wires=[0])
      qml.RY(y, wires=[1])
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
  ```

  When computing the gradient of the QNode, we can specify the trainable
  parameters to consider by passing the ``argnum`` keyword argument:

  ```pycon
  >>> qnode1 = qml.QNode(circuit, dev, diff_method="parameter-shift", argnum=[0,1])
  >>> print(qml.grad(qnode1)(x,y))
  (array(0.31434679), array(0.67949903))
  ```

  Specifying a proper subset of the trainable parameters will estimate the
  Jacobian:

  ```pycon
  >>> qnode2 = qml.QNode(circuit, dev, diff_method="parameter-shift", argnum=[0])
  >>> print(qml.grad(qnode2)(x,y))
  (array(0.31434679), array(0.))
  ```

* Allows creating differentiable observables that return custom objects such
  that the observable is supported by devices.
  [(1291)](https://github.com/PennyLaneAI/pennylane/pull/1291)

  As an example, first we define `NewObservable` class:

  ```python
  from pennylane.devices import DefaultQubit

  class NewObservable(qml.operation.Observable):
      """NewObservable"""

      num_wires = qml.operation.AnyWires
      num_params = 0
      par_domain = None

      def diagonalizing_gates(self):
          """Diagonalizing gates"""
          return []
  ```

  Once we have this new observable class, we define a `SpecialObject` class
  that can be used to encode data in an observable and a new device that supports
  our new observable and returns a `SpecialObject` as the expectation value
  (the code is shortened for brevity, the extended example can be found as a
  test in the previously referenced pull request):

  ```python
  class SpecialObject:

      def __init__(self, val):
          self.val = val

      def __mul__(self, other):
          new = SpecialObject(self.val)
          new *= other
          return new

      ...

  class DeviceSupportingNewObservable(DefaultQubit):
      name = "Device supporting NewObservable"
      short_name = "default.qubit.newobservable"
      observables = DefaultQubit.observables.union({"NewObservable"})

      def expval(self, observable, **kwargs):
          if self.shots is None and isinstance(observable, NewObservable):
              val = super().expval(qml.PauliZ(wires=0), **kwargs)
              return SpecialObject(val)

          return super().expval(observable, **kwargs)
  ```

  At this point, we can create a device that will support the differentiation
  of a `NewObservable` object:

  ```python
  dev = DeviceSupportingNewObservable(wires=1, shots=None)

  @qml.qnode(dev, diff_method="parameter-shift")
  def qnode(x):
      qml.RY(x, wires=0)
      return qml.expval(NewObservable(wires=0))
  ```

  We can then compute the jacobian of this object:

  ```pycon
  >>> result = qml.jacobian(qnode)(0.2)
  >>> print(result)
  <__main__.SpecialObject object at 0x7fd2c54721f0>
  >>> print(result.item().val)
  -0.19866933079506116
  ```

* PennyLane NumPy now includes the
  [random module's](https://numpy.org/doc/stable/reference/random/index.html#module-numpy.random)
  `Generator` objects, the recommended way of random number generation. This allows for
  random number generation using a local, rather than global seed.
  [(#1267)](https://github.com/PennyLaneAI/pennylane/pull/1267)

  ```python
  from pennylane import numpy as np

  rng = np.random.default_rng()
  random_mat1 = rng.random((3,2))
  random_mat2 = rng.standard_normal(3, requires_grad=False)
  ```

* The performance of adjoint jacobian differentiation was significantly
  improved as the method now reuses the state computed on the forward pass.
  This can be turned off to save memory with the Torch and TensorFlow
  interfaces by passing `adjoint_cache=False` during QNode creation.
  [(#1341)](https://github.com/PennyLaneAI/pennylane/pull/1341)

* The `Operator` (and by inheritance, the `Operation` and `Observable` class and their children)
  now have an `id` attribute, which can mark an operator in a circuit, for example to
  identify it on the tape by a tape transform.
  [(#1377)](https://github.com/PennyLaneAI/pennylane/pull/1377)

* The `benchmark` module was deleted, since it was outdated and is superseded by
  the new separate [benchmark repository](https://github.com/PennyLaneAI/benchmark).
  [(#1343)](https://github.com/PennyLaneAI/pennylane/pull/1343)

* Decompositions in terms of elementary gates has been added for:

  - `qml.CSWAP` [(#1306)](https://github.com/PennyLaneAI/pennylane/issues/1306)
  - `qml.SWAP` [(#1329)](https://github.com/PennyLaneAI/pennylane/pull/1329)
  - `qml.SingleExcitation` [(#1303)](https://github.com/PennyLaneAI/pennylane/pull/1303)
  - `qml.SingleExcitationPlus` and `qml.SingleExcitationMinus` [(#1278)](https://github.com/PennyLaneAI/pennylane/pull/1278)
  - `qml.DoubleExcitation` [(#1303)](https://github.com/PennyLaneAI/pennylane/pull/1303)
  - `qml.Toffoli` [(#1320)](https://github.com/PennyLaneAI/pennylane/pull/1320)
  - `qml.MultiControlledX`. [(#1287)](https://github.com/PennyLaneAI/pennylane/pull/1287)
    When controlling on three or more wires, an ancilla
    register of worker wires is required to support the decomposition.

    ```python
    ctrl_wires = [f"c{i}" for i in range(5)]
    work_wires = [f"w{i}" for i in range(3)]
    target_wires = ["t0"]
    all_wires = ctrl_wires + work_wires + target_wires

    dev = qml.device("default.qubit", wires=all_wires)

    with qml.tape.QuantumTape() as tape:
        qml.MultiControlledX(control_wires=ctrl_wires, wires=target_wires, work_wires=work_wires)
    ```

    ```pycon
    >>> tape = tape.expand(depth=1)
    >>> print(tape.draw(wire_order=qml.wires.Wires(all_wires)))

     c0: ──────────────╭C──────────────────────╭C──────────┤
     c1: ──────────────├C──────────────────────├C──────────┤
     c2: ──────────╭C──│───╭C──────────────╭C──│───╭C──────┤
     c3: ──────╭C──│───│───│───╭C──────╭C──│───│───│───╭C──┤
     c4: ──╭C──│───│───│───│───│───╭C──│───│───│───│───│───┤
     w0: ──│───│───├C──╰X──├C──│───│───│───├C──╰X──├C──│───┤
     w1: ──│───├C──╰X──────╰X──├C──│───├C──╰X──────╰X──├C──┤
     w2: ──├C──╰X──────────────╰X──├C──╰X──────────────╰X──┤
     t0: ──╰X──────────────────────╰X──────────────────────┤
    ```

* Added `qml.CPhase` as an alias for the existing `qml.ControlledPhaseShift` operation.
  [(#1319)](https://github.com/PennyLaneAI/pennylane/pull/1319).

* The `Device` class now uses caching when mapping wires.
  [(#1270)](https://github.com/PennyLaneAI/pennylane/pull/1270)

* The `Wires` class now uses caching for computing its `hash`.
  [(#1270)](https://github.com/PennyLaneAI/pennylane/pull/1270)

* Added custom gate application for Toffoli in `default.qubit`.
  [(#1249)](https://github.com/PennyLaneAI/pennylane/pull/1249)

* Added validation for noise channel parameters. Invalid noise parameters now
  raise a `ValueError`.
  [(#1357)](https://github.com/PennyLaneAI/pennylane/pull/1357)

* The device test suite now provides test cases for checking gates by comparing
  expectation values.
  [(#1212)](https://github.com/PennyLaneAI/pennylane/pull/1212)

* PennyLane's test suite is now code-formatted using `black -l 100`.
  [(#1222)](https://github.com/PennyLaneAI/pennylane/pull/1222)

* PennyLane's `qchem` package and tests are now code-formatted using `black -l 100`.
  [(#1311)](https://github.com/PennyLaneAI/pennylane/pull/1311)

<h3>Breaking changes</h3>

* The `qml.inv()` function is now deprecated with a warning to use the more general `qml.adjoint()`.
  [(#1325)](https://github.com/PennyLaneAI/pennylane/pull/1325)

* Removes support for Python 3.6 and adds support for Python 3.9.
  [(#1228)](https://github.com/XanaduAI/pennylane/pull/1228)

* The tape methods `get_resources` and `get_depth` are superseded by `specs` and will be
  deprecated after one release cycle.
  [(#1245)](https://github.com/PennyLaneAI/pennylane/pull/1245)

* Using the `qml.sample()` measurement on devices with `shots=None` continue to
  raise a warning with this functionality being fully deprecated and raising an
  error after one release cycle.
  [(#1079)](https://github.com/PennyLaneAI/pennylane/pull/1079)
  [(#1196)](https://github.com/PennyLaneAI/pennylane/pull/1196)

<h3>Bug fixes</h3>

* QNodes now display readable information when in interactive environments or when printed.
  [(#1359)](https://github.com/PennyLaneAI/pennylane/pull/1359).

* Fixes a bug with `qml.math.cast` where the `MottonenStatePreparation` operation expected
  a float type instead of double.
  [(#1400)](https://github.com/XanaduAI/pennylane/pull/1400)

* Fixes a bug where a copy of `qml.ControlledQubitUnitary` was non-functional as it did not have all the necessary information.
  [(#1411)](https://github.com/PennyLaneAI/pennylane/pull/1411)

* Warns when adjoint or reversible differentiation specified or called on a device with finite shots.
  [(#1406)](https://github.com/PennyLaneAI/pennylane/pull/1406)

* Fixes the differentiability of the operations `IsingXX` and `IsingZZ` for Autograd, Jax and Tensorflow.
  [(#1390)](https://github.com/PennyLaneAI/pennylane/pull/1390)

* Fixes a bug where multiple identical Hamiltonian terms will produce a
  different result with ``optimize=True`` using ``ExpvalCost``.
  [(#1405)](https://github.com/XanaduAI/pennylane/pull/1405)

* Fixes bug where `shots=None` was not reset when changing shots temporarily in a QNode call
  like `circuit(0.1, shots=3)`.
  [(#1392)](https://github.com/XanaduAI/pennylane/pull/1392)

* Fixes floating point errors with `diff_method="finite-diff"` and `order=1` when parameters are `float32`.
  [(#1381)](https://github.com/PennyLaneAI/pennylane/pull/1381)

* Fixes a bug where `qml.ctrl` would fail to transform gates that had no
  control defined and no decomposition defined.
  [(#1376)](https://github.com/PennyLaneAI/pennylane/pull/1376)

* Copying the `JacobianTape` now correctly also copies the `jacobian_options` attribute. This fixes a bug
  allowing the JAX interface to support adjoint differentiation.
  [(#1349)](https://github.com/PennyLaneAI/pennylane/pull/1349)

* Fixes drawing QNodes that contain multiple measurements on a single wire.
  [(#1353)](https://github.com/PennyLaneAI/pennylane/pull/1353)

* Fixes drawing QNodes with no operations.
  [(#1354)](https://github.com/PennyLaneAI/pennylane/pull/1354)

* Fixes incorrect wires in the decomposition of the `ControlledPhaseShift` operation.
  [(#1338)](https://github.com/PennyLaneAI/pennylane/pull/1338)

* Fixed tests for the `Permute` operation that used a QNode and hence expanded
  tapes twice instead of once due to QNode tape expansion and an explicit tape
  expansion call.
  [(#1318)](https://github.com/PennyLaneAI/pennylane/pull/1318).

* Prevent Hamiltonians that share wires from being multiplied together.
  [(#1273)](https://github.com/PennyLaneAI/pennylane/pull/1273)

* Fixed a bug where the custom range sequences could not be passed
  to the `StronglyEntanglingLayers` template.
  [(#1332)](https://github.com/PennyLaneAI/pennylane/pull/1332)

* Fixed a bug where `qml.sum()` and `qml.dot()` do not support the JAX interface.
  [(#1380)](https://github.com/PennyLaneAI/pennylane/pull/1380)

<h3>Documentation</h3>

* Math present in the `QubitParamShiftTape` class docstring now renders correctly.
  [(#1402)](https://github.com/PennyLaneAI/pennylane/pull/1402)

* Fix typo in the documentation of `qml.StronglyEntanglingLayers`.
  [(#1367)](https://github.com/PennyLaneAI/pennylane/pull/1367)

* Fixed typo in TensorFlow interface documentation
  [(#1312)](https://github.com/PennyLaneAI/pennylane/pull/1312)

* Fixed typos in the mathematical expressions in documentation of `qml.DoubleExcitation`.
  [(#1278)](https://github.com/PennyLaneAI/pennylane/pull/1278)

* Remove unsupported `None` option from the `qml.QNode` docstrings.
  [(#1271)](https://github.com/PennyLaneAI/pennylane/pull/1271)

* Updated the docstring of `qml.PolyXP` to reference the new location of internal
  usage.
  [(#1262)](https://github.com/PennyLaneAI/pennylane/pull/1262)

* Removes occurrences of the deprecated device argument ``analytic`` from the documentation.
  [(#1261)](https://github.com/PennyLaneAI/pennylane/pull/1261)

* Updated PyTorch and TensorFlow interface introductions.
  [(#1333)](https://github.com/PennyLaneAI/pennylane/pull/1333)

* Updates the quantum chemistry quickstart to reflect recent changes to the `qchem` module.
  [(#1227)](https://github.com/PennyLaneAI/pennylane/pull/1227)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Marius Aglitoiu, Vishnu Ajith, Juan Miguel Arrazola, Thomas Bromley, Jack Ceroni, Alaric Cheng, Miruna Daian,
Olivia Di Matteo, Tanya Garg, Christian Gogolin, Alain Delgado Gran, Diego Guala, Anthony Hayes, Ryan Hill,
Theodor Isacsson, Josh Izaac, Soran Jahangiri, Pavan Jayasinha, Nathan Killoran, Christina Lee, Ryan Levy,
Alberto Maldonado, Johannes Jakob Meyer, Romain Moyard, Ashish Panigrahi, Nahum Sá, Maria Schuld, Brian Shi,
Antal Száva, David Wierichs, Vincent Wong.


# Release 0.15.1

<h3>Bug fixes</h3>

* Fixes two bugs in the parameter-shift Hessian.
  [(#1260)](https://github.com/PennyLaneAI/pennylane/pull/1260)

  - Fixes a bug where having an unused parameter in the Autograd interface
    would result in an indexing error during backpropagation.

  - The parameter-shift Hessian only supports the two-term parameter-shift
    rule currently, so raises an error if asked to differentiate
    any unsupported gates (such as the controlled rotation gates).

* A bug which resulted in `qml.adjoint()` and `qml.inv()` failing to work with
  templates has been fixed.
  [(#1243)](https://github.com/PennyLaneAI/pennylane/pull/1243)

* Deprecation warning instances in PennyLane have been changed to `UserWarning`,
  to account for recent changes to how Python warnings are filtered in
  [PEP565](https://www.python.org/dev/peps/pep-0565/).
  [(#1211)](https://github.com/PennyLaneAI/pennylane/pull/1211)

<h3>Documentation</h3>

* Updated the order of the parameters to the `GaussianState` operation to match
  the way that the PennyLane-SF plugin uses them.
  [(#1255)](https://github.com/PennyLaneAI/pennylane/pull/1255)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Thomas Bromley, Olivia Di Matteo, Diego Guala, Anthony Hayes, Ryan Hill,
Josh Izaac, Christina Lee, Maria Schuld, Antal Száva.

# Release 0.15.0

<h3>New features since last release</h3>

<h4>Better and more flexible shot control</h4>

* Adds a new optimizer `qml.ShotAdaptiveOptimizer`, a gradient-descent optimizer where
  the shot rate is adaptively calculated using the variances of the parameter-shift gradient.
  [(#1139)](https://github.com/PennyLaneAI/pennylane/pull/1139)

  By keeping a running average of the parameter-shift gradient and the *variance* of the
  parameter-shift gradient, this optimizer frugally distributes a shot budget across the partial
  derivatives of each parameter.

  In addition, if computing the expectation value of a Hamiltonian, weighted random sampling can be
  used to further distribute the shot budget across the local terms from which the Hamiltonian is
  constructed.

  This optimizer is based on both the [iCANS1](https://quantum-journal.org/papers/q-2020-05-11-263)
  and [Rosalin](https://arxiv.org/abs/2004.06252) shot-adaptive optimizers.

  Once constructed, the cost function can be passed directly to the optimizer's `step` method.  The
  attribute `opt.total_shots_used` can be used to track the number of shots per iteration.

  ```pycon
  >>> coeffs = [2, 4, -1, 5, 2]
  >>> obs = [
  ...   qml.PauliX(1),
  ...   qml.PauliZ(1),
  ...   qml.PauliX(0) @ qml.PauliX(1),
  ...   qml.PauliY(0) @ qml.PauliY(1),
  ...   qml.PauliZ(0) @ qml.PauliZ(1)
  ... ]
  >>> H = qml.Hamiltonian(coeffs, obs)
  >>> dev = qml.device("default.qubit", wires=2, shots=100)
  >>> cost = qml.ExpvalCost(qml.templates.StronglyEntanglingLayers, H, dev)
  >>> params = qml.init.strong_ent_layers_uniform(n_layers=2, n_wires=2)
  >>> opt = qml.ShotAdaptiveOptimizer(min_shots=10)
  >>> for i in range(5):
  ...    params = opt.step(cost, params)
  ...    print(f"Step {i}: cost = {cost(params):.2f}, shots_used = {opt.total_shots_used}")
  Step 0: cost = -5.68, shots_used = 240
  Step 1: cost = -2.98, shots_used = 336
  Step 2: cost = -4.97, shots_used = 624
  Step 3: cost = -5.53, shots_used = 1054
  Step 4: cost = -6.50, shots_used = 1798
  ```

* Batches of shots can now be specified as a list, allowing measurement statistics
  to be course-grained with a single QNode evaluation.
  [(#1103)](https://github.com/PennyLaneAI/pennylane/pull/1103)

  ```pycon
  >>> shots_list = [5, 10, 1000]
  >>> dev = qml.device("default.qubit", wires=2, shots=shots_list)
  ```

  When QNodes are executed on this device, a single execution of 1015 shots will be submitted.
  However, three sets of measurement statistics will be returned; using the first 5 shots,
  second set of 10 shots, and final 1000 shots, separately.

  For example, executing a circuit with two outputs will lead to a result of shape `(3, 2)`:

  ```pycon
  >>> @qml.qnode(dev)
  ... def circuit(x):
  ...     qml.RX(x, wires=0)
  ...     qml.CNOT(wires=[0, 1])
  ...     return qml.expval(qml.PauliZ(0) @ qml.PauliX(1)), qml.expval(qml.PauliZ(0))
  >>> circuit(0.5)
  [[0.33333333 1.        ]
   [0.2        1.        ]
   [0.012      0.868     ]]
  ```

  This output remains fully differentiable.

- The number of shots can now be specified on a per-call basis when evaluating a QNode.
  [(#1075)](https://github.com/PennyLaneAI/pennylane/pull/1075).

  For this, the qnode should be called with an additional `shots` keyword argument:

  ```pycon
  >>> dev = qml.device('default.qubit', wires=1, shots=10) # default is 10
  >>> @qml.qnode(dev)
  ... def circuit(a):
  ...     qml.RX(a, wires=0)
  ...     return qml.sample(qml.PauliZ(wires=0))
  >>> circuit(0.8)
  [ 1  1  1 -1 -1  1  1  1  1  1]
  >>> circuit(0.8, shots=3)
  [ 1  1  1]
  >>> circuit(0.8)
  [ 1  1  1 -1 -1  1  1  1  1  1]
  ```

<h4>New differentiable quantum transforms</h4>

A new module is available,
[qml.transforms](https://pennylane.rtfd.io/en/stable/code/qml_transforms.html),
which contains *differentiable quantum transforms*. These are functions that act
on QNodes, quantum functions, devices, and tapes, transforming them while remaining
fully differentiable.

* A new adjoint transform has been added.
  [(#1111)](https://github.com/PennyLaneAI/pennylane/pull/1111)
  [(#1135)](https://github.com/PennyLaneAI/pennylane/pull/1135)

  This new method allows users to apply the adjoint of an arbitrary sequence of operations.

  ```python
  def subroutine(wire):
      qml.RX(0.123, wires=wire)
      qml.RY(0.456, wires=wire)

  dev = qml.device('default.qubit', wires=1)
  @qml.qnode(dev)
  def circuit():
      subroutine(0)
      qml.adjoint(subroutine)(0)
      return qml.expval(qml.PauliZ(0))
  ```

  This creates the following circuit:

  ```pycon
  >>> print(qml.draw(circuit)())
  0: --RX(0.123)--RY(0.456)--RY(-0.456)--RX(-0.123)--| <Z>
  ```

  Directly applying to a gate also works as expected.

  ```python
  qml.adjoint(qml.RX)(0.123, wires=0) # applies RX(-0.123)
  ```

* A new transform `qml.ctrl` is now available that adds control wires to subroutines.
  [(#1157)](https://github.com/PennyLaneAI/pennylane/pull/1157)

  ```python
  def my_ansatz(params):
     qml.RX(params[0], wires=0)
     qml.RZ(params[1], wires=1)

  # Create a new operation that applies `my_ansatz`
  # controlled by the "2" wire.
  my_ansatz2 = qml.ctrl(my_ansatz, control=2)

  @qml.qnode(dev)
  def circuit(params):
      my_ansatz2(params)
      return qml.state()
  ```

  This is equivalent to:

  ```python
  @qml.qnode(...)
  def circuit(params):
      qml.CRX(params[0], wires=[2, 0])
      qml.CRZ(params[1], wires=[2, 1])
      return qml.state()
  ```

* The `qml.transforms.classical_jacobian` transform has been added.
  [(#1186)](https://github.com/PennyLaneAI/pennylane/pull/1186)

  This transform returns a function to extract the Jacobian matrix of the classical part of a
  QNode, allowing the classical dependence between the QNode arguments and the quantum gate
  arguments to be extracted.

  For example, given the following QNode:

  ```pycon
  >>> @qml.qnode(dev)
  ... def circuit(weights):
  ...     qml.RX(weights[0], wires=0)
  ...     qml.RY(weights[0], wires=1)
  ...     qml.RZ(weights[2] ** 2, wires=1)
  ...     return qml.expval(qml.PauliZ(0))
  ```

  We can use this transform to extract the relationship
  :math:`f: \mathbb{R}^n \rightarrow\mathbb{R}^m` between the input QNode
  arguments :math:`w` and the gate arguments :math:`g`, for
  a given value of the QNode arguments:

  ```pycon
  >>> cjac_fn = qml.transforms.classical_jacobian(circuit)
  >>> weights = np.array([1., 1., 1.], requires_grad=True)
  >>> cjac = cjac_fn(weights)
  >>> print(cjac)
  [[1. 0. 0.]
   [1. 0. 0.]
   [0. 0. 2.]]
  ```

  The returned Jacobian has rows corresponding to gate arguments, and columns corresponding to
  QNode arguments; that is, :math:`J_{ij} = \frac{\partial}{\partial g_i} f(w_j)`.

<h4>More operations and templates</h4>

* Added the `SingleExcitation` two-qubit operation, which is useful for quantum
  chemistry applications.
  [(#1121)](https://github.com/PennyLaneAI/pennylane/pull/1121)

  It can be used to perform an SO(2) rotation in the subspace
  spanned by the states :math:`|01\rangle` and :math:`|10\rangle`.
  For example, the following circuit performs the transformation
  :math:`|10\rangle \rightarrow \cos(\phi/2)|10\rangle - \sin(\phi/2)|01\rangle`:

  ```python
  dev = qml.device('default.qubit', wires=2)

  @qml.qnode(dev)
  def circuit(phi):
      qml.PauliX(wires=0)
      qml.SingleExcitation(phi, wires=[0, 1])
  ```

  The `SingleExcitation` operation supports analytic gradients on hardware
  using only four expectation value calculations, following results from
  [Kottmann et al.](https://arxiv.org/abs/2011.05938)

* Added the `DoubleExcitation` four-qubit operation, which is useful for quantum
  chemistry applications.
  [(#1123)](https://github.com/PennyLaneAI/pennylane/pull/1123)

  It can be used to perform an SO(2) rotation in the subspace
  spanned by the states :math:`|1100\rangle` and :math:`|0011\rangle`.
  For example, the following circuit performs the transformation
  :math:`|1100\rangle\rightarrow \cos(\phi/2)|1100\rangle - \sin(\phi/2)|0011\rangle`:

  ```python
  dev = qml.device('default.qubit', wires=2)

  @qml.qnode(dev)
  def circuit(phi):
      qml.PauliX(wires=0)
      qml.PauliX(wires=1)
      qml.DoubleExcitation(phi, wires=[0, 1, 2, 3])
  ```

  The `DoubleExcitation` operation supports analytic gradients on hardware using only
  four expectation value calculations, following results from
  [Kottmann et al.](https://arxiv.org/abs/2011.05938).

* Added the `QuantumMonteCarlo` template for performing quantum Monte Carlo estimation of an
  expectation value on simulator.
  [(#1130)](https://github.com/PennyLaneAI/pennylane/pull/1130)

  The following example shows how the expectation value of sine squared over a standard normal
  distribution can be approximated:

  ```python
  from scipy.stats import norm

  m = 5
  M = 2 ** m
  n = 10
  N = 2 ** n
  target_wires = range(m + 1)
  estimation_wires = range(m + 1, n + m + 1)

  xmax = np.pi  # bound to region [-pi, pi]
  xs = np.linspace(-xmax, xmax, M)

  probs = np.array([norm().pdf(x) for x in xs])
  probs /= np.sum(probs)

  func = lambda i: np.sin(xs[i]) ** 2

  dev = qml.device("default.qubit", wires=(n + m + 1))

  @qml.qnode(dev)
  def circuit():
      qml.templates.QuantumMonteCarlo(
          probs,
          func,
          target_wires=target_wires,
          estimation_wires=estimation_wires,
      )
      return qml.probs(estimation_wires)

  phase_estimated = np.argmax(circuit()[:int(N / 2)]) / N
  expectation_estimated = (1 - np.cos(np.pi * phase_estimated)) / 2
  ```

* Added the `QuantumPhaseEstimation` template for performing quantum phase estimation for an input
  unitary matrix.
  [(#1095)](https://github.com/PennyLaneAI/pennylane/pull/1095)

  Consider the matrix corresponding to a rotation from an `RX` gate:

  ```pycon
  >>> phase = 5
  >>> target_wires = [0]
  >>> unitary = qml.RX(phase, wires=0).matrix
  ```

  The ``phase`` parameter can be estimated using ``QuantumPhaseEstimation``. For example, using five
  phase-estimation qubits:

  ```python
  n_estimation_wires = 5
  estimation_wires = range(1, n_estimation_wires + 1)

  dev = qml.device("default.qubit", wires=n_estimation_wires + 1)

  @qml.qnode(dev)
  def circuit():
      # Start in the |+> eigenstate of the unitary
      qml.Hadamard(wires=target_wires)

      QuantumPhaseEstimation(
          unitary,
          target_wires=target_wires,
          estimation_wires=estimation_wires,
      )

      return qml.probs(estimation_wires)

  phase_estimated = np.argmax(circuit()) / 2 ** n_estimation_wires

  # Need to rescale phase due to convention of RX gate
  phase_estimated = 4 * np.pi * (1 - phase)
  ```

- Added the `ControlledPhaseShift` gate as well as the `QFT` operation for applying quantum Fourier
  transforms.
  [(#1064)](https://github.com/PennyLaneAI/pennylane/pull/1064)

  ```python
  @qml.qnode(dev)
  def circuit_qft(basis_state):
      qml.BasisState(basis_state, wires=range(3))
      qml.QFT(wires=range(3))
      return qml.state()
  ```

- Added the `ControlledQubitUnitary` operation. This
  enables implementation of multi-qubit gates with a variable number of
  control qubits. It is also possible to specify a different state for the
  control qubits using the `control_values` argument (also known as a
  mixed-polarity multi-controlled operation).
  [(#1069)](https://github.com/PennyLaneAI/pennylane/pull/1069)
  [(#1104)](https://github.com/PennyLaneAI/pennylane/pull/1104)

  For example, we can  create a multi-controlled T gate using:

  ```python
  T = qml.T._matrix()
  qml.ControlledQubitUnitary(T, control_wires=[0, 1, 3], wires=2, control_values="110")
  ```

  Here, the T gate will be applied to wire `2` if control wires `0` and `1` are in
  state `1`, and control wire `3` is in state `0`. If no value is passed to
  `control_values`, the gate will be applied if all control wires are in
  the `1` state.

- Added `MultiControlledX` for multi-controlled `NOT` gates.
  This is a special case of `ControlledQubitUnitary` that applies a
  Pauli X gate conditioned on the state of an arbitrary number of
  control qubits.
  [(#1104)](https://github.com/PennyLaneAI/pennylane/pull/1104)

<h4>Support for higher-order derivatives on hardware</h4>

* Computing second derivatives and Hessians of QNodes is now supported with
  the parameter-shift differentiation method, on all machine learning interfaces.
  [(#1130)](https://github.com/PennyLaneAI/pennylane/pull/1130)
  [(#1129)](https://github.com/PennyLaneAI/pennylane/pull/1129)
  [(#1110)](https://github.com/PennyLaneAI/pennylane/pull/1110)

  Hessians are computed using the parameter-shift rule, and can be
  evaluated on both hardware and simulator devices.

  ```python
  dev = qml.device('default.qubit', wires=1)

  @qml.qnode(dev, diff_method="parameter-shift")
  def circuit(p):
      qml.RY(p[0], wires=0)
      qml.RX(p[1], wires=0)
      return qml.expval(qml.PauliZ(0))

  x = np.array([1.0, 2.0], requires_grad=True)
  ```

  ```python
  >>> hessian_fn = qml.jacobian(qml.grad(circuit))
  >>> hessian_fn(x)
  [[0.2248451 0.7651474]
   [0.7651474 0.2248451]]
  ```

* Added the function `finite_diff()` to compute finite-difference
  approximations to the gradient and the second-order derivatives of
  arbitrary callable functions.
  [(#1090)](https://github.com/PennyLaneAI/pennylane/pull/1090)

  This is useful to compute the derivative of parametrized
  `pennylane.Hamiltonian` observables with respect to their parameters.

  For example, in quantum chemistry simulations it can be used to evaluate
  the derivatives of the electronic Hamiltonian with respect to the nuclear
  coordinates:

  ```pycon
  >>> def H(x):
  ...    return qml.qchem.molecular_hamiltonian(['H', 'H'], x)[0]
  >>> x = np.array([0., 0., -0.66140414, 0., 0., 0.66140414])
  >>> grad_fn = qml.finite_diff(H, N=1)
  >>> grad = grad_fn(x)
  >>> deriv2_fn = qml.finite_diff(H, N=2, idx=[0, 1])
  >>> deriv2_fn(x)
  ```

* The JAX interface now supports all devices, including hardware devices,
  via the parameter-shift differentiation method.
  [(#1076)](https://github.com/PennyLaneAI/pennylane/pull/1076)

  For example, using the JAX interface with Cirq:

  ```python
  dev = qml.device('cirq.simulator', wires=1)
  @qml.qnode(dev, interface="jax", diff_method="parameter-shift")
  def circuit(x):
      qml.RX(x[1], wires=0)
      qml.Rot(x[0], x[1], x[2], wires=0)
      return qml.expval(qml.PauliZ(0))
  weights = jnp.array([0.2, 0.5, 0.1])
  print(circuit(weights))
  ```

  Currently, when used with the parameter-shift differentiation method,
  only a single returned expectation value or variance is supported.
  Multiple expectations/variances, as well as probability and state returns,
  are not currently allowed.

<h3>Improvements</h3>

  ```python
  dev = qml.device("default.qubit", wires=2)

  inputstate = [np.sqrt(0.2), np.sqrt(0.3), np.sqrt(0.4), np.sqrt(0.1)]

  @qml.qnode(dev)
  def circuit():
      mottonen.MottonenStatePreparation(inputstate,wires=[0, 1])
      return qml.expval(qml.PauliZ(0))
  ```

  Previously returned:

  ```pycon
  >>> print(qml.draw(circuit)())
  0: ──RY(1.57)──╭C─────────────╭C──╭C──╭C──┤ ⟨Z⟩
  1: ──RY(1.35)──╰X──RY(0.422)──╰X──╰X──╰X──┤
  ```

  In this release, it now returns:

  ```pycon
  >>> print(qml.draw(circuit)())
  0: ──RY(1.57)──╭C─────────────╭C──┤ ⟨Z⟩
  1: ──RY(1.35)──╰X──RY(0.422)──╰X──┤
  ```

- The templates are now classes inheriting
  from `Operation`, and define the ansatz in their `expand()` method. This
  change does not affect the user interface.
  [(#1138)](https://github.com/PennyLaneAI/pennylane/pull/1138)
  [(#1156)](https://github.com/PennyLaneAI/pennylane/pull/1156)
  [(#1163)](https://github.com/PennyLaneAI/pennylane/pull/1163)
  [(#1192)](https://github.com/PennyLaneAI/pennylane/pull/1192)

  For convenience, some templates have a new method that returns the expected
  shape of the trainable parameter tensor, which can be used to create
  random tensors.

  ```python
  shape = qml.templates.BasicEntanglerLayers.shape(n_layers=2, n_wires=4)
  weights = np.random.random(shape)
  qml.templates.BasicEntanglerLayers(weights, wires=range(4))
  ```

- `QubitUnitary` now validates to ensure the input matrix is two dimensional.
  [(#1128)](https://github.com/PennyLaneAI/pennylane/pull/1128)

* Most layers in Pytorch or Keras accept arbitrary dimension inputs, where each dimension barring
  the last (in the case where the actual weight function of the layer operates on one-dimensional
  vectors) is broadcast over. This is now also supported by KerasLayer and TorchLayer.
  [(#1062)](https://github.com/PennyLaneAI/pennylane/pull/1062).

  Example use:

  ```python
  dev = qml.device("default.qubit", wires=4)
  x = tf.ones((5, 4, 4))

  @qml.qnode(dev)
  def layer(weights, inputs):
      qml.templates.AngleEmbedding(inputs, wires=range(4))
      qml.templates.StronglyEntanglingLayers(weights, wires=range(4))
      return [qml.expval(qml.PauliZ(i)) for i in range(4)]

  qlayer = qml.qnn.KerasLayer(layer, {"weights": (4, 4, 3)}, output_dim=4)
  out = qlayer(x)
  ```

  The output tensor has the following shape:
  ```pycon
  >>> out.shape
  (5, 4, 4)
  ```

* If only one argument to the function `qml.grad` has the `requires_grad` attribute
  set to True, then the returned gradient will be a NumPy array, rather than a
  tuple of length 1.
  [(#1067)](https://github.com/PennyLaneAI/pennylane/pull/1067)
  [(#1081)](https://github.com/PennyLaneAI/pennylane/pull/1081)

* An improvement has been made to how `QubitDevice` generates and post-processess samples,
  allowing QNode measurement statistics to work on devices with more than 32 qubits.
  [(#1088)](https://github.com/PennyLaneAI/pennylane/pull/1088)

* Due to the addition of `density_matrix()` as a return type from a QNode, tuples are now supported
  by the `output_dim` parameter in `qnn.KerasLayer`.
  [(#1070)](https://github.com/PennyLaneAI/pennylane/pull/1070)

* Two new utility methods are provided for working with quantum tapes.
  [(#1175)](https://github.com/PennyLaneAI/pennylane/pull/1175)

  - `qml.tape.get_active_tape()` gets the currently recording tape.

  - `tape.stop_recording()` is a context manager that temporarily
    stops the currently recording tape from recording additional
    tapes or quantum operations.

  For example:

  ```pycon
  >>> with qml.tape.QuantumTape():
  ...     qml.RX(0, wires=0)
  ...     current_tape = qml.tape.get_active_tape()
  ...     with current_tape.stop_recording():
  ...         qml.RY(1.0, wires=1)
  ...     qml.RZ(2, wires=1)
  >>> current_tape.operations
  [RX(0, wires=[0]), RZ(2, wires=[1])]
  ```

* When printing `qml.Hamiltonian` objects, the terms are sorted by number of wires followed by coefficients.
  [(#981)](https://github.com/PennyLaneAI/pennylane/pull/981)

* Adds `qml.math.conj` to the PennyLane math module.
  [(#1143)](https://github.com/PennyLaneAI/pennylane/pull/1143)

  This new method will do elementwise conjugation to the given tensor-like object,
  correctly dispatching to the required tensor-manipulation framework
  to preserve differentiability.

  ```python
  >>> a = np.array([1.0 + 2.0j])
  >>> qml.math.conj(a)
  array([1.0 - 2.0j])
  ```

* The four-term parameter-shift rule, as used by the controlled rotation operations,
  has been updated to use coefficients that minimize the variance as per
  https://arxiv.org/abs/2104.05695.
  [(#1206)](https://github.com/PennyLaneAI/pennylane/pull/1206)

* A new transform `qml.transforms.invisible` has been added, to make it easier
  to transform QNodes.
  [(#1175)](https://github.com/PennyLaneAI/pennylane/pull/1175)

<h3>Breaking changes</h3>

* Devices do not have an `analytic` argument or attribute anymore.
  Instead, `shots` is the source of truth for whether a simulator
  estimates return values from a finite number of shots, or whether
  it returns analytic results (`shots=None`).
  [(#1079)](https://github.com/PennyLaneAI/pennylane/pull/1079)
  [(#1196)](https://github.com/PennyLaneAI/pennylane/pull/1196)

  ```python
  dev_analytic = qml.device('default.qubit', wires=1, shots=None)
  dev_finite_shots = qml.device('default.qubit', wires=1, shots=1000)

  def circuit():
      qml.Hadamard(wires=0)
      return qml.expval(qml.PauliZ(wires=0))

  circuit_analytic = qml.QNode(circuit, dev_analytic)
  circuit_finite_shots = qml.QNode(circuit, dev_finite_shots)
  ```

  Devices with `shots=None` return deterministic, exact results:

  ```pycon
  >>> circuit_analytic()
  0.0
  >>> circuit_analytic()
  0.0
  ```
  Devices with `shots > 0` return stochastic results estimated from
  samples in each run:

  ```pycon
  >>> circuit_finite_shots()
  -0.062
  >>> circuit_finite_shots()
  0.034
  ```

  The `qml.sample()` measurement can only be used on devices on which the number
  of shots is set explicitly.

* If creating a QNode from a quantum function with an argument named `shots`,
  a `UserWarning` is raised, warning the user that this is a reserved
  argument to change the number of shots on a per-call basis.
  [(#1075)](https://github.com/PennyLaneAI/pennylane/pull/1075)

* For devices inheriting from `QubitDevice`, the methods `expval`, `var`, `sample`
  accept two new keyword arguments --- `shot_range` and `bin_size`.
  [(#1103)](https://github.com/PennyLaneAI/pennylane/pull/1103)

  These new arguments allow for the statistics to be performed on only a subset of device samples.
  This finer level of control is accessible from the main UI by instantiating a device with a batch
  of shots.

  For example, consider the following device:

  ```pycon
  >>> dev = qml.device("my_device", shots=[5, (10, 3), 100])
  ```

  This device will execute QNodes using 135 shots, however
  measurement statistics will be **course grained** across these 135
  shots:

  * All measurement statistics will first be computed using the
    first 5 shots --- that is, `shots_range=[0, 5]`, `bin_size=5`.

  * Next, the tuple `(10, 3)` indicates 10 shots, repeated 3 times. This will use
    `shot_range=[5, 35]`, performing the expectation value in bins of size 10
    (`bin_size=10`).

  * Finally, we repeat the measurement statistics for the final 100 shots,
    `shot_range=[35, 135]`, `bin_size=100`.


* The old PennyLane core has been removed, including the following modules:
  [(#1100)](https://github.com/PennyLaneAI/pennylane/pull/1100)

  - `pennylane.variables`
  - `pennylane.qnodes`

  As part of this change, the location of the new core within the Python
  module has been moved:

  - Moves `pennylane.tape.interfaces` → `pennylane.interfaces`
  - Merges `pennylane.CircuitGraph` and `pennylane.TapeCircuitGraph`  → `pennylane.CircuitGraph`
  - Merges `pennylane.OperationRecorder` and `pennylane.TapeOperationRecorder`  →
  - `pennylane.tape.operation_recorder`
  - Merges `pennylane.measure` and `pennylane.tape.measure` → `pennylane.measure`
  - Merges `pennylane.operation` and `pennylane.tape.operation` → `pennylane.operation`
  - Merges `pennylane._queuing` and `pennylane.tape.queuing` → `pennylane.queuing`

  This has no affect on import location.

  In addition,

  - All tape-mode functions have been removed (`qml.enable_tape()`, `qml.tape_mode_active()`),
  - All tape fixtures have been deleted,
  - Tests specifically for non-tape mode have been deleted.

* The device test suite no longer accepts the `analytic` keyword.
  [(#1216)](https://github.com/PennyLaneAI/pennylane/pull/1216)

<h3>Bug fixes</h3>

* Fixes a bug where using the circuit drawer with a `ControlledQubitUnitary`
  operation raised an error.
  [(#1174)](https://github.com/PennyLaneAI/pennylane/pull/1174)

* Fixes a bug and a test where the ``QuantumTape.is_sampled`` attribute was not
  being updated.
  [(#1126)](https://github.com/PennyLaneAI/pennylane/pull/1126)

* Fixes a bug where `BasisEmbedding` would not accept inputs whose bits are all ones
  or all zeros.
  [(#1114)](https://github.com/PennyLaneAI/pennylane/pull/1114)

* The `ExpvalCost` class raises an error if instantiated
  with non-expectation measurement statistics.
  [(#1106)](https://github.com/PennyLaneAI/pennylane/pull/1106)

* Fixes a bug where decompositions would reset the differentiation method
  of a QNode.
  [(#1117)](https://github.com/PennyLaneAI/pennylane/pull/1117)

* Fixes a bug where the second-order CV parameter-shift rule would error
  if attempting to compute the gradient of a QNode with more than one
  second-order observable.
  [(#1197)](https://github.com/PennyLaneAI/pennylane/pull/1197)

* Fixes a bug where repeated Torch interface applications after expansion caused an error.
  [(#1223)](https://github.com/PennyLaneAI/pennylane/pull/1223)

* Sampling works correctly with batches of shots specified as a list.
  [(#1232)](https://github.com/PennyLaneAI/pennylane/pull/1232)

<h3>Documentation</h3>

- Updated the diagram used in the Architectural overview page of the
  Development guide such that it doesn't mention Variables.
  [(#1235)](https://github.com/PennyLaneAI/pennylane/pull/1235)

- Typos addressed in templates documentation.
  [(#1094)](https://github.com/PennyLaneAI/pennylane/pull/1094)

- Upgraded the documentation to use Sphinx 3.5.3 and the new m2r2 package.
  [(#1186)](https://github.com/PennyLaneAI/pennylane/pull/1186)

- Added `flaky` as dependency for running tests in the documentation.
  [(#1113)](https://github.com/PennyLaneAI/pennylane/pull/1113)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Shahnawaz Ahmed, Juan Miguel Arrazola, Thomas Bromley, Olivia Di Matteo, Alain Delgado Gran, Kyle
Godbey, Diego Guala, Theodor Isacsson, Josh Izaac, Soran Jahangiri, Nathan Killoran, Christina Lee,
Daniel Polatajko, Chase Roberts, Sankalp Sanand, Pritish Sehzpaul, Maria Schuld, Antal Száva, David Wierichs.


# Release 0.14.1

<h3>Bug fixes</h3>

* Fixes a testing bug where tests that required JAX would fail if JAX was not installed.
  The tests will now instead be skipped if JAX can not be imported.
  [(#1066)](https://github.com/PennyLaneAI/pennylane/pull/1066)

* Fixes a bug where inverse operations could not be differentiated
  using backpropagation on `default.qubit`.
  [(#1072)](https://github.com/PennyLaneAI/pennylane/pull/1072)

* The QNode has a new keyword argument, `max_expansion`, that determines the maximum number of times
  the internal circuit should be expanded when executed on a device. In addition, the default number
  of max expansions has been increased from 2 to 10, allowing devices that require more than two
  operator decompositions to be supported.
  [(#1074)](https://github.com/PennyLaneAI/pennylane/pull/1074)

* Fixes a bug where `Hamiltonian` objects created with non-list arguments raised an error for
  arithmetic operations. [(#1082)](https://github.com/PennyLaneAI/pennylane/pull/1082)

* Fixes a bug where `Hamiltonian` objects with no coefficients or operations would return a faulty
  result when used with `ExpvalCost`. [(#1082)](https://github.com/PennyLaneAI/pennylane/pull/1082)

<h3>Documentation</h3>

* Updates mentions of `generate_hamiltonian` to `molecular_hamiltonian` in the
  docstrings of the `ExpvalCost` and `Hamiltonian` classes.
  [(#1077)](https://github.com/PennyLaneAI/pennylane/pull/1077)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Thomas Bromley, Josh Izaac, Antal Száva.



# Release 0.14.0

<h3>New features since last release</h3>

<h4>Perform quantum machine learning with JAX</h4>

* QNodes created with `default.qubit` now support a JAX interface, allowing JAX to be used
  to create, differentiate, and optimize hybrid quantum-classical models.
  [(#947)](https://github.com/PennyLaneAI/pennylane/pull/947)

  This is supported internally via a new `default.qubit.jax` device. This device runs end to end in
  JAX, meaning that it supports all of the awesome JAX transformations (`jax.vmap`, `jax.jit`,
  `jax.hessian`, etc).

  Here is an example of how to use the new JAX interface:

  ```python
  dev = qml.device("default.qubit", wires=1)
  @qml.qnode(dev, interface="jax", diff_method="backprop")
  def circuit(x):
      qml.RX(x[1], wires=0)
      qml.Rot(x[0], x[1], x[2], wires=0)
      return qml.expval(qml.PauliZ(0))

  weights = jnp.array([0.2, 0.5, 0.1])
  grad_fn = jax.grad(circuit)
  print(grad_fn(weights))
  ```

  Currently, only `diff_method="backprop"` is supported, with plans to support more in the future.

<h4>New, faster, quantum gradient methods</h4>

* A new differentiation method has been added for use with simulators. The `"adjoint"`
  method operates after a forward pass by iteratively applying inverse gates to scan backwards
  through the circuit.
  [(#1032)](https://github.com/PennyLaneAI/pennylane/pull/1032)

  This method is similar to the reversible method, but has a lower time
  overhead and a similar memory overhead. It follows the approach provided by
  [Jones and Gacon](https://arxiv.org/abs/2009.02823). This method is only compatible with certain
  statevector-based devices such as `default.qubit`.

  Example use:

  ```python
  import pennylane as qml

  wires = 1
  device = qml.device("default.qubit", wires=wires)

  @qml.qnode(device, diff_method="adjoint")
  def f(params):
      qml.RX(0.1, wires=0)
      qml.Rot(*params, wires=0)
      qml.RX(-0.3, wires=0)
      return qml.expval(qml.PauliZ(0))

  params = [0.1, 0.2, 0.3]
  qml.grad(f)(params)
  ```

* The default logic for choosing the 'best' differentiation method has been altered
  to improve performance.
  [(#1008)](https://github.com/PennyLaneAI/pennylane/pull/1008)

  - If the quantum device provides its own gradient, this is now the preferred
    differentiation method.

  - If the quantum device natively supports classical
    backpropagation, this is now preferred over the parameter-shift rule.

    This will lead to marked speed improvement during optimization when using
    `default.qubit`, with a sight penalty on the forward-pass evaluation.

  More details are available below in the 'Improvements' section for plugin developers.

* PennyLane now supports analytical quantum gradients for noisy channels, in addition to its
  existing support for unitary operations. The noisy channels `BitFlip`, `PhaseFlip`, and
  `DepolarizingChannel` all support analytic gradients out of the box.
  [(#968)](https://github.com/PennyLaneAI/pennylane/pull/968)

* A method has been added for calculating the Hessian of quantum circuits using the second-order
  parameter shift formula.
  [(#961)](https://github.com/PennyLaneAI/pennylane/pull/961)

  The following example shows the calculation of the Hessian:

  ```python
  n_wires = 5
  weights = [2.73943676, 0.16289932, 3.4536312, 2.73521126, 2.6412488]

  dev = qml.device("default.qubit", wires=n_wires)

  with qml.tape.QubitParamShiftTape() as tape:
      for i in range(n_wires):
          qml.RX(weights[i], wires=i)

      qml.CNOT(wires=[0, 1])
      qml.CNOT(wires=[2, 1])
      qml.CNOT(wires=[3, 1])
      qml.CNOT(wires=[4, 3])

      qml.expval(qml.PauliZ(1))

  print(tape.hessian(dev))
  ```

  The Hessian is not yet supported via classical machine learning interfaces, but will
  be added in a future release.

<h4>More operations and templates</h4>

* Two new error channels, `BitFlip` and `PhaseFlip` have been added.
  [(#954)](https://github.com/PennyLaneAI/pennylane/pull/954)

  They can be used in the same manner as existing error channels:

  ```python
  dev = qml.device("default.mixed", wires=2)

  @qml.qnode(dev)
  def circuit():
      qml.RX(0.3, wires=0)
      qml.RY(0.5, wires=1)
      qml.BitFlip(0.01, wires=0)
      qml.PhaseFlip(0.01, wires=1)
      return qml.expval(qml.PauliZ(0))
  ```

* Apply permutations to wires using the `Permute` subroutine.
  [(#952)](https://github.com/PennyLaneAI/pennylane/pull/952)

  ```python
  import pennylane as qml
  dev = qml.device('default.qubit', wires=5)

  @qml.qnode(dev)
  def apply_perm():
      # Send contents of wire 4 to wire 0, of wire 2 to wire 1, etc.
      qml.templates.Permute([4, 2, 0, 1, 3], wires=dev.wires)
      return qml.expval(qml.PauliZ(0))
  ```

<h4>QNode transformations</h4>

* The `qml.metric_tensor` function transforms a QNode to produce the Fubini-Study
  metric tensor with full autodifferentiation support---even on hardware.
  [(#1014)](https://github.com/PennyLaneAI/pennylane/pull/1014)

  Consider the following QNode:

  ```python
  dev = qml.device("default.qubit", wires=3)

  @qml.qnode(dev, interface="autograd")
  def circuit(weights):
      # layer 1
      qml.RX(weights[0, 0], wires=0)
      qml.RX(weights[0, 1], wires=1)

      qml.CNOT(wires=[0, 1])
      qml.CNOT(wires=[1, 2])

      # layer 2
      qml.RZ(weights[1, 0], wires=0)
      qml.RZ(weights[1, 1], wires=2)

      qml.CNOT(wires=[0, 1])
      qml.CNOT(wires=[1, 2])
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), qml.expval(qml.PauliY(2))
  ```

  We can use the `metric_tensor` function to generate a new function, that returns the
  metric tensor of this QNode:

  ```pycon
  >>> met_fn = qml.metric_tensor(circuit)
  >>> weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)
  >>> met_fn(weights)
  tensor([[0.25  , 0.    , 0.    , 0.    ],
          [0.    , 0.25  , 0.    , 0.    ],
          [0.    , 0.    , 0.0025, 0.0024],
          [0.    , 0.    , 0.0024, 0.0123]], requires_grad=True)
  ```

  The returned metric tensor is also fully differentiable, in all interfaces.
  For example, differentiating the `(3, 2)` element:

  ```pycon
  >>> grad_fn = qml.grad(lambda x: met_fn(x)[3, 2])
  >>> grad_fn(weights)
  array([[ 0.04867729, -0.00049502,  0.        ],
         [ 0.        ,  0.        ,  0.        ]])
  ```

  Differentiation is also supported using Torch, Jax, and TensorFlow.

* Adds the new function `qml.math.cov_matrix()`. This function accepts a list of commuting
  observables, and the probability distribution in the shared observable eigenbasis after the
  application of an ansatz. It uses these to construct the covariance matrix in a *framework
  independent* manner, such that the output covariance matrix is autodifferentiable.
  [(#1012)](https://github.com/PennyLaneAI/pennylane/pull/1012)

  For example, consider the following ansatz and observable list:

  ```python3
  obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliY(2)]
  ansatz = qml.templates.StronglyEntanglingLayers
  ```

  We can construct a QNode to output the probability distribution in the shared eigenbasis of the
  observables:

  ```python
  dev = qml.device("default.qubit", wires=3)

  @qml.qnode(dev, interface="autograd")
  def circuit(weights):
      ansatz(weights, wires=[0, 1, 2])
      # rotate into the basis of the observables
      for o in obs_list:
          o.diagonalizing_gates()
      return qml.probs(wires=[0, 1, 2])
  ```

  We can now compute the covariance matrix:

  ```pycon
  >>> weights = qml.init.strong_ent_layers_normal(n_layers=2, n_wires=3)
  >>> cov = qml.math.cov_matrix(circuit(weights), obs_list)
  >>> cov
  array([[0.98707611, 0.03665537],
         [0.03665537, 0.99998377]])
  ```

  Autodifferentiation is fully supported using all interfaces:

  ```pycon
  >>> cost_fn = lambda weights: qml.math.cov_matrix(circuit(weights), obs_list)[0, 1]
  >>> qml.grad(cost_fn)(weights)[0]
  array([[[ 4.94240914e-17, -2.33786398e-01, -1.54193959e-01],
          [-3.05414996e-17,  8.40072236e-04,  5.57884080e-04],
          [ 3.01859411e-17,  8.60411436e-03,  6.15745204e-04]],

         [[ 6.80309533e-04, -1.23162742e-03,  1.08729813e-03],
          [-1.53863193e-01, -1.38700657e-02, -1.36243323e-01],
          [-1.54665054e-01, -1.89018172e-02, -1.56415558e-01]]])
  ```

* A new  `qml.draw` function is available, allowing QNodes to be easily
  drawn without execution by providing example input.
  [(#962)](https://github.com/PennyLaneAI/pennylane/pull/962)

  ```python
  @qml.qnode(dev)
  def circuit(a, w):
      qml.Hadamard(0)
      qml.CRX(a, wires=[0, 1])
      qml.Rot(*w, wires=[1])
      qml.CRX(-a, wires=[0, 1])
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  ```

  The QNode circuit structure may depend on the input arguments;
  this is taken into account by passing example QNode arguments
  to the `qml.draw()` drawing function:

  ```pycon
  >>> drawer = qml.draw(circuit)
  >>> result = drawer(a=2.3, w=[1.2, 3.2, 0.7])
  >>> print(result)
  0: ──H──╭C────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩
  1: ─────╰RX(2.3)──Rot(1.2, 3.2, 0.7)──╰RX(-2.3)──╰┤ ⟨Z ⊗ Z⟩
  ```

<h4>A faster, leaner, and more flexible core</h4>

* The new core of PennyLane, rewritten from the ground up and developed over the last few release
  cycles, has achieved feature parity and has been made the new default in PennyLane v0.14. The old
  core has been marked as deprecated, and will be removed in an upcoming release.
  [(#1046)](https://github.com/PennyLaneAI/pennylane/pull/1046)
  [(#1040)](https://github.com/PennyLaneAI/pennylane/pull/1040)
  [(#1034)](https://github.com/PennyLaneAI/pennylane/pull/1034)
  [(#1035)](https://github.com/PennyLaneAI/pennylane/pull/1035)
  [(#1027)](https://github.com/PennyLaneAI/pennylane/pull/1027)
  [(#1026)](https://github.com/PennyLaneAI/pennylane/pull/1026)
  [(#1021)](https://github.com/PennyLaneAI/pennylane/pull/1021)
  [(#1054)](https://github.com/PennyLaneAI/pennylane/pull/1054)
  [(#1049)](https://github.com/PennyLaneAI/pennylane/pull/1049)

  While high-level PennyLane code and tutorials remain unchanged, the new core
  provides several advantages and improvements:

  - **Faster and more optimized**: The new core provides various performance optimizations, reducing
    pre- and post-processing overhead, and reduces the number of quantum evaluations in certain
    cases.

  - **Support for in-QNode classical processing**: this allows for differentiable classical
    processing within the QNode.

    ```python
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface="tf")
    def circuit(p):
        qml.RX(tf.sin(p[0])**2 + p[1], wires=0)
        return qml.expval(qml.PauliZ(0))
    ```

    The classical processing functions used within the QNode must match
    the QNode interface. Here, we use TensorFlow:

    ```pycon
    >>> params = tf.Variable([0.5, 0.1], dtype=tf.float64)
    >>> with tf.GradientTape() as tape:
    ...     res = circuit(params)
    >>> grad = tape.gradient(res, params)
    >>> print(res)
    tf.Tensor(0.9460913127754935, shape=(), dtype=float64)
    >>> print(grad)
    tf.Tensor([-0.27255248 -0.32390003], shape=(2,), dtype=float64)
    ```

    As a result of this change, quantum decompositions that require classical processing
    are fully supported and end-to-end differentiable in tape mode.

  - **No more Variable wrapping**: QNode arguments no longer become `Variable`
    objects within the QNode.

    ```python
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(x):
        print("Parameter value:", x)
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))
    ```

    Internal QNode parameters can be easily inspected, printed, and manipulated:

    ```pycon
    >>> circuit(0.5)
    Parameter value: 0.5
    tensor(0.87758256, requires_grad=True)
    ```

  - **Less restrictive QNode signatures**: There is no longer any restriction on the QNode signature; the QNode can be
    defined and called following the same rules as standard Python functions.

    For example, the following QNode uses positional, named, and variable
    keyword arguments:

    ```python
    x = torch.tensor(0.1, requires_grad=True)
    y = torch.tensor([0.2, 0.3], requires_grad=True)
    z = torch.tensor(0.4, requires_grad=True)

    @qml.qnode(dev, interface="torch")
    def circuit(p1, p2=y, **kwargs):
        qml.RX(p1, wires=0)
        qml.RY(p2[0] * p2[1], wires=0)
        qml.RX(kwargs["p3"], wires=0)
        return qml.var(qml.PauliZ(0))
    ```

    When we call the QNode, we may pass the arguments by name
    even if defined positionally; any argument not provided will
    use the default value.

    ```pycon
    >>> res = circuit(p1=x, p3=z)
    >>> print(res)
    tensor(0.2327, dtype=torch.float64, grad_fn=<SelectBackward>)
    >>> res.backward()
    >>> print(x.grad, y.grad, z.grad)
    tensor(0.8396) tensor([0.0289, 0.0193]) tensor(0.8387)
    ```

    This extends to the `qnn` module, where `KerasLayer` and `TorchLayer` modules
    can be created from QNodes with unrestricted signatures.

  - **Smarter measurements:** QNodes can now measure wires more than once, as
    long as all observables are commuting:

    ```python
    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return [
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        ]
    ```

    Further, the `qml.ExpvalCost()` function allows for optimizing
    measurements to reduce the number of quantum evaluations required.

  With the new PennyLane core, there are a few small breaking changes, detailed
  below in the 'Breaking Changes' section.

<h3>Improvements</h3>

* The built-in PennyLane optimizers allow more flexible cost functions. The cost function passed to most optimizers
  may accept any combination of trainable arguments, non-trainable arguments, and keyword arguments.
  [(#959)](https://github.com/PennyLaneAI/pennylane/pull/959)
  [(#1053)](https://github.com/PennyLaneAI/pennylane/pull/1053)

  The full changes apply to:

  * `AdagradOptimizer`
  * `AdamOptimizer`
  * `GradientDescentOptimizer`
  * `MomentumOptimizer`
  * `NesterovMomentumOptimizer`
  * `RMSPropOptimizer`
  * `RotosolveOptimizer`

  The `requires_grad=False` property must mark any non-trainable constant argument.
  The `RotoselectOptimizer` allows passing only keyword arguments.

  Example use:

  ```python
  def cost(x, y, data, scale=1.0):
      return scale * (x[0]-data)**2 + scale * (y-data)**2

  x = np.array([1.], requires_grad=True)
  y = np.array([1.0])
  data = np.array([2.], requires_grad=False)

  opt = qml.GradientDescentOptimizer()

  # the optimizer step and step_and_cost methods can
  # now update multiple parameters at once
  x_new, y_new, data = opt.step(cost, x, y, data, scale=0.5)
  (x_new, y_new, data), value = opt.step_and_cost(cost, x, y, data, scale=0.5)

  # list and tuple unpacking is also supported
  params = (x, y, data)
  params = opt.step(cost, *params)
  ```

* The circuit drawer has been updated to support the inclusion of unused or inactive
  wires, by passing the `show_all_wires` argument.
  [(#1033)](https://github.com/PennyLaneAI/pennylane/pull/1033)

  ```python
  dev = qml.device('default.qubit', wires=[-1, "a", "q2", 0])

  @qml.qnode(dev)
  def circuit():
      qml.Hadamard(wires=-1)
      qml.CNOT(wires=[-1, "q2"])
      return qml.expval(qml.PauliX(wires="q2"))
  ```

  ```pycon
  >>> print(qml.draw(circuit, show_all_wires=True)())
  >>>
   -1: ──H──╭C──┤
    a: ─────│───┤
   q2: ─────╰X──┤ ⟨X⟩
    0: ─────────┤
  ```

* The logic for choosing the 'best' differentiation method has been altered
  to improve performance.
  [(#1008)](https://github.com/PennyLaneAI/pennylane/pull/1008)

  - If the device provides its own gradient, this is now the preferred
    differentiation method.

  - If a device provides additional interface-specific versions that natively support classical
    backpropagation, this is now preferred over the parameter-shift rule.

    Devices define additional interface-specific devices via their `capabilities()` dictionary. For
    example, `default.qubit` supports supplementary devices for TensorFlow, Autograd, and JAX:

    ```python
    {
      "passthru_devices": {
          "tf": "default.qubit.tf",
          "autograd": "default.qubit.autograd",
          "jax": "default.qubit.jax",
      },
    }
    ```

  As a result of this change, if the QNode `diff_method` is not explicitly provided,
  it is possible that the QNode will run on a *supplementary device* of the device that was
  specifically provided:

  ```python
  dev = qml.device("default.qubit", wires=2)
  qml.QNode(dev) # will default to backprop on default.qubit.autograd
  qml.QNode(dev, interface="tf") # will default to backprop on default.qubit.tf
  qml.QNode(dev, interface="jax") # will default to backprop on default.qubit.jax
  ```

* The `default.qubit` device has been updated so that internally it applies operations in a more
  functional style, i.e., by accepting an input state and returning an evolved state.
  [(#1025)](https://github.com/PennyLaneAI/pennylane/pull/1025)

* A new test series, `pennylane/devices/tests/test_compare_default_qubit.py`, has been added, allowing to test if
  a chosen device gives the same result as `default.qubit`.
  [(#897)](https://github.com/PennyLaneAI/pennylane/pull/897)

  Three tests are added:

  - `test_hermitian_expectation`,
  - `test_pauliz_expectation_analytic`, and
  - `test_random_circuit`.

* Adds the following agnostic tensor manipulation functions to the `qml.math` module: `abs`,
  `angle`, `arcsin`, `concatenate`, `dot`, `squeeze`, `sqrt`, `sum`, `take`, `where`. These functions are
  required to fully support end-to-end differentiable Mottonen and Amplitude embedding.
  [(#922)](https://github.com/PennyLaneAI/pennylane/pull/922)
  [(#1011)](https://github.com/PennyLaneAI/pennylane/pull/1011)

* The `qml.math` module now supports JAX.
  [(#985)](https://github.com/XanaduAI/software-docs/pull/274)

* Several improvements have been made to the `Wires` class to reduce overhead and simplify the logic
  of how wire labels are interpreted:
  [(#1019)](https://github.com/PennyLaneAI/pennylane/pull/1019)
  [(#1010)](https://github.com/PennyLaneAI/pennylane/pull/1010)
  [(#1005)](https://github.com/PennyLaneAI/pennylane/pull/1005)
  [(#983)](https://github.com/PennyLaneAI/pennylane/pull/983)
  [(#967)](https://github.com/PennyLaneAI/pennylane/pull/967)

  - If the input `wires` to a wires class instantiation `Wires(wires)` can be iterated over,
    its elements are interpreted as wire labels. Otherwise, `wires` is interpreted as a single wire label.
    The only exception to this are strings, which are always interpreted as a single
    wire label, so users can address wires with labels such as `"ancilla"`.

  - Any type can now be a wire label as long as it is hashable. The hash is used to establish
    the uniqueness of two labels.

  - Indexing wires objects now returns a label, instead of a new `Wires` object. For example:

    ```pycon
    >>> w = Wires([0, 1, 2])
    >>> w[1]
    >>> 1
    ```

  - The check for uniqueness of wires moved from `Wires` instantiation to
    the `qml.wires._process` function in order to reduce overhead from repeated
    creation of `Wires` instances.

  - Calls to the `Wires` class are substantially reduced, for example by avoiding to call
    Wires on Wires instances on `Operation` instantiation, and by using labels instead of
    `Wires` objects inside the default qubit device.

* Adds the `PauliRot` generator to the `qml.operation` module. This
  generator is required to construct the metric tensor.
  [(#963)](https://github.com/PennyLaneAI/pennylane/pull/963)

* The templates are modified to make use of the new `qml.math` module, for framework-agnostic
  tensor manipulation. This allows the template library to be differentiable
  in backpropagation mode (`diff_method="backprop"`).
  [(#873)](https://github.com/PennyLaneAI/pennylane/pull/873)

* The circuit drawer now allows for the wire order to be (optionally) modified:
  [(#992)](https://github.com/PennyLaneAI/pennylane/pull/992)

  ```pycon
  >>> dev = qml.device('default.qubit', wires=["a", -1, "q2"])
  >>> @qml.qnode(dev)
  ... def circuit():
  ...     qml.Hadamard(wires=-1)
  ...     qml.CNOT(wires=["a", "q2"])
  ...     qml.RX(0.2, wires="a")
  ...     return qml.expval(qml.PauliX(wires="q2"))
  ```

  Printing with default wire order of the device:

  ```pycon
  >>> print(circuit.draw())
    a: ─────╭C──RX(0.2)──┤
   -1: ──H──│────────────┤
   q2: ─────╰X───────────┤ ⟨X⟩
  ```

  Changing the wire order:

  ```pycon
  >>> print(circuit.draw(wire_order=["q2", "a", -1]))
   q2: ──╭X───────────┤ ⟨X⟩
    a: ──╰C──RX(0.2)──┤
   -1: ───H───────────┤
  ```

<h3>Breaking changes</h3>

* QNodes using the new PennyLane core will no longer accept ragged arrays as inputs.

* When using the new PennyLane core and the Autograd interface, non-differentiable data passed
  as a QNode argument or a gate must have the `requires_grad` property set to `False`:

  ```python
  @qml.qnode(dev)
  def circuit(weights, data):
      basis_state = np.array([1, 0, 1, 1], requires_grad=False)
      qml.BasisState(basis_state, wires=[0, 1, 2, 3])
      qml.templates.AmplitudeEmbedding(data, wires=[0, 1, 2, 3])
      qml.templates.BasicEntanglerLayers(weights, wires=[0, 1, 2, 3])
      return qml.probs(wires=0)

  data = np.array(data, requires_grad=False)
  weights = np.array(weights, requires_grad=True)
  circuit(weights, data)
  ```

<h3>Bug fixes</h3>

* Fixes an issue where if the constituent observables of a tensor product do not exist in the queue,
  an error is raised. With this fix, they are first queued before annotation occurs.
  [(#1038)](https://github.com/PennyLaneAI/pennylane/pull/1038)

* Fixes an issue with tape expansions where information about sampling
  (specifically the `is_sampled` tape attribute) was not preserved.
  [(#1027)](https://github.com/PennyLaneAI/pennylane/pull/1027)

* Tape expansion was not properly taking into devices that supported inverse operations,
  causing inverse operations to be unnecessarily decomposed. The QNode tape expansion logic, as well
  as the `Operation.expand()` method, has been modified to fix this.
  [(#956)](https://github.com/PennyLaneAI/pennylane/pull/956)

* Fixes an issue where the Autograd interface was not unwrapping non-differentiable
  PennyLane tensors, which can cause issues on some devices.
  [(#941)](https://github.com/PennyLaneAI/pennylane/pull/941)

* `qml.vqe.Hamiltonian` prints any observable with any number of strings.
  [(#987)](https://github.com/PennyLaneAI/pennylane/pull/987)

* Fixes a bug where parameter-shift differentiation would fail if the QNode
  contained a single probability output.
  [(#1007)](https://github.com/PennyLaneAI/pennylane/pull/1007)

* Fixes an issue when using trainable parameters that are lists/arrays with `tape.vjp`.
  [(#1042)](https://github.com/PennyLaneAI/pennylane/pull/1042)

* The `TensorN` observable is updated to support being copied without any parameters or wires passed.
  [(#1047)](https://github.com/PennyLaneAI/pennylane/pull/1047)

* Fixed deprecation warning when importing `Sequence` from `collections` instead of `collections.abc` in `vqe/vqe.py`.
  [(#1051)](https://github.com/PennyLaneAI/pennylane/pull/1051)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Thomas Bromley, Olivia Di Matteo, Theodor Isacsson, Josh Izaac, Christina Lee,
Alejandro Montanez, Steven Oud, Chase Roberts, Sankalp Sanand, Maria Schuld, Antal
Száva, David Wierichs, Jiahao Yao.

# Release 0.13.0

<h3>New features since last release</h3>

<h4>Automatically optimize the number of measurements</h4>

* QNodes in tape mode now support returning observables on the same wire whenever the observables are
  qubit-wise commuting Pauli words. Qubit-wise commuting observables can be evaluated with a
  *single* device run as they are diagonal in the same basis, via a shared set of single-qubit rotations.
  [(#882)](https://github.com/PennyLaneAI/pennylane/pull/882)

  The following example shows a single QNode returning the expectation values of
  the qubit-wise commuting Pauli words `XX` and `XI`:

  ```python
  qml.enable_tape()

  @qml.qnode(dev)
  def f(x):
      qml.Hadamard(wires=0)
      qml.Hadamard(wires=1)
      qml.CRot(0.1, 0.2, 0.3, wires=[1, 0])
      qml.RZ(x, wires=1)
      return qml.expval(qml.PauliX(0) @ qml.PauliX(1)), qml.expval(qml.PauliX(0))
  ```

  ```pycon
  >>> f(0.4)
  tensor([0.89431013, 0.9510565 ], requires_grad=True)
  ```

* The `ExpvalCost` class (previously `VQECost`) now provides observable optimization using the
  `optimize` argument, resulting in potentially fewer device executions.
  [(#902)](https://github.com/PennyLaneAI/pennylane/pull/902)

  This is achieved by separating the observables composing the Hamiltonian into qubit-wise
  commuting groups and evaluating those groups on a single QNode using functionality from the
  `qml.grouping` module:

  ```python
  qml.enable_tape()
  commuting_obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1)]
  H = qml.vqe.Hamiltonian([1, 1], commuting_obs)

  dev = qml.device("default.qubit", wires=2)
  ansatz = qml.templates.StronglyEntanglingLayers

  cost_opt = qml.ExpvalCost(ansatz, H, dev, optimize=True)
  cost_no_opt = qml.ExpvalCost(ansatz, H, dev, optimize=False)

  params = qml.init.strong_ent_layers_uniform(3, 2)
  ```

  Grouping these commuting observables leads to fewer device executions:

  ```pycon
  >>> cost_opt(params)
  >>> ex_opt = dev.num_executions
  >>> cost_no_opt(params)
  >>> ex_no_opt = dev.num_executions - ex_opt
  >>> print("Number of executions:", ex_no_opt)
  Number of executions: 2
  >>> print("Number of executions (optimized):", ex_opt)
  Number of executions (optimized): 1
  ```

<h4>New quantum gradient features</h4>

* Compute the analytic gradient of quantum circuits in parallel on supported devices.
  [(#840)](https://github.com/PennyLaneAI/pennylane/pull/840)

  This release introduces support for batch execution of circuits, via a new device API method
  `Device.batch_execute()`. Devices that implement this new API support submitting a batch of
  circuits for *parallel* evaluation simultaneously, which can significantly reduce the computation time.

  Furthermore, if using tape mode and a compatible device, gradient computations will
  automatically make use of the new batch API---providing a speedup during optimization.

* Gradient recipes are now much more powerful, allowing for operations to define their gradient
  via an arbitrary linear combination of circuit evaluations.
  [(#909)](https://github.com/PennyLaneAI/pennylane/pull/909)
  [(#915)](https://github.com/PennyLaneAI/pennylane/pull/915)

  With this change, gradient recipes can now be of the form
  :math:`\frac{\partial}{\partial\phi_k}f(\phi_k) = \sum_{i} c_i f(a_i \phi_k + s_i )`,
  and are no longer restricted to two-term shifts with identical (but opposite in sign) shift values.

  As a result, PennyLane now supports native analytic quantum gradients for the
  controlled rotation operations `CRX`, `CRY`, `CRZ`, and `CRot`. This allows for parameter-shift
  analytic gradients on hardware, without decomposition.

  Note that this is a breaking change for developers; please see the *Breaking Changes* section
  for more details.

* The `qnn.KerasLayer` class now supports differentiating the QNode through classical
  backpropagation in tape mode.
  [(#869)](https://github.com/PennyLaneAI/pennylane/pull/869)

  ```python
  qml.enable_tape()

  dev = qml.device("default.qubit.tf", wires=2)

  @qml.qnode(dev, interface="tf", diff_method="backprop")
  def f(inputs, weights):
      qml.templates.AngleEmbedding(inputs, wires=range(2))
      qml.templates.StronglyEntanglingLayers(weights, wires=range(2))
      return [qml.expval(qml.PauliZ(i)) for i in range(2)]

  weight_shapes = {"weights": (3, 2, 3)}

  qlayer = qml.qnn.KerasLayer(f, weight_shapes, output_dim=2)

  inputs = tf.constant(np.random.random((4, 2)), dtype=tf.float32)

  with tf.GradientTape() as tape:
      out = qlayer(inputs)

  tape.jacobian(out, qlayer.trainable_weights)
  ```

<h4>New operations, templates, and measurements</h4>

* Adds the `qml.density_matrix` QNode return with partial trace capabilities.
  [(#878)](https://github.com/PennyLaneAI/pennylane/pull/878)

  The density matrix over the provided wires is returned, with all other subsystems traced out.
  `qml.density_matrix` currently works for both the `default.qubit` and `default.mixed` devices.

  ```python
  qml.enable_tape()
  dev = qml.device("default.qubit", wires=2)

  def circuit(x):
      qml.PauliY(wires=0)
      qml.Hadamard(wires=1)
      return qml.density_matrix(wires=[1])  # wire 0 is traced out
  ```

* Adds the square-root X gate `SX`. [(#871)](https://github.com/PennyLaneAI/pennylane/pull/871)

  ```python
  dev = qml.device("default.qubit", wires=1)

  @qml.qnode(dev)
  def circuit():
      qml.SX(wires=[0])
      return qml.expval(qml.PauliZ(wires=[0]))
  ```

* Two new hardware-efficient particle-conserving templates have been implemented
  to perform VQE-based quantum chemistry simulations. The new templates apply
  several layers of the particle-conserving entanglers proposed in Figs. 2a and 2b
  of Barkoutsos *et al*., [arXiv:1805.04340](https://arxiv.org/abs/1805.04340)
  [(#875)](https://github.com/PennyLaneAI/pennylane/pull/875)
  [(#876)](https://github.com/PennyLaneAI/pennylane/pull/876)

<h4>Estimate and track resources</h4>

* The `QuantumTape` class now contains basic resource estimation functionality. The method
  `tape.get_resources()` returns a dictionary with a list of the constituent operations and the
  number of times they appear in the circuit. Similarly, `tape.get_depth()` computes the circuit depth.
  [(#862)](https://github.com/PennyLaneAI/pennylane/pull/862)

  ```pycon
  >>> with qml.tape.QuantumTape() as tape:
  ...    qml.Hadamard(wires=0)
  ...    qml.RZ(0.26, wires=1)
  ...    qml.CNOT(wires=[1, 0])
  ...    qml.Rot(1.8, -2.7, 0.2, wires=0)
  ...    qml.Hadamard(wires=1)
  ...    qml.CNOT(wires=[0, 1])
  ...    qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  >>> tape.get_resources()
  {'Hadamard': 2, 'RZ': 1, 'CNOT': 2, 'Rot': 1}
  >>> tape.get_depth()
  4
  ```

* The number of device executions over a QNode's lifetime can now be returned using `num_executions`.
  [(#853)](https://github.com/PennyLaneAI/pennylane/pull/853)

  ```pycon
  >>> dev = qml.device("default.qubit", wires=2)
  >>> @qml.qnode(dev)
  ... def circuit(x, y):
  ...    qml.RX(x, wires=[0])
  ...    qml.RY(y, wires=[1])
  ...    qml.CNOT(wires=[0, 1])
  ...    return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
  >>> for _ in range(10):
  ...    circuit(0.432, 0.12)
  >>> print(dev.num_executions)
  10
  ```

<h3>Improvements</h3>

* Support for tape mode has improved across PennyLane. The following features now work in tape mode:

  - QNode collections [(#863)](https://github.com/PennyLaneAI/pennylane/pull/863)

  - `qnn.ExpvalCost` [(#863)](https://github.com/PennyLaneAI/pennylane/pull/863)
    [(#911)](https://github.com/PennyLaneAI/pennylane/pull/911)

  - `qml.qnn.KerasLayer` [(#869)](https://github.com/PennyLaneAI/pennylane/pull/869)

  - `qml.qnn.TorchLayer` [(#865)](https://github.com/PennyLaneAI/pennylane/pull/865)

  - The `qml.qaoa` module [(#905)](https://github.com/PennyLaneAI/pennylane/pull/905)

* A new function, `qml.refresh_devices()`, has been added, allowing PennyLane to
  rescan installed PennyLane plugins and refresh the device list. In addition, the `qml.device`
  loader will attempt to refresh devices if the required plugin device cannot be found.
  This will result in an improved experience if installing PennyLane and plugins within
  a running Python session (for example, on Google Colab), and avoid the need to
  restart the kernel/runtime.
  [(#907)](https://github.com/PennyLaneAI/pennylane/pull/907)

* When using `grad_fn = qml.grad(cost)` to compute the gradient of a cost function with the Autograd
  interface, the value of the intermediate forward pass is now available via the `grad_fn.forward`
  property
  [(#914)](https://github.com/PennyLaneAI/pennylane/pull/914):

  ```python
  def cost_fn(x, y):
      return 2 * np.sin(x[0]) * np.exp(-x[1]) + x[0] ** 3 + np.cos(y)

  params = np.array([0.1, 0.5], requires_grad=True)
  data = np.array(0.65, requires_grad=False)
  grad_fn = qml.grad(cost_fn)

  grad_fn(params, data)  # perform backprop and evaluate the gradient
  grad_fn.forward  # the cost function value
  ```

* Gradient-based optimizers now have a `step_and_cost` method that returns
  both the next step as well as the objective (cost) function output.
  [(#916)](https://github.com/PennyLaneAI/pennylane/pull/916)

  ```pycon
  >>> opt = qml.GradientDescentOptimizer()
  >>> params, cost = opt.step_and_cost(cost_fn, params)
  ```

* PennyLane provides a new experimental module `qml.proc` which provides framework-agnostic processing
  functions for array and tensor manipulations.
  [(#886)](https://github.com/PennyLaneAI/pennylane/pull/886)

  Given the input tensor-like object, the call is
  dispatched to the corresponding array manipulation framework, allowing for end-to-end
  differentiation to be preserved.

  ```pycon
  >>> x = torch.tensor([1., 2.])
  >>> qml.proc.ones_like(x)
  tensor([1, 1])
  >>> y = tf.Variable([[0], [5]])
  >>> qml.proc.ones_like(y, dtype=np.complex128)
  <tf.Tensor: shape=(2, 1), dtype=complex128, numpy=
  array([[1.+0.j],
         [1.+0.j]])>
  ```

  Note that these functions are experimental, and only a subset of common functionality is
  supported. Furthermore, the names and behaviour of these functions may differ from similar
  functions in common frameworks; please refer to the function docstrings for more details.

* The gradient methods in tape mode now fully separate the quantum and classical processing. Rather
  than returning the evaluated gradients directly, they now return a tuple containing the required
  quantum and classical processing steps.
  [(#840)](https://github.com/PennyLaneAI/pennylane/pull/840)

  ```python
  def gradient_method(idx, param, **options):
      # generate the quantum tapes that must be computed
      # to determine the quantum gradient
      tapes = quantum_gradient_tapes(self)

      def processing_fn(results):
          # perform classical processing on the evaluated tapes
          # returning the evaluated quantum gradient
          return classical_processing(results)

      return tapes, processing_fn
  ```

  The `JacobianTape.jacobian()` method has been similarly modified to accumulate all gradient
  quantum tapes and classical processing functions, evaluate all quantum tapes simultaneously,
  and then apply the post-processing functions to the evaluated tape results.

* The MultiRZ gate now has a defined generator, allowing it to be used in quantum natural gradient
  optimization.
  [(#912)](https://github.com/PennyLaneAI/pennylane/pull/912)

* The CRot gate now has a `decomposition` method, which breaks the gate down into rotations
  and CNOT gates. This allows `CRot` to be used on devices that do not natively support it.
  [(#908)](https://github.com/PennyLaneAI/pennylane/pull/908)

* The classical processing in the `MottonenStatePreparation` template has been largely
  rewritten to use dense matrices and tensor manipulations wherever possible.
  This is in preparation to support differentiation through the template in the future.
  [(#864)](https://github.com/PennyLaneAI/pennylane/pull/864)

* Device-based caching has replaced QNode caching. Caching is now accessed by passing a
  `cache` argument to the device.
  [(#851)](https://github.com/PennyLaneAI/pennylane/pull/851)

  The `cache` argument should be an integer specifying the size of the cache. For example, a
  cache of size 10 is created using:

  ```pycon
  >>> dev = qml.device("default.qubit", wires=2, cache=10)
  ```

* The `Operation`, `Tensor`, and `MeasurementProcess` classes now have the `__copy__` special method
  defined.
  [(#840)](https://github.com/PennyLaneAI/pennylane/pull/840)

  This allows us to ensure that, when a shallow copy is performed of an operation, the
  mutable list storing the operation parameters is *also* shallow copied. Both the old operation and
  the copied operation will continue to share the same parameter data,
  ```pycon
  >>> import copy
  >>> op = qml.RX(0.2, wires=0)
  >>> op2 = copy.copy(op)
  >>> op.data[0] is op2.data[0]
  True
  ```

  however the *list container* is not a reference:

  ```pycon
  >>> op.data is op2.data
  False
  ```

  This allows the parameters of the copied operation to be modified, without mutating
  the parameters of the original operation.

* The `QuantumTape.copy` method has been tweaked so that
  [(#840)](https://github.com/PennyLaneAI/pennylane/pull/840):

  - Optionally, the tape's operations are shallow copied in addition to the tape by passing the
    `copy_operations=True` boolean flag. This allows the copied tape's parameters to be mutated
    without affecting the original tape's parameters. (Note: the two tapes will share parameter data
    *until* one of the tapes has their parameter list modified.)

  - Copied tapes can be cast to another `QuantumTape` subclass by passing the `tape_cls` keyword
    argument.

<h3>Breaking changes</h3>

* Updated how parameter-shift gradient recipes are defined for operations, allowing for
  gradient recipes that are specified as an arbitrary number of terms.
  [(#909)](https://github.com/PennyLaneAI/pennylane/pull/909)

  Previously, `Operation.grad_recipe` was restricted to two-term parameter-shift formulas.
  With this change, the gradient recipe now contains elements of the form
  :math:`[c_i, a_i, s_i]`, resulting in a gradient recipe of
  :math:`\frac{\partial}{\partial\phi_k}f(\phi_k) = \sum_{i} c_i f(a_i \phi_k + s_i )`.

  As this is a breaking change, all custom operations with defined gradient recipes must be
  updated to continue working with PennyLane 0.13. Note though that if `grad_recipe = None`, the
  default gradient recipe remains unchanged, and corresponds to the two terms :math:`[c_0, a_0, s_0]=[1/2, 1, \pi/2]`
  and :math:`[c_1, a_1, s_1]=[-1/2, 1, -\pi/2]` for every parameter.

- The `VQECost` class has been renamed to `ExpvalCost` to reflect its general applicability
  beyond VQE. Use of `VQECost` is still possible but will result in a deprecation warning.
  [(#913)](https://github.com/PennyLaneAI/pennylane/pull/913)

<h3>Bug fixes</h3>

* The `default.qubit.tf` device is updated to handle TensorFlow objects (e.g.,
  `tf.Variable`) as gate parameters correctly when using the `MultiRZ` and
  `CRot` operations.
  [(#921)](https://github.com/PennyLaneAI/pennylane/pull/921)

* PennyLane tensor objects are now unwrapped in BaseQNode when passed as a
  keyword argument to the quantum function.
  [(#903)](https://github.com/PennyLaneAI/pennylane/pull/903)
  [(#893)](https://github.com/PennyLaneAI/pennylane/pull/893)

* The new tape mode now prevents multiple observables from being evaluated on the same wire
  if the observables are not qubit-wise commuting Pauli words.
  [(#882)](https://github.com/PennyLaneAI/pennylane/pull/882)

* Fixes a bug in `default.qubit` whereby inverses of common gates were not being applied
  via efficient gate-specific methods, instead falling back to matrix-vector multiplication.
  The following gates were affected: `PauliX`, `PauliY`, `PauliZ`, `Hadamard`, `SWAP`, `S`,
  `T`, `CNOT`, `CZ`.
  [(#872)](https://github.com/PennyLaneAI/pennylane/pull/872)

* The `PauliRot` operation now gracefully handles single-qubit Paulis, and all-identity Paulis
  [(#860)](https://github.com/PennyLaneAI/pennylane/pull/860).

* Fixes a bug whereby binary Python operators were not properly propagating the `requires_grad`
  attribute to the output tensor.
  [(#889)](https://github.com/PennyLaneAI/pennylane/pull/889)

* Fixes a bug which prevents `TorchLayer` from doing `backward` when CUDA is enabled.
  [(#899)](https://github.com/PennyLaneAI/pennylane/pull/899)

* Fixes a bug where multi-threaded execution of `QNodeCollection` sometimes fails
  because of simultaneous queuing. This is fixed by adding thread locking during queuing.
  [(#910)](https://github.com/PennyLaneAI/pennylane/pull/918)

* Fixes a bug in `QuantumTape.set_parameters()`. The previous implementation assumed
  that the `self.trainable_parms` set would always be iterated over in increasing integer
  order. However, this is not guaranteed behaviour, and can lead to the incorrect tape parameters
  being set if this is not the case.
  [(#923)](https://github.com/PennyLaneAI/pennylane/pull/923)

* Fixes broken error message if a QNode is instantiated with an unknown exception.
  [(#930)](https://github.com/PennyLaneAI/pennylane/pull/930)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Thomas Bromley, Christina Lee, Alain Delgado Gran, Olivia Di Matteo, Anthony
Hayes, Theodor Isacsson, Josh Izaac, Soran Jahangiri, Nathan Killoran, Shumpei Kobayashi, Romain
Moyard, Zeyue Niu, Maria Schuld, Antal Száva.

# Release 0.12.0

<h3>New features since last release</h3>

<h4>New and improved simulators</h4>

* PennyLane now supports a new device, `default.mixed`, designed for
  simulating mixed-state quantum computations. This enables native
  support for implementing noisy channels in a circuit, which generally
  map pure states to mixed states.
  [(#794)](https://github.com/PennyLaneAI/pennylane/pull/794)
  [(#807)](https://github.com/PennyLaneAI/pennylane/pull/807)
  [(#819)](https://github.com/PennyLaneAI/pennylane/pull/819)

  The device can be initialized as
  ```pycon
  >>> dev = qml.device("default.mixed", wires=1)
  ```

  This allows the construction of QNodes that include non-unitary operations,
  such as noisy channels:

  ```pycon
  >>> @qml.qnode(dev)
  ... def circuit(params):
  ...     qml.RX(params[0], wires=0)
  ...     qml.RY(params[1], wires=0)
  ...     qml.AmplitudeDamping(0.5, wires=0)
  ...     return qml.expval(qml.PauliZ(0))
  >>> print(circuit([0.54, 0.12]))
  0.9257702929524184
  >>> print(circuit([0, np.pi]))
  0.0
  ```

<h4>New tools for optimizing measurements</h4>

* The new `grouping` module provides functionality for grouping simultaneously measurable Pauli word
  observables.
  [(#761)](https://github.com/PennyLaneAI/pennylane/pull/761)
  [(#850)](https://github.com/PennyLaneAI/pennylane/pull/850)
  [(#852)](https://github.com/PennyLaneAI/pennylane/pull/852)

  - The `optimize_measurements` function will take as input a list of Pauli word observables and
    their corresponding coefficients (if any), and will return the partitioned Pauli terms
    diagonalized in the measurement basis and the corresponding diagonalizing circuits.

    ```python
    from pennylane.grouping import optimize_measurements
    h, nr_qubits = qml.qchem.molecular_hamiltonian("h2", "h2.xyz")
    rotations, grouped_ops, grouped_coeffs = optimize_measurements(h.ops, h.coeffs, grouping="qwc")
    ```

    The diagonalizing circuits of `rotations` correspond to the diagonalized Pauli word groupings of
    `grouped_ops`.

  - Pauli word partitioning utilities are performed by the `PauliGroupingStrategy`
    class. An input list of Pauli words can be partitioned into mutually commuting,
    qubit-wise-commuting, or anticommuting groupings.

    For example, partitioning Pauli words into anticommutative groupings by the Recursive Largest
    First (RLF) graph colouring heuristic:

    ```python
    from pennylane import PauliX, PauliY, PauliZ, Identity
    from pennylane.grouping import group_observables
    pauli_words = [
        Identity('a') @ Identity('b'),
        Identity('a') @ PauliX('b'),
        Identity('a') @ PauliY('b'),
        PauliZ('a') @ PauliX('b'),
        PauliZ('a') @ PauliY('b'),
        PauliZ('a') @ PauliZ('b')
    ]
    groupings = group_observables(pauli_words, grouping_type='anticommuting', method='rlf')
    ```

  - Various utility functions are included for obtaining and manipulating Pauli
    words in the binary symplectic vector space representation.

    For instance, two Pauli words may be converted to their binary vector representation:

    ```pycon
    >>> from pennylane.grouping import pauli_to_binary
    >>> from pennylane.wires import Wires
    >>> wire_map = {Wires('a'): 0, Wires('b'): 1}
    >>> pauli_vec_1 = pauli_to_binary(qml.PauliX('a') @ qml.PauliY('b'))
    >>> pauli_vec_2 = pauli_to_binary(qml.PauliZ('a') @ qml.PauliZ('b'))
    >>> pauli_vec_1
    [1. 1. 0. 1.]
    >>> pauli_vec_2
    [0. 0. 1. 1.]
    ```

    Their product up to a phase may be computed by taking the sum of their binary vector
    representations, and returned in the operator representation.

    ```pycon
    >>> from pennylane.grouping import binary_to_pauli
    >>> binary_to_pauli((pauli_vec_1 + pauli_vec_2) % 2, wire_map)
    Tensor product ['PauliY', 'PauliX']: 0 params, wires ['a', 'b']
    ```

    For more details on the grouping module, see the
    [grouping module documentation](https://pennylane.readthedocs.io/en/stable/code/qml_grouping.html)


<h4>Returning the quantum state from simulators</h4>

* The quantum state of a QNode can now be returned using the `qml.state()` return function.
  [(#818)](https://github.com/XanaduAI/pennylane/pull/818)

  ```python
  import pennylane as qml

  dev = qml.device("default.qubit", wires=3)
  qml.enable_tape()

  @qml.qnode(dev)
  def qfunc(x, y):
      qml.RZ(x, wires=0)
      qml.CNOT(wires=[0, 1])
      qml.RY(y, wires=1)
      qml.CNOT(wires=[0, 2])
      return qml.state()

  >>> qfunc(0.56, 0.1)
  array([0.95985437-0.27601028j, 0.        +0.j        ,
         0.04803275-0.01381203j, 0.        +0.j        ,
         0.        +0.j        , 0.        +0.j        ,
         0.        +0.j        , 0.        +0.j        ])
  ```

  Differentiating the state is currently available when using the
  classical backpropagation differentiation method (`diff_method="backprop"`) with a compatible device,
  and when using the new tape mode.

<h4>New operations and channels</h4>

* PennyLane now includes standard channels such as the Amplitude-damping,
  Phase-damping, and Depolarizing channels, as well as the ability
  to make custom qubit channels.
  [(#760)](https://github.com/PennyLaneAI/pennylane/pull/760)
  [(#766)](https://github.com/PennyLaneAI/pennylane/pull/766)
  [(#778)](https://github.com/PennyLaneAI/pennylane/pull/778)

* The controlled-Y operation is now available via `qml.CY`. For devices that do
  not natively support the controlled-Y operation, it will be decomposed
  into `qml.RY`, `qml.CNOT`, and `qml.S` operations.
  [(#806)](https://github.com/PennyLaneAI/pennylane/pull/806)

<h4>Preview the next-generation PennyLane QNode</h4>

* The new PennyLane `tape` module provides a re-formulated QNode class, rewritten from the ground-up,
  that uses a new `QuantumTape` object to represent the QNode's quantum circuit. Tape mode
  provides several advantages over the standard PennyLane QNode.
  [(#785)](https://github.com/PennyLaneAI/pennylane/pull/785)
  [(#792)](https://github.com/PennyLaneAI/pennylane/pull/792)
  [(#796)](https://github.com/PennyLaneAI/pennylane/pull/796)
  [(#800)](https://github.com/PennyLaneAI/pennylane/pull/800)
  [(#803)](https://github.com/PennyLaneAI/pennylane/pull/803)
  [(#804)](https://github.com/PennyLaneAI/pennylane/pull/804)
  [(#805)](https://github.com/PennyLaneAI/pennylane/pull/805)
  [(#808)](https://github.com/PennyLaneAI/pennylane/pull/808)
  [(#810)](https://github.com/PennyLaneAI/pennylane/pull/810)
  [(#811)](https://github.com/PennyLaneAI/pennylane/pull/811)
  [(#815)](https://github.com/PennyLaneAI/pennylane/pull/815)
  [(#820)](https://github.com/PennyLaneAI/pennylane/pull/820)
  [(#823)](https://github.com/PennyLaneAI/pennylane/pull/823)
  [(#824)](https://github.com/PennyLaneAI/pennylane/pull/824)
  [(#829)](https://github.com/PennyLaneAI/pennylane/pull/829)

  - Support for in-QNode classical processing: Tape mode allows for differentiable classical
    processing within the QNode.

  - No more Variable wrapping: In tape mode, QNode arguments no longer become `Variable`
    objects within the QNode.

  - Less restrictive QNode signatures: There is no longer any restriction on the QNode signature;
    the QNode can be defined and called following the same rules as standard Python functions.

  - Unifying all QNodes: The tape-mode QNode merges all QNodes (including the
    `JacobianQNode` and the `PassthruQNode`) into a single unified QNode, with
    identical behaviour regardless of the differentiation type.

  - Optimizations: Tape mode provides various performance optimizations, reducing pre- and
    post-processing overhead, and reduces the number of quantum evaluations in certain cases.

  Note that tape mode is **experimental**, and does not currently have feature-parity with the
  existing QNode. [Feedback and bug reports](https://github.com/PennyLaneAI/pennylane/issues) are
  encouraged and will help improve the new tape mode.

  Tape mode can be enabled globally via the `qml.enable_tape` function, without changing your
  PennyLane code:

  ```python
  qml.enable_tape()
  dev = qml.device("default.qubit", wires=1)

  @qml.qnode(dev, interface="tf")
  def circuit(p):
      print("Parameter value:", p)
      qml.RX(tf.sin(p[0])**2 + p[1], wires=0)
      return qml.expval(qml.PauliZ(0))
  ```

  For more details, please see the [tape mode
  documentation](https://pennylane.readthedocs.io/en/stable/code/qml_tape.html).

<h3>Improvements</h3>

* QNode caching has been introduced, allowing the QNode to keep track of the results of previous
  device executions and reuse those results in subsequent calls.
  Note that QNode caching is only supported in the new and experimental tape-mode.
  [(#817)](https://github.com/PennyLaneAI/pennylane/pull/817)

  Caching is available by passing a `caching` argument to the QNode:

  ```python
  dev = qml.device("default.qubit", wires=2)
  qml.enable_tape()

  @qml.qnode(dev, caching=10)  # cache up to 10 evaluations
  def qfunc(x):
      qml.RX(x, wires=0)
      qml.RX(0.3, wires=1)
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.PauliZ(1))

  qfunc(0.1)  # first evaluation executes on the device
  qfunc(0.1)  # second evaluation accesses the cached result
  ```

* Sped up the application of certain gates in `default.qubit` by using array/tensor
  manipulation tricks. The following gates are affected: `PauliX`, `PauliY`, `PauliZ`,
  `Hadamard`, `SWAP`, `S`, `T`, `CNOT`, `CZ`.
  [(#772)](https://github.com/PennyLaneAI/pennylane/pull/772)

* The computation of marginal probabilities has been made more efficient for devices
  with a large number of wires, achieving in some cases a 5x speedup.
  [(#799)](https://github.com/PennyLaneAI/pennylane/pull/799)

* Adds arithmetic operations (addition, tensor product,
  subtraction, and scalar multiplication) between `Hamiltonian`,
  `Tensor`, and `Observable` objects, and inline arithmetic
  operations between Hamiltonians and other observables.
  [(#765)](https://github.com/PennyLaneAI/pennylane/pull/765)

  Hamiltonians can now easily be defined as sums of observables:

  ```pycon3
  >>> H = 3 * qml.PauliZ(0) - (qml.PauliX(0) @ qml.PauliX(1)) + qml.Hamiltonian([4], [qml.PauliZ(0)])
  >>> print(H)
  (7.0) [Z0] + (-1.0) [X0 X1]
  ```

* Adds `compare()` method to `Observable` and `Hamiltonian` classes, which allows
  for comparison between observable quantities.
  [(#765)](https://github.com/PennyLaneAI/pennylane/pull/765)

  ```pycon3
  >>> H = qml.Hamiltonian([1], [qml.PauliZ(0)])
  >>> obs = qml.PauliZ(0) @ qml.Identity(1)
  >>> print(H.compare(obs))
  True
  ```

  ```pycon3
  >>> H = qml.Hamiltonian([2], [qml.PauliZ(0)])
  >>> obs = qml.PauliZ(1) @ qml.Identity(0)
  >>> print(H.compare(obs))
  False
  ```

* Adds `simplify()` method to the `Hamiltonian` class.
  [(#765)](https://github.com/PennyLaneAI/pennylane/pull/765)

  ```pycon3
  >>> H = qml.Hamiltonian([1, 2], [qml.PauliZ(0), qml.PauliZ(0) @ qml.Identity(1)])
  >>> H.simplify()
  >>> print(H)
  (3.0) [Z0]
  ```

* Added a new bit-flip mixer to the `qml.qaoa` module.
  [(#774)](https://github.com/PennyLaneAI/pennylane/pull/774)

* Summation of two `Wires` objects is now supported and will return
  a `Wires` object containing the set of all wires defined by the
  terms in the summation.
  [(#812)](https://github.com/PennyLaneAI/pennylane/pull/812)

<h3>Breaking changes</h3>

* The PennyLane NumPy module now returns scalar (zero-dimensional) arrays where
  Python scalars were previously returned.
  [(#820)](https://github.com/PennyLaneAI/pennylane/pull/820)
  [(#833)](https://github.com/PennyLaneAI/pennylane/pull/833)

  For example, this affects array element indexing, and summation:

  ```pycon
  >>> x = np.array([1, 2, 3], requires_grad=False)
  >>> x[0]
  tensor(1, requires_grad=False)
  >>> np.sum(x)
  tensor(6, requires_grad=True)
  ```

  This may require small updates to user code. A convenience method, `np.tensor.unwrap()`,
  has been added to help ease the transition. This converts PennyLane NumPy tensors
  to standard NumPy arrays and Python scalars:

  ```pycon
  >>> x = np.array(1.543, requires_grad=False)
  >>> x.unwrap()
  1.543
  ```

  Note, however, that information regarding array differentiability will be
  lost.

* The device capabilities dictionary has been redesigned, for clarity and robustness. In particular,
  the capabilities dictionary is now inherited from the parent class, various keys have more
  expressive names, and all keys are now defined in the base device class. For more details, please
  [refer to the developer
  documentation](https://pennylane.readthedocs.io/en/stable/development/plugins.html#device-capabilities).
  [(#781)](https://github.com/PennyLaneAI/pennylane/pull/781/files)

<h3>Bug fixes</h3>

* Changed to use lists for storing variable values inside `BaseQNode`
  allowing complex matrices to be passed to `QubitUnitary`.
  [(#773)](https://github.com/PennyLaneAI/pennylane/pull/773)

* Fixed a bug within `default.qubit`, resulting in greater efficiency
  when applying a state vector to all wires on the device.
  [(#849)](https://github.com/PennyLaneAI/pennylane/pull/849)

<h3>Documentation</h3>

* Equations have been added to the `qml.sample` and `qml.probs` docstrings
  to clarify the mathematical foundation of the performed measurements.
  [(#843)](https://github.com/PennyLaneAI/pennylane/pull/843)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Aroosa Ijaz, Juan Miguel Arrazola, Thomas Bromley, Jack Ceroni, Alain Delgado Gran, Josh Izaac,
Soran Jahangiri, Nathan Killoran, Robert Lang, Cedric Lin, Olivia Di Matteo, Nicolás Quesada, Maria
Schuld, Antal Száva.

# Release 0.11.0

<h3>New features since last release</h3>

<h4>New and improved simulators</h4>

* Added a new device, `default.qubit.autograd`, a pure-state qubit simulator written using Autograd.
  This device supports classical backpropagation (`diff_method="backprop"`); this can
  be faster than the parameter-shift rule for computing quantum gradients
  when the number of parameters to be optimized is large.
  [(#721)](https://github.com/XanaduAI/pennylane/pull/721)

  ```pycon
  >>> dev = qml.device("default.qubit.autograd", wires=1)
  >>> @qml.qnode(dev, diff_method="backprop")
  ... def circuit(x):
  ...     qml.RX(x[1], wires=0)
  ...     qml.Rot(x[0], x[1], x[2], wires=0)
  ...     return qml.expval(qml.PauliZ(0))
  >>> weights = np.array([0.2, 0.5, 0.1])
  >>> grad_fn = qml.grad(circuit)
  >>> print(grad_fn(weights))
  array([-2.25267173e-01, -1.00864546e+00,  6.93889390e-18])
  ```

  See the [device documentation](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.devices.default_qubit_autograd.DefaultQubitAutograd.html) for more details.

* A new experimental C++ state-vector simulator device is now available, `lightning.qubit`. It
  uses the C++ Eigen library to perform fast linear algebra calculations for simulating quantum
  state-vector evolution.

  `lightning.qubit` is currently in beta; it can be installed via `pip`:

  ```console
  $ pip install pennylane-lightning
  ```

  Once installed, it can be used as a PennyLane device:

  ```pycon
  >>> dev = qml.device("lightning.qubit", wires=2)
  ```

  For more details, please see the [lightning qubit documentation](https://pennylane-lightning.readthedocs.io).

<h4>New algorithms and templates</h4>

* Added built-in QAOA functionality via the new `qml.qaoa` module.
  [(#712)](https://github.com/PennyLaneAI/pennylane/pull/712)
  [(#718)](https://github.com/PennyLaneAI/pennylane/pull/718)
  [(#741)](https://github.com/PennyLaneAI/pennylane/pull/741)
  [(#720)](https://github.com/PennyLaneAI/pennylane/pull/720)

  This includes the following features:

  * New `qml.qaoa.x_mixer` and `qml.qaoa.xy_mixer` functions for defining Pauli-X and XY
    mixer Hamiltonians.

  * MaxCut: The `qml.qaoa.maxcut` function allows easy construction of the cost Hamiltonian
    and recommended mixer Hamiltonian for solving the MaxCut problem for a supplied graph.

  * Layers: `qml.qaoa.cost_layer` and `qml.qaoa.mixer_layer` take cost and mixer
    Hamiltonians, respectively, and apply the corresponding QAOA cost and mixer layers
    to the quantum circuit

  For example, using PennyLane to construct and solve a MaxCut problem with QAOA:

  ```python
  wires = range(3)
  graph = Graph([(0, 1), (1, 2), (2, 0)])
  cost_h, mixer_h = qaoa.maxcut(graph)

  def qaoa_layer(gamma, alpha):
      qaoa.cost_layer(gamma, cost_h)
      qaoa.mixer_layer(alpha, mixer_h)

  def antatz(params, **kwargs):

      for w in wires:
          qml.Hadamard(wires=w)

      # repeat the QAOA layer two times
      qml.layer(qaoa_layer, 2, params[0], params[1])

  dev = qml.device('default.qubit', wires=len(wires))
  cost_function = qml.VQECost(ansatz, cost_h, dev)
  ```

* Added an `ApproxTimeEvolution` template to the PennyLane templates module, which
  can be used to implement Trotterized time-evolution under a Hamiltonian.
  [(#710)](https://github.com/XanaduAI/pennylane/pull/710)

  <img src="https://pennylane.readthedocs.io/en/latest/_static/templates/subroutines/approx_time_evolution.png" width=50%/>

* Added a `qml.layer` template-constructing function, which takes a unitary, and
  repeatedly applies it on a set of wires to a given depth.
  [(#723)](https://github.com/PennyLaneAI/pennylane/pull/723)

  ```python
  def subroutine():
      qml.Hadamard(wires=[0])
      qml.CNOT(wires=[0, 1])
      qml.PauliX(wires=[1])

  dev = qml.device('default.qubit', wires=3)

  @qml.qnode(dev)
  def circuit():
      qml.layer(subroutine, 3)
      return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
  ```

  This creates the following circuit:
  ```pycon
  >>> circuit()
  >>> print(circuit.draw())
  0: ──H──╭C──X──H──╭C──X──H──╭C──X──┤ ⟨Z⟩
  1: ─────╰X────────╰X────────╰X─────┤ ⟨Z⟩
  ```

* Added the `qml.utils.decompose_hamiltonian` function. This function can be used to
  decompose a Hamiltonian into a linear combination of Pauli operators.
  [(#671)](https://github.com/XanaduAI/pennylane/pull/671)

  ```pycon
  >>> A = np.array(
  ... [[-2, -2+1j, -2, -2],
  ... [-2-1j,  0,  0, -1],
  ... [-2,  0, -2, -1],
  ... [-2, -1, -1,  0]])
  >>> coeffs, obs_list = decompose_hamiltonian(A)
  ```

<h4>New device features</h4>

* It is now possible to specify custom wire labels, such as `['anc1', 'anc2', 0, 1, 3]`, where the labels
  can be strings or numbers.
  [(#666)](https://github.com/XanaduAI/pennylane/pull/666)

  Custom wire labels are defined by passing a list to the `wires` argument when creating the device:

  ```pycon
  >>> dev = qml.device("default.qubit", wires=['anc1', 'anc2', 0, 1, 3])
  ```

  Quantum operations should then be invoked with these custom wire labels:

  ``` pycon
  >>> @qml.qnode(dev)
  >>> def circuit():
  ...    qml.Hadamard(wires='anc2')
  ...    qml.CNOT(wires=['anc1', 3])
  ...    ...
  ```

  The existing behaviour, in which the number of wires is specified on device initialization,
  continues to work as usual. This gives a default behaviour where wires are labelled
  by consecutive integers.

  ```pycon
  >>> dev = qml.device("default.qubit", wires=5)
  ```

* An integrated device test suite has been added, which can be used
  to run basic integration tests on core or external devices.
  [(#695)](https://github.com/PennyLaneAI/pennylane/pull/695)
  [(#724)](https://github.com/PennyLaneAI/pennylane/pull/724)
  [(#733)](https://github.com/PennyLaneAI/pennylane/pull/733)

  The test can be invoked against a particular device by calling the `pl-device-test`
  command line program:

  ```console
  $ pl-device-test --device=default.qubit --shots=1234 --analytic=False
  ```

  If the tests are run on external devices, the device and its dependencies must be
  installed locally. For more details, please see the
  [plugin test documentation](http://pennylane.readthedocs.io/en/latest/code/api/pennylane.devices.tests.html).

<h3>Improvements</h3>

* The functions implementing the quantum circuits building the Unitary Coupled-Cluster
  (UCCSD) VQE ansatz have been improved, with a more consistent naming convention and
  improved docstrings.
  [(#748)](https://github.com/PennyLaneAI/pennylane/pull/748)

  The changes include:

  - The terms *1particle-1hole (ph)* and *2particle-2hole (pphh)* excitations
    were replaced with the names *single* and *double* excitations, respectively.

  - The non-differentiable arguments in the `UCCSD` template were renamed accordingly:
    `ph` → `s_wires`, `pphh` → `d_wires`

  - The term *virtual*, previously used to refer the *unoccupied* orbitals, was discarded.

  - The Usage Details sections were updated and improved.

* Added support for TensorFlow 2.3 and PyTorch 1.6.
  [(#725)](https://github.com/PennyLaneAI/pennylane/pull/725)

* Returning probabilities is now supported from photonic QNodes.
  As with qubit QNodes, photonic QNodes returning probabilities are
  end-to-end differentiable.
  [(#699)](https://github.com/XanaduAI/pennylane/pull/699/)

  ```pycon
  >>> dev = qml.device("strawberryfields.fock", wires=2, cutoff_dim=5)
  >>> @qml.qnode(dev)
  ... def circuit(a):
  ...     qml.Displacement(a, 0, wires=0)
  ...     return qml.probs(wires=0)
  >>> print(circuit(0.5))
  [7.78800783e-01 1.94700196e-01 2.43375245e-02 2.02812704e-03 1.26757940e-04]
  ```

<h3>Breaking changes</h3>

* The `pennylane.plugins` and `pennylane.beta.plugins` folders have been renamed to
  `pennylane.devices` and `pennylane.beta.devices`, to reflect their content better.
  [(#726)](https://github.com/XanaduAI/pennylane/pull/726)

<h3>Bug fixes</h3>

* The PennyLane interface conversion functions can now convert QNodes with
  pre-existing interfaces.
  [(#707)](https://github.com/XanaduAI/pennylane/pull/707)

<h3>Documentation</h3>

* The interfaces section of the documentation has been renamed to 'Interfaces and training',
  and updated with the latest variable handling details.
  [(#753)](https://github.com/PennyLaneAI/pennylane/pull/753)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Thomas Bromley, Jack Ceroni, Alain Delgado Gran, Shadab Hussain, Theodor
Isacsson, Josh Izaac, Nathan Killoran, Maria Schuld, Antal Száva, Nicola Vitucci.

# Release 0.10.0

<h3>New features since last release</h3>

<h4>New and improved simulators</h4>

* Added a new device, `default.qubit.tf`, a pure-state qubit simulator written using TensorFlow.
  As a result, it supports classical backpropagation as a means to compute the Jacobian. This can
  be faster than the parameter-shift rule for computing quantum gradients
  when the number of parameters to be optimized is large.

  `default.qubit.tf` is designed to be used with end-to-end classical backpropagation
  (`diff_method="backprop"`) with the TensorFlow interface. This is the default method
  of differentiation when creating a QNode with this device.

  Using this method, the created QNode is a 'white-box' that is
  tightly integrated with your TensorFlow computation, including
  [AutoGraph](https://www.tensorflow.org/guide/function) support:

  ```pycon
  >>> dev = qml.device("default.qubit.tf", wires=1)
  >>> @tf.function
  ... @qml.qnode(dev, interface="tf", diff_method="backprop")
  ... def circuit(x):
  ...     qml.RX(x[1], wires=0)
  ...     qml.Rot(x[0], x[1], x[2], wires=0)
  ...     return qml.expval(qml.PauliZ(0))
  >>> weights = tf.Variable([0.2, 0.5, 0.1])
  >>> with tf.GradientTape() as tape:
  ...     res = circuit(weights)
  >>> print(tape.gradient(res, weights))
  tf.Tensor([-2.2526717e-01 -1.0086454e+00  1.3877788e-17], shape=(3,), dtype=float32)
  ```

  See the `default.qubit.tf`
  [documentation](https://pennylane.ai/en/stable/code/api/pennylane.beta.plugins.DefaultQubitTF.html)
  for more details.

* The [default.tensor plugin](https://github.com/XanaduAI/pennylane/blob/master/pennylane/beta/plugins/default_tensor.py)
  has been significantly upgraded. It now allows two different
  tensor network representations to be used: `"exact"` and `"mps"`. The former uses a
  exact factorized representation of quantum states, while the latter uses a matrix product state
  representation.
  ([#572](https://github.com/XanaduAI/pennylane/pull/572))
  ([#599](https://github.com/XanaduAI/pennylane/pull/599))

<h4>New machine learning functionality and integrations</h4>

* PennyLane QNodes can now be converted into Torch layers, allowing for creation of quantum and
  hybrid models using the `torch.nn` API.
  [(#588)](https://github.com/XanaduAI/pennylane/pull/588)

  A PennyLane QNode can be converted into a `torch.nn` layer using the `qml.qnn.TorchLayer` class:

  ```pycon
  >>> @qml.qnode(dev)
  ... def qnode(inputs, weights_0, weight_1):
  ...    # define the circuit
  ...    # ...

  >>> weight_shapes = {"weights_0": 3, "weight_1": 1}
  >>> qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
  ```

  A hybrid model can then be easily constructed:

  ```pycon
  >>> model = torch.nn.Sequential(qlayer, torch.nn.Linear(2, 2))
  ```

* Added a new "reversible" differentiation method which can be used in simulators, but not hardware.

  The reversible approach is similar to backpropagation, but trades off extra computation for
  enhanced memory efficiency. Where backpropagation caches the state tensors at each step during
  a simulated evolution, the reversible method only caches the final pre-measurement state.

  Compared to the parameter-shift method, the reversible method can be faster or slower,
  depending on the density and location of parametrized gates in a circuit
  (circuits with higher density of parametrized gates near the end of the circuit will see a benefit).
  [(#670)](https://github.com/XanaduAI/pennylane/pull/670)

  ```pycon
  >>> dev = qml.device("default.qubit", wires=2)
  ... @qml.qnode(dev, diff_method="reversible")
  ... def circuit(x):
  ...     qml.RX(x, wires=0)
  ...     qml.RX(x, wires=0)
  ...     qml.CNOT(wires=[0,1])
  ...     return qml.expval(qml.PauliZ(0))
  >>> qml.grad(circuit)(0.5)
  (array(-0.47942554),)
  ```

<h4>New templates and cost functions</h4>

* Added the new templates `UCCSD`, `SingleExcitationUnitary`, and`DoubleExcitationUnitary`,
  which together implement the Unitary Coupled-Cluster Singles and Doubles (UCCSD) ansatz
  to perform VQE-based quantum chemistry simulations using PennyLane-QChem.
  [(#622)](https://github.com/XanaduAI/pennylane/pull/622)
  [(#638)](https://github.com/XanaduAI/pennylane/pull/638)
  [(#654)](https://github.com/XanaduAI/pennylane/pull/654)
  [(#659)](https://github.com/XanaduAI/pennylane/pull/659)
  [(#622)](https://github.com/XanaduAI/pennylane/pull/622)

* Added module `pennylane.qnn.cost` with class `SquaredErrorLoss`. The module contains classes
  to calculate losses and cost functions on circuits with trainable parameters.
  [(#642)](https://github.com/XanaduAI/pennylane/pull/642)

<h3>Improvements</h3>

* Improves the wire management by making the `Operator.wires` attribute a `wires` object.
  [(#666)](https://github.com/XanaduAI/pennylane/pull/666)

* A significant improvement with respect to how QNodes and interfaces mark quantum function
  arguments as differentiable when using Autograd, designed to improve performance and make
  QNodes more intuitive.
  [(#648)](https://github.com/XanaduAI/pennylane/pull/648)
  [(#650)](https://github.com/XanaduAI/pennylane/pull/650)

  In particular, the following changes have been made:

  - A new `ndarray` subclass `pennylane.numpy.tensor`, which extends NumPy arrays with
    the keyword argument and attribute `requires_grad`. Tensors which have `requires_grad=False`
    are treated as non-differentiable by the Autograd interface.

  - A new subpackage `pennylane.numpy`, which wraps `autograd.numpy` such that NumPy functions
    accept the `requires_grad` keyword argument, and allows Autograd to differentiate
    `pennylane.numpy.tensor` objects.

  - The `argnum` argument to `qml.grad` is now optional; if not provided, arguments explicitly
    marked as `requires_grad=False` are excluded for the list of differentiable arguments.
    The ability to pass `argnum` has been retained for backwards compatibility, and
    if present the old behaviour persists.

* The QNode Torch interface now inspects QNode positional arguments.
  If any argument does not have the attribute `requires_grad=True`, it
  is automatically excluded from quantum gradient computations.
  [(#652)](https://github.com/XanaduAI/pennylane/pull/652)
  [(#660)](https://github.com/XanaduAI/pennylane/pull/660)

* The QNode TF interface now inspects QNode positional arguments.
  If any argument is not being watched by a `tf.GradientTape()`,
  it is automatically excluded from quantum gradient computations.
  [(#655)](https://github.com/XanaduAI/pennylane/pull/655)
  [(#660)](https://github.com/XanaduAI/pennylane/pull/660)

* QNodes have two new public methods: `QNode.set_trainable_args()` and `QNode.get_trainable_args()`.
  These are designed to be called by interfaces, to specify to the QNode which of its
  input arguments are differentiable. Arguments which are non-differentiable will not be converted
  to PennyLane Variable objects within the QNode.
  [(#660)](https://github.com/XanaduAI/pennylane/pull/660)

* Added `decomposition` method to PauliX, PauliY, PauliZ, S, T, Hadamard, and PhaseShift gates, which
  decomposes each of these gates into rotation gates.
  [(#668)](https://github.com/XanaduAI/pennylane/pull/668)

* The `CircuitGraph` class now supports serializing contained circuit operations
  and measurement basis rotations to an OpenQASM2.0 script via the new
  `CircuitGraph.to_openqasm()` method.
  [(#623)](https://github.com/XanaduAI/pennylane/pull/623)

<h3>Breaking changes</h3>

* Removes support for Python 3.5.
  [(#639)](https://github.com/XanaduAI/pennylane/pull/639)

<h3>Documentation</h3>

* Various small typos were fixed.

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Thomas Bromley, Jack Ceroni, Alain Delgado Gran, Theodor Isacsson, Josh Izaac,
Nathan Killoran, Maria Schuld, Antal Száva, Nicola Vitucci.


# Release 0.9.0

<h3>New features since last release</h3>

<h4>New machine learning integrations</h4>

* PennyLane QNodes can now be converted into Keras layers, allowing for creation of quantum and
  hybrid models using the Keras API.
  [(#529)](https://github.com/XanaduAI/pennylane/pull/529)

  A PennyLane QNode can be converted into a Keras layer using the `KerasLayer` class:

  ```python
  from pennylane.qnn import KerasLayer

  @qml.qnode(dev)
  def circuit(inputs, weights_0, weight_1):
     # define the circuit
     # ...

  weight_shapes = {"weights_0": 3, "weight_1": 1}
  qlayer = qml.qnn.KerasLayer(circuit, weight_shapes, output_dim=2)
  ```

  A hybrid model can then be easily constructed:

  ```python
  model = tf.keras.models.Sequential([qlayer, tf.keras.layers.Dense(2)])
  ```

* Added a new type of QNode, `qml.qnodes.PassthruQNode`. For simulators which are coded in an
  external library which supports automatic differentiation, PennyLane will treat a PassthruQNode as
  a "white box", and rely on the external library to directly provide gradients via backpropagation.
  This can be more efficient than the using parameter-shift rule for a large number of parameters.
  [(#488)](https://github.com/XanaduAI/pennylane/pull/488)

  Currently this behaviour is supported by PennyLane's `default.tensor.tf` device backend,
  compatible with the `'tf'` interface using TensorFlow 2:

  ```python
  dev = qml.device('default.tensor.tf', wires=2)

  @qml.qnode(dev, diff_method="backprop")
  def circuit(params):
      qml.RX(params[0], wires=0)
      qml.RX(params[1], wires=1)
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.PauliZ(0))

  qnode = PassthruQNode(circuit, dev)
  params = tf.Variable([0.3, 0.1])

  with tf.GradientTape() as tape:
      tape.watch(params)
      res = qnode(params)

  grad = tape.gradient(res, params)
  ```

<h4>New optimizers</h4>

* Added the `qml.RotosolveOptimizer`, a gradient-free optimizer
  that minimizes the quantum function by updating each parameter,
  one-by-one, via a closed-form expression while keeping other parameters
  fixed.
  [(#636)](https://github.com/XanaduAI/pennylane/pull/636)
  [(#539)](https://github.com/XanaduAI/pennylane/pull/539)

* Added the `qml.RotoselectOptimizer`, which uses Rotosolve to
  minimizes a quantum function with respect to both the
  rotation operations applied and the rotation parameters.
  [(#636)](https://github.com/XanaduAI/pennylane/pull/636)
  [(#539)](https://github.com/XanaduAI/pennylane/pull/539)

  For example, given a quantum function `f` that accepts parameters `x`
  and a list of corresponding rotation operations `generators`,
  the Rotoselect optimizer will, at each step, update both the parameter
  values and the list of rotation gates to minimize the loss:

  ```pycon
  >>> opt = qml.optimize.RotoselectOptimizer()
  >>> x = [0.3, 0.7]
  >>> generators = [qml.RX, qml.RY]
  >>> for _ in range(100):
  ...     x, generators = opt.step(f, x, generators)
  ```


<h4>New operations</h4>

* Added the `PauliRot` gate, which performs an arbitrary
  Pauli rotation on multiple qubits, and the `MultiRZ` gate,
  which performs a rotation generated by a tensor product
  of Pauli Z operators.
  [(#559)](https://github.com/XanaduAI/pennylane/pull/559)

  ```python
  dev = qml.device('default.qubit', wires=4)

  @qml.qnode(dev)
  def circuit(angle):
      qml.PauliRot(angle, "IXYZ", wires=[0, 1, 2, 3])
      return [qml.expval(qml.PauliZ(wire)) for wire in [0, 1, 2, 3]]
  ```

  ```pycon
  >>> circuit(0.4)
  [1.         0.92106099 0.92106099 1.        ]
  >>> print(circuit.draw())
   0: ──╭RI(0.4)──┤ ⟨Z⟩
   1: ──├RX(0.4)──┤ ⟨Z⟩
   2: ──├RY(0.4)──┤ ⟨Z⟩
   3: ──╰RZ(0.4)──┤ ⟨Z⟩
  ```

  If the `PauliRot` gate is not supported on the target device, it will
  be decomposed into `Hadamard`, `RX` and `MultiRZ` gates. Note that
  identity gates in the Pauli word result in untouched wires:

  ```pycon
  >>> print(circuit.draw())
   0: ───────────────────────────────────┤ ⟨Z⟩
   1: ──H──────────╭RZ(0.4)──H───────────┤ ⟨Z⟩
   2: ──RX(1.571)──├RZ(0.4)──RX(-1.571)──┤ ⟨Z⟩
   3: ─────────────╰RZ(0.4)──────────────┤ ⟨Z⟩
  ```

  If the `MultiRZ` gate is not supported, it will be decomposed into
  `CNOT` and `RZ` gates:

  ```pycon
  >>> print(circuit.draw())
   0: ──────────────────────────────────────────────────┤ ⟨Z⟩
   1: ──H──────────────╭X──RZ(0.4)──╭X──────H───────────┤ ⟨Z⟩
   2: ──RX(1.571)──╭X──╰C───────────╰C──╭X──RX(-1.571)──┤ ⟨Z⟩
   3: ─────────────╰C───────────────────╰C──────────────┤ ⟨Z⟩
  ```

* PennyLane now provides `DiagonalQubitUnitary` for diagonal gates, that are e.g.,
  encountered in IQP circuits. These kinds of gates can be evaluated much faster on
  a simulator device.
  [(#567)](https://github.com/XanaduAI/pennylane/pull/567)

  The gate can be used, for example, to efficiently simulate oracles:

  ```python
  dev = qml.device('default.qubit', wires=3)

  # Function as a bitstring
  f = np.array([1, 0, 0, 1, 1, 0, 1, 0])

  @qml.qnode(dev)
  def circuit(weights1, weights2):
      qml.templates.StronglyEntanglingLayers(weights1, wires=[0, 1, 2])

      # Implements the function as a phase-kickback oracle
      qml.DiagonalQubitUnitary((-1)**f, wires=[0, 1, 2])

      qml.templates.StronglyEntanglingLayers(weights2, wires=[0, 1, 2])
      return [qml.expval(qml.PauliZ(w)) for w in range(3)]
  ```

* Added the `TensorN` CVObservable that can represent the tensor product of the
  `NumberOperator` on photonic backends.
  [(#608)](https://github.com/XanaduAI/pennylane/pull/608)

<h4>New templates</h4>

* Added the `ArbitraryUnitary` and `ArbitraryStatePreparation` templates, which use
  `PauliRot` gates to perform an arbitrary unitary and prepare an arbitrary basis
  state with the minimal number of parameters.
  [(#590)](https://github.com/XanaduAI/pennylane/pull/590)

  ```python
  dev = qml.device('default.qubit', wires=3)

  @qml.qnode(dev)
  def circuit(weights1, weights2):
        qml.templates.ArbitraryStatePreparation(weights1, wires=[0, 1, 2])
        qml.templates.ArbitraryUnitary(weights2, wires=[0, 1, 2])
        return qml.probs(wires=[0, 1, 2])
  ```

* Added the `IQPEmbedding` template, which encodes inputs into the diagonal gates of an
  IQP circuit.
  [(#605)](https://github.com/XanaduAI/pennylane/pull/605)

  <img src="https://pennylane.readthedocs.io/en/latest/_images/iqp.png"
  width=50%></img>

* Added the `SimplifiedTwoDesign` template, which implements the circuit
  design of [Cerezo et al. (2020)](<https://arxiv.org/abs/2001.00550>).
  [(#556)](https://github.com/XanaduAI/pennylane/pull/556)

  <img src="https://pennylane.readthedocs.io/en/latest/_images/simplified_two_design.png"
  width=50%></img>

* Added the `BasicEntanglerLayers` template, which is a simple layer architecture
  of rotations and CNOT nearest-neighbour entanglers.
  [(#555)](https://github.com/XanaduAI/pennylane/pull/555)

  <img src="https://pennylane.readthedocs.io/en/latest/_images/basic_entangler.png"
  width=50%></img>

* PennyLane now offers a broadcasting function to easily construct templates:
  `qml.broadcast()` takes single quantum operations or other templates and applies
  them to wires in a specific pattern.
  [(#515)](https://github.com/XanaduAI/pennylane/pull/515)
  [(#522)](https://github.com/XanaduAI/pennylane/pull/522)
  [(#526)](https://github.com/XanaduAI/pennylane/pull/526)
  [(#603)](https://github.com/XanaduAI/pennylane/pull/603)

  For example, we can use broadcast to repeat a custom template
  across multiple wires:

  ```python
  from pennylane.templates import template

  @template
  def mytemplate(pars, wires):
      qml.Hadamard(wires=wires)
      qml.RY(pars, wires=wires)

  dev = qml.device('default.qubit', wires=3)

  @qml.qnode(dev)
  def circuit(pars):
      qml.broadcast(mytemplate, pattern="single", wires=[0,1,2], parameters=pars)
      return qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> circuit([1, 1, 0.1])
  -0.841470984807896
  >>> print(circuit.draw())
   0: ──H──RY(1.0)──┤ ⟨Z⟩
   1: ──H──RY(1.0)──┤
   2: ──H──RY(0.1)──┤
  ```

  For other available patterns, see the
  [broadcast function documentation](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.broadcast.html).

<h3>Breaking changes</h3>

* The `QAOAEmbedding` now uses the new `MultiRZ` gate as a `ZZ` entangler,
  which changes the convention. While
  previously, the `ZZ` gate in the embedding was implemented as

  ```python
  CNOT(wires=[wires[0], wires[1]])
  RZ(2 * parameter, wires=wires[0])
  CNOT(wires=[wires[0], wires[1]])
  ```

  the `MultiRZ` corresponds to

  ```python
  CNOT(wires=[wires[1], wires[0]])
  RZ(parameter, wires=wires[0])
  CNOT(wires=[wires[1], wires[0]])
  ```

  which differs in the factor of `2`, and fixes a bug in the
  wires that the `CNOT` was applied to.
  [(#609)](https://github.com/XanaduAI/pennylane/pull/609)

* Probability methods are handled by `QubitDevice` and device method
  requirements are modified to simplify plugin development.
  [(#573)](https://github.com/XanaduAI/pennylane/pull/573)

* The internal variables `All` and `Any` to mark an `Operation` as acting on all or any
  wires have been renamed to `AllWires` and `AnyWires`.
  [(#614)](https://github.com/XanaduAI/pennylane/pull/614)

<h3>Improvements</h3>

* A new `Wires` class was introduced for the internal
  bookkeeping of wire indices.
  [(#615)](https://github.com/XanaduAI/pennylane/pull/615)

* Improvements to the speed/performance of the `default.qubit` device.
  [(#567)](https://github.com/XanaduAI/pennylane/pull/567)
  [(#559)](https://github.com/XanaduAI/pennylane/pull/559)

* Added the `"backprop"` and `"device"` differentiation methods to the `qnode`
  decorator.
  [(#552)](https://github.com/XanaduAI/pennylane/pull/552)

  - `"backprop"`: Use classical backpropagation. Default on simulator
    devices that are classically end-to-end differentiable.
    The returned QNode can only be used with the same machine learning
    framework (e.g., `default.tensor.tf` simulator with the `tensorflow` interface).

  - `"device"`: Queries the device directly for the gradient.

  Using the `"backprop"` differentiation method with the `default.tensor.tf`
  device, the created QNode is a 'white-box', and is tightly integrated with
  the overall TensorFlow computation:

  ```python
  >>> dev = qml.device("default.tensor.tf", wires=1)
  >>> @qml.qnode(dev, interface="tf", diff_method="backprop")
  >>> def circuit(x):
  ...     qml.RX(x[1], wires=0)
  ...     qml.Rot(x[0], x[1], x[2], wires=0)
  ...     return qml.expval(qml.PauliZ(0))
  >>> vars = tf.Variable([0.2, 0.5, 0.1])
  >>> with tf.GradientTape() as tape:
  ...     res = circuit(vars)
  >>> tape.gradient(res, vars)
  <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-2.2526717e-01, -1.0086454e+00,  1.3877788e-17], dtype=float32)>
  ```

* The circuit drawer now displays inverted operations, as well as wires
  where probabilities are returned from the device:
  [(#540)](https://github.com/XanaduAI/pennylane/pull/540)

  ```python
  >>> @qml.qnode(dev)
  ... def circuit(theta):
  ...     qml.RX(theta, wires=0)
  ...     qml.CNOT(wires=[0, 1])
  ...     qml.S(wires=1).inv()
  ...     return qml.probs(wires=[0, 1])
  >>> circuit(0.2)
  array([0.99003329, 0.        , 0.        , 0.00996671])
  >>> print(circuit.draw())
  0: ──RX(0.2)──╭C───────╭┤ Probs
  1: ───────────╰X──S⁻¹──╰┤ Probs
  ```

* You can now evaluate the metric tensor of a VQE Hamiltonian via the new
  `VQECost.metric_tensor` method. This allows `VQECost` objects to be directly
  optimized by the quantum natural gradient optimizer (`qml.QNGOptimizer`).
  [(#618)](https://github.com/XanaduAI/pennylane/pull/618)

* The input check functions in `pennylane.templates.utils` are now public
  and visible in the API documentation.
  [(#566)](https://github.com/XanaduAI/pennylane/pull/566)

* Added keyword arguments for step size and order to the `qnode` decorator, as well as
  the `QNode` and `JacobianQNode` classes. This enables the user to set the step size
  and order when using finite difference methods. These options are also exposed when
  creating QNode collections.
  [(#530)](https://github.com/XanaduAI/pennylane/pull/530)
  [(#585)](https://github.com/XanaduAI/pennylane/pull/585)
  [(#587)](https://github.com/XanaduAI/pennylane/pull/587)

* The decomposition for the `CRY` gate now uses the simpler form `RY @ CNOT @ RY @ CNOT`
  [(#547)](https://github.com/XanaduAI/pennylane/pull/547)

* The underlying queuing system was refactored, removing the `qml._current_context`
  property that held the currently active `QNode` or `OperationRecorder`. Now, all
  objects that expose a queue for operations inherit from `QueuingContext` and
  register their queue globally.
  [(#548)](https://github.com/XanaduAI/pennylane/pull/548)

* The PennyLane repository has a new benchmarking tool which supports the comparison of different git revisions.
  [(#568)](https://github.com/XanaduAI/pennylane/pull/568)
  [(#560)](https://github.com/XanaduAI/pennylane/pull/560)
  [(#516)](https://github.com/XanaduAI/pennylane/pull/516)

<h3>Documentation</h3>

* Updated the development section by creating a landing page with links to sub-pages
  containing specific guides.
  [(#596)](https://github.com/XanaduAI/pennylane/pull/596)

* Extended the developer's guide by a section explaining how to add new templates.
  [(#564)](https://github.com/XanaduAI/pennylane/pull/564)

<h3>Bug fixes</h3>

* `tf.GradientTape().jacobian()` can now be evaluated on QNodes using the TensorFlow interface.
  [(#626)](https://github.com/XanaduAI/pennylane/pull/626)

* `RandomLayers()` is now compatible with the qiskit devices.
  [(#597)](https://github.com/XanaduAI/pennylane/pull/597)

* `DefaultQubit.probability()` now returns the correct probability when called with
  `device.analytic=False`.
  [(#563)](https://github.com/XanaduAI/pennylane/pull/563)

* Fixed a bug in the `StronglyEntanglingLayers` template, allowing it to
  work correctly when applied to a single wire.
  [(544)](https://github.com/XanaduAI/pennylane/pull/544)

* Fixed a bug when inverting operations with decompositions; operations marked as inverted
  are now correctly inverted when the fallback decomposition is called.
  [(#543)](https://github.com/XanaduAI/pennylane/pull/543)

* The `QNode.print_applied()` method now correctly displays wires where
  `qml.prob()` is being returned.
  [#542](https://github.com/XanaduAI/pennylane/pull/542)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ville Bergholm, Lana Bozanic, Thomas Bromley, Theodor Isacsson, Josh Izaac, Nathan Killoran,
Maggie Li, Johannes Jakob Meyer, Maria Schuld, Sukin Sim, Antal Száva.

# Release 0.8.1

<h3>Improvements</h3>

* Beginning of support for Python 3.8, with the test suite
  now being run in a Python 3.8 environment.
  [(#501)](https://github.com/XanaduAI/pennylane/pull/501)

<h3>Documentation</h3>

* Present templates as a gallery of thumbnails showing the
  basic circuit architecture.
  [(#499)](https://github.com/XanaduAI/pennylane/pull/499)

<h3>Bug fixes</h3>

* Fixed a bug where multiplying a QNode parameter by 0 caused a divide
  by zero error when calculating the parameter shift formula.
  [(#512)](https://github.com/XanaduAI/pennylane/pull/512)

* Fixed a bug where the shape of differentiable QNode arguments
  was being cached on the first construction, leading to indexing
  errors if the QNode was re-evaluated if the argument changed shape.
  [(#505)](https://github.com/XanaduAI/pennylane/pull/505)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ville Bergholm, Josh Izaac, Johannes Jakob Meyer, Maria Schuld, Antal Száva.

# Release 0.8.0

<h3>New features since last release</h3>

* Added a quantum chemistry package, `pennylane.qchem`, which supports
  integration with OpenFermion, Psi4, PySCF, and OpenBabel.
  [(#453)](https://github.com/XanaduAI/pennylane/pull/453)

  Features include:

  - Generate the qubit Hamiltonians directly starting with the atomic structure of the molecule.
  - Calculate the mean-field (Hartree-Fock) electronic structure of molecules.
  - Allow to define an active space based on the number of active electrons and active orbitals.
  - Perform the fermionic-to-qubit transformation of the electronic Hamiltonian by
    using different functions implemented in OpenFermion.
  - Convert OpenFermion's QubitOperator to a Pennylane `Hamiltonian` class.
  - Perform a Variational Quantum Eigensolver (VQE) computation with this Hamiltonian in PennyLane.

  Check out the [quantum chemistry quickstart](https://pennylane.readthedocs.io/en/latest/introduction/chemistry.html), as well the quantum chemistry and VQE tutorials.

* PennyLane now has some functions and classes for creating and solving VQE
  problems. [(#467)](https://github.com/XanaduAI/pennylane/pull/467)

  - `qml.Hamiltonian`: a lightweight class for representing qubit Hamiltonians
  - `qml.VQECost`: a class for quickly constructing a differentiable cost function
    given a circuit ansatz, Hamiltonian, and one or more devices

    ```python
    >>> H = qml.vqe.Hamiltonian(coeffs, obs)
    >>> cost = qml.VQECost(ansatz, hamiltonian, dev, interface="torch")
    >>> params = torch.rand([4, 3])
    >>> cost(params)
    tensor(0.0245, dtype=torch.float64)
    ```

* Added a circuit drawing feature that provides a text-based representation
  of a QNode instance. It can be invoked via `qnode.draw()`. The user can specify
  to display variable names instead of variable values and choose either an ASCII
  or Unicode charset.
  [(#446)](https://github.com/XanaduAI/pennylane/pull/446)

  Consider the following circuit as an example:
  ```python3
  @qml.qnode(dev)
  def qfunc(a, w):
      qml.Hadamard(0)
      qml.CRX(a, wires=[0, 1])
      qml.Rot(w[0], w[1], w[2], wires=[1])
      qml.CRX(-a, wires=[0, 1])

      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  ```

  We can draw the circuit after it has been executed:

  ```python
  >>> result = qfunc(2.3, [1.2, 3.2, 0.7])
  >>> print(qfunc.draw())
   0: ──H──╭C────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩
   1: ─────╰RX(2.3)──Rot(1.2, 3.2, 0.7)──╰RX(-2.3)──╰┤ ⟨Z ⊗ Z⟩
  >>> print(qfunc.draw(charset="ascii"))
   0: --H--+C----------------------------+C---------+| <Z @ Z>
   1: -----+RX(2.3)--Rot(1.2, 3.2, 0.7)--+RX(-2.3)--+| <Z @ Z>
  >>> print(qfunc.draw(show_variable_names=True))
   0: ──H──╭C─────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩
   1: ─────╰RX(a)──Rot(w[0], w[1], w[2])──╰RX(-1*a)──╰┤ ⟨Z ⊗ Z⟩
  ```

* Added `QAOAEmbedding` and its parameter initialization
  as a new trainable template.
  [(#442)](https://github.com/XanaduAI/pennylane/pull/442)

  <img src="https://pennylane.readthedocs.io/en/latest/_images/qaoa_layers.png"
  width=70%></img>

* Added the `qml.probs()` measurement function, allowing QNodes
  to differentiate variational circuit probabilities
  on simulators and hardware.
  [(#432)](https://github.com/XanaduAI/pennylane/pull/432)

  ```python
  @qml.qnode(dev)
  def circuit(x):
      qml.Hadamard(wires=0)
      qml.RY(x, wires=0)
      qml.RX(x, wires=1)
      qml.CNOT(wires=[0, 1])
      return qml.probs(wires=[0])
  ```
  Executing this circuit gives the marginal probability of wire 1:
  ```python
  >>> circuit(0.2)
  [0.40066533 0.59933467]
  ```
  QNodes that return probabilities fully support autodifferentiation.

* Added the convenience load functions `qml.from_pyquil`, `qml.from_quil` and
  `qml.from_quil_file` that convert pyQuil objects and Quil code to PennyLane
  templates. This feature requires version 0.8 or above of the PennyLane-Forest
  plugin.
  [(#459)](https://github.com/XanaduAI/pennylane/pull/459)

* Added a `qml.inv` method that inverts templates and sequences of Operations.
  Added a `@qml.template` decorator that makes templates return the queued Operations.
  [(#462)](https://github.com/XanaduAI/pennylane/pull/462)

  For example, using this function to invert a template inside a QNode:

  ```python3
      @qml.template
      def ansatz(weights, wires):
          for idx, wire in enumerate(wires):
              qml.RX(weights[idx], wires=[wire])

          for idx in range(len(wires) - 1):
              qml.CNOT(wires=[wires[idx], wires[idx + 1]])

      dev = qml.device('default.qubit', wires=2)

      @qml.qnode(dev)
      def circuit(weights):
          qml.inv(ansatz(weights, wires=[0, 1]))
          return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    ```

* Added the `QNodeCollection` container class, that allows independent
  QNodes to be stored and evaluated simultaneously. Experimental support
  for asynchronous evaluation of contained QNodes is provided with the
  `parallel=True` keyword argument.
  [(#466)](https://github.com/XanaduAI/pennylane/pull/466)

* Added a high level `qml.map` function, that maps a quantum
  circuit template over a list of observables or devices, returning
  a `QNodeCollection`.
  [(#466)](https://github.com/XanaduAI/pennylane/pull/466)

  For example:

  ```python3
  >>> def my_template(params, wires, **kwargs):
  >>>    qml.RX(params[0], wires=wires[0])
  >>>    qml.RX(params[1], wires=wires[1])
  >>>    qml.CNOT(wires=wires)

  >>> obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliX(1)]
  >>> dev = qml.device("default.qubit", wires=2)
  >>> qnodes = qml.map(my_template, obs_list, dev, measure="expval")
  >>> qnodes([0.54, 0.12])
  array([-0.06154835  0.99280864])
  ```

* Added high level `qml.sum`, `qml.dot`, `qml.apply` functions
  that act on QNode collections.
  [(#466)](https://github.com/XanaduAI/pennylane/pull/466)

  `qml.apply` allows vectorized functions to act over the entire QNode
  collection:
  ```python
  >>> qnodes = qml.map(my_template, obs_list, dev, measure="expval")
  >>> cost = qml.apply(np.sin, qnodes)
  >>> cost([0.54, 0.12])
  array([-0.0615095  0.83756375])
  ```

  `qml.sum` and `qml.dot` take the sum of a QNode collection, and a
  dot product of tensors/arrays/QNode collections, respectively.

<h3>Breaking changes</h3>

* Deprecated the old-style `QNode` such that only the new-style `QNode` and its syntax can be used,
  moved all related files from the `pennylane/beta` folder to `pennylane`.
  [(#440)](https://github.com/XanaduAI/pennylane/pull/440)

<h3>Improvements</h3>

* Added the `Tensor.prune()` method and the `Tensor.non_identity_obs` property for extracting
  non-identity instances from the observables making up a `Tensor` instance.
  [(#498)](https://github.com/XanaduAI/pennylane/pull/498)

* Renamed the `expt.tensornet` and `expt.tensornet.tf` devices to `default.tensor` and
  `default.tensor.tf`.
  [(#495)](https://github.com/XanaduAI/pennylane/pull/495)

* Added a serialization method to the `CircuitGraph` class that is used to create a unique
  hash for each quantum circuit graph.
  [(#470)](https://github.com/XanaduAI/pennylane/pull/470)

* Added the `Observable.eigvals` method to return the eigenvalues of observables.
  [(#449)](https://github.com/XanaduAI/pennylane/pull/449)

* Added the `Observable.diagonalizing_gates` method to return the gates
  that diagonalize an observable in the computational basis.
  [(#454)](https://github.com/XanaduAI/pennylane/pull/454)

* Added the `Operator.matrix` method to return the matrix representation
  of an operator in the computational basis.
  [(#454)](https://github.com/XanaduAI/pennylane/pull/454)

* Added a `QubitDevice` class which implements common functionalities of plugin devices such that
  plugin devices can rely on these implementations. The new `QubitDevice` also includes
  a new `execute` method, which allows for more convenient plugin design. In addition, `QubitDevice`
  also unifies the way samples are generated on qubit-based devices.
  [(#452)](https://github.com/XanaduAI/pennylane/pull/452)
  [(#473)](https://github.com/XanaduAI/pennylane/pull/473)

* Improved documentation of `AmplitudeEmbedding` and `BasisEmbedding` templates.
  [(#441)](https://github.com/XanaduAI/pennylane/pull/441)
  [(#439)](https://github.com/XanaduAI/pennylane/pull/439)

* Codeblocks in the documentation now have a 'copy' button for easily
  copying examples.
  [(#437)](https://github.com/XanaduAI/pennylane/pull/437)

<h3>Documentation</h3>

* Update the developers plugin guide to use QubitDevice.
  [(#483)](https://github.com/XanaduAI/pennylane/pull/483)

<h3>Bug fixes</h3>

* Fixed a bug in `CVQNode._pd_analytic`, where non-descendant observables were not
  Heisenberg-transformed before evaluating the partial derivatives when using the
  order-2 parameter-shift method, resulting in an erroneous Jacobian for some circuits.
  [(#433)](https://github.com/XanaduAI/pennylane/pull/433)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Ville Bergholm, Alain Delgado Gran, Olivia Di Matteo,
Theodor Isacsson, Josh Izaac, Soran Jahangiri, Nathan Killoran, Johannes Jakob Meyer,
Zeyue Niu, Maria Schuld, Antal Száva.

# Release 0.7.0

<h3>New features since last release</h3>

* Custom padding constant in `AmplitudeEmbedding` is supported (see 'Breaking changes'.)
  [(#419)](https://github.com/XanaduAI/pennylane/pull/419)

* `StronglyEntanglingLayer` and `RandomLayer` now work with a single wire.
  [(#409)](https://github.com/XanaduAI/pennylane/pull/409)
  [(#413)](https://github.com/XanaduAI/pennylane/pull/413)

* Added support for applying the inverse of an `Operation` within a circuit.
  [(#377)](https://github.com/XanaduAI/pennylane/pull/377)

* Added an `OperationRecorder()` context manager, that allows templates
  and quantum functions to be executed while recording events. The
  recorder can be used with and without QNodes as a debugging utility.
  [(#388)](https://github.com/XanaduAI/pennylane/pull/388)

* Operations can now specify a decomposition that is used when the desired operation
  is not supported on the target device.
  [(#396)](https://github.com/XanaduAI/pennylane/pull/396)

* The ability to load circuits from external frameworks as templates
  has been added via the new `qml.load()` function. This feature
  requires plugin support --- this initial release provides support
  for Qiskit circuits and QASM files when `pennylane-qiskit` is installed,
  via the functions `qml.from_qiskit` and `qml.from_qasm`.
  [(#418)](https://github.com/XanaduAI/pennylane/pull/418)

* An experimental tensor network device has been added
  [(#416)](https://github.com/XanaduAI/pennylane/pull/416)
  [(#395)](https://github.com/XanaduAI/pennylane/pull/395)
  [(#394)](https://github.com/XanaduAI/pennylane/pull/394)
  [(#380)](https://github.com/XanaduAI/pennylane/pull/380)

* An experimental tensor network device which uses TensorFlow for
  backpropagation has been added
  [(#427)](https://github.com/XanaduAI/pennylane/pull/427)

* Custom padding constant in `AmplitudeEmbedding` is supported (see 'Breaking changes'.)
  [(#419)](https://github.com/XanaduAI/pennylane/pull/419)

<h3>Breaking changes</h3>

* The `pad` parameter in `AmplitudeEmbedding()` is now either `None` (no automatic padding), or a
  number that is used as the padding constant.
  [(#419)](https://github.com/XanaduAI/pennylane/pull/419)

* Initialization functions now return a single array of weights per function. Utilities for multi-weight templates
  `Interferometer()` and `CVNeuralNetLayers()` are provided.
  [(#412)](https://github.com/XanaduAI/pennylane/pull/412)

* The single layer templates `RandomLayer()`, `CVNeuralNetLayer()` and `StronglyEntanglingLayer()`
  have been turned into private functions `_random_layer()`, `_cv_neural_net_layer()` and
  `_strongly_entangling_layer()`. Recommended use is now via the corresponding `Layers()` templates.
  [(#413)](https://github.com/XanaduAI/pennylane/pull/413)

<h3>Improvements</h3>

* Added extensive input checks in templates.
  [(#419)](https://github.com/XanaduAI/pennylane/pull/419)

* Templates integration tests are rewritten - now cover keyword/positional argument passing,
  interfaces and combinations of templates.
  [(#409)](https://github.com/XanaduAI/pennylane/pull/409)
  [(#419)](https://github.com/XanaduAI/pennylane/pull/419)

* State vector preparation operations in the `default.qubit` plugin can now be
  applied to subsets of wires, and are restricted to being the first operation
  in a circuit.
  [(#346)](https://github.com/XanaduAI/pennylane/pull/346)

* The `QNode` class is split into a hierarchy of simpler classes.
  [(#354)](https://github.com/XanaduAI/pennylane/pull/354)
  [(#398)](https://github.com/XanaduAI/pennylane/pull/398)
  [(#415)](https://github.com/XanaduAI/pennylane/pull/415)
  [(#417)](https://github.com/XanaduAI/pennylane/pull/417)
  [(#425)](https://github.com/XanaduAI/pennylane/pull/425)

* Added the gates U1, U2 and U3 parametrizing arbitrary unitaries on 1, 2 and 3
  qubits and the Toffoli gate to the set of qubit operations.
  [(#396)](https://github.com/XanaduAI/pennylane/pull/396)

* Changes have been made to accomodate the movement of the main function
  in `pytest._internal` to `pytest._internal.main` in pip 19.3.
  [(#404)](https://github.com/XanaduAI/pennylane/pull/404)

* Added the templates `BasisStatePreparation` and `MottonenStatePreparation` that use
  gates to prepare a basis state and an arbitrary state respectively.
  [(#336)](https://github.com/XanaduAI/pennylane/pull/336)

* Added decompositions for `BasisState` and `QubitStateVector` based on state
  preparation templates.
  [(#414)](https://github.com/XanaduAI/pennylane/pull/414)

* Replaces the pseudo-inverse in the quantum natural gradient optimizer
  (which can be numerically unstable) with `np.linalg.solve`.
  [(#428)](https://github.com/XanaduAI/pennylane/pull/428)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ville Bergholm, Josh Izaac, Nathan Killoran, Angus Lowe, Johannes Jakob Meyer,
Oluwatobi Ogunbayo, Maria Schuld, Antal Száva.

# Release 0.6.1

<h3>New features since last release</h3>

* Added a `print_applied` method to QNodes, allowing the operation
  and observable queue to be printed as last constructed.
  [(#378)](https://github.com/XanaduAI/pennylane/pull/378)

<h3>Improvements</h3>

* A new `Operator` base class is introduced, which is inherited by both the
  `Observable` class and the `Operation` class.
  [(#355)](https://github.com/XanaduAI/pennylane/pull/355)

* Removed deprecated `@abstractproperty` decorators
  in `_device.py`.
  [(#374)](https://github.com/XanaduAI/pennylane/pull/374)

* The `CircuitGraph` class is updated to deal with `Operation` instances directly.
  [(#344)](https://github.com/XanaduAI/pennylane/pull/344)

* Comprehensive gradient tests have been added for the interfaces.
  [(#381)](https://github.com/XanaduAI/pennylane/pull/381)

<h3>Documentation</h3>

* The new restructured documentation has been polished and updated.
  [(#387)](https://github.com/XanaduAI/pennylane/pull/387)
  [(#375)](https://github.com/XanaduAI/pennylane/pull/375)
  [(#372)](https://github.com/XanaduAI/pennylane/pull/372)
  [(#370)](https://github.com/XanaduAI/pennylane/pull/370)
  [(#369)](https://github.com/XanaduAI/pennylane/pull/369)
  [(#367)](https://github.com/XanaduAI/pennylane/pull/367)
  [(#364)](https://github.com/XanaduAI/pennylane/pull/364)

* Updated the development guides.
  [(#382)](https://github.com/XanaduAI/pennylane/pull/382)
  [(#379)](https://github.com/XanaduAI/pennylane/pull/379)

* Added all modules, classes, and functions to the API section
  in the documentation.
  [(#373)](https://github.com/XanaduAI/pennylane/pull/373)

<h3>Bug fixes</h3>

* Replaces the existing `np.linalg.norm` normalization with hand-coded
  normalization, allowing `AmplitudeEmbedding` to be used with differentiable
  parameters. AmplitudeEmbedding tests have been added and improved.
  [(#376)](https://github.com/XanaduAI/pennylane/pull/376)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ville Bergholm, Josh Izaac, Nathan Killoran, Maria Schuld, Antal Száva

# Release 0.6.0

<h3>New features since last release</h3>

* The devices `default.qubit` and `default.gaussian` have a new initialization parameter
  `analytic` that indicates if expectation values and variances should be calculated
  analytically and not be estimated from data.
  [(#317)](https://github.com/XanaduAI/pennylane/pull/317)

* Added C-SWAP gate to the set of qubit operations
  [(#330)](https://github.com/XanaduAI/pennylane/pull/330)

* The TensorFlow interface has been renamed from `"tfe"` to `"tf"`, and
  now supports TensorFlow 2.0.
  [(#337)](https://github.com/XanaduAI/pennylane/pull/337)

* Added the S and T gates to the set of qubit operations.
  [(#343)](https://github.com/XanaduAI/pennylane/pull/343)

* Tensor observables are now supported within the `expval`,
  `var`, and `sample` functions, by using the `@` operator.
  [(#267)](https://github.com/XanaduAI/pennylane/pull/267)


<h3>Breaking changes</h3>

* The argument `n` specifying the number of samples in the method `Device.sample` was removed.
  Instead, the method will always return `Device.shots` many samples.
  [(#317)](https://github.com/XanaduAI/pennylane/pull/317)

<h3>Improvements</h3>

* The number of shots / random samples used to estimate expectation values and variances, `Device.shots`,
  can now be changed after device creation.
  [(#317)](https://github.com/XanaduAI/pennylane/pull/317)

* Unified import shortcuts to be under qml in qnode.py
  and test_operation.py
  [(#329)](https://github.com/XanaduAI/pennylane/pull/329)

* The quantum natural gradient now uses `scipy.linalg.pinvh` which is more efficient for symmetric matrices
  than the previously used `scipy.linalg.pinv`.
  [(#331)](https://github.com/XanaduAI/pennylane/pull/331)

* The deprecated `qml.expval.Observable` syntax has been removed.
  [(#267)](https://github.com/XanaduAI/pennylane/pull/267)

* Remainder of the unittest-style tests were ported to pytest.
  [(#310)](https://github.com/XanaduAI/pennylane/pull/310)

* The `do_queue` argument for operations now only takes effect
  within QNodes. Outside of QNodes, operations can now be instantiated
  without needing to specify `do_queue`.
  [(#359)](https://github.com/XanaduAI/pennylane/pull/359)

<h3>Documentation</h3>

* The docs are rewritten and restructured to contain a code introduction section as well as an API section.
  [(#314)](https://github.com/XanaduAI/pennylane/pull/275)

* Added Ising model example to the tutorials
  [(#319)](https://github.com/XanaduAI/pennylane/pull/319)

* Added tutorial for QAOA on MaxCut problem
  [(#328)](https://github.com/XanaduAI/pennylane/pull/328)

* Added QGAN flow chart figure to its tutorial
  [(#333)](https://github.com/XanaduAI/pennylane/pull/333)

* Added missing figures for gallery thumbnails of state-preparation
  and QGAN tutorials
  [(#326)](https://github.com/XanaduAI/pennylane/pull/326)

* Fixed typos in the state preparation tutorial
  [(#321)](https://github.com/XanaduAI/pennylane/pull/321)

* Fixed bug in VQE tutorial 3D plots
  [(#327)](https://github.com/XanaduAI/pennylane/pull/327)

<h3>Bug fixes</h3>

* Fixed typo in measurement type error message in qnode.py
  [(#341)](https://github.com/XanaduAI/pennylane/pull/341)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Shahnawaz Ahmed, Ville Bergholm, Aroosa Ijaz, Josh Izaac, Nathan Killoran, Angus Lowe,
Johannes Jakob Meyer, Maria Schuld, Antal Száva, Roeland Wiersema.

# Release 0.5.0

<h3>New features since last release</h3>

* Adds a new optimizer, `qml.QNGOptimizer`, which optimizes QNodes using
  quantum natural gradient descent. See https://arxiv.org/abs/1909.02108
  for more details.
  [(#295)](https://github.com/XanaduAI/pennylane/pull/295)
  [(#311)](https://github.com/XanaduAI/pennylane/pull/311)

* Adds a new QNode method, `QNode.metric_tensor()`,
  which returns the block-diagonal approximation to the Fubini-Study
  metric tensor evaluated on the attached device.
  [(#295)](https://github.com/XanaduAI/pennylane/pull/295)

* Sampling support: QNodes can now return a specified number of samples
  from a given observable via the top-level `pennylane.sample()` function.
  To support this on plugin devices, there is a new `Device.sample` method.

  Calculating gradients of QNodes that involve sampling is not possible.
  [(#256)](https://github.com/XanaduAI/pennylane/pull/256)

* `default.qubit` has been updated to provide support for sampling.
  [(#256)](https://github.com/XanaduAI/pennylane/pull/256)

* Added controlled rotation gates to PennyLane operations and `default.qubit` plugin.
  [(#251)](https://github.com/XanaduAI/pennylane/pull/251)

<h3>Breaking changes</h3>

* The method `Device.supported` was removed, and replaced with the methods
  `Device.supports_observable` and `Device.supports_operation`.
  Both methods can be called with string arguments (`dev.supports_observable('PauliX')`) and
  class arguments (`dev.supports_observable(qml.PauliX)`).
  [(#276)](https://github.com/XanaduAI/pennylane/pull/276)

* The following CV observables were renamed to comply with the new Operation/Observable
  scheme: `MeanPhoton` to `NumberOperator`, `Homodyne` to `QuadOperator` and `NumberState` to `FockStateProjector`.
  [(#254)](https://github.com/XanaduAI/pennylane/pull/254)

<h3>Improvements</h3>

* The `AmplitudeEmbedding` function now provides options to normalize and
  pad features to ensure a valid state vector is prepared.
  [(#275)](https://github.com/XanaduAI/pennylane/pull/275)

* Operations can now optionally specify generators, either as existing PennyLane
  operations, or by providing a NumPy array.
  [(#295)](https://github.com/XanaduAI/pennylane/pull/295)
  [(#313)](https://github.com/XanaduAI/pennylane/pull/313)

* Adds a `Device.parameters` property, so that devices can view a dictionary mapping free
  parameters to operation parameters. This will allow plugin devices to take advantage
  of parametric compilation.
  [(#283)](https://github.com/XanaduAI/pennylane/pull/283)

* Introduces two enumerations: `Any` and `All`, representing any number of wires
  and all wires in the system respectively. They can be imported from
  `pennylane.operation`, and can be used when defining the `Operation.num_wires`
  class attribute of operations.
  [(#277)](https://github.com/XanaduAI/pennylane/pull/277)

  As part of this change:

  - `All` is equivalent to the integer 0, for backwards compatibility with the
    existing test suite

  - `Any` is equivalent to the integer -1 to allow numeric comparison
    operators to continue working

  - An additional validation is now added to the `Operation` class,
    which will alert the user that an operation with `num_wires = All`
    is being incorrectly.

* The one-qubit rotations in `pennylane.plugins.default_qubit` no longer depend on Scipy's `expm`. Instead
  they are calculated with Euler's formula.
  [(#292)](https://github.com/XanaduAI/pennylane/pull/292)

* Creates an `ObservableReturnTypes` enumeration class containing `Sample`,
  `Variance` and `Expectation`. These new values can be assigned to the `return_type`
  attribute of an `Observable`.
  [(#290)](https://github.com/XanaduAI/pennylane/pull/290)

* Changed the signature of the `RandomLayer` and `RandomLayers` templates to have a fixed seed by default.
  [(#258)](https://github.com/XanaduAI/pennylane/pull/258)

* `setup.py` has been cleaned up, removing the non-working shebang,
  and removing unused imports.
  [(#262)](https://github.com/XanaduAI/pennylane/pull/262)

<h3>Documentation</h3>

* A documentation refactor to simplify the tutorials and
  include Sphinx-Gallery.
  [(#291)](https://github.com/XanaduAI/pennylane/pull/291)

  - Examples and tutorials previously split across the `examples/`
    and `doc/tutorials/` directories, in a mixture of ReST and Jupyter notebooks,
    have been rewritten as Python scripts with ReST comments in a single location,
    the `examples/` folder.

  - Sphinx-Gallery is used to automatically build and run the tutorials.
    Rendered output is displayed in the Sphinx documentation.

  - Links are provided at the top of every tutorial page for downloading the
    tutorial as an executable python script, downloading the tutorial
    as a Jupyter notebook, or viewing the notebook on GitHub.

  - The tutorials table of contents have been moved to a single quick start page.

* Fixed a typo in `QubitStateVector`.
  [(#296)](https://github.com/XanaduAI/pennylane/pull/296)

* Fixed a typo in the `default_gaussian.gaussian_state` function.
  [(#293)](https://github.com/XanaduAI/pennylane/pull/293)

* Fixed a typo in the gradient recipe within the `RX`, `RY`, `RZ`
  operation docstrings.
  [(#248)](https://github.com/XanaduAI/pennylane/pull/248)

* Fixed a broken link in the tutorial documentation, as a
  result of the `qml.expval.Observable` deprecation.
  [(#246)](https://github.com/XanaduAI/pennylane/pull/246)

<h3>Bug fixes</h3>

* Fixed a bug where a `PolyXP` observable would fail if applied to subsets
  of wires on `default.gaussian`.
  [(#277)](https://github.com/XanaduAI/pennylane/pull/277)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Simon Cross, Aroosa Ijaz, Josh Izaac, Nathan Killoran, Johannes Jakob Meyer,
Rohit Midha, Nicolás Quesada, Maria Schuld, Antal Száva, Roeland Wiersema.

# Release 0.4.0

<h3>New features since last release</h3>

* `pennylane.expval()` is now a top-level *function*, and is no longer
  a package of classes. For now, the existing `pennylane.expval.Observable`
  interface continues to work, but will raise a deprecation warning.
  [(#232)](https://github.com/XanaduAI/pennylane/pull/232)

* Variance support: QNodes can now return the variance of observables,
  via the top-level `pennylane.var()` function. To support this on
  plugin devices, there is a new `Device.var` method.

  The following observables support analytic gradients of variances:

  - All qubit observables (requiring 3 circuit evaluations for involutory
    observables such as `Identity`, `X`, `Y`, `Z`; and 5 circuit evals for
    non-involutary observables, currently only `qml.Hermitian`)

  - First-order CV observables (requiring 5 circuit evaluations)

  Second-order CV observables support numerical variance gradients.

* `pennylane.about()` function added, providing details
  on current PennyLane version, installed plugins, Python,
  platform, and NumPy versions [(#186)](https://github.com/XanaduAI/pennylane/pull/186)

* Removed the logic that allowed `wires` to be passed as a positional
  argument in quantum operations. This allows us to raise more useful
  error messages for the user if incorrect syntax is used.
  [(#188)](https://github.com/XanaduAI/pennylane/pull/188)

* Adds support for multi-qubit expectation values of the `pennylane.Hermitian()`
  observable [(#192)](https://github.com/XanaduAI/pennylane/pull/192)

* Adds support for multi-qubit expectation values in `default.qubit`.
  [(#202)](https://github.com/XanaduAI/pennylane/pull/202)

* Organize templates into submodules [(#195)](https://github.com/XanaduAI/pennylane/pull/195).
  This included the following improvements:

  - Distinguish embedding templates from layer templates.

  - New random initialization functions supporting the templates available
    in the new submodule `pennylane.init`.

  - Added a random circuit template (`RandomLayers()`), in which rotations and 2-qubit gates are randomly
    distributed over the wires

  - Add various embedding strategies

<h3>Breaking changes</h3>

* The `Device` methods `expectations`, `pre_expval`, and `post_expval` have been
  renamed to `observables`, `pre_measure`, and `post_measure` respectively.
  [(#232)](https://github.com/XanaduAI/pennylane/pull/232)

<h3>Improvements</h3>

* `default.qubit` plugin now uses `np.tensordot` when applying quantum operations
  and evaluating expectations, resulting in significant speedup
  [(#239)](https://github.com/XanaduAI/pennylane/pull/239),
  [(#241)](https://github.com/XanaduAI/pennylane/pull/241)

* PennyLane now allows division of quantum operation parameters by a constant
  [(#179)](https://github.com/XanaduAI/pennylane/pull/179)

* Portions of the test suite are in the process of being ported to pytest.
  Note: this is still a work in progress.

  Ported tests include:

  - `test_ops.py`
  - `test_about.py`
  - `test_classical_gradients.py`
  - `test_observables.py`
  - `test_measure.py`
  - `test_init.py`
  - `test_templates*.py`
  - `test_ops.py`
  - `test_variable.py`
  - `test_qnode.py` (partial)

<h3>Bug fixes</h3>

* Fixed a bug in `Device.supported`, which would incorrectly
  mark an operation as supported if it shared a name with an
  observable [(#203)](https://github.com/XanaduAI/pennylane/pull/203)

* Fixed a bug in `Operation.wires`, by explicitly casting the
  type of each wire to an integer [(#206)](https://github.com/XanaduAI/pennylane/pull/206)

* Removed code in PennyLane which configured the logger,
  as this would clash with users' configurations
  [(#208)](https://github.com/XanaduAI/pennylane/pull/208)

* Fixed a bug in `default.qubit`, in which `QubitStateVector` operations
  were accidentally being cast to `np.float` instead of `np.complex`.
  [(#211)](https://github.com/XanaduAI/pennylane/pull/211)


<h3>Contributors</h3>

This release contains contributions from:

Shahnawaz Ahmed, riveSunder, Aroosa Ijaz, Josh Izaac, Nathan Killoran, Maria Schuld.

# Release 0.3.1

<h3>Bug fixes</h3>

* Fixed a bug where the interfaces submodule was not correctly being packaged via setup.py

# Release 0.3.0

<h3>New features since last release</h3>

* PennyLane now includes a new `interfaces` submodule, which enables QNode integration with additional machine learning libraries.
* Adds support for an experimental PyTorch interface for QNodes
* Adds support for an experimental TensorFlow eager execution interface for QNodes
* Adds a PyTorch+GPU+QPU tutorial to the documentation
* Documentation now includes links and tutorials including the new [PennyLane-Forest](https://github.com/rigetti/pennylane-forest) plugin.

<h3>Improvements</h3>

* Printing a QNode object, via `print(qnode)` or in an interactive terminal, now displays more useful information regarding the QNode,
  including the device it runs on, the number of wires, it's interface, and the quantum function it uses:

  ```python
  >>> print(qnode)
  <QNode: device='default.qubit', func=circuit, wires=2, interface=PyTorch>
  ```

<h3>Contributors</h3>

This release contains contributions from:

Josh Izaac and Nathan Killoran.


# Release 0.2.0

<h3>New features since last release</h3>

* Added the `Identity` expectation value for both CV and qubit models [(#135)](https://github.com/XanaduAI/pennylane/pull/135)
* Added the `templates.py` submodule, containing some commonly used QML models to be used as ansatz in QNodes [(#133)](https://github.com/XanaduAI/pennylane/pull/133)
* Added the `qml.Interferometer` CV operation [(#152)](https://github.com/XanaduAI/pennylane/pull/152)
* Wires are now supported as free QNode parameters [(#151)](https://github.com/XanaduAI/pennylane/pull/151)
* Added ability to update stepsizes of the optimizers [(#159)](https://github.com/XanaduAI/pennylane/pull/159)

<h3>Improvements</h3>

* Removed use of hardcoded values in the optimizers, made them parameters (see [#131](https://github.com/XanaduAI/pennylane/pull/131) and [#132](https://github.com/XanaduAI/pennylane/pull/132))
* Created the new `PlaceholderExpectation`, to be used when both CV and qubit expval modules contain expectations with the same name
* Provide a way for plugins to view the operation queue _before_ applying operations. This allows for on-the-fly modifications of
  the queue, allowing hardware-based plugins to support the full range of qubit expectation values. [(#143)](https://github.com/XanaduAI/pennylane/pull/143)
* QNode return values now support _any_ form of sequence, such as lists, sets, etc. [(#144)](https://github.com/XanaduAI/pennylane/pull/144)
* CV analytic gradient calculation is now more robust, allowing for operations which may not themselves be differentiated, but have a
  well defined `_heisenberg_rep` method, and so may succeed operations that are analytically differentiable [(#152)](https://github.com/XanaduAI/pennylane/pull/152)

<h3>Bug fixes</h3>

* Fixed a bug where the variational classifier example was not batching when learning parity (see [#128](https://github.com/XanaduAI/pennylane/pull/128) and [#129](https://github.com/XanaduAI/pennylane/pull/129))
* Fixed an inconsistency where some initial state operations were documented as accepting complex parameters - all operations
  now accept real values [(#146)](https://github.com/XanaduAI/pennylane/pull/146)

<h3>Contributors</h3>

This release contains contributions from:

Christian Gogolin, Josh Izaac, Nathan Killoran, and Maria Schuld.


# Release 0.1.0

Initial public release.

<h3>Contributors</h3>
This release contains contributions from:

Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, and Nathan Killoran.
