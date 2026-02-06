
# Release 0.16.0

<h4>First class support for quantum kernels</h4>

* The new `qp.kernels` module provides basic functionalities for [working with quantum
  kernels](https://pennylane.readthedocs.io/en/stable/code/qml_kernels.html) as
  well as post-processing methods to mitigate sampling errors and device noise:
  [(#1102)](https://github.com/PennyLaneAI/pennylane/pull/1102)

  ```python

  num_wires = 6
  wires = range(num_wires)
  dev = qp.device('default.qubit', wires=wires)

  @qp.qnode(dev)
  def kernel_circuit(x1, x2):
      qp.templates.AngleEmbedding(x1, wires=wires)
      qp.adjoint(qp.templates.AngleEmbedding)(x2, wires=wires)
      return qp.probs(wires)

  kernel = lambda x1, x2: kernel_circuit(x1, x2)[0]
  X_train = np.random.random((10, 6))
  X_test = np.random.random((5, 6))

  # Create symmetric square kernel matrix (for training)
  K = qp.kernels.square_kernel_matrix(X_train, kernel)

  # Compute kernel between test and training data.
  K_test = qp.kernels.kernel_matrix(X_train, X_test, kernel)
  K1 = qp.kernels.mitigate_depolarizing_noise(K, num_wires, method='single')
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

  The function `qp.grouping.pauli_group` provides a generator to
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
  >>> pauli_word = qp.PauliZ('b')  # corresponds to Pauli 'IZ'
  >>> pauli_word_to_matrix(pauli_word, wire_map=wire_map)
  array([[ 1.,  0.,  0.,  0.],
         [ 0., -1.,  0., -0.],
         [ 0.,  0.,  1.,  0.],
         [ 0., -0.,  0., -1.]])
  ```

<h4>New transforms</h4>

* The `qp.specs` QNode transform creates a function that returns specifications or
  details about the QNode, including depth, number of gates, and number of
  gradient executions required.
  [(#1245)](https://github.com/PennyLaneAI/pennylane/pull/1245)

  For example:

  ```python
  dev = qp.device('default.qubit', wires=4)

  @qp.qnode(dev, diff_method='parameter-shift')
  def circuit(x, y):
      qp.RX(x[0], wires=0)
      qp.Toffoli(wires=(0, 1, 2))
      qp.CRY(x[1], wires=(0, 1))
      qp.Rot(x[2], x[3], y, wires=0)
      return qp.expval(qp.PauliZ(0)), qp.expval(qp.PauliX(1))
  ```

  We can now use the `qp.specs` transform to generate a function that returns
  details and resource information:

  ```pycon
  >>> x = np.array([0.05, 0.1, 0.2, 0.3], requires_grad=True)
  >>> y = np.array(0.4, requires_grad=False)
  >>> specs_func = qp.specs(circuit)
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

* Adds a decorator `@qp.qfunc_transform` to easily create a transformation
  that modifies the behaviour of a quantum function.
  [(#1315)](https://github.com/PennyLaneAI/pennylane/pull/1315)

  For example, consider the following transform, which scales the parameter of
  all `RX` gates by :math:`x \rightarrow \sin(a) \sqrt{x}`, and the parameters
  of all `RY` gates by :math:`y \rightarrow \cos(a * b) y`:

  ```python
  @qp.qfunc_transform
  def my_transform(tape, a, b):
      for op in tape.operations + tape.measurements:
          if op.name == "RX":
              x = op.parameters[0]
              qp.RX(qp.math.sin(a) * qp.math.sqrt(x), wires=op.wires)
          elif op.name == "RY":
              y = op.parameters[0]
              qp.RX(qp.math.cos(a * b) * y, wires=op.wires)
          else:
              op.queue()
  ```

  We can now apply this transform to any quantum function:

  ```python
  dev = qp.device("default.qubit", wires=2)

  def ansatz(x):
      qp.Hadamard(wires=0)
      qp.RX(x[0], wires=0)
      qp.RY(x[1], wires=1)
      qp.CNOT(wires=[0, 1])

  @qp.qnode(dev)
  def circuit(params, transform_weights):
      qp.RX(0.1, wires=0)

      # apply the transform to the ansatz
      my_transform(*transform_weights)(ansatz)(params)

      return qp.expval(qp.PauliZ(1))
  ```

  We can print this QNode to show that the qfunc transform is taking place:

  ```pycon
  >>> x = np.array([0.5, 0.3], requires_grad=True)
  >>> transform_weights = np.array([0.1, 0.6], requires_grad=True)
  >>> print(qp.draw(circuit)(x, transform_weights))
   0: ──RX(0.1)────H──RX(0.0706)──╭C──┤
   1: ──RX(0.299)─────────────────╰X──┤ ⟨Z⟩
  ```

  Evaluating the QNode, as well as the derivative, with respect to the gate
  parameter *and* the transform weights:

  ```pycon
  >>> circuit(x, transform_weights)
  tensor(0.00672829, requires_grad=True)
  >>> qp.grad(circuit)(x, transform_weights)
  (array([ 0.00671711, -0.00207359]), array([6.69695008e-02, 3.73694364e-06]))
  ```

* Adds a `hamiltonian_expand` tape transform. This takes a tape ending in
  `qp.expval(H)`, where `H` is a Hamiltonian, and maps it to a collection
  of tapes which can be executed and passed into a post-processing function yielding
  the expectation value.
  [(#1142)](https://github.com/PennyLaneAI/pennylane/pull/1142)

  Example use:

  ```python
  H = qp.PauliZ(0) + 3 * qp.PauliZ(0) @ qp.PauliX(1)

  with qp.tape.QuantumTape() as tape:
      qp.Hadamard(wires=1)
      qp.expval(H)

  tapes, fn = qp.transforms.hamiltonian_expand(tape)
  ```

  We can now evaluate the transformed tapes, and apply the post-processing
  function:

  ```pycon
  >>> dev = qp.device("default.qubit", wires=3)
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

  dev = qp.device("default.qubit", wires=(n + m + 1))

  def fn():
      qp.templates.MottonenStatePreparation(np.sqrt(probs), wires=a_wires)
      r_unitary(qp.RY, r_rotations, control_wires=a_wires[::-1], target_wire=target_wire)

  @qp.qnode(dev)
  def qmc():
      qp.quantum_monte_carlo(fn, wires, target_wire, estimation_wires)()
      return qp.probs(estimation_wires)

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
  >>> cost, mixer, mapping = qp.qaoa.max_weight_cycle(g)
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
  [qp.qaoa.cycle](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.qaoa.cycle.html)
  module.


<h4>Extended operations and templates</h4>

* Added functionality to compute the sparse matrix representation of a `qp.Hamiltonian` object.
  [(#1394)](https://github.com/PennyLaneAI/pennylane/pull/1394)

  ```python
  coeffs = [1, -0.45]
  obs = [qp.PauliZ(0) @ qp.PauliZ(1), qp.PauliY(0) @ qp.PauliZ(1)]
  H = qp.Hamiltonian(coeffs, obs)
  H_sparse = qp.utils.sparse_hamiltonian(H)
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
  import pennylane as qp
  from pennylane import numpy as np

  electrons = 2
  qubits = 4

  hf_state = qp.qchem.hf_state(electrons, qubits)
  singles, doubles = qp.qchem.excitations(electrons, qubits)
  ```

  Now we can use the template ``AllSinglesDoubles`` to define the
  quantum circuit,

  ```python
  from pennylane.templates import AllSinglesDoubles

  wires = range(qubits)

  dev = qp.device('default.qubit', wires=wires)

  @qp.qnode(dev)
  def circuit(weights, hf_state, singles, doubles):
      AllSinglesDoubles(weights, wires, hf_state, singles, doubles)
      return qp.expval(qp.PauliZ(0))

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
  dev = qp.device('default.qubit', wires = 4)
  a = 0
  b = 1

  @qp.qnode(dev)
  def circuit():
      qp.BasisState(np.array([a, b]), wires=[1, 2])
      qp.QubitCarry(wires=[0, 1, 2, 3])
      qp.CNOT(wires=[1, 2])
      qp.QubitSum(wires=[0, 1, 2])
      return qp.probs(wires=[3, 2])

  probs = circuit()
  bitstrings = tuple(itertools.product([0, 1], repeat = 2))
  indx = np.argwhere(probs == 1).flatten()[0]
  output = bitstrings[indx]
  ```

  ```pycon
  >>> print(output)
  (0, 1)
  ```

* Added the `qp.Projector` observable, which is available on all devices
  inheriting from the `QubitDevice` class.
  [(#1356)](https://github.com/PennyLaneAI/pennylane/pull/1356)
  [(#1368)](https://github.com/PennyLaneAI/pennylane/pull/1368)

  Using `qp.Projector`, we can define the basis state projectors to use when
  computing expectation values. Let us take for example a circuit that prepares
  Bell states:

  ```python
  dev = qp.device("default.qubit", wires=2)

  @qp.qnode(dev)
  def circuit(basis_state):
      qp.Hadamard(wires=[0])
      qp.CNOT(wires=[0, 1])
      return qp.expval(qp.Projector(basis_state, wires=[0, 1]))
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

  - The IsingXX gate `qp.IsingXX` [(#1194)](https://github.com/PennyLaneAI/pennylane/pull/1194)
  - The IsingZZ gate `qp.IsingZZ` [(#1199)](https://github.com/PennyLaneAI/pennylane/pull/1199)
  - The ISWAP gate `qp.ISWAP` [(#1298)](https://github.com/PennyLaneAI/pennylane/pull/1298)
  - The reset error noise channel `qp.ResetError` [(#1321)](https://github.com/PennyLaneAI/pennylane/pull/1321)


<h3>Improvements</h3>

* The ``argnum`` keyword argument can now be specified for a QNode to define a
  subset of trainable parameters used to estimate the Jacobian.
  [(#1371)](https://github.com/PennyLaneAI/pennylane/pull/1371)

  For example, consider two trainable parameters and a quantum function:

  ```python
  dev = qp.device("default.qubit", wires=2)

  x = np.array(0.543, requires_grad=True)
  y = np.array(-0.654, requires_grad=True)

  def circuit(x,y):
      qp.RX(x, wires=[0])
      qp.RY(y, wires=[1])
      qp.CNOT(wires=[0, 1])
      return qp.expval(qp.PauliZ(0) @ qp.PauliX(1))
  ```

  When computing the gradient of the QNode, we can specify the trainable
  parameters to consider by passing the ``argnum`` keyword argument:

  ```pycon
  >>> qnode1 = qp.QNode(circuit, dev, diff_method="parameter-shift", argnum=[0,1])
  >>> print(qp.grad(qnode1)(x,y))
  (array(0.31434679), array(0.67949903))
  ```

  Specifying a proper subset of the trainable parameters will estimate the
  Jacobian:

  ```pycon
  >>> qnode2 = qp.QNode(circuit, dev, diff_method="parameter-shift", argnum=[0])
  >>> print(qp.grad(qnode2)(x,y))
  (array(0.31434679), array(0.))
  ```

* Allows creating differentiable observables that return custom objects such
  that the observable is supported by devices.
  [(1291)](https://github.com/PennyLaneAI/pennylane/pull/1291)

  As an example, first we define `NewObservable` class:

  ```python
  from pennylane.devices import DefaultQubit

  class NewObservable(qp.operation.Observable):
      """NewObservable"""

      num_wires = qp.operation.AnyWires
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
              val = super().expval(qp.PauliZ(wires=0), **kwargs)
              return SpecialObject(val)

          return super().expval(observable, **kwargs)
  ```

  At this point, we can create a device that will support the differentiation
  of a `NewObservable` object:

  ```python
  dev = DeviceSupportingNewObservable(wires=1, shots=None)

  @qp.qnode(dev, diff_method="parameter-shift")
  def qnode(x):
      qp.RY(x, wires=0)
      return qp.expval(NewObservable(wires=0))
  ```

  We can then compute the jacobian of this object:

  ```pycon
  >>> result = qp.jacobian(qnode)(0.2)
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

  - `qp.CSWAP` [(#1306)](https://github.com/PennyLaneAI/pennylane/issues/1306)
  - `qp.SWAP` [(#1329)](https://github.com/PennyLaneAI/pennylane/pull/1329)
  - `qp.SingleExcitation` [(#1303)](https://github.com/PennyLaneAI/pennylane/pull/1303)
  - `qp.SingleExcitationPlus` and `qp.SingleExcitationMinus` [(#1278)](https://github.com/PennyLaneAI/pennylane/pull/1278)
  - `qp.DoubleExcitation` [(#1303)](https://github.com/PennyLaneAI/pennylane/pull/1303)
  - `qp.Toffoli` [(#1320)](https://github.com/PennyLaneAI/pennylane/pull/1320)
  - `qp.MultiControlledX`. [(#1287)](https://github.com/PennyLaneAI/pennylane/pull/1287)
    When controlling on three or more wires, an ancilla
    register of worker wires is required to support the decomposition.

    ```python
    ctrl_wires = [f"c{i}" for i in range(5)]
    work_wires = [f"w{i}" for i in range(3)]
    target_wires = ["t0"]
    all_wires = ctrl_wires + work_wires + target_wires

    dev = qp.device("default.qubit", wires=all_wires)

    with qp.tape.QuantumTape() as tape:
        qp.MultiControlledX(control_wires=ctrl_wires, wires=target_wires, work_wires=work_wires)
    ```

    ```pycon
    >>> tape = tape.expand(depth=1)
    >>> print(tape.draw(wire_order=qp.wires.Wires(all_wires)))

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

* Added `qp.CPhase` as an alias for the existing `qp.ControlledPhaseShift` operation.
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

* The `qp.inv()` function is now deprecated with a warning to use the more general `qp.adjoint()`.
  [(#1325)](https://github.com/PennyLaneAI/pennylane/pull/1325)

* Removes support for Python 3.6 and adds support for Python 3.9.
  [(#1228)](https://github.com/XanaduAI/pennylane/pull/1228)

* The tape methods `get_resources` and `get_depth` are superseded by `specs` and will be
  deprecated after one release cycle.
  [(#1245)](https://github.com/PennyLaneAI/pennylane/pull/1245)

* Using the `qp.sample()` measurement on devices with `shots=None` continue to
  raise a warning with this functionality being fully deprecated and raising an
  error after one release cycle.
  [(#1079)](https://github.com/PennyLaneAI/pennylane/pull/1079)
  [(#1196)](https://github.com/PennyLaneAI/pennylane/pull/1196)

<h3>Bug fixes</h3>

* QNodes now display readable information when in interactive environments or when printed.
  [(#1359)](https://github.com/PennyLaneAI/pennylane/pull/1359).

* Fixes a bug with `qp.math.cast` where the `MottonenStatePreparation` operation expected
  a float type instead of double.
  [(#1400)](https://github.com/XanaduAI/pennylane/pull/1400)

* Fixes a bug where a copy of `qp.ControlledQubitUnitary` was non-functional as it did not have all the necessary information.
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

* Fixes a bug where `qp.ctrl` would fail to transform gates that had no
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

* Fixed a bug where `qp.sum()` and `qp.dot()` do not support the JAX interface.
  [(#1380)](https://github.com/PennyLaneAI/pennylane/pull/1380)

<h3>Documentation</h3>

* Math present in the `QubitParamShiftTape` class docstring now renders correctly.
  [(#1402)](https://github.com/PennyLaneAI/pennylane/pull/1402)

* Fix typo in the documentation of `qp.StronglyEntanglingLayers`.
  [(#1367)](https://github.com/PennyLaneAI/pennylane/pull/1367)

* Fixed typo in TensorFlow interface documentation
  [(#1312)](https://github.com/PennyLaneAI/pennylane/pull/1312)

* Fixed typos in the mathematical expressions in documentation of `qp.DoubleExcitation`.
  [(#1278)](https://github.com/PennyLaneAI/pennylane/pull/1278)

* Remove unsupported `None` option from the `qp.QNode` docstrings.
  [(#1271)](https://github.com/PennyLaneAI/pennylane/pull/1271)

* Updated the docstring of `qp.PolyXP` to reference the new location of internal
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

