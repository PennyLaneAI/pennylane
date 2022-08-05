:orphan:

# Release 0.25.0-dev (development release)

<h3>New features since last release</h3>

* Added readout error functionality to the MixedDevice.
  [(#2786)](https://github.com/PennyLaneAI/pennylane/pull/2786)

  Readout error has been added by applying a BitFlip channel to the wires
  measured after the diagonalizing gates corresponding to the measurement
  observable has been applied. The probability of the readout error occurring
  should be passed when creating the device.

  ```pycon
  >>> dev = qml.device("default.mixed", wires=2, readout_prob=0.1)
  >>> @qml.qnode(dev)
  ... def circuit():
  ...     return qml.expval(qml.PauliZ(0))
  >>> print(circuit())
  0.8
  ```

* Operations for quantum chemistry now also support parameter broadcasting
  in their numerical representations.
  [(#2726)](https://github.com/PennyLaneAI/pennylane/pull/2726)

  Similar to standard parametrized operations, quantum chemistry operations now
  also work with broadcasted parameters:

  ```pycon
  >>> op = qml.SingleExcitation(np.array([0.3, 1.2, -0.7]), wires=[0, 1])
  >>> op.matrix().shape
  (3, 4, 4)
  ```

* The gradient transform `qml.gradients.param_shift` now accepts the new Boolean keyword
  argument `broadcast`. If it is set to `True`, broadcasting is used to compute the derivative.
  [(#2749)](https://github.com/PennyLaneAI/pennylane/pull/2749)

  For example, for the circuit

  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def circuit(x, y):
      qml.RX(x, wires=0)
      qml.CRY(y, wires=[0, 1])
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  ```

  we may compute the derivative via

  ```pycon
  >>> x, y = np.array([0.4, 0.23], requires_grad=True)
  >>> qml.gradients.param_shift(circuit, broadcast=True)(x, y)
  (tensor(-0.38429095, requires_grad=True),
   tensor(0.00899816, requires_grad=True))
  ```

  Note that `QuantumTapes`/`QNodes` with multiple return values and shot vectors are not supported
  yet and the operations with trainable parameters are required to support broadcasting when using
  `broadcast=True`. One way of checking the latter is the `Attribute` `supports_broadcasting`:

  ```pycon
  >>> qml.RX in qml.ops.qubit.attributes.supports_broadcasting
  True
  ```

* Functionality for estimating the number of non-Clifford gates and logical qubits needed to
  implement quantum phase estimation algorithms for simulating materials and molecules is added to
  the new `qml.resource` module. Quantum algorithms in first quantization using a plane-wave basis
  and in second quantization with a double-factorized Hamiltonian are supported.
  [(#2646)](https://github.com/PennyLaneAI/pennylane/pull/2646)
  [(#2653)](https://github.com/PennyLaneAI/pennylane/pull/2653)
  [(#2665)](https://github.com/PennyLaneAI/pennylane/pull/2665)
  [(#2694)](https://github.com/PennyLaneAI/pennylane/pull/2694)
  [(#2720)](https://github.com/PennyLaneAI/pennylane/pull/2720)
  [(#2723)](https://github.com/PennyLaneAI/pennylane/pull/2723)
  [(#2746)](https://github.com/PennyLaneAI/pennylane/pull/2746)
  [(#2796)](https://github.com/PennyLaneAI/pennylane/pull/2796)
  [(#2797)](https://github.com/PennyLaneAI/pennylane/pull/2797)
  [(#2874)](https://github.com/PennyLaneAI/pennylane/pull/2874)
  [(#2644)](https://github.com/PennyLaneAI/pennylane/pull/2644)

  The resource estimation algorithms are implemented as classes inherited from the `Operation`
  class. The number of non-Clifford gates and logical qubits for implementing each algorithm can be
  estimated by initiating the class for a given system. For the first quantization algorithm, the
  number of plane waves, number of electrons and the unit cell volume (in atomic units) are needed
  to initiate the `FirstQuantization` class. The resource can then be estimated as

  ```python
  import pennylane as qml
  from pennylane import numpy as np
  
  n = 100000        # number of plane waves
  eta = 156         # number of electrons
  omega = 1145.166  # unit cell volume
  
  algo = FirstQuantization(n, eta, omega)
  
  # print the number of non-Clifford gates and logical qubits
  print(algo.gates, algo.qubits)
  ```
  
  ```pycon
  1.10e+13, 4416
  ```
  
  For the second quantization algorithm, the one- and two-electron integrals are needed to initiate
  the `DoubleFactorization` class which creates a double-factorized Hamiltonian and computes the
  number of non-Clifford gates and logical qubits for simulating the Hamiltonian:

  ```python
  import pennylane as qml
  from pennylane import numpy as np
  
  symbols  = ['O', 'H', 'H']
  geometry = np.array([[0.00000000,  0.00000000,  0.28377432],
                       [0.00000000,  1.45278171, -1.00662237],
                       [0.00000000, -1.45278171, -1.00662237]], requires_grad = False)
  
  mol = qml.qchem.Molecule(symbols, geometry, basis_name='sto-3g')
  core, one, two = qml.qchem.electron_integrals(mol)()
  algo = DoubleFactorization(one, two)
  
  # print the number of non-Clifford gates and logical qubits
  print(algo.gates, algo.qubits)
  ```

  ```pycon
  103969925, 290
  ```

  The methods of the `FirstQuantization` and the `DoubleFactorization` classes can be also accessed
  individually. For instance, the logical qubits can be computed by providing the inputs needed for
  this estimation without initiating the class.

  ```python
  n = 100000
  eta = 156
  omega = 169.69608
  error = 0.01
  qml.resource.FirstQuantization.qubit_cost(n, eta, omega, error)
  ```
  
  ```pycon
  4377
  ```

  In addition to the number of non-Clifford gates and logical qubits, some other quantities such as
  the 1-norm of the Hamiltonian and double factorization of the second-quantized Hamiltonian can be
  obtained either by initiating the classes or by directly calling the functions.

* `DefaultQubit` devices now natively support parameter broadcasting
  and `qml.gradients.param_shift` allows to make use of broadcasting.
  [(#2627)](https://github.com/PennyLaneAI/pennylane/pull/2627)

  Instead of utilizing the `broadcast_expand` transform, `DefaultQubit`-based
  devices now are able to directly execute broadcasted circuits, providing
  a faster way of executing the same circuit at varied parameter positions.

  Given a standard `QNode`,

  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def circuit(x, y):
      qml.RX(x, wires=0)
      qml.RY(y, wires=0)
      return qml.expval(qml.PauliZ(0))
  ```

  we can call it with broadcasted parameters:

  ```pycon
  >>> x = np.array([0.4, 1.2, 0.6], requires_grad=True)
  >>> y = np.array([0.9, -0.7, 4.2], requires_grad=True)
  >>> circuit(x, y)
  tensor([ 0.5725407 ,  0.2771465 , -0.40462972], requires_grad=True)
  ```

  It's also possible to broadcast only some parameters:

  ```pycon
  >>> x = np.array([0.4, 1.2, 0.6], requires_grad=True)
  >>> y = np.array(0.23, requires_grad=True)
  >>> circuit(x, y)
  tensor([0.89680614, 0.35281557, 0.80360155], requires_grad=True)
  ```

* Added the new optimizer, `qml.SPSAOptimizer` that implements the simultaneous
  perturbation stochastic approximation method based on
  [An Overview of the Simultaneous Perturbation Method for Efficient Optimization](https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF).
  [(#2661)](https://github.com/PennyLaneAI/pennylane/pull/2661)

  It is a suitable optimizer for cost functions whose evaluation may involve
  noise, as optimization with SPSA may significantly decrease the number of
  quantum executions for the entire optimization.

  ```pycon
  >>> dev = qml.device("default.qubit", wires=1)
  >>> def circuit(params):
  ...     qml.RX(params[0], wires=0)
  ...     qml.RY(params[1], wires=0)
  >>> coeffs = [1, 1]
  >>> obs = [qml.PauliX(0), qml.PauliZ(0)]
  >>> H = qml.Hamiltonian(coeffs, obs)
  >>> @qml.qnode(dev)
  ... def cost(params):
  ...     circuit(params)
  ...     return qml.expval(H)
  >>> params = np.random.normal(0, np.pi, (2), requires_grad=True)
  >>> print(params)
  [-5.92774911 -4.26420843]
  >>> print(cost(params))
  0.43866366253270167
  >>> max_iterations = 50
  >>> opt = qml.SPSAOptimizer(maxiter=max_iterations)
  >>> for _ in range(max_iterations):
  ...     params, energy = opt.step_and_cost(cost, params)
  >>> print(params)
  [-6.21193761 -2.99360548]
  >>> print(energy)
  -1.1258709813834058
  ```

* Differentiable zero-noise-extrapolation error mitigation via ``qml.transforms.mitigate_with_zne`` with ``qml.transforms.fold_global`` and ``qml.transforms.poly_extrapolate``.
  [(#2757)](https://github.com/PennyLaneAI/pennylane/pull/2757)

  When using a noisy or real device, you can now create a differentiable mitigated qnode that internally executes folded circuits that increase the noise and extrapolating with a polynomial fit back to zero noise. There will be an accompanying demo on this, see [(PennyLaneAI/qml/529)](https://github.com/PennyLaneAI/qml/pull/529).

  ```python
  # Describe noise
  noise_gate = qml.DepolarizingChannel
  noise_strength = 0.1

  # Load devices
  dev_ideal = qml.device("default.mixed", wires=n_wires)
  dev_noisy = qml.transforms.insert(noise_gate, noise_strength)(dev_ideal)

  scale_factors = [1, 2, 3]
  @mitigate_with_zne(
    scale_factors,
    qml.transforms.fold_global,
    qml.transforms.poly_extrapolate,
    extrapolate_kwargs={'order': 2}
  )
  @qml.qnode(dev_noisy)
  def qnode_mitigated(theta):
      qml.RY(theta, wires=0)
      return qml.expval(qml.PauliX(0))

  theta = np.array(0.5, requires_grad=True)
  grad = qml.grad(qnode_mitigated)
  >>> grad(theta)
  0.5712737447327619
  ```

* The quantum information module now supports computation of relative entropy.
  [(#2772)](https://github.com/PennyLaneAI/pennylane/pull/2772)

  It includes a function in `qml.math`:

  ```pycon
  >>> rho = np.array([[0.3, 0], [0, 0.7]])
  >>> sigma = np.array([[0.5, 0], [0, 0.5]])
  >>> qml.math.relative_entropy(rho, sigma)
  tensor(0.08228288, requires_grad=True)
  ```

  as well as a QNode transform:

  ```python
  dev = qml.device('default.qubit', wires=2)

  @qml.qnode(dev)
  def circuit(param):
      qml.RY(param, wires=0)
      qml.CNOT(wires=[0, 1])
      return qml.state()
  ```

  ```pycon
  >>> relative_entropy_circuit = qml.qinfo.relative_entropy(circuit, circuit, wires0=[0], wires1=[0])
  >>> x, y = np.array(0.4), np.array(0.6)
  >>> relative_entropy_circuit((x,), (y,))
  0.017750012490703237
  ```
  
* New PennyLane-inspired `sketch` and `sketch_dark` styles are now available for
  drawing circuit diagram graphics.
  [(#2709)](https://github.com/PennyLaneAI/pennylane/pull/2709)

* Added `QutritDevice` as an abstract base class for qutrit devices.
  ([#2781](https://github.com/PennyLaneAI/pennylane/pull/2781), [#2782](https://github.com/PennyLaneAI/pennylane/pull/2782))
* Added operation `qml.QutritUnitary` for applying user-specified unitary operations on qutrit devices.
  [(#2699)](https://github.com/PennyLaneAI/pennylane/pull/2699)

* Added `default.qutrit` plugin for pure state simulation of qutrits. Currently supports operation `qml.QutritUnitary` and measurements `qml.state()`, `qml.probs()`.
  [(#2783)](https://github.com/PennyLaneAI/pennylane/pull/2783)

  ```pycon
  >>> dev = qml.device("default.qutrit", wires=1)
  >>> @qml.qnode(dev)
  ... def circuit(U):
  ...     qml.QutritUnitary(U, wires=0)
  ...     return qml.probs(wires=0)
  >>> U = np.array([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
  >>> print(circuit(U))
  [0.5 0.5 0. ]
  ```
  
**Operator Arithmetic:**

* `default.qubit` now will natively execute any operation that defines a matrix except
  for trainable `Pow` operations. This includes custom operations, `GroverOperator`, `QFT`,
  `U1`, `U2`, `U3`, and arithmetic operations. The existance of a matrix is determined by the
  `Operator.has_matrix` property.

* When adjoint differentiation is requested, circuits are now decomposed so
  that all trainable operations have a generator.
  [(#2836)](https://github.com/PennyLaneAI/pennylane/pull/2836)

* Adds the `Controlled` symbolic operator to represent a controlled version of any
  operation.
  [(#2634)](https://github.com/PennyLaneAI/pennylane/pull/2634)

* Adds a base class `qml.ops.op_math.SymbolicOp` for single-operator symbolic
  operators such as `Adjoint` and `Pow`.
  [(#2721)](https://github.com/PennyLaneAI/pennylane/pull/2721)

* A `Sum` symbolic class is added that allows users to represent the sum of operators.
  [(#2475)](https://github.com/PennyLaneAI/pennylane/pull/2475)

  The `Sum` class provides functionality like any other PennyLane operator. We can
  get the matrix, eigenvalues, terms, diagonalizing gates and more.

  ```pycon
  >>> summed_op = qml.op_sum(qml.PauliX(0), qml.PauliZ(0))
  >>> summed_op
  PauliX(wires=[0]) + PauliZ(wires=[0])
  >>> qml.matrix(summed_op)
  array([[ 1,  1],
         [ 1, -1]])
  >>> summed_op.terms()
  ([1.0, 1.0], (PauliX(wires=[0]), PauliZ(wires=[0])))
  ```

  The `summed_op` can also be used inside a `qnode` as an observable.
  If the circuit is parameterized, then we can also differentiate through the
  sum observable.

  ```python
  sum_op = Sum(qml.PauliX(0), qml.PauliZ(1))
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev, grad_method="best")
  def circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(weights[2], wires=1)
        return qml.expval(sum_op)
  ```

  ```pycon
  >>> weights = qnp.array([0.1, 0.2, 0.3], requires_grad=True)
  >>> qml.grad(circuit)(weights)
  tensor([-0.09347337, -0.18884787, -0.28818254], requires_grad=True)
  ```

* Added `__add__` and `__pow__` dunder methods to the `qml.operation.Operator` class so that users can combine operators
  more naturally. [(#2807)](https://github.com/PennyLaneAI/pennylane/pull/2807)

  ```pycon
  >>> sum_op = qml.RX(phi=1.23, wires=0) + qml.RZ(phi=3.14, wires=0)
  >>> sum_op
  RX(1.23, wires=[0]) + RZ(3.14, wires=[0])
  >>> exp_op = qml.RZ(1.0, wires=0) ** 2
  >>> exp_op
  RZ**2(1.0, wires=[0])
  ```

* Added `__mul__` and `__matmul__` dunder methods to the `qml.operation.Operator` class so
  that users can combine operators more naturally.
  [(#2891)](https://github.com/PennyLaneAI/pennylane/pull/2891)

  ```pycon
  >>> prod_op = qml.RX(1, wires=0) @ qml.RY(2, wires=0)
  >>> prod_op
  PauliX(wires=[0]) @ PauliZ(wires=[1])
  >>> sprod_op = 6 * qml.RX(1, 0)
  >>> sprod_op
  6*(RX(1, wires=[0]))
  ```

* Added support for addition of operators and scalars. [(#2849)](https://github.com/PennyLaneAI/pennylane/pull/2849)

  ```pycon
  >>> sum_op = 5 + qml.PauliX(0)
  >>> sum_op.matrix()
  array([[5., 1.],
         [1., 5.]])
  ```

  Added `__neg__` and `__sub__` dunder methods to the `qml.operation.Operator` class so that users
  can negate and substract operators more naturally.

  ```pycon
  >>> -(-qml.PauliZ(0) + qml.PauliX(0)).matrix()
  array([[ 1, -1],
        [-1, -1]])
  ```

* A `SProd` symbolic class is added that allows users to represent the scalar product
of operators. [(#2622)](https://github.com/PennyLaneAI/pennylane/pull/2622)

  We can get the matrix, eigenvalues, terms, diagonalizing gates and more.

  ```pycon
  >>> sprod_op = qml.s_prod(2.0, qml.PauliX(0))
  >>> sprod_op
  2.0*(PauliX(wires=[0]))
  >>> sprod_op.matrix()
  array([[ 0., 2.],
         [ 2., 0.]])
  >>> sprod_op.terms()
  ([2.0], [PauliX(wires=[0])])
  ```

  The `sprod_op` can also be used inside a `qnode` as an observable.
  If the circuit is parameterized, then we can also differentiate through the observable.

  ```python
  dev = qml.device("default.qubit", wires=1)

  @qml.qnode(dev, grad_method="best")
  def circuit(scalar, theta):
        qml.RX(theta, wires=0)
        return qml.expval(qml.s_prod(scalar, qml.Hadamard(wires=0)))
  ```

  ```pycon
  >>> scalar, theta = (1.2, 3.4)
  >>> qml.grad(circuit, argnum=[0,1])(scalar, theta)
  (array(-0.68362956), array(0.21683382))
  ```

* Added `arithmetic_depth` property and `simplify` method to the `Operator`, `Sum`, `Adjoint`
and `SProd` operators so that users can reduce the depth of nested operators.

```pycon
>>> sum_op = qml.ops.Sum(qml.RX(phi=1.23, wires=0), qml.ops.Sum(qml.RZ(phi=3.14, wires=0), qml.PauliZ(0)))
>>> sum_op.arithmetic_depth
2
>>> simplified_op = sum_op.simplify()
>>> simplified_op.arithmetic_depth
1
```

* A `Prod` symbolic class is added that allows users to represent the Prod of operators.
  [(#2625)](https://github.com/PennyLaneAI/pennylane/pull/2625)

  The `Prod` class provides functionality like any other PennyLane operator. We can
  get the matrix, eigenvalues, terms, diagonalizing gates and more.

  ```pycon
  >>> prop_op = Prod(qml.PauliX(0), qml.PauliZ(0))
  >>> prop_op
  PauliX(wires=[0]) @ PauliZ(wires=[0])
  >>> qml.matrix(prop_op)
  array([[ 0,  -1],
         [ 1,   0]])
  >>> prop_op.terms()
  ([1.0], [PauliX(wires=[0]) @ PauliZ(wires=[0])])
  ```

  The `prod_op` can also be used inside a `qnode` as an observable.
  If the circuit is parameterized, then we can also differentiate through the
  product observable.

  ```python
  prod_op = Prod(qml.PauliZ(wires=0), qml.Hadamard(wires=1))
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def circuit(weights):
      qml.RX(weights[0], wires=0)
      return qml.expval(prod_op)
  ```

  ```pycon
  >>> weights = qnp.array([0.1], requires_grad=True)
  >>> qml.grad(circuit)(weights)
  tensor([-0.07059288589999416], requires_grad=True)
  ```
  
  The `prod_op` can also be used inside a `qnode` as an operation which,
  if parameterized, can be differentiated.

  ```python
  dev = qml.device("default.qubit", wires=3)

  @qml.qnode(dev)
  def circuit(theta):
      qml.prod(qml.PauliZ(0), qml.RX(theta, 1))
      return qml.expval(qml.PauliZ(1))
  ```

  ```pycon
  >>> circuit(1.23)
  tensor(0.33423773, requires_grad=True)
  >>> qml.grad(circuit)(1.23)
  -0.9424888019316975
  ```

* New FlipSign operator that flips the sign for a given basic state. [(#2780)](https://github.com/PennyLaneAI/pennylane/pull/2780)

* Added `qml.counts` which samples from the supplied observable returning the number of counts
  for each sample.
  [(#2686)](https://github.com/PennyLaneAI/pennylane/pull/2686)
  [(#2839)](https://github.com/PennyLaneAI/pennylane/pull/2839)
  [(#2876)](https://github.com/PennyLaneAI/pennylane/pull/2876)

  Note that the change included creating a new `Counts` measurement type in `measurements.py`.

  `qml.counts` can be used to obtain counted raw samples in the computational basis:

  ```pycon
  >>> dev = qml.device("default.qubit", wires=2, shots=1000)
  >>>
  >>> @qml.qnode(dev)
  >>> def circuit():
  ...     qml.Hadamard(wires=0)
  ...     qml.CNOT(wires=[0, 1])
  ...     return qml.counts()
  >>> result = circuit()
  >>> print(result)
  {'00': 495, '11': 505}
  ```

  Counts can also be obtained when sampling the eigenstates of an observable:

  ```pycon
  >>> dev = qml.device("default.qubit", wires=2, shots=1000)
  >>>
  >>> @qml.qnode(dev)
  >>> def circuit():
  ...   qml.Hadamard(wires=0)
  ...   qml.CNOT(wires=[0, 1])
  ...   return qml.counts(qml.PauliZ(0)), qml.counts(qml.PauliZ(1))
  >>> result = circuit()
  >>> print(result)
  ({-1: 470, 1: 530}, {-1: 470, 1: 530})
  ```

<h3>Improvements</h3>

* The efficiency of the Hartree-Fock workflow is improved by removing the repetitive basis set
  normalisation steps and modifying how the permutational symmetries are applied to avoid repetitive
  electron repulsion integral calculations.
  [(#2850)](https://github.com/PennyLaneAI/pennylane/pull/2850)

* The coefficients of the non-differentiable molecular Hamiltonians generated with openfermion have
  `requires_grad = False` by default.
  [(#2865)](https://github.com/PennyLaneAI/pennylane/pull/2865)

* A small performance upgrade to the `compute_matrix` method
  of broadcastable parametric operations.
  [(#2726)](https://github.com/PennyLaneAI/pennylane/pull/2726)

* Jacobians are cached with the Autograd interface when using the
  parameter-shift rule.
  [(#2645)](https://github.com/PennyLaneAI/pennylane/pull/2645)

* The `qml.state` and `qml.density_matrix` measurements now support custom wire
  labels.
  [(#2779)](https://github.com/PennyLaneAI/pennylane/pull/2779)

* Add trivial behaviour logic to `qml.operation.expand_matrix`.
  [(#2785)](https://github.com/PennyLaneAI/pennylane/issues/2785)

* Adds a new function to compare operators. `qml.equal` can be used to compare equality of parametric operators taking
  into account their interfaces and trainability.
  [(#2651)](https://github.com/PennyLaneAI/pennylane/pull/2651)

* The `default.mixed` device now supports backpropagation with the `"jax"` interface.
  [(#2754)](https://github.com/PennyLaneAI/pennylane/pull/2754)

* Quantum channels such as `qml.BitFlip` now support abstract tensors. This allows
  their usage inside QNodes decorated by `tf.function`, `jax.jit`, or `jax.vmap`:

  ```python
  dev = qml.device("default.mixed", wires=1)

  @qml.qnode(dev, diff_method="backprop", interface="jax")
  def circuit(t):
      qml.PauliX(wires=0)
      qml.ThermalRelaxationError(0.1, t, 1.4, 0.1, wires=0)
      return qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> x = jnp.array([0.8, 1.0, 1.2])
  >>> jax.vmap(circuit)(x)
  DeviceArray([-0.78849435, -0.8287073 , -0.85608006], dtype=float32)
  ```

* Added an `are_pauli_words_qwc` function which checks if certain
  Pauli words are pairwise qubit-wise commuting. This new function improves performance when measuring hamiltonians
  with many commuting terms.
  [(#2789)](https://github.com/PennyLaneAI/pennylane/pull/2798)

* Adjoint differentiation now uses the adjoint symbolic wrapper instead of in-place inversion.
  [(#2855)](https://github.com/PennyLaneAI/pennylane/pull/2855)

* Automatic circuit cutting is improved by making better partition imbalance derivations.
  Now it is more likely to generate optimal cuts for larger circuits.
  [(#2517)](https://github.com/PennyLaneAI/pennylane/pull/2517)

<h3>Breaking changes</h3>

* The deprecated `qml.hf` module is removed. The `qml.hf` functionality is fully supported by
  `qml.qchem`.
  [(#2795)](https://github.com/PennyLaneAI/pennylane/pull/2795)

* Custom devices inheriting from `DefaultQubit` or `QubitDevice` can break due to the introduction
  of parameter broadcasting.
  [(#2627)](https://github.com/PennyLaneAI/pennylane/pull/2627)

  A custom device should only break if all three following statements hold simultaneously:

  1. The custom device inherits from `DefaultQubit`, not `QubitDevice`.
  2. The device implements custom methods in the simulation pipeline that are incompatible
     with broadcasting (for example `expval`, `apply_operation` or `analytic_probability`).
  3. The custom device maintains the flag `"supports_broadcasting": False` in its `capabilities`
     dictionary *or* it overwrites `Device.batch_transform` without applying `broadcast_expand`
     (or both).

  The `capabilities["supports_broadcasting"]` is set to `True` for
  `DefaultQubit`. Therefore typically, the easiest fix will be to change the
  `capabilities["supports_broadcasting"]` flag to `False` for the child device
  and/or to include a call to `broadcast_expand` in
  `CustomDevice.batch_transform`, similar to how `Device.batch_transform` calls
  it.

  Separately from the above, custom devices that inherit from `QubitDevice` and implement a
  custom `_gather` method need to allow for the kwarg `axis` to be passed to this `_gather` method.

* PennyLane now depends on newer versions (>=2.7) of the `semantic_version` package,
  which provides an updated API that is incompatible which versions of the package prior to 2.7.
  If you run into issues relating to this package, please reinstall PennyLane.
  [(#2744)](https://github.com/PennyLaneAI/pennylane/pull/2744)
  [(#2767)](https://github.com/PennyLaneAI/pennylane/pull/2767)

* The argument `argnum` of the function `qml.batch_input` has been redefined: now it indicates the
  indices of the batched parameters, which need to be non-trainable, in the quantum tape. Consequently, its default
  value (set to 0) has been removed.
  [(#2873)](https://github.com/PennyLaneAI/pennylane/pull/2873)

  Before this breaking change, one could call `qml.batch_input` without any arguments when using
  batched inputs as the first argument of the quantum circuit.

  ```python
  dev = qml.device("default.qubit", wires=2, shots=None)

  @qml.batch_input()  # argnum = 0
  @qml.qnode(dev, diff_method="parameter-shift", interface="tf")
  def circuit(inputs, weights):  # argument `inputs` is batched
      qml.RY(weights[0], wires=0)
      qml.AngleEmbedding(inputs, wires=range(2), rotation="Y")
      qml.RY(weights[1], wires=1)
      return qml.expval(qml.PauliZ(1))
  ```

  With this breaking change, users must set a value to `argnum` specifying the index of the
  batched inputs with respect to all quantum tape parameters. In this example the quantum tape
  parameters are `[ weights[0], inputs, weights[1] ]`, thus `argnum` should be set to 1, specifying
  that `inputs` is batched:

  ```python
  dev = qml.device("default.qubit", wires=2, shots=None)

  @qml.batch_input(argnum=1)
  @qml.qnode(dev, diff_method="parameter-shift", interface="tf")
  def circuit(inputs, weights):
      qml.RY(weights[0], wires=0)
      qml.AngleEmbedding(inputs, wires=range(2), rotation="Y")
      qml.RY(weights[1], wires=1)
      return qml.expval(qml.PauliZ(1))
  ```

* Adds `expm` to the `pennylane.math` module for matrix exponentiation.
  [(#2890)](https://github.com/PennyLaneAI/pennylane/pull/2890)

<h3>Deprecations</h3>

<h3>Documentation</h3>

* Added a dedicated docstring for the `QubitDevice.sample` method.
  [(#2812)](https://github.com/PennyLaneAI/pennylane/pull/2812)

* Optimization examples of using JAXopt and Optax with the JAX interface have
  been added.
  [(#2769)](https://github.com/PennyLaneAI/pennylane/pull/2769)

<h3>Bug fixes</h3>

* Updated IsingXY gate doc-string.
  [(#2858)](https://github.com/PennyLaneAI/pennylane/pull/2858)

* Fixes a bug where the parameter-shift gradient breaks when using both
  custom `grad_recipe`s that contain unshifted terms and recipes that
  do not contains any unshifted terms.
  [(#2834)](https://github.com/PennyLaneAI/pennylane/pull/2834)

* Fixes mixed CPU-GPU data-locality issues for Torch interface.
  [(#2830)](https://github.com/PennyLaneAI/pennylane/pull/2830)

* Fixes a bug where the parameter-shift Hessian of circuits with untrainable
  parameters might be computed with respect to the wrong parameters or
  might raise an error.
  [(#2822)](https://github.com/PennyLaneAI/pennylane/pull/2822)

* Fixes a bug where the custom implementation of the `states_to_binary` device
  method was not used.
  [(#2809)](https://github.com/PennyLaneAI/pennylane/pull/2809)

* `qml.grouping.group_observables` now works when individual wire
  labels are iterable.
  [(#2752)](https://github.com/PennyLaneAI/pennylane/pull/2752)

* The adjoint of an adjoint has a correct `expand` result.
  [(#2766)](https://github.com/PennyLaneAI/pennylane/pull/2766)

* Fix the ability to return custom objects as the expectation value of a QNode with the Autograd interface.
  [(#2808)](https://github.com/PennyLaneAI/pennylane/pull/2808)

* The WireCut operator now raises an error when instantiating it with an empty list.
  [(#2826)](https://github.com/PennyLaneAI/pennylane/pull/2826)

* Allow hamiltonians with grouped observables to be measured on devices
  which were transformed using `qml.transform.insert()`.
  [(#2857)](https://github.com/PennyLaneAI/pennylane/pull/2857)

* Fixes a bug where `qml.batch_input` raised an error when using a batched operator that was not
  located at the beginning of the circuit. In addition, now `qml.batch_input` raises an error when
  using trainable batched inputs, which avoids an unwanted behaviour with duplicated parameters.
  [(#2873)](https://github.com/PennyLaneAI/pennylane/pull/2873)

* Calling `qml.equal` with nested operators now raises a NotImplementedError.
  [(#2877)](https://github.com/PennyLaneAI/pennylane/pull/2877)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Samuel Banning, Juan Miguel Arrazola, Utkarsh Azad, David Ittah, Soran Jahangiri, Edward Jiang,
Ankit Khandelwal, Meenu Kumari, Christina Lee, Sergio Martínez-Losa, Albert Mitjans Coma,
Ixchel Meza Chavez, Romain Moyard, Zeyue Niu, Lee James O'Riordan, Mudit Pandey, Bogdan Reznychenko,
Shuli Shu, Jay Soni, Modjtaba Shokrian-Zini, Antal Száva, David Wierichs, Moritz Willmann
