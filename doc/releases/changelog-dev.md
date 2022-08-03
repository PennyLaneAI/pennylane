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

* New PennyLane-inspired `sketch` and `sketch_dark` styles are now available for drawing circuit diagram graphics.
  [(#2709)](https://github.com/PennyLaneAI/pennylane/pull/2709)

* Added `QutritDevice` as an abstract base class for qutrit devices.
  [(#2781)](https://github.com/PennyLaneAI/pennylane/pull/2781)
  * Added operation `qml.QutritUnitary` for applying user-specified unitary operations on qutrit devices.
  [(#2699)](https://github.com/PennyLaneAI/pennylane/pull/2699)

**Operator Arithmetic:**

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

  ```
  >>> weights = qnp.array([0.1, 0.2, 0.3], requires_grad=True)
  >>> qml.grad(circuit)(weights)
  tensor([-0.09347337, -0.18884787, -0.28818254], requires_grad=True)
  ```
* New FlipSign operator that flips the sign for a given basic state. [(#2780)](https://github.com/PennyLaneAI/pennylane/pull/2780)


<h3>Improvements</h3>

* Jacobians are cached with the Autograd interface when using the
  parameter-shift rule.
  [(#2645)](https://github.com/PennyLaneAI/pennylane/pull/2645)

* Samples can be grouped into counts by passing the `counts=True` flag to `qml.sample`.
  [(#2686)](https://github.com/PennyLaneAI/pennylane/pull/2686)

  Note that the change included creating a new `Counts` measurement type in `measurements.py`.

  `counts=True` can be set when obtaining raw samples in the computational basis:

  ```pycon
  >>> dev = qml.device("default.qubit", wires=2, shots=1000)
  >>>
  >>> @qml.qnode(dev)
  >>> def circuit():
  ...     qml.Hadamard(wires=0)
  ...     qml.CNOT(wires=[0, 1])
  ...     # passing the counts flag
  ...     return qml.sample(counts=True)   
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
  ...   return qml.sample(qml.PauliZ(0), counts=True), qml.sample(qml.PauliZ(1), counts=True)
  >>> result = circuit()
  >>> print(result)
  [tensor({-1: 526, 1: 474}, dtype=object, requires_grad=True)
   tensor({-1: 526, 1: 474}, dtype=object, requires_grad=True)]
  ```

* The `qml.state` and `qml.density_matrix` measurements now support custom wire
  labels.
  [(#2779)](https://github.com/PennyLaneAI/pennylane/pull/2779)

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

<h3>Breaking changes</h3>

* PennyLane now depends on newer versions (>=2.7) of the `semantic_version` package,
  which provides an updated API that is incompatible which versions of the package prior to 2.7.
  If you run into issues relating to this package, please reinstall PennyLane.
  [(#2744)](https://github.com/PennyLaneAI/pennylane/pull/2744)
  [(#2767)](https://github.com/PennyLaneAI/pennylane/pull/2767)

<h3>Deprecations</h3>

<h3>Documentation</h3>

* Added a dedicated docstring for the `QubitDevice.sample` method.
  [(#2812)](https://github.com/PennyLaneAI/pennylane/pull/2812)

* Optimization examples of using JAXopt and Optax with the JAX interface have
  been added.
  [(#2769)](https://github.com/PennyLaneAI/pennylane/pull/2769)

<h3>Bug fixes</h3>

* Fixes a bug where the custom implementation of the `states_to_binary` device
  method was not used.
  [(#2809)](https://github.com/PennyLaneAI/pennylane/pull/2809)
  
* `qml.grouping.group_observables` now works when individual wire
  labels are iterable.
  [(#2752)](https://github.com/PennyLaneAI/pennylane/pull/2752)

* The adjoint of an adjoint has a correct `expand` result.
  [(#2766)](https://github.com/PennyLaneAI/pennylane/pull/2766)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

David Ittah, Edward Jiang, Ankit Khandelwal, Meenu Kumari, Christina Lee, Sergio Martínez-Losa,
Ixchel Meza Chavez, Lee James O'Riordan, Mudit Pandey, Bogdan Reznychenko,
Jay Soni, Antal Száva, David Wierichs, Moritz Willmann
