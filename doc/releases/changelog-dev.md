:orphan:

# Release 0.25.0-dev (development release)

<h3>New features since last release</h3>

<h4>Estimate computational requirements ⚙️</h4>

* Functionality for estimating molecular simulation computations has been added.
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

  The new `qml.resource` module allows you to estimate the number of non-[Clifford gates](https://en.wikipedia.org/wiki/Clifford_gates) 
  and logical qubits needed to implement quantum phase estimation algorithms 
  for simulating materials and molecules. This includes support for quantum 
  algorithms using first and second quantization with specific bases:

  - First quantization using a plane-wave basis via the `FirstQuantization` class:

  ```python
  import pennylane as qml
  from pennylane import numpy as np
  
  n = 100000        # number of plane waves
  eta = 156         # number of electrons
  omega = 1145.166  # unit cell volume in atomic units
  
  algo = FirstQuantization(n, eta, omega)
  ```
  
  ```pycon
  >>> print(algo.gates, algo.qubits)
  1.10e+13, 4416
  ```

  - Second quantization with a double-factorized Hamiltonian via the 
  `DoubleFactorization` class: 
 
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
  ```

  ```pycon
  >>> print(algo.gates, algo.qubits)
  103969925, 290
  ```

  The methods of the `FirstQuantization` and the `DoubleFactorization` classes 
  can be also accessed individually without initiating the class:

  - The number of logical qubits with `qubit_cost`:

  ```python
  n = 100000
  eta = 156
  omega = 169.69608
  error = 0.01
  ```
  
  ```pycon
  >>> qml.resource.FirstQuantization.qubit_cost(n, eta, omega, error)
  4377
  ```

  - The number of non-Clifford gates with `gate_cost` (TODO: check this):

   ```python
    n = 100000
    eta = 156
    omega = 169.69608
    error = 0.01
    ```
    
    ```pycon
    >>> qml.resource.FirstQuantization.gate_cost(n, eta, omega, error)
    TODO
    ``` 

<h4>Differentiable error mitigation 🧑‍🔧</h4>

  TODO 

* Differentiable zero-noise-extrapolation error mitigation is now available.
  [(#2757)](https://github.com/PennyLaneAI/pennylane/pull/2757)

  When using a noisy device (simulator or real hardware), you can now create a 
  differentiable-mitigated qnode that internally executes *folded* circuits that 
  increase the noise and extrapolating with a polynomial fit back to zero noise. 
  There will be an accompanying demo on this, see [(PennyLaneAI/qml/529)](https://github.com/PennyLaneAI/qml/pull/529).

  ``qml.transforms.mitigate_with_zne`` with ``qml.transforms.fold_global`` and ``qml.transforms.poly_extrapolate``.

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
  
<h4>Native support for parameter broadcasting for `DefaultQubit` devices 📡</h4>

* `DefaultQubit` devices now natively support parameter broadcasting.
  [(#2627)](https://github.com/PennyLaneAI/pennylane/pull/2627)
  
  Instead of utilizing the `broadcast_expand` transform, `DefaultQubit`-based
  devices now are able to directly execute broadcasted circuits, providing
  a faster way of executing the same circuit at varied parameter positions

  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def circuit(x, y):
      qml.RX(x, wires=0)
      qml.RY(y, wires=0)
      return qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> x = np.array([0.4, 1.2, 0.6], requires_grad=True)
  >>> y = np.array([0.9, -0.7, 4.2], requires_grad=True)
  >>> circuit(x, y)
  tensor([ 0.5725407 ,  0.2771465 , -0.40462972], requires_grad=True)
  ```

  It's also possible to broadcast only *some* parameters:

  ```pycon
  >>> x = np.array([0.4, 1.2, 0.6], requires_grad=True)
  >>> y = np.array(0.23, requires_grad=True)
  >>> circuit(x, y)
  tensor([0.89680614, 0.35281557, 0.80360155], requires_grad=True)
  ```

<h4>The SPSA optimizer 🦾</h4>

* The [simultaneous perturbation stochastic approximation (SPSA) optimizer](https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF) 
  is available via `qml.SPSAOptimizer`.
  [(#2661)](https://github.com/PennyLaneAI/pennylane/pull/2661)

  The SPSA optimizer is suitable for cost functions whose evaluation may involve
  noise, as optimization with SPSA may significantly decrease the number of
  quantum executions for the entire optimization. Use the SPSA optimizer like you
  would any other optimizer:

  ```python
  max_iterations = 50
  opt = qml.SPSA(maxiter=max_iterations) # TODO: check documentation for SPSA

  for _ in range(max_iterations):
      params, cost = opt.step_and_cost(cost, params)
  ```  


<h4>Relative entropy now available in `qml.qinfo`</h4>

* The quantum information module now supports computation of [relative entropy](https://en.wikipedia.org/wiki/Quantum_relative_entropy).
  [(#2772)](https://github.com/PennyLaneAI/pennylane/pull/2772)

  To calculate relative entropy, one requires two quantum states and subspaces over 
  which you would like to calculate the relative entropy. We've enabled two ways 
  to calculate the relative entropy:
  
  - A QNode transform via `qml.qinfo.relative_entropy`:

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

  - Relative entropy support in `qml.math` for flexible post-processing:

  ```pycon
  >>> rho = np.array([[0.3, 0], [0, 0.7]])
  >>> sigma = np.array([[0.5, 0], [0, 0.5]])
  >>> qml.math.relative_entropy(rho, sigma)
  tensor(0.08228288, requires_grad=True)
  ```

* New PennyLane-inspired `sketch` and `sketch_dark` styles are now available for drawing circuit diagram graphics.
  [(#2709)](https://github.com/PennyLaneAI/pennylane/pull/2709)

**Operator Arithmetic:**

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

  ```python
  >>> summed_op = qml.RX(phi=1.23, wires=0) + qml.RZ(phi=3.14, wires=0)
  >>> summed_op
  RX(1.23, wires=[0]) + RZ(3.14, wires=[0])
  >>> exp_op = qml.RZ(1.0, wires=0) ** 2
  >>> exp_op
  RZ**2(1.0, wires=[0])
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

<h3>Breaking changes</h3>

* The deprecated `qml.hf` module is removed. The `qml.hf` functionality is fully supported by
  `qml.qchem`.
  [(#2795)](https://github.com/PennyLaneAI/pennylane/pull/2795)

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

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Utkarsh Azad, Samuel Banning, Isaac De Vlugt, David Ittah, Soran Jahangiri, Edward Jiang,
Ankit Khandelwal, Christina Lee, Sergio Martínez-Losa, Albert Mitjans Coma, Ixchel Meza Chavez,
Romain Moyard, Lee James O'Riordan, Mudit Pandey, Bogdan Reznychenko, Shuli Shu, Jay Soni,
Modjtaba Shokrian-Zini, Antal Száva, David Wierichs, Moritz Willmann
