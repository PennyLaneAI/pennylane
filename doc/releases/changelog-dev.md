:orphan:

# Release 0.25.0-dev (development release)

<h3>New features since last release</h3>

<h4>Estimate computational resource requirements 🧠</h4>

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

  The new `qml.resource` module allows you to estimate the number of 
  non-[Clifford gates](https://en.wikipedia.org/wiki/Clifford_gates) and logical 
  qubits needed to implement quantum phase estimation algorithms for simulating 
  materials and molecules. This includes support for quantum algorithms using first 
  and second quantization with specific bases:

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

  - The number of logical qubits with `qubit_cost`
  - The number of non-Clifford gates with `gate_cost`

  ```python
  n = 100000
  eta = 156
  omega = 169.69608
  error = 0.01
  ```
  
  ```pycon
  >>> qml.resource.FirstQuantization.qubit_cost(n, eta, omega, error)
  4377
  >>> qml.resource.FirstQuantization.gate_cost(n, eta, omega, error)
  3676557345574 
  ```

<h4>Differentiable error mitigation ⚙️</h4>

* Differentiable zero-noise-extrapolation error mitigation is now available.
  [(#2757)](https://github.com/PennyLaneAI/pennylane/pull/2757)

  In zero-noise-extrapolation (ZNE), the noise of a circuit is artificially increased by _folding_
  the circuit. By executing an expectation value with increasing noise, we can extrapolate back to
  zero noise by means of a polynomial fit. More details on this can be found in the details section
  of the [documentation](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.transforms.fold_global.html)
  and a soon-to-appear demo [(PennyLaneAI/qml/529)](https://github.com/PennyLaneAI/qml/pull/529).

  The new feature in this release is that ZNE that is now *differentiable*, meaning that you can now elevate any variational
  quantum algorithm that is written in PennyLane to a *mitigated* algorithm with improved results on noisy hardware while maintaining differentiability throughout.

  In order to do so, use the `qml.transforms.mitigate_with_zne` transform on your QNode and provide the PennyLane proprietary
  `qml.transforms.fold_global` folding function and `qml.transforms.poly_extrapolate` extrapolation function. Here is an example for a noisy simulation device
  where we mitigate a QNode and still are able to compute the gradient:

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

<h4>More native support for parameter broadcasting with `DefaultQubit` devices 📡</h4>

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
  
* Parameter-shift gradients now allow for parameter broadcasting. 
  [(#2749)](https://github.com/PennyLaneAI/pennylane/pull/2749)

  The gradient transform `qml.gradients.param_shift` now accepts the new Boolean 
  keyword argument `broadcast`. If it is set to `True`, broadcasting is used to 
  compute the derivative:

  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def circuit(x, y):
      qml.RX(x, wires=0)
      qml.CRY(y, wires=[0, 1])
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  ```

  ```pycon
  >>> x, y = np.array([0.4, 0.23], requires_grad=True)
  >>> qml.gradients.param_shift(circuit, broadcast=True)(x, y)
  (tensor(-0.38429095, requires_grad=True),
   tensor(0.00899816, requires_grad=True))
  ```

  To illustrate the speedup, consider the following figure which shows, for a 
  constant-depth circuit with Pauli rotations and controlled Pauli rotations, the 
  time required to compute `qml.gradients.param_shift(circuit, broadcast=False)(params)`
  ("No broadcasting") and `qml.gradients.param_shift(circuit, broadcast=True)(params)` 
  ("Broadcasting") as a function of the number of qubits.

  <img src="https://pennylane.readthedocs.io/en/latest/_images/default_qubit_native_broadcast_speedup.png" width=70%/>

  For transparency, the base code used for this example is given below:

  ```python
  dev = qml.device("default.qubit", shots=None, wires=num_wires) 

  def constant_depth_Pauli_rotations(param, wires):
      [qml.Hadamard(w) for w in wires]
      [qml.RX(p, w) for p, w in zip(param[0], wires)]
      [qml.RY(p, w) for p, w in zip(param[1], wires)]
      qml.broadcast(qml.CRX, wires=wires, pattern="ring", parameters=param[2])
      [qml.RY(p, w) for p, w in zip(param[3], wires)]
      [qml.RZ(p, w) for p, w in zip(param[4], wires)]
      qml.broadcast(qml.CRX, wires=wires, pattern="ring", parameters=param[5])

      return qml.probs(wires)

  for num_wires in list(range(3, 13)):
    dev = qml.device("default.qubit", wires=num_wires)
    circuit = qml.QNode(constant_depth_Pauli_rotations, device=dev)
    param = np.random.random(ansatz.shape_fn(num_wires))
    
    # time qml.gradients.param_shift(circuit, broadcast=False)(param) 
    # time qml.gradients.param_shift(circuit, broadcast=True)(param)
  ```

  Note that `QNodes` with multiple return values and shot vectors are not supported
  at this time and the Operators with trainable parameters are required to support 
  broadcasting when using `broadcast=True`. One way of checking the latter is with 
  `qml.ops.qubit.attributes.supports_broadcasting`:

  ```pycon
  >>> qml.RX in qml.ops.qubit.attributes.supports_broadcasting
  True
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

<h4>Relative entropy is now available in `qml.qinfo` 💥</h4>

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

  - Support in `qml.math` for flexible post-processing:

  ```pycon
  >>> rho = np.array([[0.3, 0], [0, 0.7]])
  >>> sigma = np.array([[0.5, 0], [0, 0.5]])
  >>> qml.math.relative_entropy(rho, sigma)
  tensor(0.08228288, requires_grad=True)
  ```

<h4>More drawing styles 🎨</h4>

* New PennyLane-inspired `sketch` and `sketch_dark` styles are now available for 
  drawing circuit diagram graphics.
  [(#2709)](https://github.com/PennyLaneAI/pennylane/pull/2709)

<h4>Quality-of-life upgrades to Operator arithmetic ➕➖✖️</h4>

TODO

* Many convenient additions have been made regarding Operator arithmetic.
  [(#2475)](https://github.com/PennyLaneAI/pennylane/pull/2475)
  [(#2622)](https://github.com/PennyLaneAI/pennylane/pull/2622)
  [(#2807)](https://github.com/PennyLaneAI/pennylane/pull/2807)
  [(#2849)](https://github.com/PennyLaneAI/pennylane/pull/2849)
  
  Many quality-of-life features have been added to how we handle Operators. This
  includes:

  - Add Operators via `qml.op_sum` to create a new Operator whose matrix, terms, 
  and eigenvalues can be accessed as per usual.

  ```pycon
  >>> summed_op = qml.op_sum(qml.PauliX(0), qml.PauliZ(1))
  >>> summed_op
  PauliX(wires=[0]) + PauliZ(wires=[1])
  >>> qml.matrix(summed_op)
  array([[ 1,  0,  1,  0],
         [ 0, -1,  0,  1],
         [ 1,  0,  1,  0],
         [ 0,  1,  0, -1]]) 
  >>> summed_op.terms()
  ([1.0, 1.0], (PauliX(wires=[0]), PauliZ(wires=[1])))
  ```

  Summed Operators can also be used inside of a QNode as observables while maintaining
  differentiability:

  ```python
  summed_op = qml.op_sum(qml.PauliX(0), qml.PauliZ(1))

  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def circuit(weights):
      qml.RX(weights[0], wires=0)
      qml.RY(weights[1], wires=1)
      qml.CNOT(wires=[0, 1])
      qml.RX(weights[2], wires=1)
      return qml.expval(summed_op)
  ```

  ```pycon
  >>> weights = qnp.array([0.1, 0.2, 0.3], requires_grad=True)
  >>> qml.grad(circuit)(weights)
  array([-0.09347337, -0.18884787, -0.28818254])
  ```

  - You can now add scalars to Operators, where the interpretation is that the 
  scalar is a properly-sized identity matrix. 

  ```pycon
  >>> sum_op = 5 + qml.PauliX(0)
  >>> sum_op.matrix()
  array([[5., 1.],
         [1., 5.]])
  ```

* Added `__add__` and `__pow__` dunder methods to the `qml.operation.Operator` class 
  so that users can combine operators more naturally.

  ```python
  >>> summed_op = qml.RX(phi=1.23, wires=0) + qml.RZ(phi=3.14, wires=0)
  >>> summed_op
  RX(1.23, wires=[0]) + RZ(3.14, wires=[0])
  >>> exp_op = qml.RZ(1.0, wires=0) ** 2
  >>> exp_op
  RZ**2(1.0, wires=[0])
  ```

* Added support for addition of operators and scalars. 

* A `SProd` symbolic class is added that allows users to represent the scalar product
of operators.

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

* Adds the `Controlled` symbolic operator to represent a controlled version of any
  operation.
  [(#2634)](https://github.com/PennyLaneAI/pennylane/pull/2634)

* New FlipSign operator that flips the sign for a given basic state. 
  [(#2780)](https://github.com/PennyLaneAI/pennylane/pull/2780)

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

<h3>Improvements 📈</h3>

* The efficiency of the Hartree-Fock workflow has been improved by removing 
  repetitive steps.
  [(#2850)](https://github.com/PennyLaneAI/pennylane/pull/2850)

* The coefficients of the non-differentiable molecular Hamiltonians generated 
  with openfermion now have `requires_grad = False` by default.
  [(#2865)](https://github.com/PennyLaneAI/pennylane/pull/2865)

* Upgraded performance of the `compute_matrix` method of broadcastable 
  parametric operations.
  [(#2726)](https://github.com/PennyLaneAI/pennylane/pull/2726)

* Jacobians are now cached with the Autograd interface when using the
  parameter-shift rule.
  [(#2645)](https://github.com/PennyLaneAI/pennylane/pull/2645)

* The `qml.state` and `qml.density_matrix` measurements now support custom wire
  labels.
  [(#2779)](https://github.com/PennyLaneAI/pennylane/pull/2779)

* Add trivial behaviour logic to `qml.operation.expand_matrix`.
  [(#2785)](https://github.com/PennyLaneAI/pennylane/issues/2785)

* Adds a new function to compare operators. `qml.equal` can be used to compare equality 
  of parametric operators taking into account their interfaces and trainability.
  [(#2651)](https://github.com/PennyLaneAI/pennylane/pull/2651)

* The `default.mixed` device now supports backpropagation with the `"jax"` interface.
  [(#2754)](https://github.com/PennyLaneAI/pennylane/pull/2754)

* Added an `are_pauli_words_qwc` function which checks if certain
  Pauli words are pairwise qubit-wise commuting. This new function improves performance 
  when measuring hamiltonians with many commuting terms.
  [(#2789)](https://github.com/PennyLaneAI/pennylane/pull/2798)

* Adjoint differentiation now uses the adjoint symbolic wrapper instead of in-place 
  inversion.
  [(#2855)](https://github.com/PennyLaneAI/pennylane/pull/2855)

<h3>Breaking changes 💔</h3>

* The deprecated `qml.hf` module is removed. The `qml.hf` functionality is now fully 
  supported by `qml.qchem`.
  [(#2795)](https://github.com/PennyLaneAI/pennylane/pull/2795)

* PennyLane now depends on newer versions (>=2.7) of the `semantic_version` package,
  which provides an updated API that is incompatible which versions of the package 
  prior to 2.7. If you run into issues relating to this package, please reinstall 
  PennyLane.
  [(#2744)](https://github.com/PennyLaneAI/pennylane/pull/2744)
  [(#2767)](https://github.com/PennyLaneAI/pennylane/pull/2767)

<h3>Deprecations 👋</h3>

🦗

<h3>Documentation 📕</h3>

* Added a dedicated docstring for the `QubitDevice.sample` method.
  [(#2812)](https://github.com/PennyLaneAI/pennylane/pull/2812)

* Optimization examples of using JAXopt and Optax with the JAX interface have
  been added.
  [(#2769)](https://github.com/PennyLaneAI/pennylane/pull/2769)

<h3>Bug fixes 🐞</h3>

* Updated IsingXY gate docstring.
  [(#2858)](https://github.com/PennyLaneAI/pennylane/pull/2858)

* Fixed a bug where the parameter-shift gradient breaks when using both
  custom `grad_recipe`s that contain unshifted terms and recipes that
  do not contain any unshifted terms.
  [(#2834)](https://github.com/PennyLaneAI/pennylane/pull/2834)

* Fixed mixed CPU-GPU data-locality issues for the Torch interface.
  [(#2830)](https://github.com/PennyLaneAI/pennylane/pull/2830)

* Fixed a bug where the parameter-shift Hessian of circuits with untrainable
  parameters might be computed with respect to the wrong parameters or
  might raise an error.
  [(#2822)](https://github.com/PennyLaneAI/pennylane/pull/2822)

* Fixed a bug where the custom implementation of the `states_to_binary` device
  method was not used.
  [(#2809)](https://github.com/PennyLaneAI/pennylane/pull/2809)

* `qml.grouping.group_observables` now works when individual wire
  labels are iterable.
  [(#2752)](https://github.com/PennyLaneAI/pennylane/pull/2752)

* The adjoint of an adjoint now has a correct `expand` result.
  [(#2766)](https://github.com/PennyLaneAI/pennylane/pull/2766)

* Fixed the ability to return custom objects as the expectation value of a QNode with 
  the Autograd interface.
  [(#2808)](https://github.com/PennyLaneAI/pennylane/pull/2808)

* The WireCut Operator now raises an error when instantiating it with an empty list.
  [(#2826)](https://github.com/PennyLaneAI/pennylane/pull/2826)

* Hamiltonians with grouped observables are now allowed to be measured on devices 
  which were transformed using `qml.transform.insert()`.
  [(#2857)](https://github.com/PennyLaneAI/pennylane/pull/2857)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Utkarsh Azad, Samuel Banning, Isaac De Vlugt, David Ittah, Soran Jahangiri, Edward Jiang,
Ankit Khandelwal, Christina Lee, Sergio Martínez-Losa, Albert Mitjans Coma, Ixchel Meza Chavez,
Romain Moyard, Lee James O'Riordan, Mudit Pandey, Bogdan Reznychenko, Shuli Shu, Jay Soni,
Modjtaba Shokrian-Zini, Antal Száva, David Wierichs, Moritz Willmann
