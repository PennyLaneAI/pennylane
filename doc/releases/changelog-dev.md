:orphan:

# Release 0.26.0-dev (development release)

<h3>New features since last release</h3>

* Added more than one rotation for qml.BasicEntangler

  BasicEntangler can now handle more than one rotation. The 
  parameters for each gate can be defined independently, an
  error occurs if the parameters are not of the expected shape.

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

=======
* Added `QutritDevice` as an abstract base class for qutrit devices.
  [#2781](https://github.com/PennyLaneAI/pennylane/pull/2781)
  [#2782](https://github.com/PennyLaneAI/pennylane/pull/2782)
  
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
  
* Added `qml.THermitian` observable for measuring user-specified Hermitian matrix observables for qutrit circuits.
  [#2784](https://github.com/PennyLaneAI/pennylane/pull/2784)

<h3>Improvements</h3>

* Automatic circuit cutting is improved by making better partition imbalance derivations.
  Now it is more likely to generate optimal cuts for larger circuits.
  [(#2517)](https://github.com/PennyLaneAI/pennylane/pull/2517)

* The `qml.simplify` method now can compute the adjoint and power of specific operators.
  [(#2922)](https://github.com/PennyLaneAI/pennylane/pull/2922)

  ```pycon
  >>> adj_op = qml.adjoint(qml.RX(1, 0))
  >>> qml.simplify(adj_op)
  RX(-1, wires=[0])
  ```

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):


Miguel Esteban Villalobos, Samuel Banning, Juan Miguel Arrazola, Utkarsh Azad, David Ittah, Soran Jahangiri, Edward Jiang,
Ankit Khandelwal, Meenu Kumari, Christina Lee, Sergio Martínez-Losa, Albert Mitjans Coma,
Ixchel Meza Chavez, Romain Moyard, Lee James O'Riordan, Mudit Pandey, Bogdan Reznychenko,
Shuli Shu, Jay Soni, Modjtaba Shokrian-Zini, Antal Száva, David Wierichs, Moritz Willmann, Olivia Di Matteo,
Josh Izaac,
Korbinian Kottmann,
Zeyue Niu,
Mudit Pandey,
Antal Száva
