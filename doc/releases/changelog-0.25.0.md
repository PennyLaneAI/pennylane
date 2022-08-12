:orphan:

# Release 0.25.0 (current release)

<h3>New features since last release</h3>

<h4>Estimate computational resource requirements 🧠</h4>

* Functionality for estimating molecular simulation computations has been added with `qml.resource`.
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

  The new [resource](https://pennylane.readthedocs.io/en/stable/code/qml_resource.html) module allows you to estimate the number of 
  non-[Clifford gates](https://en.wikipedia.org/wiki/Clifford_gates) and logical 
  qubits needed to implement [quantum phase estimation](https://codebook.xanadu.ai/P.1) 
  algorithms for simulating materials and molecules. This includes support for quantum 
  algorithms using [first](https://en.wikipedia.org/wiki/First_quantization) and [second](https://en.wikipedia.org/wiki/Second_quantization) quantization with specific bases:

  - [First quantization](https://en.wikipedia.org/wiki/First_quantization) using a plane-wave basis via the `FirstQuantization` class:

    ```python
    >>> n = 100000        # number of plane waves
    >>> eta = 156         # number of electrons
    >>> omega = 1145.166  # unit cell volume in atomic units
    >>> algo = FirstQuantization(n, eta, omega)
    >>> algo.gates
    1.10e+13
    >>> algo.qubits
    4416
    ```

  - [Second quantization](https://en.wikipedia.org/wiki/Second_quantization) with a double-factorized Hamiltonian via the 
    `DoubleFactorization` class: 
 
    ```python
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
  can be also accessed individually without instantiating an instance of the class:

  + The number of logical qubits with `qubit_cost`
  + The number of non-Clifford gates with `gate_cost`

    ```pycon
    >>> qml.resource.FirstQuantization.qubit_cost(100000, 156, 169.69608, 0.01)
    4377
    >>> qml.resource.FirstQuantization.gate_cost(100000, 156, 169.69608, 0.01)
    3676557345574 
    ```

<h4>Differentiable error mitigation ⚙️</h4>

* Differentiable zero-noise-extrapolation (ZNE) error mitigation is now available.
  [(#2757)](https://github.com/PennyLaneAI/pennylane/pull/2757)

  Elevate any variational quantum algorithm to a *mitigated* algorithm with improved 
  results on noisy hardware while maintaining differentiability throughout.

  In order to do so, use the `qml.transforms.mitigate_with_zne` transform on your QNode and provide the PennyLane proprietary
  `qml.transforms.fold_global` folding function and `qml.transforms.poly_extrapolate` extrapolation function. 
  Here is an example for a noisy simulation device where we mitigate a QNode and are still 
  able to compute the gradient:

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
  ```

  ```pycon
  >>> theta = np.array(0.5, requires_grad=True)
  >>> qml.grad(qnode_mitigated)(theta)
  0.5712737447327619
  ```

<h4>More native support for parameter broadcasting 📡</h4>

* `default.qubit` now natively supports parameter broadcasting, providing 
  increased performance when executing the same circuit at various parameter positions
  compared to manually looping over parameters, or directly using the `qml.transforms.broadcast_expand` transform.
  [(#2627)](https://github.com/PennyLaneAI/pennylane/pull/2627)

  ```python
  dev = qml.device("default.qubit", wires=1)

  @qml.qnode(dev)
  def circuit(x):
      qml.RX(x, wires=0)
      return qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> circuit(np.array([0.1, 0.3, 0.2]))
  tensor([0.99500417, 0.95533649, 0.98006658], requires_grad=True) 
  ```

* Parameter-shift gradients now allow for parameter broadcasting internally,
  which can result in a significant speedup when computing gradients of
  circuits with many parameters.
  [(#2749)](https://github.com/PennyLaneAI/pennylane/pull/2749)

  The gradient transform `qml.gradients.param_shift` now accepts the keyword argument 
  `broadcast`. If set to `True`, broadcasting is used to compute the derivative:

  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def circuit(x, y):
      qml.RX(x, wires=0)
      qml.RY(y, wires=1)
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  ```

  ```pycon
  >>> x = np.array([np.pi/3, np.pi/2], requires_grad=True)
  >>> y = np.array([np.pi/6, np.pi/5], requires_grad=True)
  >>> qml.gradients.param_shift(circuit, broadcast=True)(x, y)
  (tensor([[-0.9330127,  0.       ],
           [ 0.       , -0.9330127]], requires_grad=True),
  tensor([[0.25, 0.  ],
          [0.  , 0.25]], requires_grad=True))
  ```

  The following simpler example also makes use of broadcasting:

  ```python
  @qml.qnode(dev, diff_method="parameter-shift", broadcast=True)
  def circuit(x, y):
      qml.RX(x, wires=0)
      qml.RY(y, wires=1)
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  ```

  ```pycon
  >>> x = np.array(0.1, requires_grad=True)
  >>> y = np.array(0.4, requires_grad=True)
  >>> qml.grad(circuit)(x, y)
  (array(-0.09195267), array(-0.38747287))
  ```

  where only 2, instead of 4 with `broadcast=False`, circuits are created internally.

  To illustrate the speedup, for a constant-depth circuit with Pauli rotations and controlled Pauli rotations, the 
  time required to compute `qml.gradients.param_shift(circuit, broadcast=False)(params)`
  ("No broadcasting") and `qml.gradients.param_shift(circuit, broadcast=True)(params)` 
  ("Broadcasting") as a function of the number of qubits is given
  [here](https://pennylane.readthedocs.io/en/stable/_images/param_shift_broadcast_speedup.png).

* Operations for quantum chemistry now support parameter broadcasting.
  [(#2726)](https://github.com/PennyLaneAI/pennylane/pull/2726)

  ```pycon
  >>> op = qml.SingleExcitation(np.array([0.3, 1.2, -0.7]), wires=[0, 1])
  >>> op.matrix().shape
  (3, 4, 4)
  ```

<h4>Intuitive operator arithmetic 🧮</h4>

* New functionality for representing the sum, product, and scalar-product 
  of operators is available.
  [(#2475)](https://github.com/PennyLaneAI/pennylane/pull/2475)
  [(#2625)](https://github.com/PennyLaneAI/pennylane/pull/2625)
  [(#2622)](https://github.com/PennyLaneAI/pennylane/pull/2622)
  [(#2721)](https://github.com/PennyLaneAI/pennylane/pull/2721)

  The following functionalities have been added to facilitate creating new operators 
  whose matrix, terms, and eigenvalues can be accessed as per usual, while maintaining 
  differentiability. operators created from these new features can be used within
  QNodes as Operations or as observables (where physically applicable). 

  - Summing any number of operators via `qml.op_sum` results in a "summed" operator:

    ```pycon
    >>> ops_to_sum = [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(0)] 
    >>> summed_ops = qml.op_sum(*ops_to_sum)
    >>> summed_ops
    PauliX(wires=[0]) + PauliY(wires=[1]) + PauliZ(wires=[0])
    >>> qml.matrix(summed_ops)
    array([[ 1.+0.j,  0.-1.j,  1.+0.j,  0.+0.j],
           [ 0.+1.j,  1.+0.j,  0.+0.j,  1.+0.j],
           [ 1.+0.j,  0.+0.j, -1.+0.j,  0.-1.j],
           [ 0.+0.j,  1.+0.j,  0.+1.j, -1.+0.j]])
    >>> summed_ops.terms()
    ([1.0, 1.0, 1.0], (PauliX(wires=[0]), PauliY(wires=[1]), PauliZ(wires=[0])))
    ```

  - Multiplying any number of operators via `qml.prod` results in a "product" operator, 
    where the matrix product or tensor product is used correspondingly:

    ```pycon
    >>> theta = 1.23
    >>> prod_op = qml.prod(qml.PauliZ(0), qml.RX(theta, 1))
    >>> prod_op
    PauliZ(wires=[0]) @ RX(1.23, wires=[1]) 
    >>> qml.eigvals(prod_op)
    [-1.39373197 -0.23981492  0.23981492  1.39373197]
    ```
  
  - Taking the product of a coefficient and an operator via `qml.s_prod` produces a 
    "scalar-product" operator:

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

  Each of these new functionalities can be used within QNodes as operators or observables, 
  where applicable, while also maintaining differentiability. For example:

  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def circuit(angles):
      qml.prod(qml.PauliZ(0), qml.RY(angles[0], 1))
      qml.op_sum(qml.PauliX(1), qml.RY(angles[1], 0))

      return qml.expval(qml.op_sum(qml.PauliX(0), qml.PauliZ(1)))
  ```

  ```pycon
  >>> angles = np.array([1.23, 4.56], requires_grad=True)
  >>> circuit(angles)
  0.33423772712450256
  >>> qml.grad(circuit)(angles)
  [-0.9424888  0.]
  ```
  
* All PennyLane operators can now be added, subtracted, multiplied, scaled, and raised to 
  powers using `+`, `-`, `@`, `*`, `**`, respectively.
  [(#2849)](https://github.com/PennyLaneAI/pennylane/pull/2849)
  [(#2825)](https://github.com/PennyLaneAI/pennylane/pull/2825)
  [(#2891)](https://github.com/PennyLaneAI/pennylane/pull/2891)
  
  - You can now add scalars to operators, where the interpretation is that the 
    scalar is a properly-sized identity matrix;

    ```pycon
    >>> sum_op = 5 + qml.PauliX(0)
    >>> sum_op.matrix()
    array([[5., 1.],
           [1., 5.]])
    ```
    
  - The `+` and `-` operators can be used to combine all Pennylane operators:

    ```pycon
    >>> sum_op = qml.RX(phi=1.23, wires=0) + qml.RZ(phi=3.14, wires=0) - qml.RY(phi=0.12, wires=0)
    >>> sum_op
    RX(1.23, wires=[0]) + RZ(3.14, wires=[0]) + -1*(RY(0.12, wires=[0]))
    >>> qml.matrix(sum_op)
    array([[-0.18063077-0.99999968j,  0.05996401-0.57695852j],
           [-0.05996401-0.57695852j, -0.18063077+0.99999968j]])
    ```
    Note that the behavior of `+` with *observables* is different; it still creates 
    a Hamiltonian.
  
  - The `*` and `@` operators can be used to scale and compose all PennyLane operators.
  
    ```pycon
    >>> prod_op = 2*qml.RX(1, wires=0) @ qml.RY(2, wires=0)
    >>> prod_op
    2*(RX(1, wires=[0])) @ RY(2, wires=[0])
    >>> qml.matrix(prod_op)
    array([[ 0.94831976-0.80684536j, -1.47692053-0.51806945j],
           [ 1.47692053-0.51806945j,  0.94831976+0.80684536j]])
     ```
  
  - The `**` operator can be used to raise PennyLane operators to a power.
  
    ```pycon       
    >>> exp_op = qml.RZ(1.0, wires=0) ** 2
    >>> exp_op
    RZ**2(1.0, wires=[0])
    >>> qml.matrix(exp_op)
    array([[0.54030231-0.84147098j, 0.        +0.j        ],
           [0.        +0.j        , 0.54030231+0.84147098j]])
    ```

* A new class called `Controlled` is available in `qml.ops.op_math` to help represent 
  a controlled version of any operation. 
  [(#2634)](https://github.com/PennyLaneAI/pennylane/pull/2634)

  ```python
  from pennylane.ops.op_math import Controlled

  @qml.qnode(qml.device('default.qubit', wires=3))
  def circuit():
      qml.PauliX(0)
      Controlled(qml.Hadamard(2), control_wires=(0,1), control_values=[1,0])
      return qml.probs(wires=range(3))
  ```

  ```pycon
  >>> circuit()
  tensor([0. , 0. , 0. , 0. , 0.5, 0.5, 0. , 0. ], requires_grad=True)
  >>> print(qml.draw(circuit)())
  0: ──X─╭●─┤ ╭Probs
  1: ────├○─┤ ├Probs
  2: ────╰H─┤ ╰Probs
  ```

* Arithmetic operations can now be simplified using `qml.simplify`.
  [(#2835)](https://github.com/PennyLaneAI/pennylane/pull/2835)
  [(#2854)](https://github.com/PennyLaneAI/pennylane/pull/2854)
  
  ```pycon 
  >>> op = qml.adjoint(qml.adjoint(qml.RX(x, wires=0))) 
  >>> op 
  Adjoint(Adjoint(RX))(tensor([1.04719755, 1.57079633], requires_grad=True), wires=[0]) 
  >>> qml.simplify(op) 
  RX(tensor([1.04719755, 1.57079633], requires_grad=True), wires=[0]) 
  ```

* A new function called `qml.equal` can be used to compare the equality 
  of parametric operators.
  [(#2651)](https://github.com/PennyLaneAI/pennylane/pull/2651)

  ```pycon
  >>> qml.equal(qml.RX(1.23, 0), qml.RX(1.23, 0))
  True
  >>> qml.equal(qml.RY(4.56, 0), qml.RY(7.89, 0))
  False
  ```

<h4>Marvelous mixed state features 🙌</h4>

* The `default.mixed` device now supports [backpropagation](https://pennylane.readthedocs.io/en/stable/introduction/unsupported_gradients.html#backpropagation) with the `"jax"` interface, which can result in significant speedups.
  [(#2754)](https://github.com/PennyLaneAI/pennylane/pull/2754)
  [(#2776)](https://github.com/PennyLaneAI/pennylane/pull/2776)

  ```python
  dev = qml.device("default.mixed", wires=2)

  @qml.qnode(dev, diff_method="backprop", interface="jax")
  def circuit(angles):
      qml.RX(angles[0], wires=0)
      qml.RY(angles[1], wires=1)
      return qml.expval(qml.PauliZ(0) + qml.PauliZ(1))
  ```

  ```pycon
  >>> angles = np.array([np.pi/6, np.pi/5], requires_grad=True)
  >>> qml.grad(circuit)(angles)
  array([-0.8660254 , -0.25881905])
  ```
  
  Additionally, quantum channels now support Jax and TensorFlow tensors.
  This allows quantum channels to be used inside QNodes decorated by `tf.function`, 
  `jax.jit`, or `jax.vmap`.

* The `default.mixed` device now supports readout error.
  [(#2786)](https://github.com/PennyLaneAI/pennylane/pull/2786)

  A new keyword argument called `readout_prob` can be specified when creating a
  `default.mixed` device. Any circuits running on a `default.mixed` device with a 
  finite `readout_prob` (upper-bounded by 1) will alter the measurements performed 
  at the end of the circuit similarly to how a `qml.BitFlip` channel would affect 
  circuit measurements:

  ```pycon
  >>> dev = qml.device("default.mixed", wires=2, readout_prob=0.1)
  >>> @qml.qnode(dev)
  ... def circuit():
  ...     return qml.expval(qml.PauliZ(0))
  >>> circuit()
  0.8
  ```
  
<h4>Relative entropy is now available in qml.qinfo 💥</h4>

* The quantum information module now supports computation of [relative entropy](https://en.wikipedia.org/wiki/Quantum_relative_entropy).
  [(#2772)](https://github.com/PennyLaneAI/pennylane/pull/2772)

  We've enabled two cases for calculating the relative entropy:
  
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

<h4>New measurements, operators, and more! ✨</h4>

* A new measurement called `qml.counts` is available.
  [(#2686)](https://github.com/PennyLaneAI/pennylane/pull/2686)
  [(#2839)](https://github.com/PennyLaneAI/pennylane/pull/2839)
  [(#2876)](https://github.com/PennyLaneAI/pennylane/pull/2876)

  QNodes with `shots != None` that return `qml.counts` will yield a dictionary 
  whose keys are bitstrings representing computational basis states that were measured, 
  and whose values are the corresponding counts (i.e., how many times that computational 
  basis state was measured):

  ```python
  dev = qml.device("default.qubit", wires=2, shots=1000)

  @qml.qnode(dev)
  def circuit():
      qml.Hadamard(wires=0)
      qml.CNOT(wires=[0, 1])
      return qml.counts()
  ```

  ```pycon
  >>> circuit()
  {'00': 495, '11': 505}
  ```

  `qml.counts` can also accept observables, where the resulting dictionary is ordered
  by the eigenvalues of the observable.

  ```python
  dev = qml.device("default.qubit", wires=2, shots=1000)

  @qml.qnode(dev)
  def circuit():
      qml.Hadamard(wires=0)
      qml.CNOT(wires=[0, 1])
      return qml.counts(qml.PauliZ(0)), qml.counts(qml.PauliZ(1))
  ```

  ```pycon
  >>> circuit()
  ({-1: 470, 1: 530}, {-1: 470, 1: 530})
  ```
  
* A new experimental return type for QNodes with multiple measurements has been added.
  [(#2814)](https://github.com/PennyLaneAI/pennylane/pull/2814)
  [(#2815)](https://github.com/PennyLaneAI/pennylane/pull/2815)
  [(#2860)](https://github.com/PennyLaneAI/pennylane/pull/2860)

  QNodes returning a list or tuple of different measurements return an intuitive 
  data structure via `qml.enable_return()`, where the individual measurements are 
  separated into their own tensors:

  ```python
  qml.enable_return()
  dev = qml.device("default.qubit", wires=2)
  
  @qml.qnode(dev)
  def circuit(x):
      qml.Hadamard(wires=[0])
      qml.CRX(x, wires=[0, 1])
      return (qml.probs(wires=[0]), qml.vn_entropy(wires=[0]), qml.probs(wires=0), qml.expval(wires=1))
  ```
  ```pycon
  >>> circuit(0.5)
  (tensor([0.5, 0.5], requires_grad=True), tensor(0.08014815, requires_grad=True), tensor([0.5, 0.5], requires_grad=True), tensor(0.93879128, requires_grad=True))
  ```

  In addition, QNodes that utilize this new return type support backpropagation.
  This new return type can be disabled thereafter via `qml.disable_return()`.

* An operator called `qml.FlipSign` is now available.
  [(#2780)](https://github.com/PennyLaneAI/pennylane/pull/2780)

  Mathematically, `qml.FlipSign` functions as follows: 
  $\text{FlipSign}(n) \vert m \rangle = (-1)^\delta_{n,m} \vert m \rangle$, where 
  $\vert m \rangle$ is an arbitrary qubit state and $n$ is a qubit configuration:

  ```python
  basis_state = [0, 1]

  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def circuit():
    for wire in list(range(2)):
          qml.Hadamard(wires = wire)
    qml.FlipSign(basis_state, wires = list(range(2)))
    return qml.sample()
  ```

  ```pycon
  >>> circuit()
  tensor([ 0.5+0.j  -0.5+0.j 0.5+0.j  0.5+0.j], requires_grad=True)
  ```

* The [simultaneous perturbation stochastic approximation (SPSA) optimizer](https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF) 
  is available via `qml.SPSAOptimizer`.
  [(#2661)](https://github.com/PennyLaneAI/pennylane/pull/2661)

  The SPSA optimizer is suitable for cost functions whose evaluation may involve
  noise. Use the SPSA optimizer like you would any other optimizer:

  ```python
  max_iterations = 50
  opt = qml.SPSAOptimizer(maxiter=max_iterations) 

  for _ in range(max_iterations):
      params, cost = opt.step_and_cost(cost, params)
  ```  
  
<h4>More drawing styles 🎨</h4>

* New PennyLane-inspired `sketch` and `sketch_dark` styles are now available for 
  drawing circuit diagram graphics.
  [(#2709)](https://github.com/PennyLaneAI/pennylane/pull/2709)

<h3>Improvements 📈</h3>

* `default.qubit` now natively executes any operation that defines a matrix except
  for trainable `Pow` operations. 
  [(#2836)](https://github.com/PennyLaneAI/pennylane/pull/2836)

* Added `expm` to the `qml.math` module for matrix exponentiation.
  [(#2890)](https://github.com/PennyLaneAI/pennylane/pull/2890)
  
* When adjoint differentiation is requested, circuits are now decomposed so
  that all trainable operations have a generator.
  [(#2836)](https://github.com/PennyLaneAI/pennylane/pull/2836)
  
* A warning is emitted for measurements
  - `qml.state`
  - `qml.density_matrix`
  - `qml.vn_entropy`
  - `qml.mutual_info`
  when using a device with finite shots or a shot list because the results of
  these measurements are always analytic.
  [(#2918)](https://github.com/PennyLaneAI/pennylane/pull/2918)

* The efficiency of the Hartree-Fock workflow has been improved by removing 
  repetitive steps.
  [(#2850)](https://github.com/PennyLaneAI/pennylane/pull/2850)

* The coefficients of the non-differentiable molecular Hamiltonians generated 
  with openfermion now have `requires_grad = False` by default.
  [(#2865)](https://github.com/PennyLaneAI/pennylane/pull/2865)

* Upgraded performance of the `compute_matrix` method of broadcastable 
  parametric operations.
  [(#2759)](https://github.com/PennyLaneAI/pennylane/pull/2759)

* Jacobians are now cached with the Autograd interface when using the
  parameter-shift rule.
  [(#2645)](https://github.com/PennyLaneAI/pennylane/pull/2645)

* The `qml.state` and `qml.density_matrix` measurements now support custom wire
  labels.
  [(#2779)](https://github.com/PennyLaneAI/pennylane/pull/2779)

* Add trivial behaviour logic to `qml.operation.expand_matrix`.
  [(#2785)](https://github.com/PennyLaneAI/pennylane/issues/2785)

* Added an `are_pauli_words_qwc` function which checks if certain
  Pauli words are pairwise qubit-wise commuting. This new function improves performance 
  when measuring hamiltonians with many commuting terms.
  [(#2789)](https://github.com/PennyLaneAI/pennylane/pull/2798)

* Adjoint differentiation now uses the adjoint symbolic wrapper instead of in-place 
  inversion.
  [(#2855)](https://github.com/PennyLaneAI/pennylane/pull/2855)

<h3>Breaking changes 💔</h3>

* The deprecated `qml.hf` module is removed. Users with code that calls `qml.hf` 
  can simply replace `qml.hf` with `qml.qchem` in most cases, or refer to the 
  [documentation](https://pennylane.readthedocs.io/en/stable/code/qml_qchem.html) 
  and [demos](https://pennylane.ai/qml/demos_quantum-chemistry.html) for more 
  information.
  [(#2795)](https://github.com/PennyLaneAI/pennylane/pull/2795)

* `default.qubit` now uses `stopping_condition` to specify support for anything with a matrix.
  To override this behavior in inheriting devices and to support only a specific subset of operations,
  developers need to override `stopping_condition`.
  [(#2836)](https://github.com/PennyLaneAI/pennylane/pull/2836)

* Custom devices inheriting from `DefaultQubit` or `QubitDevice` can break due to the introduction
  of parameter broadcasting.
  [(#2627)](https://github.com/PennyLaneAI/pennylane/pull/2627)

  A custom device should only break if all three following statements hold simultaneously:

  1. The custom device inherits from `DefaultQubit`, not `QubitDevice`.
  2. The device implements custom methods in the simulation pipeline that are incompatible
     with broadcasting (for example `expval`, `apply_operation` or `analytic_probability`).
  3. The custom device maintains the flag `"supports_broadcasting": True` in its `capabilities`
     dictionary *or* it overwrites `Device.batch_transform` without applying `broadcast_expand`
     (or both).

  The `capabilities["supports_broadcasting"]` is set to `True` for
  `DefaultQubit`. Typically, the easiest fix will be to change the
  `capabilities["supports_broadcasting"]` flag to `False` for the child device
  and/or to include a call to `broadcast_expand` in
  `CustomDevice.batch_transform`, similar to how `Device.batch_transform` calls
  it.

  Separately from the above, custom devices that inherit from `QubitDevice` and implement a
  custom `_gather` method need to allow for the kwarg `axis` to be passed to this `_gather` method.

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

* PennyLane now depends on newer versions (>=2.7) of the `semantic_version` package,
  which provides an updated API that is incompatible which versions of the package 
  prior to 2.7. If you run into issues relating to this package, please reinstall 
  PennyLane.
  [(#2744)](https://github.com/PennyLaneAI/pennylane/pull/2744)
  [(#2767)](https://github.com/PennyLaneAI/pennylane/pull/2767)

<h3>Documentation 📕</h3>

* Added a dedicated docstring for the `QubitDevice.sample` method.
  [(#2812)](https://github.com/PennyLaneAI/pennylane/pull/2812)

* Optimization examples of using JAXopt and Optax with the JAX interface have
  been added.
  [(#2769)](https://github.com/PennyLaneAI/pennylane/pull/2769)

* Updated IsingXY gate docstring.
  [(#2858)](https://github.com/PennyLaneAI/pennylane/pull/2858)

<h3>Bug fixes 🐞</h3>

* Cleaned up interactions between operator arithmetic and batching by
  testing supported cases and adding errors when batching is not supported.
  [(#2900)](https://github.com/PennyLaneAI/pennylane/pull/2900)

* Fixed a bug where the parameter-shift rule wasn't defined for `qml.kUpCCGSD`.
  [(#2913)](https://github.com/PennyLaneAI/pennylane/pull/2913)  

* Reworked the Hermiticity check in `qml.Hermitian` by using `qml.math` calls
  because calling `.conj()` on an `EagerTensor` from TensorFlow raised an
  error.
  [(#2895)](https://github.com/PennyLaneAI/pennylane/pull/2895)

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

* The WireCut operator now raises an error when instantiating it with an empty list.
  [(#2826)](https://github.com/PennyLaneAI/pennylane/pull/2826)

* Hamiltonians with grouped observables are now allowed to be measured on devices 
  which were transformed using `qml.transform.insert()`.
  [(#2857)](https://github.com/PennyLaneAI/pennylane/pull/2857)

* Fixes a bug where `qml.batch_input` raised an error when using a batched operator that was not
  located at the beginning of the circuit. In addition, now `qml.batch_input` raises an error when
  using trainable batched inputs, which avoids an unwanted behaviour with duplicated parameters.
  [(#2873)](https://github.com/PennyLaneAI/pennylane/pull/2873)

* Calling `qml.equal` with nested operators now raises a `NotImplementedError`.
  [(#2877)](https://github.com/PennyLaneAI/pennylane/pull/2877)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Utkarsh Azad, Samuel Banning, Prajwal Borkar, Isaac De Vlugt, Olivia Di Matteo, Kristiyan Dilov, 
David Ittah, Josh Izaac, Soran Jahangiri, Edward Jiang, Ankit Khandelwal, Korbinian Kottmann, Meenu Kumari, 
Christina Lee, Sergio Martínez-Losa, Albert Mitjans Coma, Ixchel Meza Chavez, Romain Moyard, Lee James O'Riordan,
Mudit Pandey, Bogdan Reznychenko, Shuli Shu, Jay Soni, Modjtaba Shokrian-Zini, Antal Száva, David Wierichs, 
Moritz Willmann

