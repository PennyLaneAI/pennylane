
# Release 0.29.0

<h3>New features since last release</h3>

<h4>Pulse programming 🔊</h4>

* Support for creating pulse-based circuits that describe evolution under a time-dependent
  Hamiltonian has now been added, as well as the ability to execute and differentiate these
  pulse-based circuits on simulator.
  [(#3586)](https://github.com/PennyLaneAI/pennylane/pull/3586)
  [(#3617)](https://github.com/PennyLaneAI/pennylane/pull/3617)
  [(#3645)](https://github.com/PennyLaneAI/pennylane/pull/3645)
  [(#3652)](https://github.com/PennyLaneAI/pennylane/pull/3652)
  [(#3665)](https://github.com/PennyLaneAI/pennylane/pull/3665)
  [(#3673)](https://github.com/PennyLaneAI/pennylane/pull/3673)
  [(#3706)](https://github.com/PennyLaneAI/pennylane/pull/3706)
  [(#3730)](https://github.com/PennyLaneAI/pennylane/pull/3730)

  A time-dependent Hamiltonian can be created using `qml.pulse.ParametrizedHamiltonian`, which 
  holds information representing a linear combination of operators
  with parametrized coefficents and can be constructed as follows:

  ```python
  from jax import numpy as jnp

  f1 = lambda p, t: p * jnp.sin(t) * (t - 1)
  f2 = lambda p, t: p[0] * jnp.cos(p[1]* t ** 2)

  XX = qml.PauliX(0) @ qml.PauliX(1)
  YY = qml.PauliY(0) @ qml.PauliY(1)
  ZZ = qml.PauliZ(0) @ qml.PauliZ(1)

  H =  2 * ZZ + f1 * XX + f2 * YY
  ```

  ```pycon
  >>> H
  ParametrizedHamiltonian: terms=3
  >>> p1 = jnp.array(1.2)
  >>> p2 = jnp.array([2.3, 3.4])
  >>> H((p1, p2), t=0.5)
  (2*(PauliZ(wires=[0]) @ PauliZ(wires=[1]))) + ((-0.2876553231625218*(PauliX(wires=[0]) @ PauliX(wires=[1]))) + (1.517961235535459*(PauliY(wires=[0]) @ PauliY(wires=[1]))))
  ```

  The time-dependent Hamiltonian can be used within a circuit with `qml.evolve`:

  ```python
  def pulse_circuit(params, time):
      qml.evolve(H)(params, time)
      return qml.expval(qml.PauliX(0) @ qml.PauliY(1))
  ```

  Pulse-based circuits can be executed and differentiated on the `default.qubit.jax` simulator using
  JAX as an interface:

  ```pycon
  >>> dev = qml.device("default.qubit.jax", wires=2)
  >>> qnode = qml.QNode(pulse_circuit, dev, interface="jax")
  >>> params = (p1, p2)
  >>> qnode(params, time=0.5)
  Array(0.72153819, dtype=float64)
  >>> jax.grad(qnode)(params, time=0.5)
  (Array(-0.11324919, dtype=float64),
   Array([-0.64399616,  0.06326374], dtype=float64))
  ```

  Check out the [qml.pulse](https://docs.pennylane.ai/en/stable/code/qml_pulse.html) documentation
  page for more details!

<h4>Special unitary operation 🌞</h4>

* A new operation `qml.SpecialUnitary` has been added, providing access to an arbitrary
  unitary gate via a parametrization in the Pauli basis.
  [(#3650)](https://github.com/PennyLaneAI/pennylane/pull/3650)
  [(#3651)](https://github.com/PennyLaneAI/pennylane/pull/3651)
  [(#3674)](https://github.com/PennyLaneAI/pennylane/pull/3674)

  `qml.SpecialUnitary` creates a unitary that exponentiates a linear combination of all possible Pauli words in lexicographical order — except for the identity operator — for `num_wires` wires, of which there are `4**num_wires - 1`. As its first argument,
  `qml.SpecialUnitary` takes a list of the `4**num_wires - 1` parameters that are the coefficients of the linear combination. 

  To see all possible Pauli words for `num_wires` wires, you can use the `qml.ops.qubit.special_unitary.pauli_basis_strings` function:

  ```pycon
  >>> qml.ops.qubit.special_unitary.pauli_basis_strings(1) # 4**1-1 = 3 Pauli words
  ['X', 'Y', 'Z']
  >>> qml.ops.qubit.special_unitary.pauli_basis_strings(2) # 4**2-1 = 15 Pauli words
  ['IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']
  ```

  To use `qml.SpecialUnitary`, for example, on a single qubit, we may define

  ```pycon
  >>> thetas = np.array([0.2, 0.1, -0.5])
  >>> U = qml.SpecialUnitary(thetas, 0)
  >>> qml.matrix(U)
  array([[ 0.8537127 -0.47537233j,  0.09507447+0.19014893j],
         [-0.09507447+0.19014893j,  0.8537127 +0.47537233j]])
  ```

  A single non-zero entry in the parameters will create a Pauli rotation:

  ```pycon
  >>> x = 0.412
  >>> theta = x * np.array([1, 0, 0]) # The first entry belongs to the Pauli word "X"
  >>> su = qml.SpecialUnitary(theta, wires=0)
  >>> rx = qml.RX(-2 * x, 0) # RX introduces a prefactor -0.5 that has to be compensated
  >>> qml.math.allclose(qml.matrix(su), qml.matrix(rx))
  True
  ```

  This operation can be differentiated with hardware-compatible methods like parameter shifts
  and it supports parameter broadcasting/batching, but not both at the same time. Learn more by
  visiting the
  [qml.SpecialUnitary](https://docs.pennylane.ai/en/stable/code/api/pennylane.SpecialUnitary.html)
  documentation.

<h4>Always differentiable 📈</h4>

* The Hadamard test gradient transform is now available via `qml.gradients.hadamard_grad`. This transform 
  is also available as a differentiation method within `QNode`s.
  [(#3625)](https://github.com/PennyLaneAI/pennylane/pull/3625)
  [(#3736)](https://github.com/PennyLaneAI/pennylane/pull/3736)

  `qml.gradients.hadamard_grad` is a hardware-compatible transform that calculates the
  gradient of a quantum circuit using the Hadamard test. Note that the device requires an
  auxiliary wire to calculate the gradient.

  ```pycon
  >>> dev = qml.device("default.qubit", wires=2)
  >>> @qml.qnode(dev)
  ... def circuit(params):
  ...     qml.RX(params[0], wires=0)
  ...     qml.RY(params[1], wires=0)
  ...     qml.RX(params[2], wires=0)
  ...     return qml.expval(qml.PauliZ(0))
  >>> params = np.array([0.1, 0.2, 0.3], requires_grad=True)
  >>> qml.gradients.hadamard_grad(circuit)(params)
  (tensor(-0.3875172, requires_grad=True),
   tensor(-0.18884787, requires_grad=True),
   tensor(-0.38355704, requires_grad=True))
  ```

  This transform can be registered directly as the quantum gradient transform to use during
  autodifferentiation:

  ```pycon
  >>> dev = qml.device("default.qubit", wires=2)
  >>> @qml.qnode(dev, interface="jax", diff_method="hadamard")
  ... def circuit(params):
  ...     qml.RX(params[0], wires=0)
  ...     qml.RY(params[1], wires=0)
  ...     qml.RX(params[2], wires=0)
  ...     return qml.expval(qml.PauliZ(0))
  >>> params = jax.numpy.array([0.1, 0.2, 0.3])
  >>> jax.jacobian(circuit)(params)
  Array([-0.3875172 , -0.18884787, -0.38355705], dtype=float32)
  ```

* The gradient transform `qml.gradients.spsa_grad` is now registered as a
  differentiation method for QNodes.
  [(#3440)](https://github.com/PennyLaneAI/pennylane/pull/3440)

  The SPSA gradient transform can now be used implicitly by marking a QNode
  as differentiable with SPSA. It can be selected via

  ```pycon
  >>> dev = qml.device("default.qubit", wires=1)
  >>> @qml.qnode(dev, interface="jax", diff_method="spsa", h=0.05, num_directions=20)
  ... def circuit(x):
  ...     qml.RX(x, 0)
  ...     return qml.expval(qml.PauliZ(0))
  >>> jax.jacobian(circuit)(jax.numpy.array(0.5))
  Array(-0.4792258, dtype=float32, weak_type=True)
  ```

  The argument `num_directions` determines how many directions of simultaneous
  perturbation are used and therefore the number of circuit evaluations, up
  to a prefactor. See the
  [SPSA gradient transform documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.gradients.spsa_grad.html) for details.
  Note: The full SPSA optimization method is already available as `qml.SPSAOptimizer`.

* The default interface is now `auto`. There is no need to specify the interface anymore; it is automatically
  determined by checking your QNode parameters.
  [(#3677)](https://github.com/PennyLaneAI/pennylane/pull/3677)
  [(#3752)](https://github.com/PennyLaneAI/pennylane/pull/3752)
  [(#3829)](https://github.com/PennyLaneAI/pennylane/pull/3829)

  ```python
  import jax
  import jax.numpy as jnp

  qml.enable_return()
  a = jnp.array(0.1)
  b = jnp.array(0.2)

  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def circuit(a, b):
      qml.RY(a, wires=0)
      qml.RX(b, wires=1)
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliY(1))
  ```

  ```pycon
  >>> circuit(a, b)
  (Array(0.9950042, dtype=float32), Array(-0.19767681, dtype=float32))
  >>> jac = jax.jacobian(circuit)(a, b)
  >>> jac
  (Array(-0.09983341, dtype=float32, weak_type=True), Array(0.01983384, dtype=float32, weak_type=True))
  ```

* The JAX-JIT interface now supports higher-order gradient computation with the new return types system.
  [(#3498)](https://github.com/PennyLaneAI/pennylane/pull/3498)

  Here is an example of using JAX-JIT to compute the Hessian of a circuit:

  ```python
  import pennylane as qp
  import jax
  from jax import numpy as jnp

  jax.config.update("jax_enable_x64", True)

  qml.enable_return()

  dev = qml.device("default.qubit", wires=2)

  @jax.jit
  @qml.qnode(dev, interface="jax-jit", diff_method="parameter-shift", max_diff=2)
  def circuit(a, b):
      qml.RY(a, wires=0)
      qml.RX(b, wires=1)
      return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

  a, b = jnp.array(1.0), jnp.array(2.0)
  ```

  ```pycon
  >>> jax.hessian(circuit, argnums=[0, 1])(a, b)
  (((Array(-0.54030231, dtype=float64, weak_type=True),
     Array(0., dtype=float64, weak_type=True)),
    (Array(-1.76002563e-17, dtype=float64, weak_type=True),
     Array(0., dtype=float64, weak_type=True))),
   ((Array(0., dtype=float64, weak_type=True),
     Array(-1.00700085e-17, dtype=float64, weak_type=True)),
    (Array(0., dtype=float64, weak_type=True),
    Array(0.41614684, dtype=float64, weak_type=True))))
  ```

* The `qchem` workflow has been modified to support both Autograd and JAX frameworks.
  [(#3458)](https://github.com/PennyLaneAI/pennylane/pull/3458)
  [(#3462)](https://github.com/PennyLaneAI/pennylane/pull/3462)
  [(#3495)](https://github.com/PennyLaneAI/pennylane/pull/3495)

  The JAX interface is automatically used when the differentiable parameters are JAX objects. Here
  is an example for computing the Hartree-Fock energy gradients with respect to the atomic
  coordinates.

  ```python
  import pennylane as qp
  from pennylane import numpy as np
  import jax

  symbols = ["H", "H"]
  geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

  mol = qml.qchem.Molecule(symbols, geometry)

  args = [jax.numpy.array(mol.coordinates)]
  ```

  ```pycon
  >>> jax.grad(qml.qchem.hf_energy(mol))(*args)
  Array([[ 0.       ,  0.       ,  0.3650435],
         [ 0.       ,  0.       , -0.3650435]], dtype=float64)
  ```

* The kernel matrix utility functions in `qml.kernels` are now autodifferentiation-compatible.
  In addition, they support batching, for example for quantum kernel execution with shot vectors.
  [(#3742)](https://github.com/PennyLaneAI/pennylane/pull/3742)

  This allows for the following:
  
  ```python
  dev = qml.device('default.qubit', wires=2, shots=(100, 100))
  @qml.qnode(dev)
  def circuit(x1, x2):
      qml.templates.AngleEmbedding(x1, wires=dev.wires)
      qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=dev.wires)
      return qml.probs(wires=dev.wires)

  kernel = lambda x1, x2: circuit(x1, x2)
  ```

  We can then compute the kernel matrix on a set of 4 (random) feature
  vectors `X` but using two sets of 100 shots each via

  ```pycon
  >>> X = np.random.random((4, 2))
  >>> qml.kernels.square_kernel_matrix(X, kernel)[:, 0]
  tensor([[[1.  , 0.86, 0.88, 0.92],
           [0.86, 1.  , 0.75, 0.97],
           [0.88, 0.75, 1.  , 0.91],
           [0.92, 0.97, 0.91, 1.  ]],
          [[1.  , 0.93, 0.91, 0.92],
           [0.93, 1.  , 0.8 , 1.  ],
           [0.91, 0.8 , 1.  , 0.91],
           [0.92, 1.  , 0.91, 1.  ]]], requires_grad=True)
  ```

  Note that we have extracted the first probability vector entry for each 100-shot evaluation.

<h4>Smartly decompose Hamiltonian evolution 💯</h4>

* Hamiltonian evolution using `qml.evolve` or `qml.exp` can now be decomposed into operations.
  [(#3691)](https://github.com/PennyLaneAI/pennylane/pull/3691)
  [(#3777)](https://github.com/PennyLaneAI/pennylane/pull/3777)

  If the time-evolved Hamiltonian is equivalent to another PennyLane operation, then that
  operation is returned as the decomposition:

  ```pycon
  >>> exp_op = qml.evolve(qml.PauliX(0) @ qml.PauliX(1))
  >>> exp_op.decomposition()
  [IsingXX((2+0j), wires=[0, 1])]
  ```
  
  If the Hamiltonian is a Pauli word, then the decomposition is provided as a
  `qml.PauliRot` operation:

  ```pycon
  >>> qml.evolve(qml.PauliZ(0) @ qml.PauliX(1)).decomposition()
  [PauliRot((2+0j), ZX, wires=[0, 1])]
  ```
  
  Otherwise, the Hamiltonian is a linear combination of operators and the Suzuki-Trotter
  decomposition is used:

  ```pycon
  >>> qml.evolve(qml.sum(qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)), num_steps=2).decomposition()
  [RX((1+0j), wires=[0]),
   RY((1+0j), wires=[0]),
   RZ((1+0j), wires=[0]),
   RX((1+0j), wires=[0]),
   RY((1+0j), wires=[0]),
   RZ((1+0j), wires=[0])]
  ```

<h4>Tools for quantum chemistry and other applications 🛠️</h4>

* A new method called `qml.qchem.givens_decomposition` has been added, which decomposes a unitary into a sequence
  of Givens rotation gates with phase shifts and a diagonal phase matrix.
  [(#3573)](https://github.com/PennyLaneAI/pennylane/pull/3573)

  ```python
  unitary = np.array([[ 0.73678+0.27511j, -0.5095 +0.10704j, -0.06847+0.32515j],
                      [-0.21271+0.34938j, -0.38853+0.36497j,  0.61467-0.41317j],
                      [ 0.41356-0.20765j, -0.00651-0.66689j,  0.32839-0.48293j]])

  phase_mat, ordered_rotations = qml.qchem.givens_decomposition(unitary)
  ```

  ```pycon
  >>> phase_mat
  tensor([-0.20604358+0.9785369j , -0.82993272+0.55786114j,
          0.56230612-0.82692833j], requires_grad=True)
  >>> ordered_rotations
  [(tensor([[-0.65087861-0.63937521j, -0.40933651-0.j        ],
            [-0.29201359-0.28685265j,  0.91238348-0.j        ]], requires_grad=True),
    (0, 1)),
  (tensor([[ 0.47970366-0.33308926j, -0.8117487 -0.j        ],
            [ 0.66677093-0.46298215j,  0.5840069 -0.j        ]], requires_grad=True),
    (1, 2)),
  (tensor([[ 0.36147547+0.73779454j, -0.57008306-0.j        ],
            [ 0.2508207 +0.51194108j,  0.82158706-0.j        ]], requires_grad=True),
    (0, 1))]
  ```

* A new template called `qml.BasisRotation` has been added, which performs a basis transformation defined by a set of
  fermionic ladder operators.
  [(#3573)](https://github.com/PennyLaneAI/pennylane/pull/3573)

  ```python
  import pennylane as qp
  from pennylane import numpy as np

  V = np.array([[ 0.53672126+0.j        , -0.1126064 -2.41479668j],
                [-0.1126064 +2.41479668j,  1.48694623+0.j        ]])
  eigen_vals, eigen_vecs = np.linalg.eigh(V)
  umat = eigen_vecs.T
  wires = range(len(umat))
  def circuit():
      qml.adjoint(qml.BasisRotation(wires=wires, unitary_matrix=umat))
      for idx, eigenval in enumerate(eigen_vals):
          qml.RZ(eigenval, wires=[idx])
      qml.BasisRotation(wires=wires, unitary_matrix=umat)
  ```

  ```pycon
  >>> circ_unitary = qml.matrix(circuit)()
  >>> np.round(circ_unitary/circ_unitary[0][0], 3)
  tensor([[ 1.   -0.j   , -0.   +0.j   , -0.   +0.j   , -0.   +0.j   ],
          [-0.   +0.j   , -0.516-0.596j, -0.302-0.536j, -0.   +0.j   ],
          [-0.   +0.j   ,  0.35 +0.506j, -0.311-0.724j, -0.   +0.j   ],
          [-0.   +0.j   , -0.   +0.j   , -0.   +0.j   , -0.438+0.899j]], requires_grad=True)
  ```

* A new function called `qml.qchem.load_basisset` has been added to extract `qml.qchem` basis set data from the Basis Set Exchange
  library.
  [(#3363)](https://github.com/PennyLaneAI/pennylane/pull/3363)

* A new function called `qml.math.max_entropy` has been added to compute the maximum entropy of a quantum state.
  [(#3594)](https://github.com/PennyLaneAI/pennylane/pull/3594)

* A new template called `qml.TwoLocalSwapNetwork` has been added that implements a canonical 2-complete linear (2-CCL) swap network
  described in [arXiv:1905.05118](https://arxiv.org/abs/1905.05118).
  [(#3447)](https://github.com/PennyLaneAI/pennylane/pull/3447)

  ```python3
  dev = qml.device('default.qubit', wires=5)
  weights = np.random.random(size=qml.templates.TwoLocalSwapNetwork.shape(len(dev.wires)))
  acquaintances = lambda index, wires, param: (qml.CRY(param, wires=index)
                                   if np.abs(wires[0]-wires[1]) else qml.CRZ(param, wires=index))
  @qml.qnode(dev)
  def swap_network_circuit():
      qml.templates.TwoLocalSwapNetwork(dev.wires, acquaintances, weights, fermionic=False)
      return qml.state()
  ```

  ```pycon
  >>> print(weights)
  tensor([0.20308242, 0.91906199, 0.67988804, 0.81290256, 0.08708985,
          0.81860084, 0.34448344, 0.05655892, 0.61781612, 0.51829044], requires_grad=True)
  >>> print(qml.draw(swap_network_circuit, expansion_strategy = 'device')())
  0: ─╭●────────╭SWAP─────────────────╭●────────╭SWAP─────────────────╭●────────╭SWAP─┤  State
  1: ─╰RY(0.20)─╰SWAP─╭●────────╭SWAP─╰RY(0.09)─╰SWAP─╭●────────╭SWAP─╰RY(0.62)─╰SWAP─┤  State
  2: ─╭●────────╭SWAP─╰RY(0.68)─╰SWAP─╭●────────╭SWAP─╰RY(0.34)─╰SWAP─╭●────────╭SWAP─┤  State
  3: ─╰RY(0.92)─╰SWAP─╭●────────╭SWAP─╰RY(0.82)─╰SWAP─╭●────────╭SWAP─╰RY(0.52)─╰SWAP─┤  State
  4: ─────────────────╰RY(0.81)─╰SWAP─────────────────╰RY(0.06)─╰SWAP─────────────────┤  State
  ```

<h3>Improvements 🛠</h3>

<h4>Pulse programming</h4>

* A new function called `qml.pulse.pwc` has been added as a convenience function for defining a `qml.pulse.ParametrizedHamiltonian`.
  This function can be used to create a callable coefficient by setting
  the timespan over which the function should be non-zero. The resulting callable
  can be passed an array of parameters and a time.
  [(#3645)](https://github.com/PennyLaneAI/pennylane/pull/3645)

  ```pycon
  >>> timespan = (2, 4)
  >>> f = qml.pulse.pwc(timespan)
  >>> f * qml.PauliX(0)
  ParametrizedHamiltonian: terms=1
  ```

  The `params` array will be used as bin values evenly distributed over the timespan,
  and the parameter `t` will determine which of the bins is returned.

  ```pycon
  >>> f(params=[1.2, 2.3, 3.4, 4.5], t=3.9)
  DeviceArray(4.5, dtype=float32)
  >>> f(params=[1.2, 2.3, 3.4, 4.5], t=6)  # zero outside the range (2, 4)
  DeviceArray(0., dtype=float32)
  ```

* A new function called`qml.pulse.pwc_from_function` has been added as a decorator for defining a
  `qml.pulse.ParametrizedHamiltonian`.
  This function can be used to decorate a function and create a piecewise constant
  approximation of it.
  [(#3645)](https://github.com/PennyLaneAI/pennylane/pull/3645)

  ```pycon
  >>> @qml.pulse.pwc_from_function((2, 4), num_bins=10)
  ... def f1(p, t):
  ...     return p * t
  ```

  The resulting function approximates the same of `p**2 * t` on the interval `t=(2, 4)`
  in 10 bins, and returns zero outside the interval.

  ```pycon
  # t=2 and t=2.1 are within the same bin
  >>> f1(3, 2), f1(3, 2.1)
  (DeviceArray(6., dtype=float32), DeviceArray(6., dtype=float32))
  # next bin
  >>> f1(3, 2.2)
  DeviceArray(6.6666665, dtype=float32)
  # outside the interval t=(2, 4)
  >>> f1(3, 5)
  DeviceArray(0., dtype=float32)
  ```

* Add `ParametrizedHamiltonianPytree` class, which is a pytree jax object representing a parametrized
  Hamiltonian, where the matrix computation is delayed to improve performance.
  [(#3779)](https://github.com/PennyLaneAI/pennylane/pull/3779)

<h4>Operations and batching</h4>

* The function `qml.dot` has been updated to compute the dot product between a vector and a list of operators.
  [(#3586)](https://github.com/PennyLaneAI/pennylane/pull/3586)

  ```pycon
  >>> coeffs = np.array([1.1, 2.2])
  >>> ops = [qml.PauliX(0), qml.PauliY(0)]
  >>> qml.dot(coeffs, ops)
  (1.1*(PauliX(wires=[0]))) + (2.2*(PauliY(wires=[0])))
  >>> qml.dot(coeffs, ops, pauli=True)
  1.1 * X(0) + 2.2 * Y(0)
  ```

* `qml.evolve` returns the evolution of an `Operator` or a `ParametrizedHamiltonian`.
  [(#3617)](https://github.com/PennyLaneAI/pennylane/pull/3617)
  [(#3706)](https://github.com/PennyLaneAI/pennylane/pull/3706)

* `qml.ControlledQubitUnitary` now inherits from `qml.ops.op_math.ControlledOp`, which defines `decomposition`, `expand`, and `sparse_matrix` rather than raising an error.
  [(#3450)](https://github.com/PennyLaneAI/pennylane/pull/3450)

* Parameter broadcasting support has been added for the `qml.ops.op_math.Controlled` class if the base operator supports
  broadcasting.
  [(#3450)](https://github.com/PennyLaneAI/pennylane/pull/3450)

* The `qml.generator` function now checks if the generator is Hermitian, rather than whether it is a subclass of
  `Observable`. This allows it to return valid generators from `SymbolicOp` and `CompositeOp` classes.
  [(#3485)](https://github.com/PennyLaneAI/pennylane/pull/3485)

* The `qml.equal` function has been extended to compare `Prod` and `Sum` operators.
  [(#3516)](https://github.com/PennyLaneAI/pennylane/pull/3516)

* `qml.purity` has been added as a measurement process for purity
  [(#3551)](https://github.com/PennyLaneAI/pennylane/pull/3551)

* In-place inversion has been removed for qutrit operations in preparation for the
  removal of in-place inversion.
  [(#3566)](https://github.com/PennyLaneAI/pennylane/pull/3566)

* The `qml.utils.sparse_hamiltonian` function has been moved to thee `qml.Hamiltonian.sparse_matrix` method.
  [(#3585)](https://github.com/PennyLaneAI/pennylane/pull/3585)

* The `qml.pauli.PauliSentence.operation()` method has been improved to avoid instantiating an `SProd` operator when
  the coefficient is equal to 1.
  [(#3595)](https://github.com/PennyLaneAI/pennylane/pull/3595)

* Batching is now allowed in all `SymbolicOp` operators, which include `Exp`, `Pow` and `SProd`.
  [(#3597)](https://github.com/PennyLaneAI/pennylane/pull/3597)

* The `Sum` and `Prod` operations now have broadcasted operands.
  [(#3611)](https://github.com/PennyLaneAI/pennylane/pull/3611)

* The XYX single-qubit unitary decomposition has been implemented.
  [(#3628)](https://github.com/PennyLaneAI/pennylane/pull/3628)

* All dunder methods now return `NotImplemented`, allowing the right dunder method (e.g. `__radd__`)
  of the other class to be called.
  [(#3631)](https://github.com/PennyLaneAI/pennylane/pull/3631)

* The `qml.GellMann` operators now include their index when displayed.
  [(#3641)](https://github.com/PennyLaneAI/pennylane/pull/3641)

* `qml.ops.ctrl_decomp_zyz` has been added to compute the decomposition of a controlled single-qubit operation given
  a single-qubit operation and the control wires.
  [(#3681)](https://github.com/PennyLaneAI/pennylane/pull/3681)

* `qml.pauli.is_pauli_word` now supports `Prod` and `SProd` operators, and it returns `False` when a
  `Hamiltonian` contains more than one term.
  [(#3692)](https://github.com/PennyLaneAI/pennylane/pull/3692)

* `qml.pauli.pauli_word_to_string` now supports `Prod`, `SProd` and `Hamiltonian` operators.
  [(#3692)](https://github.com/PennyLaneAI/pennylane/pull/3692)

* `qml.ops.op_math.Controlled` can now decompose single qubit target operations more effectively using the ZYZ
  decomposition.
  [(#3726)](https://github.com/PennyLaneAI/pennylane/pull/3726)

 * The `qml.qchem.Molecule` class raises an error when the molecule has an odd number of electrons or 
   when the spin multiplicity is not 1.
   [(#3748)](https://github.com/PennyLaneAI/pennylane/pull/3748)

* `qml.qchem.basis_rotation` now accounts for spin, allowing it to perform Basis Rotation Groupings
  for molecular hamiltonians.
  [(#3714)](https://github.com/PennyLaneAI/pennylane/pull/3714)
  [(#3774)](https://github.com/PennyLaneAI/pennylane/pull/3774)

* The gradient transforms work for the new return type system with non-trivial classical jacobians.
  [(#3776)](https://github.com/PennyLaneAI/pennylane/pull/3776)

* The `default.mixed` device has received a performance improvement for multi-qubit operations.
  This also allows to apply channels that act on more than seven qubits, which was not possible before.
  [(#3584)](https://github.com/PennyLaneAI/pennylane/pull/3584)

* `qml.dot` now groups coefficients together.
  [(#3691)](https://github.com/PennyLaneAI/pennylane/pull/3691)

  ```pycon
  >>> qml.dot(coeffs=[2, 2, 2], ops=[qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)])
  2*(PauliX(wires=[0]) + PauliY(wires=[1]) + PauliZ(wires=[2]))
  ```

* `qml.generator` now supports operators with `Sum` and `Prod` generators.
  [(#3691)](https://github.com/PennyLaneAI/pennylane/pull/3691)

* The `Sum._sort` method now takes into account the name of the operator when sorting.
  [(#3691)](https://github.com/PennyLaneAI/pennylane/pull/3691)

* A new tape transform called `qml.transforms.sign_expand` has been added. It implements the optimal decomposition of a fast-forwardable Hamiltonian that minimizes the variance of its estimator in the Single-Qubit-Measurement from [arXiv:2207.09479](https://arxiv.org/abs/2207.09479).
  [(#2852)](https://github.com/PennyLaneAI/pennylane/pull/2852)

<h4>Differentiability and interfaces</h4>

* The `qml.math` module now also contains a submodule for
  fast Fourier transforms, `qml.math.fft`.
  [(#1440)](https://github.com/PennyLaneAI/pennylane/pull/1440)

  The submodule in particular provides differentiable
  versions of the following functions, available in all common
  interfaces for PennyLane

  * [fft](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html)
  * [ifft](https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html)
  * [fft2](https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html)
  * [ifft2](https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft2.html)

  Note that the output of the derivative of these functions
  may differ when used with complex-valued inputs, due to different
  conventions on complex-valued derivatives.

* Validation has been added on gradient keyword arguments when initializing a QNode — if unexpected keyword arguments are passed,
  a `UserWarning` is raised. A list of the current expected gradient function keyword arguments can be accessed via
  `qml.gradients.SUPPORTED_GRADIENT_KWARGS`.
  [(#3526)](https://github.com/PennyLaneAI/pennylane/pull/3526)

* The `numpy` version has been constrained to `<1.24`.
  [(#3563)](https://github.com/PennyLaneAI/pennylane/pull/3563)

* Support for two-qubit unitary decomposition with JAX-JIT has been added.
  [(#3569)](https://github.com/PennyLaneAI/pennylane/pull/3569)

* `qml.math.size` now supports PyTorch tensors.
  [(#3606)](https://github.com/PennyLaneAI/pennylane/pull/3606)

* Most quantum channels are now fully differentiable on all interfaces.
  [(#3612)](https://github.com/PennyLaneAI/pennylane/pull/3612)

* `qml.math.matmul` now supports PyTorch and Autograd tensors.
  [(#3613)](https://github.com/PennyLaneAI/pennylane/pull/3613)

* Add `qml.math.detach`, which detaches a tensor from its trace. This stops
  automatic gradient computations.
  [(#3674)](https://github.com/PennyLaneAI/pennylane/pull/3674)

* Add `typing.TensorLike` type.
  [(#3675)](https://github.com/PennyLaneAI/pennylane/pull/3675)

* `qml.QuantumMonteCarlo` template is now JAX-JIT compatible when passing `jax.numpy` arrays to the template.
  [(#3734)](https://github.com/PennyLaneAI/pennylane/pull/3734)

* `DefaultQubitJax` now supports evolving the state vector when executing `qml.pulse.ParametrizedEvolution`
  gates.
  [(#3743)](https://github.com/PennyLaneAI/pennylane/pull/3743)

* `SProd.sparse_matrix` now supports interface-specific variables with a single element as the `scalar`.
  [(#3770)](https://github.com/PennyLaneAI/pennylane/pull/3770)

* Added `argnum` argument to `metric_tensor`. By passing a sequence of indices referring to trainable tape parameters,
  the metric tensor is only computed with respect to these parameters. This reduces the number of tapes that have to
  be run.
  [(#3587)](https://github.com/PennyLaneAI/pennylane/pull/3587)

* The parameter-shift derivative of variances saves a redundant evaluation of the
  corresponding unshifted expectation value tape, if possible
  [(#3744)](https://github.com/PennyLaneAI/pennylane/pull/3744)

<h4>Next generation device API</h4>

* The `apply_operation` single-dispatch function is added to `devices/qubit` that applies an operation
  to a state and returns a new state.
  [(#3637)](https://github.com/PennyLaneAI/pennylane/pull/3637)

* The `preprocess` function is added to `devices/qubit` that validates, expands, and transforms a batch
  of `QuantumTape` objects to abstract preprocessing details away from the device.
  [(#3708)](https://github.com/PennyLaneAI/pennylane/pull/3708)

* The `create_initial_state` function is added to `devices/qubit` that returns an initial state for an execution.
  [(#3683)](https://github.com/PennyLaneAI/pennylane/pull/3683)

* The `simulate` function is added to `devices/qubit` that turns a single quantum tape into a measurement result.
  The function only supports state based measurements with either no observables or observables with diagonalizing gates.
  It supports simultaneous measurement of non-commuting observables.
  [(#3700)](https://github.com/PennyLaneAI/pennylane/pull/3700)

* The `ExecutionConfig` data class has been added.
  [(#3649)](https://github.com/PennyLaneAI/pennylane/pull/3649)

* The `StatePrep` class has been added as an interface that state-prep operators must implement.
  [(#3654)](https://github.com/PennyLaneAI/pennylane/pull/3654)

* `qml.QubitStateVector` now implements the `StatePrep` interface.
  [(#3685)](https://github.com/PennyLaneAI/pennylane/pull/3685)

* `qml.BasisState` now implements the `StatePrep` interface.
  [(#3693)](https://github.com/PennyLaneAI/pennylane/pull/3693)

* New Abstract Base Class for devices `Device` is added to the `devices.experimental` submodule.
  This interface is still in experimental mode and not integrated with the rest of pennylane.
  [(#3602)](https://github.com/PennyLaneAI/pennylane/pull/3602)

<h4>Other improvements</h4>

* Writing Hamiltonians to a file using the `qml.data` module has been improved by employing a condensed writing format.
  [(#3592)](https://github.com/PennyLaneAI/pennylane/pull/3592)

* Lazy-loading in the `qml.data.Dataset.read()` method is more universally supported.
  [(#3605)](https://github.com/PennyLaneAI/pennylane/pull/3605)

* The `qchem.Molecule` class raises an error when the molecule has an odd number of electrons or
  when the spin multiplicity is not 1.
  [(#3748)](https://github.com/PennyLaneAI/pennylane/pull/3748)

* `qml.draw` and `qml.draw_mpl` have been updated to draw any quantum function,
  which allows for visualizing only part of a complete circuit/QNode.
  [(#3760)](https://github.com/PennyLaneAI/pennylane/pull/3760)

* The string representation of a Measurement Process now includes the `_eigvals`
  property if it is set.
  [(#3820)](https://github.com/PennyLaneAI/pennylane/pull/3820)

<h3>Breaking changes 💔</h3>

* The argument `mode` in execution has been replaced by the boolean `grad_on_execution` in the new execution pipeline.
  [(#3723)](https://github.com/PennyLaneAI/pennylane/pull/3723)

* `qml.VQECost` has been removed.
  [(#3735)](https://github.com/PennyLaneAI/pennylane/pull/3735)

* The default interface is now `auto`.
  [(#3677)](https://github.com/PennyLaneAI/pennylane/pull/3677)
  [(#3752)](https://github.com/PennyLaneAI/pennylane/pull/3752)
  [(#3829)](https://github.com/PennyLaneAI/pennylane/pull/3829)

  The interface is determined during the QNode call instead of the
  initialization. It means that the `gradient_fn` and `gradient_kwargs` are only defined on the QNode at the beginning
  of the call. Moreover, without specifying the interface it is not possible to guarantee that the device will not be changed
  during the call if you are using backprop (such as `default.qubit` changing to `default.qubit.jax`) whereas before it was happening at
  initialization.

* The tape method `get_operation` can also now return the operation index in the tape, and it can be
  activated by setting the `return_op_index` to `True`: `get_operation(idx, return_op_index=True)`. It will become
  the default in version `0.30`.
  [(#3667)](https://github.com/PennyLaneAI/pennylane/pull/3667)

* `Operation.inv()` and the `Operation.inverse` setter have been removed. Please use `qml.adjoint` or `qml.pow` instead.
  [(#3618)](https://github.com/PennyLaneAI/pennylane/pull/3618)

  For example, instead of

  ```pycon
  >>> qml.PauliX(0).inv()
  ```

  use

  ```pycon
  >>> qml.adjoint(qml.PauliX(0))
  ```

* The `Operation.inverse` property has been removed completely.
  [(#3725)](https://github.com/PennyLaneAI/pennylane/pull/3725)

* The target wires of `qml.ControlledQubitUnitary` are no longer available via `op.hyperparameters["u_wires"]`.
  Instead, they can be accesses via `op.base.wires` or `op.target_wires`.
  [(#3450)](https://github.com/PennyLaneAI/pennylane/pull/3450)

* The tape constructed by a `QNode` is no longer queued to surrounding contexts.
  [(#3509)](https://github.com/PennyLaneAI/pennylane/pull/3509)

* Nested operators like `Tensor`, `Hamiltonian`, and `Adjoint` now remove their owned operators
  from the queue instead of updating their metadata to have an `"owner"`.
  [(#3282)](https://github.com/PennyLaneAI/pennylane/pull/3282)

* `qml.qchem.scf`, `qml.RandomLayers.compute_decomposition`, and `qml.Wires.select_random` now use
  local random number generators instead of global random number generators. This may lead to slightly
  different random numbers and an independence of the results from the global random number generation state.
  Please provide a seed to each individual function instead if you want controllable results.
  [(#3624)](https://github.com/PennyLaneAI/pennylane/pull/3624)

* `qml.transforms.measurement_grouping` has been removed. Users should use `qml.transforms.hamiltonian_expand`
  instead.
  [(#3701)](https://github.com/PennyLaneAI/pennylane/pull/3701)

* `op.simplify()` for operators which are linear combinations of Pauli words will use a builtin Pauli representation
  to more efficiently compute the simplification of the operator.
  [(#3481)](https://github.com/PennyLaneAI/pennylane/pull/3481)

* All `Operator`'s input parameters that are lists are cast into vanilla numpy arrays.
  [(#3659)](https://github.com/PennyLaneAI/pennylane/pull/3659)

* `QubitDevice.expval` no longer permutes an observable's wire order before passing
  it to `QubitDevice.probability`. The associated downstream changes for `default.qubit`
  have been made, but this may still affect expectations for other devices that inherit
  from `QubitDevice` and override `probability` (or any other helper functions that take
  a wire order such as `marginal_prob`, `estimate_probability` or `analytic_probability`).
  [(#3753)](https://github.com/PennyLaneAI/pennylane/pull/3753)

<h3>Deprecations 👋</h3>

* `qml.utils.sparse_hamiltonian` function has been deprecated, and usage will now raise a warning.
  Instead, one should use the `qml.Hamiltonian.sparse_matrix` method.
  [(#3585)](https://github.com/PennyLaneAI/pennylane/pull/3585)

* The `collections` module has been deprecated.
  [(#3686)](https://github.com/PennyLaneAI/pennylane/pull/3686)
  [(#3687)](https://github.com/PennyLaneAI/pennylane/pull/3687)

* `qml.op_sum` has been deprecated. Users should use `qml.sum` instead.
  [(#3686)](https://github.com/PennyLaneAI/pennylane/pull/3686)

* The use of `Evolution` directly has been deprecated. Users should use `qml.evolve` instead.
  This new function changes the sign of the given parameter.
  [(#3706)](https://github.com/PennyLaneAI/pennylane/pull/3706)

* Use of `qml.dot` with a `QNodeCollection` has been deprecated.
  [(#3586)](https://github.com/PennyLaneAI/pennylane/pull/3586)

<h3>Documentation 📝</h3>

* Revise note on GPU support in the [circuit introduction](https://docs.pennylane.ai/en/stable/introduction/circuits.html#defining-a-device).
[(#3836)](https://github.com/PennyLaneAI/pennylane/pull/3836)

* Make warning about vanilla version of NumPy for differentiation more prominent.
  [(#3838)](https://github.com/PennyLaneAI/pennylane/pull/3838)

* The documentation for `qml.operation` has been improved.
  [(#3664)](https://github.com/PennyLaneAI/pennylane/pull/3664)

* The code example in `qml.SparseHamiltonian` has been updated with the correct wire range.
  [(#3643)](https://github.com/PennyLaneAI/pennylane/pull/3643)

* A hyperlink has been added in the text for a URL in the `qml.qchem.mol_data` docstring.
  [(#3644)](https://github.com/PennyLaneAI/pennylane/pull/3644)

* A typo was corrected in the documentation for `qml.math.vn_entropy`.
[(#3740)](https://github.com/PennyLaneAI/pennylane/pull/3740)

<h3>Bug fixes 🐛</h3>

* Fixed a bug where measuring ``qml.probs`` in the computational basis with non-commuting
  measurements returned incorrect results. Now an error is raised.
  [(#3811)](https://github.com/PennyLaneAI/pennylane/pull/3811)

* Fixed a bug where measuring ``qml.probs`` in the computational basis with non-commuting
  measurements returned incorrect results. Now an error is raised.
  [(#3811)](https://github.com/PennyLaneAI/pennylane/pull/3811)

* Fixed a bug in the drawer where nested controlled operations would output
  the label of the operation being controlled, rather than the control values.
  [(#3745)](https://github.com/PennyLaneAI/pennylane/pull/3745)

* Fixed a bug in `qml.transforms.metric_tensor` where prefactors of operation generators were taken
  into account multiple times, leading to wrong outputs for non-standard operations.
  [(#3579)](https://github.com/PennyLaneAI/pennylane/pull/3579)

* Local random number generators are now used where possible to avoid mutating the global random state.
  [(#3624)](https://github.com/PennyLaneAI/pennylane/pull/3624)

* The `networkx` version change being broken has been fixed by selectively skipping a `qcut` TensorFlow-JIT test.
  [(#3609)](https://github.com/PennyLaneAI/pennylane/pull/3609)
  [(#3619)](https://github.com/PennyLaneAI/pennylane/pull/3619)

* Fixed the wires for the `Y` decomposition in the ZX calculus transform.
  [(#3598)](https://github.com/PennyLaneAI/pennylane/pull/3598)

* `qml.pauli.PauliWord` is now pickle-able.
  [(#3588)](https://github.com/PennyLaneAI/pennylane/pull/3588)

* Child classes of `QuantumScript` now return their own type when using `SomeChildClass.from_queue`.
  [(#3501)](https://github.com/PennyLaneAI/pennylane/pull/3501)

* A typo has been fixed in the calculation and error messages in `operation.py`
  [(#3536)](https://github.com/PennyLaneAI/pennylane/pull/3536)

* `qml.data.Dataset.write()` now ensures that any lazy-loaded values are loaded before they are written to a file.
  [(#3605)](https://github.com/PennyLaneAI/pennylane/pull/3605)

* `Tensor._batch_size` is now set to `None` during initialization, copying and `map_wires`.
  [(#3642)](https://github.com/PennyLaneAI/pennylane/pull/3642)
  [(#3661)](https://github.com/PennyLaneAI/pennylane/pull/3661)

* `Tensor.has_matrix` is now set to `True`.
  [(#3647)](https://github.com/PennyLaneAI/pennylane/pull/3647)

* Fixed typo in the example of `qml.IsingZZ` gate decomposition.
  [(#3676)](https://github.com/PennyLaneAI/pennylane/pull/3676)

* Fixed a bug that made tapes/qnodes using `qml.Snapshot` incompatible with `qml.drawer.tape_mpl`.
  [(#3704)](https://github.com/PennyLaneAI/pennylane/pull/3704)

* `Tensor._pauli_rep` is set to `None` during initialization and `Tensor.data` has been added to its setter.
  [(#3722)](https://github.com/PennyLaneAI/pennylane/pull/3722)

* `qml.math.ndim` has been redirected to `jnp.ndim` when using it on a `jax` tensor.
  [(#3730)](https://github.com/PennyLaneAI/pennylane/pull/3730)

* Implementations of `marginal_prob` (and subsequently, `qml.probs`) now return
  probabilities with the expected wire order.
  [(#3753)](https://github.com/PennyLaneAI/pennylane/pull/3753)

  This bug affected most probabilistic measurement processes on devices that
  inherit from `QubitDevice` when the measured wires are out of order with
  respect to the device wires and 3 or more wires are measured. The assumption
  was that marginal probabilities would be computed with the device's state
  and wire order, then re-ordered according to the measurement process wire
  order. Instead, the re-ordering went in the inverse direction (that is, from
  measurement process wire order to device wire order). This is now fixed. Note
  that this only occurred for 3 or more measured wires because this mapping is
  identical otherwise. More details and discussion of this bug can be found in
  [the original bug report](https://github.com/PennyLaneAI/pennylane/issues/3741).

* Empty iterables can no longer be returned from QNodes.
  [(#3769)](https://github.com/PennyLaneAI/pennylane/pull/3769)

* The keyword arguments for `qml.equal` now are used when comparing the observables of a 
  Measurement Process. The eigvals of measurements are only requested if both observables are `None`,
  saving computational effort.
  [(#3820)](https://github.com/PennyLaneAI/pennylane/pull/3820)

* Only converts input to `qml.Hermitian` to a numpy array if the input is a list.
  [(#3820)](https://github.com/PennyLaneAI/pennylane/pull/3820)

<h3>Contributors ✍</h3>

This release contains contributions from (in alphabetical order):

Gian-Luca Anselmetti,
Guillermo Alonso-Linaje,
Juan Miguel Arrazola,
Ikko Ashimine,
Utkarsh Azad,
Miriam Beddig,
Cristian Boghiu,
Thomas Bromley,
Astral Cai,
Isaac De Vlugt,
Olivia Di Matteo,
Lillian M. A. Frederiksen,
Soran Jahangiri,
Korbinian Kottmann,
Christina Lee,
Albert Mitjans Coma,
Romain Moyard,
Mudit Pandey,
Borja Requena,
Matthew Silverman,
Jay Soni,
Antal Száva,
Frederik Wilde,
David Wierichs,
Moritz Willmann.
