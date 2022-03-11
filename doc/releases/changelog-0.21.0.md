:orphan:

# Release 0.21.0

<h3>New features since last release</h3>

<h4>Reduce qubit requirements of simulating Hamiltonians ⚛️</h4>

* Functions for tapering qubits based on molecular symmetries have been added,
  following results from [Setia et al](https://arxiv.org/abs/1910.14644).
  [(#1966)](https://github.com/PennyLaneAI/pennylane/pull/1966)
  [(#1974)](https://github.com/PennyLaneAI/pennylane/pull/1974)
  [(#2041)](https://github.com/PennyLaneAI/pennylane/pull/2041)
  [(#2042)](https://github.com/PennyLaneAI/pennylane/pull/2042)

  With this functionality, a molecular Hamiltonian and the corresponding Hartree-Fock (HF) state can be transformed to a new Hamiltonian and HF state that acts on a reduced number of qubits, respectively.
  ```python
  # molecular geometry
  symbols = ["He", "H"]
  geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4588684632]])
  mol = qml.hf.Molecule(symbols, geometry, charge=1)

  # generate the qubit Hamiltonian
  H = qml.hf.generate_hamiltonian(mol)(geometry)

  # determine Hamiltonian symmetries
  generators, paulix_ops = qml.hf.generate_symmetries(H, len(H.wires))
  opt_sector = qml.hf.optimal_sector(H, generators, mol.n_electrons)

  # taper the Hamiltonian
  H_tapered = qml.hf.transform_hamiltonian(H, generators, paulix_ops, opt_sector)
  ```

  We can compare the number of qubits required by the original Hamiltonian
  and the tapered Hamiltonian:

  ```pycon
  >>> len(H.wires)
  4
  >>> len(H_tapered.wires)
  2
  ```

  For quantum chemistry algorithms, the Hartree-Fock state can also be tapered:

  ```python
  n_elec = mol.n_electrons
  n_qubits = mol.n_orbitals * 2

  hf_tapered = qml.hf.transform_hf(
      generators, paulix_ops, opt_sector, n_elec, n_qubits
  )
  ```
  ```pycon
  >>> hf_tapered
  tensor([1, 1], requires_grad=True)
  ```

<h4>New tensor network templates 🪢</h4>

* Quantum circuits with the shape
  of a matrix product state tensor network can now be easily implemented
  using the new `qml.MPS` template, based on the work
  [arXiv:1803.11537](https://arxiv.org/abs/1803.11537).
  [(#1871)](https://github.com/PennyLaneAI/pennylane/pull/1871)

  ```python
  def block(weights, wires):
      qml.CNOT(wires=[wires[0], wires[1]])
      qml.RY(weights[0], wires=wires[0])
      qml.RY(weights[1], wires=wires[1])

  n_wires = 4
  n_block_wires = 2
  n_params_block = 2
  template_weights = np.array([[0.1, -0.3], [0.4, 0.2], [-0.15, 0.5]], requires_grad=True)

  dev = qml.device("default.qubit", wires=range(n_wires))

  @qml.qnode(dev)
  def circuit(weights):
      qml.MPS(range(n_wires), n_block_wires, block, n_params_block, weights)
      return qml.expval(qml.PauliZ(wires=n_wires - 1))
  ```

  The resulting circuit is:
  ```pycon
  >>> print(qml.draw(circuit, expansion_strategy="device")(template_weights))
  0: ──╭C──RY(0.1)───────────────────────────────┤
  1: ──╰X──RY(-0.3)──╭C──RY(0.4)─────────────────┤
  2: ────────────────╰X──RY(0.2)──╭C──RY(-0.15)──┤
  3: ─────────────────────────────╰X──RY(0.5)────┤ ⟨Z⟩
  ```

* Added a template for tree tensor networks, `qml.TTN`.
  [(#2043)](https://github.com/PennyLaneAI/pennylane/pull/2043)
  ```python
  def block(weights, wires):
      qml.CNOT(wires=[wires[0], wires[1]])
      qml.RY(weights[0], wires=wires[0])
      qml.RY(weights[1], wires=wires[1])

  n_wires = 4
  n_block_wires = 2
  n_params_block = 2
  n_blocks = qml.MPS.get_n_blocks(range(n_wires), n_block_wires)
  template_weights = [[0.1, -0.3]] * n_blocks

  dev = qml.device("default.qubit", wires=range(n_wires))

  @qml.qnode(dev)
  def circuit(template_weights):
      qml.TTN(range(n_wires), n_block_wires, block, n_params_block, template_weights)
      return qml.expval(qml.PauliZ(wires=n_wires - 1))
  ```
  The resulting circuit is:
  ```pycon
  >>> print(qml.draw(circuit, expansion_strategy="device")(template_weights))
  0: ──╭C──RY(0.1)─────────────────┤
  1: ──╰X──RY(-0.3)──╭C──RY(0.1)───┤
  2: ──╭C──RY(0.1)───│─────────────┤
  3: ──╰X──RY(-0.3)──╰X──RY(-0.3)──┤ ⟨Z⟩
  ```

<h4>Generalized RotosolveOptmizer 📉</h4>

* The `RotosolveOptimizer` has been generalized to arbitrary frequency spectra
  in the cost function. Also note the changes in behaviour listed under *Breaking
  changes*.
  [(#2081)](https://github.com/PennyLaneAI/pennylane/pull/2081)

  Previously, the RotosolveOptimizer only supported variational circuits using
  special gates such as single-qubit Pauli rotations. Now, circuits with
  arbitrary gates are supported natively without decomposition, as long as the
  frequencies of the gate parameters are known. This new generalization extends
  the Rotosolve optimization method to a larger class of circuits, and can
  reduce the cost of the optimization compared to decomposing all gates to
  single-qubit rotations.

  Consider the QNode
  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def qnode(x, Y):
      qml.RX(2.5 * x, wires=0)
      qml.CNOT(wires=[0, 1])
      qml.RZ(0.3 * Y[0], wires=0)
      qml.CRY(1.1 * Y[1], wires=[1, 0])
      return qml.expval(qml.PauliX(0) @ qml.PauliZ(1))

  x = np.array(0.8, requires_grad=True)
  Y = np.array([-0.2, 1.5], requires_grad=True)
  ```

  Its frequency spectra can be easily obtained via `qml.fourier.qnode_spectrum`:
  ```pycon
  >>> spectra = qml.fourier.qnode_spectrum(qnode)(x, Y)
  >>> spectra
  {'x': {(): [-2.5, 0.0, 2.5]},
   'Y': {(0,): [-0.3, 0.0, 0.3], (1,): [-1.1, -0.55, 0.0, 0.55, 1.1]}}
  ```

  We may then initialize the `RotosolveOptimizer` and minimize the QNode cost function
  by providing this information about the frequency spectra. We also compare the cost at
  each step to the initial cost.
  ```pycon
  >>> cost_init = qnode(x, Y)
  >>> opt = qml.RotosolveOptimizer()
  >>> for _ in range(2):
  ...     x, Y = opt.step(qnode, x, Y, spectra=spectra)
  ...     print(f"New cost: {np.round(qnode(x, Y), 3)}; Initial cost: {np.round(cost_init, 3)}")
  New cost: 0.0; Initial cost: 0.706
  New cost: -1.0; Initial cost: 0.706
  ```

  The optimization with `RotosolveOptimizer` is performed in substeps. The minimal cost
  of these substeps can be retrieved by setting `full_output=True`.
  ```pycon
  >>> x = np.array(0.8, requires_grad=True)
  >>> Y = np.array([-0.2, 1.5], requires_grad=True)
  >>> opt = qml.RotosolveOptimizer()
  >>> for _ in range(2):
  ...     (x, Y), history = opt.step(qnode, x, Y, spectra=spectra, full_output=True)
  ...     print(f"New cost: {np.round(qnode(x, Y), 3)} reached via substeps {np.round(history, 3)}")
  New cost: 0.0 reached via substeps [-0.  0.  0.]
  New cost: -1.0 reached via substeps [-1. -1. -1.]
  ```
  However, note that these intermediate minimal values are evaluations of the
  *reconstructions* that Rotosolve creates and uses internally for the optimization,
  and not of the original objective function. For noisy cost functions, these intermediate
  evaluations may differ significantly from evaluations of the original cost function.

<h4>Improved JAX support 💻</h4>

* The JAX interface now supports evaluating vector-valued QNodes.
  [(#2110)](https://github.com/PennyLaneAI/pennylane/pull/2110)

  Vector-valued QNodes include those with:
  * `qml.probs`;
  * `qml.state`;
  * `qml.sample` or
  * multiple `qml.expval` / `qml.var` measurements.

  Consider a QNode that returns basis-state probabilities:
  ```python
  dev = qml.device('default.qubit', wires=2)
  x = jnp.array(0.543)
  y = jnp.array(-0.654)

  @qml.qnode(dev, diff_method="parameter-shift", interface="jax")
  def circuit(x, y):
      qml.RX(x, wires=[0])
      qml.RY(y, wires=[1])
      qml.CNOT(wires=[0, 1])
      return qml.probs(wires=[1])
  ```
  The QNode can be evaluated and its jacobian can be computed:
  ```pycon
  >>> circuit(x, y)
  DeviceArray([0.8397495 , 0.16025047], dtype=float32)
  >>> jax.jacobian(circuit, argnums=[0, 1])(x, y)
  (DeviceArray([-0.2050439,  0.2050439], dtype=float32, weak_type=True),
   DeviceArray([ 0.26043, -0.26043], dtype=float32, weak_type=True))
  ```
  Note that `jax.jit` is not yet supported for vector-valued QNodes.

<h4>Speedier quantum natural gradient ⚡</h4>

* A new function for computing the metric tensor on simulators,
  `qml.adjoint_metric_tensor`, has been added, that uses classically
  efficient methods to massively improve performance.
  [(#1992)](https://github.com/PennyLaneAI/pennylane/pull/1992)

  This method, detailed in [Jones (2020)](https://arxiv.org/abs/2011.02991),
  computes the metric tensor using four copies of the state vector and
  a number of operations that scales quadratically in the number of trainable
  parameters.

  Note that as it makes use of state cloning, it is inherently classical
  and can only be used with statevector simulators and `shots=None`.

  It is particularly useful for larger circuits for which backpropagation requires
  inconvenient or even unfeasible amounts of storage, but is slower.
  Furthermore, the adjoint method is only available for analytic computation, not
  for measurements simulation with `shots!=None`.

  ```python
  dev = qml.device("default.qubit", wires=3)

  @qml.qnode(dev)
  def circuit(x, y):
      qml.Rot(*x[0], wires=0)
      qml.Rot(*x[1], wires=1)
      qml.Rot(*x[2], wires=2)
      qml.CNOT(wires=[0, 1])
      qml.CNOT(wires=[1, 2])
      qml.CNOT(wires=[2, 0])
      qml.RY(y[0], wires=0)
      qml.RY(y[1], wires=1)
      qml.RY(y[0], wires=2)
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), qml.expval(qml.PauliY(1))

  x = np.array([[0.2, 0.4, -0.1], [-2.1, 0.5, -0.2], [0.1, 0.7, -0.6]], requires_grad=False)
  y = np.array([1.3, 0.2], requires_grad=True)
  ```

  ```pycon
  >>> qml.adjoint_metric_tensor(circuit)(x, y)
  tensor([[ 0.25495723, -0.07086695],
          [-0.07086695,  0.24945606]], requires_grad=True)
  ```

  Computational cost

  The adjoint method uses :math:`2P^2+4P+1` gates and state cloning operations if the circuit
  is composed only of trainable gates, where :math:`P` is the number of trainable operations.
  If non-trainable gates are included, each of them is applied about :math:`n^2-n` times, where
  :math:`n` is the number of trainable operations that follow after the respective
  non-trainable operation in the circuit. This means that non-trainable gates later in the
  circuit are executed less often, making the adjoint method a bit cheaper if such gates
  appear later.
  The adjoint method requires memory for 4 independent state vectors, which corresponds roughly
  to storing a state vector of a system with 2 additional qubits.

<h4>Compute the Hessian on hardware ⬆️</h4>

* A new gradient transform `qml.gradients.param_shift_hessian` has been added
  to directly compute the Hessian (2nd order partial derivative matrix) of
  QNodes on hardware.
  [(#1884)](https://github.com/PennyLaneAI/pennylane/pull/1884)

  The function generates parameter-shifted tapes which allow the Hessian to be
  computed analytically on hardware and software devices. Compared to using an
  auto-differentiation framework to compute the Hessian via parameter shifts,
  this function will use fewer device invocations and can be used to inspect
  the parameter-shifted "Hessian tapes" directly. The function remains fully
  differentiable on all supported PennyLane interfaces.

  Additionally, the parameter-shift Hessian comes with a new batch transform decorator
  `@qml.gradients.hessian_transform`, which can be used to create custom Hessian functions.

  The following code demonstrates how to use the parameter-shift Hessian:

  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def circuit(x):
      qml.RX(x[0], wires=0)
      qml.RY(x[1], wires=0)
      return qml.expval(qml.PauliZ(0))

  x = np.array([0.1, 0.2], requires_grad=True)

  hessian = qml.gradients.param_shift_hessian(circuit)(x)
  ```
  ```pycon
  >>> hessian
  tensor([[-0.97517033,  0.01983384],
          [ 0.01983384, -0.97517033]], requires_grad=True)
  ```

<h3>Improvements</h3>

* The `qml.transforms.insert` transform now supports adding operation after or
  before certain specific gates.
  [(#1980)](https://github.com/PennyLaneAI/pennylane/pull/1980)

* Added a modified version of the `simplify` function to the `hf` module.
  [(#2103)](https://github.com/PennyLaneAI/pennylane/pull/2103)

  This function combines redundant terms in a Hamiltonian and eliminates terms
  with a coefficient smaller than a cutoff value. The new function makes
  construction of molecular Hamiltonians more efficient. For LiH, as an
  example, the time to construct the Hamiltonian is reduced roughly by a factor
  of 20.

* The QAOA module now accepts both NetworkX and RetworkX graphs as function inputs.
  [(#1791)](https://github.com/PennyLaneAI/pennylane/pull/1791)

* The `CircuitGraph`, used to represent circuits via directed acyclic graphs, now
  uses RetworkX for its internal representation. This results in significant speedup
  for algorithms that rely on a directed acyclic graph representation.
  [(#1791)](https://github.com/PennyLaneAI/pennylane/pull/1791)

* For subclasses of `Operator` where the number of parameters
  is known before instantiation, the
  `num_params` is reverted back to being a static property. This allows to
  programmatically know the number of parameters before an operator is
  instantiated without changing the user interface. A test was added to ensure
  that different ways of defining `num_params` work as expected.
  [(#2101)](https://github.com/PennyLaneAI/pennylane/pull/2101)
  [(#2135)](https://github.com/PennyLaneAI/pennylane/pull/2135)

* A `WireCut` operator has been added for manual wire cut placement
  when constructing a QNode.
  [(#2093)](https://github.com/PennyLaneAI/pennylane/pull/2093)

* The new function `qml.drawer.tape_text` produces a string drawing of a tape. This function
  differs in implementation and minor stylistic details from the old string circuit drawing
  infrastructure.
  [(#1885)](https://github.com/PennyLaneAI/pennylane/pull/1885)

* The `RotosolveOptimizer` now raises an error if no trainable arguments are
  detected, instead of silently skipping update steps for all arguments.
  [(#2109)](https://github.com/PennyLaneAI/pennylane/pull/2109)

* The function `qml.math.safe_squeeze` is introduced and `gradient_transform` allows
  for QNode argument axes of size `1`.
  [(#2080)](https://github.com/PennyLaneAI/pennylane/pull/2080)

  `qml.math.safe_squeeze` wraps `qml.math.squeeze`, with slight modifications:

  - When provided the `axis` keyword argument, axes that do not have size `1` will be
    ignored, instead of raising an error.

  - The keyword argument `exclude_axis` allows to explicitly exclude axes from the
    squeezing.

* The `adjoint` transform now raises and error whenever the object it is applied to
  is not callable.
  [(#2060)](https://github.com/PennyLaneAI/pennylane/pull/2060)

  An example is a list of operations to which one might apply `qml.adjoint`:

  ```python
  dev = qml.device("default.qubit", wires=2)
  @qml.qnode(dev)
  def circuit_wrong(params):
      # Note the difference:                  v                         v
      qml.adjoint(qml.templates.AngleEmbedding(params, wires=dev.wires))
      return qml.state()

  @qml.qnode(dev)
  def circuit_correct(params):
      # Note the difference:                  v                         v
      qml.adjoint(qml.templates.AngleEmbedding)(params, wires=dev.wires)
      return qml.state()

  params = list(range(1, 3))
  ```

  Evaluating `circuit_wrong(params)` now raises a `ValueError` and if we apply
  `qml.adjoint` correctly, we get

  ```pycon
  >>> circuit_correct(params)
  [ 0.47415988+0.j          0.         0.73846026j  0.         0.25903472j
   -0.40342268+0.j        ]
  ```

* A precision argument has been added to the tape's ``to_openqasm`` function
  to control the precision of parameters.
  [(#2071)](https://github.com/PennyLaneAI/pennylane/pull/2071)

* Interferometer now has a `shape` method.
  [(#1946)](https://github.com/PennyLaneAI/pennylane/pull/1946)

* The Barrier and Identity operations now support the `adjoint` method.
  [(#2062)](https://github.com/PennyLaneAI/pennylane/pull/2062)
  [(#2063)](https://github.com/PennyLaneAI/pennylane/pull/2063)

* `qml.BasisStatePreparation` now supports the `batch_params` decorator.
  [(#2091)](https://github.com/PennyLaneAI/pennylane/pull/2091)

* Added a new `multi_dispatch` decorator that helps ease the definition of new functions
  inside PennyLane. The decorator is used throughout the math module, demonstrating use cases.
  [(#2082)](https://github.com/PennyLaneAI/pennylane/pull/2084)
  [(#2096)](https://github.com/PennyLaneAI/pennylane/pull/2096)

  We can decorate a function, indicating the arguments that are
  tensors handled by the interface:

  ```pycon
  >>> @qml.math.multi_dispatch(argnum=[0, 1])
  ... def some_function(tensor1, tensor2, option, like):
  ...     # the interface string is stored in ``like``.
  ...     ...
  ```

  Previously, this was done using the private utility function `_multi_dispatch`.

  ```pycon
  >>> def some_function(tensor1, tensor2, option):
  ...     interface = qml.math._multi_dispatch([tensor1, tensor2])
  ...     ...
  ```

* The `IsingZZ` gate was added to the `diagonal_in_z_basis` attribute. For this
  an explicit `_eigvals` method was added.
  [(#2113)](https://github.com/PennyLaneAI/pennylane/pull/2113)

* The `IsingXX`, `IsingYY` and `IsingZZ` gates were added to
  the `composable_rotations` attribute.
  [(#2113)](https://github.com/PennyLaneAI/pennylane/pull/2113)

<h3>Breaking changes</h3>

* QNode arguments will no longer be considered trainable by default when using
  the Autograd interface. In order to obtain derivatives with respect to a parameter,
  it should be instantiated via PennyLane's NumPy wrapper using the `requires_grad=True`
  attribute. The previous behaviour was deprecated in version v0.19.0 of PennyLane.
  [(#2116)](https://github.com/PennyLaneAI/pennylane/pull/2116)
  [(#2125)](https://github.com/PennyLaneAI/pennylane/pull/2125)
  [(#2139)](https://github.com/PennyLaneAI/pennylane/pull/2139)
  [(#2148)](https://github.com/PennyLaneAI/pennylane/pull/2148)
  [(#2156)](https://github.com/PennyLaneAI/pennylane/pull/2156)

  ```python
  from pennylane import numpy as np

  @qml.qnode(qml.device("default.qubit", wires=2))
  def circuit(x):
    ...

  x = np.array([0.1, 0.2], requires_grad=True)
  qml.grad(circuit)(x)
  ```

  For the `qml.grad` and `qml.jacobian` functions, trainability can alternatively be
  indicated via the `argnum` keyword:

  ```python
  import numpy as np

  @qml.qnode(qml.device("default.qubit", wires=2))
  def circuit(hyperparam, param):
    ...

  x = np.array([0.1, 0.2])
  qml.grad(circuit, argnum=1)(0.5, x)
  ```

* `qml.jacobian` now follows a different convention regarding its output shape.
  [(#2059)](https://github.com/PennyLaneAI/pennylane/pull/2059)

  Previously, `qml.jacobian` would attempt to stack the Jacobian for multiple
  QNode arguments, which succeeded whenever the arguments have the same shape.
  In this case, the stacked Jacobian would also be transposed, leading to the
  output shape `(*reverse_QNode_args_shape, *reverse_output_shape, num_QNode_args)`

  If no stacking and transposing occurs, the output shape instead is a `tuple`
  where each entry corresponds to one QNode argument and has the shape
  `(*output_shape, *QNode_arg_shape)`.

  This breaking change alters the behaviour in the first case and removes the attempt
  to stack and transpose, so that the output always has the shape of the second
  type.

  Note that the behaviour is unchanged --- that is, the Jacobian tuple is unpacked into
  a single Jacobian --- if `argnum=None` and there is only one QNode argument
  with respect to which the differentiation takes place, or if an integer
  is provided as `argnum`.

  A workaround that allowed `qml.jacobian` to differentiate multiple QNode arguments
  will no longer support higher-order derivatives. In such cases, combining multiple
  arguments into a single array is recommended.

* `qml.metric_tensor`, `qml.adjoint_metric_tensor` and `qml.transforms.classical_jacobian`
  now follow a different convention regarding their output shape when being used
  with the Autograd interface
  [(#2059)](https://github.com/PennyLaneAI/pennylane/pull/2059)

  See the previous entry for details. This breaking change immediately follows from
  the change in `qml.jacobian` whenever `hybrid=True` is used in the above methods.

* The behaviour of `RotosolveOptimizer` has been changed regarding
  its keyword arguments.
  [(#2081)](https://github.com/PennyLaneAI/pennylane/pull/2081)

  The keyword arguments `optimizer` and `optimizer_kwargs` for the
  `RotosolveOptimizer` have been renamed to `substep_optimizer`
  and `substep_kwargs`, respectively. Furthermore they have been
  moved from `step` and `step_and_cost` to the initialization `__init__`.

  The keyword argument `num_freqs` has been renamed to `nums_frequency`
  and is expected to take a different shape now:
  Previously, it was expected to be an `int` or a list of entries, with
  each entry in turn being either an `int` or a `list` of `int` entries.
  Now the expected structure is a nested dictionary, matching the
  formatting expected by
  [qml.fourier.reconstruct](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.fourier.reconstruct.html)
  This also matches the expected formatting of the new keyword arguments
  `spectra` and `shifts`.

  For more details, see the
  [RotosolveOptimizer documentation](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.RotosolveOptimizer.html).

<h3>Deprecations</h3>

* Deprecates the caching ability provided by `QubitDevice`.
  [(#2154)](https://github.com/PennyLaneAI/pennylane/pull/2154)

  Going forward, the preferred way is to use the caching abilities of the
  QNode:
  ```python
  dev = qml.device("default.qubit", wires=2)

  cache = {}

  @qml.qnode(dev, diff_method='parameter-shift', cache=cache)
  def circuit():
      qml.RY(0.345, wires=0)
      return qml.expval(qml.PauliZ(0))
  ```
  ```pycon
  >>> for _ in range(10):
  ...    circuit()
  >>> dev.num_executions
  1
  ```

<h3>Bug fixes</h3>

* Fixes a bug where an incorrect number of executions are recorded by
  a QNode using a custom cache with `diff_method="backprop"`.
  [(#2171)](https://github.com/PennyLaneAI/pennylane/pull/2171)

* Fixes a bug where the `default.qubit.jax` device can't be used with `diff_method=None` and jitting.
  [(#2136)](https://github.com/PennyLaneAI/pennylane/pull/2136)

* Fixes a bug where the Torch interface was not properly unwrapping Torch tensors
  to NumPy arrays before executing gradient tapes on devices.
  [(#2117)](https://github.com/PennyLaneAI/pennylane/pull/2117)

* Fixes a bug for the TensorFlow interface where the dtype of input tensors was
  not cast.
  [(#2120)](https://github.com/PennyLaneAI/pennylane/pull/2120)

* Fixes a bug where batch transformed QNodes would fail to apply batch transforms
  provided by the underlying device.
  [(#2111)](https://github.com/PennyLaneAI/pennylane/pull/2111)

* An error is now raised during QNode creation if backpropagation is requested on a device with
  finite shots specified.
  [(#2114)](https://github.com/PennyLaneAI/pennylane/pull/2114)

* Pytest now ignores any `DeprecationWarning` raised within autograd's `numpy_wrapper` module.
  Other assorted minor test warnings are fixed.
  [(#2007)](https://github.com/PennyLaneAI/pennylane/pull/2007)

* Fixes a bug where the QNode was not correctly diagonalizing qubit-wise
  commuting observables.
  [(#2097)](https://github.com/PennyLaneAI/pennylane/pull/2097)

* Fixes a bug in `gradient_transform` where the hybrid differentiation
  of circuits with a single parametrized gate failed and QNode argument
  axes of size `1` where removed from the output gradient.
  [(#2080)](https://github.com/PennyLaneAI/pennylane/pull/2080)

* The available `diff_method` options for QNodes has been corrected in both the
  error messages and the documentation.
  [(#2078)](https://github.com/PennyLaneAI/pennylane/pull/2078)

* Fixes a bug in `DefaultQubit` where the second derivative of QNodes at
  positions corresponding to vanishing state vector amplitudes is wrong.
  [(#2057)](https://github.com/PennyLaneAI/pennylane/pull/2057)

* Fixes a bug where PennyLane didn't require v0.20.0 of PennyLane-Lightning,
  but raised an error with versions of Lightning earlier than v0.20.0 due to
  the new batch execution pipeline.
  [(#2033)](https://github.com/PennyLaneAI/pennylane/pull/2033)

* Fixes a bug in `classical_jacobian` when used with Torch, where the
  Jacobian of the preprocessing was also computed for non-trainable
  parameters.
  [(#2020)](https://github.com/PennyLaneAI/pennylane/pull/2020)

* Fixes a bug in queueing of the `two_qubit_decomposition` method that
  originally led to circuits with >3 two-qubit unitaries failing when passed
  through the `unitary_to_rot` optimization transform.
  [(#2015)](https://github.com/PennyLaneAI/pennylane/pull/2015)

* Fixes a bug which allows using `jax.jit` to be compatible with circuits
  which return `qml.probs` when the `default.qubit.jax` is provided with a custom shot
  vector.
  [(#2028)](https://github.com/PennyLaneAI/pennylane/pull/2028)

* Updated the `adjoint()` method for non-parametric qubit operations to
  solve a bug where repeated `adjoint()` calls don't return the correct
  operator.
  [(#2133)](https://github.com/PennyLaneAI/pennylane/pull/2133)

* Fixed a bug in `insert()` which prevented operations that inherited
  from multiple classes to be inserted.
  [(#2172)](https://github.com/PennyLaneAI/pennylane/pull/2172)

<h3>Documentation</h3>

* Fixes an error in the signs of equations in the `DoubleExcitation` page.
  [(#2072)](https://github.com/PennyLaneAI/pennylane/pull/2072)

* Extends the interfaces description page to explicitly mention device
  compatibility.
  [(#2031)](https://github.com/PennyLaneAI/pennylane/pull/2031)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Ali Asadi, Utkarsh Azad, Sam Banning, Thomas Bromley,
Esther Cruz, Olivia Di Matteo, Christian Gogolin, Diego Guala, Anthony Hayes,
David Ittah, Josh Izaac, Soran Jahangiri, Edward Jiang, Ankit Khandelwal,
Nathan Killoran, Korbinian Kottmann, Christina Lee, Romain Moyard, Lee James
O'Riordan, Maria Schuld, Jay Soni, Antal Száva, David Wierichs, Shaoming Zhang.
