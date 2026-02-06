
# Release 0.19.0

<h3>New features since last release</h3>

<h4>Differentiable Hartree-Fock solver</h4>

* A differentiable Hartree-Fock (HF) solver has been added. It can be used to construct molecular
  Hamiltonians that can be differentiated with respect to nuclear coordinates and basis-set
  parameters. [(#1610)](https://github.com/PennyLaneAI/pennylane/pull/1610)

  The HF solver computes the integrals over basis functions, constructs the relevant matrices, and
  performs self-consistent-field iterations to obtain a set of optimized molecular orbital
  coefficients. These coefficients and the computed integrals over basis functions are used to
  construct the one- and two-body electron integrals in the molecular orbital basis which can be
  used to generate a differentiable second-quantized Hamiltonian in the fermionic and qubit basis.

  The following code shows the construction of the Hamiltonian for the hydrogen molecule where the
  geometry of the molecule is differentiable.

  ```python
  symbols = ["H", "H"]
  geometry = np.array([[0.0000000000, 0.0000000000, -0.6943528941],
                       [0.0000000000, 0.0000000000,  0.6943528941]], requires_grad=True)

  mol = qml.hf.Molecule(symbols, geometry)
  args_mol = [geometry]

  hamiltonian = qml.hf.generate_hamiltonian(mol)(*args_mol)
  ```
  ```pycon
  >>> hamiltonian.coeffs
  tensor([-0.09041082+0.j,  0.17220382+0.j,  0.17220382+0.j,
           0.16893367+0.j,  0.04523101+0.j, -0.04523101+0.j,
          -0.04523101+0.j,  0.04523101+0.j, -0.22581352+0.j,
           0.12092003+0.j, -0.22581352+0.j,  0.16615103+0.j,
           0.16615103+0.j,  0.12092003+0.j,  0.17464937+0.j], requires_grad=True)
  ```

  The generated Hamiltonian can be used in a circuit where the atomic
  coordinates and circuit parameters are optimized simultaneously.

  ```python
  symbols = ["H", "H"]
  geometry = np.array([[0.0000000000, 0.0000000000, 0.0],
                       [0.0000000000, 0.0000000000, 2.0]], requires_grad=True)

  mol = qml.hf.Molecule(symbols, geometry)

  dev = qml.device("default.qubit", wires=4)
  params = [np.array([0.0], requires_grad=True)]

  def generate_circuit(mol):
      @qml.qnode(dev)
      def circuit(*args):
          qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
          qml.DoubleExcitation(*args[0][0], wires=[0, 1, 2, 3])
          return qml.expval(qml.hf.generate_hamiltonian(mol)(*args[1:]))
      return circuit

  for n in range(25):

      mol = qml.hf.Molecule(symbols, geometry)
      args = [params, geometry] # initial values of the differentiable parameters

      g_params = qml.grad(generate_circuit(mol), argnum = 0)(*args)
      params = params - 0.5 * g_params[0]

      forces = qml.grad(generate_circuit(mol), argnum = 1)(*args)
      geometry = geometry - 0.5 * forces

      print(f'Step: {n}, Energy: {generate_circuit(mol)(*args)}, Maximum Force: {forces.max()}')
  ```
  In addition, the new Hartree-Fock solver can further be used to optimize the
  basis set parameters. For details, please refer to the [differentiable
  Hartree-Fock solver
  documentation](https://pennylane.readthedocs.io/en/latest/code/qml_hf.html#using-the-differentiable-hf-solver).

<h4>Integration with Mitiq</h4>

* Error mitigation using the zero-noise extrapolation method is now available through the
  `transforms.mitigate_with_zne` transform. This transform can integrate with the
  [Mitiq](https://mitiq.readthedocs.io/en/stable/) package for unitary folding and extrapolation
  functionality.
  [(#1813)](https://github.com/PennyLaneAI/pennylane/pull/1813)

  Consider the following noisy device:

  ```python
  noise_strength = 0.05
  dev = qml.device("default.mixed", wires=2)
  dev = qml.transforms.insert(qml.AmplitudeDamping, noise_strength)(dev)
  ```

  We can mitigate the effects of this noise for circuits run on this device by using the added
  transform:

  ```python
  from mitiq.zne.scaling import fold_global
  from mitiq.zne.inference import RichardsonFactory

  n_wires = 2
  n_layers = 2

  shapes = qml.SimplifiedTwoDesign.shape(n_wires, n_layers)
  np.random.seed(0)
  w1, w2 = [np.random.random(s) for s in shapes]

  @qml.transforms.mitigate_with_zne([1, 2, 3], fold_global, RichardsonFactory.extrapolate)
  @qml.beta.qnode(dev)
  def circuit(w1, w2):
      qml.SimplifiedTwoDesign(w1, w2, wires=range(2))
      return qml.expval(qml.PauliZ(0))
  ```

  Now, when we execute `circuit`, errors will be automatically mitigated:

  ```pycon
  >>> circuit(w1, w2)
  0.19113067083636542
  ```

<h4>Powerful new transforms</h4>

* The unitary matrix corresponding to a quantum circuit can now be generated using the new
  `get_unitary_matrix()` transform.
  [(#1609)](https://github.com/PennyLaneAI/pennylane/pull/1609)
  [(#1786)](https://github.com/PennyLaneAI/pennylane/pull/1786)

  This transform is fully differentiable across all supported PennyLane autodiff frameworks.

  ```python
  def circuit(theta):
      qml.RX(theta, wires=1)
      qml.PauliZ(wires=0)
      qml.CNOT(wires=[0, 1])
  ```

  ```pycon
  >>> theta = torch.tensor(0.3, requires_grad=True)
  >>> matrix = qml.transforms.get_unitary_matrix(circuit)(theta)
  >>> print(matrix)
  tensor([[ 0.9888+0.0000j,  0.0000+0.0000j,  0.0000-0.1494j,  0.0000+0.0000j],
        [ 0.0000+0.0000j,  0.0000+0.1494j,  0.0000+0.0000j, -0.9888+0.0000j],
        [ 0.0000-0.1494j,  0.0000+0.0000j,  0.9888+0.0000j,  0.0000+0.0000j],
        [ 0.0000+0.0000j, -0.9888+0.0000j,  0.0000+0.0000j,  0.0000+0.1494j]],
       grad_fn=<MmBackward>)
  >>> loss = torch.real(torch.trace(matrix))
  >>> loss.backward()
  >>> theta.grad
  tensor(-0.1494)
  ```

* Arbitrary two-qubit unitaries can now be decomposed into elementary gates. This
  functionality has been incorporated into the `qml.transforms.unitary_to_rot` transform, and is
  available separately as `qml.transforms.two_qubit_decomposition`.
  [(#1552)](https://github.com/PennyLaneAI/pennylane/pull/1552)

  As an example, consider the following randomly-generated matrix and circuit that uses it:

  ```python
  U = np.array([
      [-0.03053706-0.03662692j,  0.01313778+0.38162226j, 0.4101526 -0.81893687j, -0.03864617+0.10743148j],
      [-0.17171136-0.24851809j,  0.06046239+0.1929145j, -0.04813084-0.01748555j, -0.29544883-0.88202604j],
      [ 0.39634931-0.78959795j, -0.25521689-0.17045233j, -0.1391033 -0.09670952j, -0.25043606+0.18393466j],
      [ 0.29599198-0.19573188j,  0.55605806+0.64025769j, 0.06140516+0.35499559j,  0.02674726+0.1563311j ]
  ])

  dev = qml.device('default.qubit', wires=2)

  @qml.qnode(dev)
  @qml.transforms.unitary_to_rot
  def circuit(x, y):
      qml.QubitUnitary(U, wires=[0, 1])
      return qml.expval(qml.PauliZ(wires=0))
  ```

  If we run the circuit, we can see the new decomposition:

  ```pycon
  >>> circuit(0.3, 0.4)
  tensor(-0.81295986, requires_grad=True)
  >>> print(qml.draw(circuit)(0.3, 0.4))
  0: ──Rot(2.78, 0.242, -2.28)──╭X──RZ(0.176)───╭C─────────────╭X──Rot(-3.87, 0.321, -2.09)──┤ ⟨Z⟩
  1: ──Rot(4.64, 2.69, -1.56)───╰C──RY(-0.883)──╰X──RY(-1.47)──╰C──Rot(1.68, 0.337, 0.587)───┤
  ```

* A new transform, `@qml.batch_params`, has been added, that makes QNodes
  handle a batch dimension in trainable parameters.
  [(#1710)](https://github.com/PennyLaneAI/pennylane/pull/1710)
  [(#1761)](https://github.com/PennyLaneAI/pennylane/pull/1761)

  This transform will create multiple circuits, one per batch dimension.
  As a result, it is both simulator and hardware compatible.

  ```python
  @qml.batch_params
  @qml.beta.qnode(dev)
  def circuit(x, weights):
      qml.RX(x, wires=0)
      qml.RY(0.2, wires=1)
      qml.templates.StronglyEntanglingLayers(weights, wires=[0, 1, 2])
      return qml.expval(qml.Hadamard(0))
  ```

  The `qml.batch_params` decorator allows us to pass arguments `x` and `weights`
  that have a batch dimension. For example,

  ```pycon
  >>> batch_size = 3
  >>> x = np.linspace(0.1, 0.5, batch_size)
  >>> weights = np.random.random((batch_size, 10, 3, 3))
  ```

  If we evaluate the QNode with these inputs, we will get an output
  of shape ``(batch_size,)``:

  ```pycon
  >>> circuit(x, weights)
  tensor([0.08569816, 0.12619101, 0.21122004], requires_grad=True)
  ```

* The `insert` transform has now been added, providing a way to insert single-qubit operations into
  a quantum circuit. The transform can apply to quantum functions, tapes, and devices.
  [(#1795)](https://github.com/PennyLaneAI/pennylane/pull/1795)

  The following QNode can be transformed to add noise to the circuit:

  ```python
  dev = qml.device("default.mixed", wires=2)

  @qml.qnode(dev)
  @qml.transforms.insert(qml.AmplitudeDamping, 0.2, position="end")
  def f(w, x, y, z):
      qml.RX(w, wires=0)
      qml.RY(x, wires=1)
      qml.CNOT(wires=[0, 1])
      qml.RY(y, wires=0)
      qml.RX(z, wires=1)
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  ```

  Executions of this circuit will differ from the noise-free value:

  ```pycon
  >>> f(0.9, 0.4, 0.5, 0.6)
  tensor(0.754847, requires_grad=True)
  >>> print(qml.draw(f)(0.9, 0.4, 0.5, 0.6))
   0: ──RX(0.9)──╭C──RY(0.5)──AmplitudeDamping(0.2)──╭┤ ⟨Z ⊗ Z⟩
   1: ──RY(0.4)──╰X──RX(0.6)──AmplitudeDamping(0.2)──╰┤ ⟨Z ⊗ Z⟩
  ```

* Common tape expansion functions are now available in `qml.transforms`,
  alongside a new `create_expand_fn` function for easily creating expansion functions
  from stopping criteria.
  [(#1734)](https://github.com/PennyLaneAI/pennylane/pull/1734)
  [(#1760)](https://github.com/PennyLaneAI/pennylane/pull/1760)

  `create_expand_fn` takes the default depth to which the expansion function
  should expand a tape, a stopping criterion, an optional device, and a docstring to be set for the
  created function.
  The stopping criterion must take a queuable object and return a boolean.

  For example, to create an expansion function that decomposes all trainable,
  multi-parameter operations:

  ```python
  >>> stop_at = ~(qml.operation.has_multipar & qml.operation.is_trainable)
  >>> expand_fn = qml.transforms.create_expand_fn(depth=5, stop_at=stop_at)
  ```

  The created expansion function can be used within a custom transform.
  Devices can also be provided, producing expansion functions that decompose
  tapes to support the native gate set of the device.

<h4>Batch execution of circuits</h4>

* A new, experimental QNode has been added, that adds support for batch execution of circuits,
  custom quantum gradient support, and arbitrary order derivatives. This QNode is available via
  `qml.beta.QNode`, and `@qml.beta.qnode`.
  [(#1642)](https://github.com/PennyLaneAI/pennylane/pull/1642)
  [(#1646)](https://github.com/PennyLaneAI/pennylane/pull/1646)
  [(#1651)](https://github.com/PennyLaneAI/pennylane/pull/1651)
  [(#1804)](https://github.com/PennyLaneAI/pennylane/pull/1804)

  It differs from the standard QNode in several ways:

  - Custom gradient transforms can be specified as the differentiation method:

    ```python
    @qml.gradients.gradient_transform
    def my_gradient_transform(tape):
        ...
        return tapes, processing_fn

    @qml.beta.qnode(dev, diff_method=my_gradient_transform)
    def circuit():
    ```

  - Arbitrary :math:`n`-th order derivatives are supported on hardware using
    gradient transforms such as the parameter-shift rule. To specify that an :math:`n`-th
    order derivative of a QNode will be computed, the `max_diff` argument should be set.
    By default, this is set to 1 (first-order derivatives only).

  - Internally, if multiple circuits are generated for execution simultaneously, they
    will be packaged into a single job for execution on the device. This can lead to
    significant performance improvement when executing the QNode on remote
    quantum hardware.

  - When decomposing the circuit, the default decomposition strategy will prioritize
    decompositions that result in the smallest number of parametrized operations
    required to satisfy the differentiation method. Additional decompositions required
    to satisfy the native gate set of the quantum device will be performed later, by the
    device at execution time. While this may lead to a slight increase in classical processing,
    it significantly reduces the number of circuit evaluations needed to compute
    gradients of complex unitaries.

  In an upcoming release, this QNode will replace the existing one. If you come across any bugs
  while using this QNode, please let us know via a [bug
  report](https://github.com/PennyLaneAI/pennylane/issues/new?assignees=&labels=bug+%3Abug%3A&template=bug_report.yml&title=%5BBUG%5D)
  on our GitHub bug tracker.

  Currently, this beta QNode does not support the following features:

  - Non-mutability via the `mutable` keyword argument
  - The `reversible` QNode differentiation method
  - The ability to specify a `dtype` when using PyTorch and TensorFlow.

  It is also not tested with the `qml.qnn` module.

<h4>New operations and templates</h4>

* Added a new operation `OrbitalRotation`, which implements the spin-adapted spatial orbital
  rotation gate. [(#1665)](https://github.com/PennyLaneAI/pennylane/pull/1665)

  An example circuit that uses `OrbitalRotation` operation is:

  ```python
  dev = qml.device('default.qubit', wires=4)

  @qml.qnode(dev)
  def circuit(phi):
      qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
      qml.OrbitalRotation(phi, wires=[0, 1, 2, 3])
      return qml.state()
  ```

  If we run this circuit, we will get the following output

  ```pycon
  >>> circuit(0.1)
  array([ 0.        +0.j,  0.        +0.j,  0.        +0.j,
          0.00249792+0.j,  0.        +0.j,  0.        +0.j,
          -0.04991671+0.j,  0.        +0.j,  0.        +0.j,
          -0.04991671+0.j,  0.        +0.j,  0.        +0.j,
          0.99750208+0.j,  0.        +0.j,  0.        +0.j,
          0.        +0.j])
  ```

* Added a new template `GateFabric`, which implements a local, expressive, quantum-number-preserving
  ansatz proposed by Anselmetti *et al.* in [arXiv:2104.05692](https://arxiv.org/abs/2104.05695).
  [(#1687)](https://github.com/PennyLaneAI/pennylane/pull/1687)

  An example of a circuit using `GateFabric` template is:

  ```python
  coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
  H, qubits = qml.qchem.molecular_hamiltonian(["H", "H"], coordinates)
  ref_state = qml.qchem.hf_state(electrons=2, orbitals=qubits)

  dev = qml.device('default.qubit', wires=qubits)

  @qml.qnode(dev)
  def ansatz(weights):
      qml.templates.GateFabric(weights, wires=[0,1,2,3],
                                  init_state=ref_state, include_pi=True)
      return qml.expval(H)
  ```

  For more details, see the [GateFabric
  documentation](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.GateFabric.html).

* Added a new template `kUpCCGSD`, which implements a unitary coupled cluster ansatz with
  generalized singles and pair doubles excitation operators, proposed by Joonho Lee *et al.*
  in [arXiv:1810.02327](https://arxiv.org/abs/1810.02327).
  [(#1743)](https://github.com/PennyLaneAI/pennylane/pull/1743)

  An example of a circuit using `kUpCCGSD` template is:

  ```python
  coordinates = np.array([0.0, 0.0, -0.6614, 0.0, 0.0, 0.6614])
  H, qubits = qml.qchem.molecular_hamiltonian(["H", "H"], coordinates)
  ref_state = qml.qchem.hf_state(electrons=2, orbitals=qubits)

  dev = qml.device('default.qubit', wires=qubits)

  @qml.qnode(dev)
  def ansatz(weights):
      qml.templates.kUpCCGSD(weights, wires=[0,1,2,3], k=0, delta_sz=0,
                                  init_state=ref_state)
      return qml.expval(H)
  ```

<h4>Improved utilities for quantum compilation and characterization</h4>

* The new `qml.fourier.qnode_spectrum` function extends the former
  `qml.fourier.spectrum` function
  and takes classical processing of QNode arguments into account.
  The frequencies are computed per (requested) QNode argument instead
  of per gate `id`. The gate `id`s are ignored.
  [(#1681)](https://github.com/PennyLaneAI/pennylane/pull/1681)
  [(#1720)](https://github.com/PennyLaneAI/pennylane/pull/1720)

  Consider the following example, which uses non-trainable inputs `x`, `y` and `z`
  as well as trainable parameters `w` as arguments to the QNode.

  ```python
  import pennylane as qp
  import numpy as np

  n_qubits = 3
  dev = qml.device("default.qubit", wires=n_qubits)

  @qml.qnode(dev)
  def circuit(x, y, z, w):
      for i in range(n_qubits):
          qml.RX(0.5*x[i], wires=i)
          qml.Rot(w[0,i,0], w[0,i,1], w[0,i,2], wires=i)
          qml.RY(2.3*y[i], wires=i)
          qml.Rot(w[1,i,0], w[1,i,1], w[1,i,2], wires=i)
          qml.RX(z, wires=i)
      return qml.expval(qml.PauliZ(wires=0))

  x = np.array([1., 2., 3.])
  y = np.array([0.1, 0.3, 0.5])
  z = -1.8
  w = np.random.random((2, n_qubits, 3))
  ```

  This circuit looks as follows:

  ```pycon
  >>> print(qml.draw(circuit)(x, y, z, w))
  0: ──RX(0.5)──Rot(0.598, 0.949, 0.346)───RY(0.23)──Rot(0.693, 0.0738, 0.246)──RX(-1.8)──┤ ⟨Z⟩
  1: ──RX(1)────Rot(0.0711, 0.701, 0.445)──RY(0.69)──Rot(0.32, 0.0482, 0.437)───RX(-1.8)──┤
  2: ──RX(1.5)──Rot(0.401, 0.0795, 0.731)──RY(1.15)──Rot(0.756, 0.38, 0.38)─────RX(-1.8)──┤
  ```

  Applying the `qml.fourier.qnode_spectrum` function to the circuit for the non-trainable
  parameters, we obtain:

  ```pycon
  >>> spec = qml.fourier.qnode_spectrum(circuit, encoding_args={"x", "y", "z"})(x, y, z, w)
  >>> for inp, freqs in spec.items():
  ...     print(f"{inp}: {freqs}")
  "x": {(0,): [-0.5, 0.0, 0.5], (1,): [-0.5, 0.0, 0.5], (2,): [-0.5, 0.0, 0.5]}
  "y": {(0,): [-2.3, 0.0, 2.3], (1,): [-2.3, 0.0, 2.3], (2,): [-2.3, 0.0, 2.3]}
  "z": {(): [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]}
  ```

  We can see that all three parameters in the QNode arguments ``x`` and ``y``
  contribute the spectrum of a Pauli rotation ``[-1.0, 0.0, 1.0]``, rescaled with the
  prefactor of the respective parameter in the circuit.
  The three ``RX`` rotations using the parameter ``z`` accumulate, yielding a more
  complex frequency spectrum.

  For details on how to control for which parameters the spectrum is computed,
  a comparison to `qml.fourier.circuit_spectrum`, and other usage details, please see the
  [fourier.qnode_spectrum docstring](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.fourier.qnode_spectrum.html).

* Two new methods were added to the Device API, allowing PennyLane devices
  increased control over circuit decompositions.
  [(#1651)](https://github.com/PennyLaneAI/pennylane/pull/1651)

  - `Device.expand_fn(tape) -> tape`: expands a tape such that it is supported by the device. By
    default, performs the standard device-specific gate set decomposition done in the default
    QNode. Devices may overwrite this method in order to define their own decomposition logic.

    Note that the numerical result after applying this method should remain unchanged; PennyLane
    will assume that the expanded tape returns exactly the same value as the original tape when
    executed.

  - `Device.batch_transform(tape) -> (tapes, processing_fn)`: preprocesses the tape in the case
    where the device needs to generate multiple circuits to execute from the input circuit. The
    requirement of a post-processing function makes this distinct to the `expand_fn` method above.

    By default, this method applies the transform

    .. math:: \left\langle \sum_i c_i h_i\right\rangle → \sum_i c_i \left\langle h_i \right\rangle

    if `expval(H)` is present on devices that do not natively support Hamiltonians with
    non-commuting terms.

* A new class has been added to store operator attributes, such as `self_inverses`,
  and `composable_rotation`, as a list of operation names.
  [(#1763)](https://github.com/PennyLaneAI/pennylane/pull/1763)

  A number of such attributes, for the purpose of compilation transforms, can be found
  in `ops/qubit/attributes.py`, but the class can also be used to create your own. For
  example, we can create a new Attribute, `pauli_ops`, like so:

  ```pycon
  >>> from pennylane.ops.qubit.attributes import Attribute
  >>> pauli_ops = Attribute(["PauliX", "PauliY", "PauliZ"])
  ```

  We can check either a string or an Operation for inclusion in this set:

  ```pycon
  >>> qml.PauliX(0) in pauli_ops
  True
  >>> "Hadamard" in pauli_ops
  False
  ```

  We can also dynamically add operators to the sets at runtime. This is useful
  for adding custom operations to the attributes such as `composable_rotations`
  and ``self_inverses`` that are used in compilation transforms. For example,
  suppose you have created a new Operation, `MyGate`, which you know to be its
  own inverse. Adding it to the set, like so

  ```pycon
  >>> from pennylane.ops.qubit.attributes import self_inverses
  >>> self_inverses.add("MyGate")
  ```

  will enable the gate to be considered by the `cancel_inverses` compilation
  transform if two such gates are adjacent in a circuit.

<h3>Improvements</h3>

* The `qml.metric_tensor` transform has been improved with regards to
  both function and performance.
  [(#1638)](https://github.com/PennyLaneAI/pennylane/pull/1638)
  [(#1721)](https://github.com/PennyLaneAI/pennylane/pull/1721)

  - If the underlying device supports batch execution of circuits, the quantum circuits required to
    compute the metric tensor elements will be automatically submitted as a batched job. This can
    lead to significant performance improvements for devices with a non-trivial job submission
    overhead.

  - Previously, the transform would only return the metric tensor with respect to gate arguments,
    and ignore any classical processing inside the QNode, even very trivial classical processing
    such as parameter permutation. The metric tensor now takes into account classical processing,
    and returns the metric tensor with respect to QNode arguments, not simply gate arguments:

    ```pycon
    >>> @qml.qnode(dev)
    ... def circuit(x):
    ...     qml.Hadamard(wires=1)
    ...     qml.RX(x[0], wires=0)
    ...     qml.CNOT(wires=[0, 1])
    ...     qml.RY(x[1] ** 2, wires=1)
    ...     qml.RY(x[1], wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> x = np.array([0.1, 0.2], requires_grad=True)
    >>> qml.metric_tensor(circuit)(x)
    array([[0.25      , 0.        ],
           [0.        , 0.28750832]])
    ```

    To revert to the previous behaviour of returning the metric tensor with respect to gate
    arguments, `qml.metric_tensor(qnode, hybrid=False)` can be passed.

    ```pycon
    >>> qml.metric_tensor(circuit, hybrid=False)(x)
    array([[0.25      , 0.        , 0.        ],
           [0.        , 0.25      , 0.        ],
           [0.        , 0.        , 0.24750832]])
    ```

  - The metric tensor transform now works with a larger set of operations. In particular,
    all operations that have a single variational parameter and define a generator are now
    supported. In addition to a reduction in decomposition overhead, the change
    also results in fewer circuit evaluations.

* The expansion rule in the `qml.metric_tensor` transform has been changed.
  [(#1721)](https://github.com/PennyLaneAI/pennylane/pull/1721)

  If `hybrid=False`, the changed expansion rule might lead to a changed output.

* The `ApproxTimeEvolution` template can now be used with Hamiltonians that have
  trainable coefficients.
  [(#1789)](https://github.com/PennyLaneAI/pennylane/pull/1789)

  Resulting QNodes can be differentiated with respect to both the time parameter
  *and* the Hamiltonian coefficients.

  ```python
  dev = qml.device('default.qubit', wires=2)
  obs = [qml.PauliX(0) @ qml.PauliY(1), qml.PauliY(0) @ qml.PauliX(1)]

  @qml.qnode(dev)
  def circuit(coeffs, t):
      H = qml.Hamiltonian(coeffs, obs)
      qml.templates.ApproxTimeEvolution(H, t, 2)
      return qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> t = np.array(0.54, requires_grad=True)
  >>> coeffs = np.array([-0.6, 2.0], requires_grad=True)
  >>> qml.grad(circuit)(coeffs, t)
  (array([-1.07813375, -1.07813375]), array(-2.79516158))
  ```

  All differentiation methods, including backpropagation and the parameter-shift
  rule, are supported.

* Quantum function transforms and batch transforms can now be applied to devices.
  Once applied to a device, any quantum function executed on the
  modified device will be transformed prior to execution.
  [(#1809)](https://github.com/PennyLaneAI/pennylane/pull/1809)
  [(#1810)](https://github.com/PennyLaneAI/pennylane/pull/1810)

  ```python
  dev = qml.device("default.mixed", wires=1)
  dev = qml.transforms.merge_rotations()(dev)

  @qml.beta.qnode(dev)
  def f(w, x, y, z):
      qml.RX(w, wires=0)
      qml.RX(x, wires=0)
      qml.RX(y, wires=0)
      qml.RX(z, wires=0)
      return qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> print(f(0.9, 0.4, 0.5, 0.6))
   -0.7373937155412453
  >>> print(qml.draw(f, expansion_strategy="device")(0.9, 0.4, 0.5, 0.6))
   0: ──RX(2.4)──┤ ⟨Z⟩
  ```

* It is now possible to draw QNodes that have been transformed by a 'batch transform'; that is,
  a transform that maps a single QNode into multiple circuits under the hood. Examples of
  batch transforms include `@qml.metric_tensor` and `@qml.gradients`.
  [(#1762)](https://github.com/PennyLaneAI/pennylane/pull/1762)

  For example, consider the parameter-shift rule, which generates two circuits per parameter;
  one circuit that has the parameter shifted forward, and another that has the parameter shifted
  backwards:

  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.gradients.param_shift
  @qml.beta.qnode(dev)
  def circuit(x):
      qml.RX(x, wires=0)
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.PauliZ(wires=0))
  ```

  ```pycon
  >>> print(qml.draw(circuit)(0.6))
   0: ──RX(2.17)──╭C──┤ ⟨Z⟩
   1: ────────────╰X──┤

   0: ──RX(-0.971)──╭C──┤ ⟨Z⟩
   1: ──────────────╰X──┤
  ```

* Support for differentiable execution of batches of circuits has been
  extended to the JAX interface for scalar functions, via the beta
  `pennylane.interfaces.batch` module.
  [(#1634)](https://github.com/PennyLaneAI/pennylane/pull/1634)
  [(#1685)](https://github.com/PennyLaneAI/pennylane/pull/1685)

  For example using the `execute` function from the `pennylane.interfaces.batch` module:

  ```python
  from pennylane.interfaces.batch import execute

  def cost_fn(x):
      with qml.tape.JacobianTape() as tape1:
          qml.RX(x[0], wires=[0])
          qml.RY(x[1], wires=[1])
          qml.CNOT(wires=[0, 1])
          qml.var(qml.PauliZ(0) @ qml.PauliX(1))

      with qml.tape.JacobianTape() as tape2:
          qml.RX(x[0], wires=0)
          qml.RY(x[0], wires=1)
          qml.CNOT(wires=[0, 1])
          qml.probs(wires=1)

      result = execute(
        [tape1, tape2], dev,
        gradient_fn=qml.gradients.param_shift,
        interface="autograd"
      )
      return (result[0] + result[1][0, 0])[0]

  res = jax.grad(cost_fn)(params)
  ```

* All qubit operations have been re-written to use the `qml.math` framework
  for internal classical processing and the generation of their matrix representations.
  As a result these representations are now fully differentiable, and the
  framework-specific device classes no longer need to maintain framework-specific
  versions of these matrices.
  [(#1749)](https://github.com/PennyLaneAI/pennylane/pull/1749)
  [(#1802)](https://github.com/PennyLaneAI/pennylane/pull/1802)

* The use of `expval(H)`, where `H` is a cost Hamiltonian generated by the `qaoa` module,
  has been sped up. This was achieved by making PennyLane decompose a circuit with an `expval(H)`
  measurement into subcircuits if the `Hamiltonian.grouping_indices` attribute is set, and setting
  this attribute in the relevant `qaoa` module functions.
  [(#1718)](https://github.com/PennyLaneAI/pennylane/pull/1718)

* Operations can now have gradient recipes that depend on the state of the operation.
  [(#1674)](https://github.com/PennyLaneAI/pennylane/pull/1674)

  For example, this allows for gradient recipes that are parameter dependent:

  ```python
  class RX(qml.RX):

      @property
      def grad_recipe(self):
          # The gradient is given by [f(2x) - f(0)] / (2 sin(x)), by subsituting
          # shift = x into the two term parameter-shift rule.
          x = self.data[0]
          c = 0.5 / np.sin(x)
          return ([[c, 0.0, 2 * x], [-c, 0.0, 0.0]],)
  ```

* Shots can now be passed as a runtime argument to transforms that execute circuits in batches,
  similarly to QNodes. [(#1707)](https://github.com/PennyLaneAI/pennylane/pull/1707)

  An example of such a transform are the gradient transforms in the
  `qml.gradients` module. As a result, we can now call gradient transforms
  (such as `qml.gradients.param_shift`) and set the number of shots at runtime.

  ```pycon
  >>> dev = qml.device("default.qubit", wires=1, shots=1000)
  >>> @qml.beta.qnode(dev)
  ... def circuit(x):
  ...     qml.RX(x, wires=0)
  ...     return qml.expval(qml.PauliZ(0))
  >>> grad_fn = qml.gradients.param_shift(circuit)
  >>> param = np.array(0.564, requires_grad=True)
  >>> grad_fn(param, shots=[(1, 10)]).T
  array([[-1., -1., -1., -1., -1.,  0., -1.,  0., -1.,  0.]])
  >>> param2 = np.array(0.1233, requires_grad=True)
  >>> grad_fn(param2, shots=None)
  array([[-0.12298782]])
  ```

* Templates are now top level imported and can be used directly e.g. `qml.QFT(wires=0)`.
  [(#1779)](https://github.com/PennyLaneAI/pennylane/pull/1779)

* `qml.probs` now accepts an attribute `op` that allows to rotate the computational basis and get
  the probabilities in the rotated basis.
  [(#1692)](https://github.com/PennyLaneAI/pennylane/pull/1692)

* Refactored the `expand_fn` functionality in the Device class to avoid any
  edge cases leading to failures with plugins.
  [(#1838)](https://github.com/PennyLaneAI/pennylane/pull/1838)

* Updated the `qml.QNGOptimizer.step_and_cost` method to avoid the use of
  deprecated functionality.
  [(#1834)](https://github.com/PennyLaneAI/pennylane/pull/1834)

* Added a custom `torch.to_numpy` implementation to
  `pennylane/math/single_dispatch.py` to ensure compabilitity with
  PyTorch 1.10.
  [(#1824)](https://github.com/PennyLaneAI/pennylane/pull/1824)
  [(#1825)](https://github.com/PennyLaneAI/pennylane/pull/1825)

* The default for an `Operation`'s `control_wires` attribute is now an empty `Wires`
  object instead of the attribute raising a `NonImplementedError`.
  [(#1821)](https://github.com/PennyLaneAI/pennylane/pull/1821)

* `qml.circuit_drawer.MPLDrawer` will now automatically rotate and resize text to fit inside
  the rectangle created by the `box_gate` method.
  [(#1764)](https://github.com/PennyLaneAI/pennylane/pull/1764)

* Operators now have a `label` method to determine how they are drawn.  This will
  eventually override the `RepresentationResolver` class.
  [(#1678)](https://github.com/PennyLaneAI/pennylane/pull/1678)

* The operation `label` method now supports string variables.
  [(#1815)](https://github.com/PennyLaneAI/pennylane/pull/1815)

* A new utility class `qml.BooleanFn` is introduced. It wraps a function that takes a single
  argument and returns a Boolean.
  [(#1734)](https://github.com/PennyLaneAI/pennylane/pull/1734)

  After wrapping, `qml.BooleanFn` can be called like the wrapped function, and
  multiple instances can be manipulated and combined with the bitwise operators
  `&`, `|` and `~`.

* There is a new utility function `qml.math.is_independent` that checks whether
  a callable is independent of its arguments.
  [(#1700)](https://github.com/PennyLaneAI/pennylane/pull/1700)

  This function is experimental and might behave differently than expected.

  Note that the test relies on both numerical and analytical checks, except
  when using the PyTorch interface which only performs a numerical check.
  It is known that there are edge cases on which this test will yield wrong
  results, in particular non-smooth functions may be problematic.
  For details, please refer to the
  [is_indpendent docstring](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.math.is_independent.html).

* The `qml.beta.QNode` now supports the `qml.qnn` module.
  [(#1748)](https://github.com/PennyLaneAI/pennylane/pull/1748)

* `@qml.beta.QNode` now supports the `qml.specs` transform.
  [(#1739)](https://github.com/PennyLaneAI/pennylane/pull/1739)

* `qml.circuit_drawer.drawable_layers` and `qml.circuit_drawer.drawable_grid` process a list of
  operations to layer positions for drawing.
  [(#1639)](https://github.com/PennyLaneAI/pennylane/pull/1639)

* `qml.transforms.batch_transform` now accepts `expand_fn`s that take additional arguments and
  keyword arguments. In fact, `expand_fn` and `transform_fn` now **must** have the same signature.
  [(#1721)](https://github.com/PennyLaneAI/pennylane/pull/1721)

* The `qml.batch_transform` decorator is now ignored during Sphinx builds, allowing
  the correct signature to display in the built documentation.
  [(#1733)](https://github.com/PennyLaneAI/pennylane/pull/1733)

* The tests for qubit operations are split into multiple files.
  [(#1661)](https://github.com/PennyLaneAI/pennylane/pull/1661)

* The transform for the Jacobian of the classical preprocessing within a QNode,
  `qml.transforms.classical_jacobian`, now takes a keyword argument `argnum` to specify
  the QNode argument indices with respect to which the Jacobian is computed.
  [(#1645)](https://github.com/PennyLaneAI/pennylane/pull/1645)

  An example for the usage of ``argnum`` is

  ```python
  @qml.qnode(dev)
  def circuit(x, y, z):
      qml.RX(qml.math.sin(x), wires=0)
      qml.CNOT(wires=[0, 1])
      qml.RY(y ** 2, wires=1)
      qml.RZ(1 / z, wires=1)
      return qml.expval(qml.PauliZ(0))

  jac_fn = qml.transforms.classical_jacobian(circuit, argnum=[1, 2])
  ```

  The Jacobian can then be computed at specified parameters.

  ```pycon
  >>> x, y, z = np.array([0.1, -2.5, 0.71])
  >>> jac_fn(x, y, z)
  (array([-0., -5., -0.]), array([-0.        , -0.        , -1.98373339]))
  ```

  The returned arrays are the derivatives of the three parametrized gates in the circuit
  with respect to `y` and `z` respectively.

  There also are explicit tests for `classical_jacobian` now, which previously was tested
  implicitly via its use in the `metric_tensor` transform.

  For more usage details, please see the
  [classical Jacobian docstring](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.transforms.classical_jacobian.html).

* A new utility function `qml.math.is_abstract(tensor)` has been added. This function
  returns `True` if the tensor is *abstract*; that is, it has no value or shape.
  This can occur if within a function that has been just-in-time compiled.
  [(#1845)](https://github.com/PennyLaneAI/pennylane/pull/1845)

* ``qml.circuit_drawer.CircuitDrawer`` can accept a string for the ``charset`` keyword, instead of
  a ``CharSet`` object. [(#1640)](https://github.com/PennyLaneAI/pennylane/pull/1640)

* ``qml.math.sort`` will now return only the sorted torch tensor and not the corresponding indices,
  making sort consistent across interfaces.
  [(#1691)](https://github.com/PennyLaneAI/pennylane/pull/1691)

* Specific QNode execution options are now re-used by batch transforms
  to execute transformed QNodes.
  [(#1708)](https://github.com/PennyLaneAI/pennylane/pull/1708)

* To standardize across all optimizers, `qml.optimize.AdamOptimizer` now also uses `accumulation`
  (in form of `collections.namedtuple`) to keep track of running quantities. Before it used three
  variables `fm`, `sm` and `t`. [(#1757)](https://github.com/PennyLaneAI/pennylane/pull/1757)

<h3>Breaking changes</h3>

* The operator attributes `has_unitary_generator`, `is_composable_rotation`,
  `is_self_inverse`, `is_symmetric_over_all_wires`, and
  `is_symmetric_over_control_wires` have been removed as attributes from the
  base class. They have been replaced by the sets that store the names of
  operations with similar properties in `ops/qubit/attributes.py`.
  [(#1763)](https://github.com/PennyLaneAI/pennylane/pull/1763)

* The `qml.inv` function has been removed, `qml.adjoint` should be used
  instead.
  [(#1778)](https://github.com/PennyLaneAI/pennylane/pull/1778)

* The input signature of an `expand_fn` used in a `batch_transform`
  now **must** have the same signature as the provided `transform_fn`,
  and vice versa.
  [(#1721)](https://github.com/PennyLaneAI/pennylane/pull/1721)

* The `default.qubit.torch` device automatically determines if computations
  should be run on a CPU or a GPU and doesn't take a `torch_device` argument
  anymore.
  [(#1705)](https://github.com/PennyLaneAI/pennylane/pull/1705)

* The utility function `qml.math.requires_grad` now returns `True` when using Autograd
  if and only if the `requires_grad=True` attribute is set on the NumPy array. Previously,
  this function would return `True` for *all* NumPy arrays and Python floats, unless
  `requires_grad=False` was explicitly set.
  [(#1638)](https://github.com/PennyLaneAI/pennylane/pull/1638)

* The operation `qml.Interferometer` has been renamed `qml.InterferometerUnitary` in order to
  distinguish it from the template `qml.templates.Interferometer`.
  [(#1714)](https://github.com/PennyLaneAI/pennylane/pull/1714)

* The `qml.transforms.invisible` decorator has been replaced with `qml.tape.stop_recording`, which
  may act as a context manager as well as a decorator to ensure that contained logic is
  non-recordable or non-queueable within a QNode or quantum tape context.
  [(#1754)](https://github.com/PennyLaneAI/pennylane/pull/1754)

* Templates `SingleExcitationUnitary` and `DoubleExcitationUnitary` have been renamed
  to `FermionicSingleExcitation` and `FermionicDoubleExcitation`, respectively.
  [(#1822)](https://github.com/PennyLaneAI/pennylane/pull/1822)

<h3>Deprecations</h3>

* Allowing cost functions to be differentiated using `qml.grad` or
  `qml.jacobian` without explicitly marking parameters as trainable is being
  deprecated, and will be removed in an upcoming release.
  Please specify the `requires_grad` attribute for every argument, or specify
  `argnum` when using `qml.grad` or `qml.jacobian`.
  [(#1773)](https://github.com/PennyLaneAI/pennylane/pull/1773)

  The following raises a warning in v0.19.0 and will raise an error in
  an upcoming release:

  ```python
  import pennylane as qp

  dev = qml.device('default.qubit', wires=1)

  @qml.qnode(dev)
  def test(x):
      qml.RY(x, wires=[0])
      return qml.expval(qml.PauliZ(0))

  par = 0.3
  qml.grad(test)(par)
  ```

  Preferred approaches include specifying the `requires_grad` attribute:

  ```python
  import pennylane as qp
  from pennylane import numpy as np

  dev = qml.device('default.qubit', wires=1)

  @qml.qnode(dev)
  def test(x):
      qml.RY(x, wires=[0])
      return qml.expval(qml.PauliZ(0))

  par = np.array(0.3, requires_grad=True)
  qml.grad(test)(par)
  ```

  Or specifying the `argnum` argument when using `qml.grad` or `qml.jacobian`:

  ```python
  import pennylane as qp

  dev = qml.device('default.qubit', wires=1)

  @qml.qnode(dev)
  def test(x):
      qml.RY(x, wires=[0])
      return qml.expval(qml.PauliZ(0))

  par = 0.3
  qml.grad(test, argnum=0)(par)
  ```

  <img src="https://pennylane.readthedocs.io/en/latest/_static/requires_grad.png" style="width: 100%;"/>

* The `default.tensor` device from the beta folder is no longer maintained
  and has been deprecated. It will be removed in future releases.
  [(#1851)](https://github.com/PennyLaneAI/pennylane/pull/1851)

* The `qml.metric_tensor` and `qml.QNGOptimizer` keyword argument `diag_approx`
  is deprecated.
  Approximations can be controlled with the more fine-grained `approx` keyword
  argument, with `approx="block-diag"` (the default) reproducing the old
  behaviour.
  [(#1721)](https://github.com/PennyLaneAI/pennylane/pull/1721)
  [(#1834)](https://github.com/PennyLaneAI/pennylane/pull/1834)

* The `template` decorator is now deprecated with a warning message and will be removed
  in release `v0.20.0`. It has been removed from different PennyLane functions.
  [(#1794)](https://github.com/PennyLaneAI/pennylane/pull/1794)
  [(#1808)](https://github.com/PennyLaneAI/pennylane/pull/1808)

* The `qml.fourier.spectrum` function has been renamed to
  `qml.fourier.circuit_spectrum`, in order to clearly separate the new
  `qnode_spectrum` function from this one.  `qml.fourier.spectrum` is now an
  alias for `circuit_spectrum` but is flagged for deprecation and will be
  removed soon.
  [(#1681)](https://github.com/PennyLaneAI/pennylane/pull/1681)

* The `init` module, which contains functions to generate random parameter
  tensors for templates, is flagged for deprecation and will be removed in the
  next release cycle.  Instead, the templates' `shape` method can be used to
  get the desired shape of the tensor, which can then be generated manually.
  [(#1689)](https://github.com/PennyLaneAI/pennylane/pull/1689)

  To generate the parameter tensors, the `np.random.normal` and
  `np.random.uniform` functions can be used (just like in the `init` module).
  Considering the default arguments of these functions as of NumPy v1.21, some
  non-default options were used by the `init` module:

  * All functions generating normally distributed parameters used
    `np.random.normal` by passing `scale=0.1`;

  * Most functions generating uniformly distributed parameters (except for
    certain CVQNN initializers) used `np.random.uniform` by passing
    `high=2*math.pi`;

  * The `cvqnn_layers_r_uniform`, `cvqnn_layers_a_uniform`,
    `cvqnn_layers_kappa_uniform` functions used `np.random.uniform` by passing
    `high=0.1`.

* The `QNode.draw` method has been deprecated, and will be removed in an
  upcoming release.  Please use the `qml.draw` transform instead.
  [(#1746)](https://github.com/PennyLaneAI/pennylane/pull/1746)

* The `QNode.metric_tensor` method has been deprecated, and will be removed in
  an upcoming release.  Please use the `qml.metric_tensor` transform instead.
  [(#1638)](https://github.com/PennyLaneAI/pennylane/pull/1638)

* The `pad` parameter of the `qml.AmplitudeEmbedding` template has been removed.
  It has instead been renamed to the `pad_with` parameter.
  [(#1805)](https://github.com/PennyLaneAI/pennylane/pull/1805)

<h3>Bug fixes</h3>

* Fixes a bug where `qml.math.dot` failed to work with `@tf.function` autograph mode.
  [(#1842)](https://github.com/PennyLaneAI/pennylane/pull/1842)

* Fixes a bug where in rare instances the parameters of a tape are returned unsorted
  by `Tape.get_parameters`.
  [(#1836)](https://github.com/PennyLaneAI/pennylane/pull/1836)

* Fixes a bug with the arrow width in the `measure` of `qml.circuit_drawer.MPLDrawer`.
  [(#1823)](https://github.com/PennyLaneAI/pennylane/pull/1823)

* The helper functions `qml.math.block_diag` and `qml.math.scatter_element_add`
  now are entirely differentiable when using Autograd.  Previously only indexed
  entries of the block diagonal could be differentiated, while the derivative
  w.r.t to the second argument of `qml.math.scatter_element_add` dispatched to
  NumPy instead of Autograd.
  [(#1816)](https://github.com/PennyLaneAI/pennylane/pull/1816)
  [(#1818)](https://github.com/PennyLaneAI/pennylane/pull/1818)

* Fixes a bug such that the original shot vector information of a
  device is preserved, so that outside the context manager the device
  remains unchanged.
  [(#1792)](https://github.com/PennyLaneAI/pennylane/pull/1792)

* Modifies `qml.math.take` to be compatible with a breaking change
  released in JAX 0.2.24 and ensure that PennyLane supports this JAX
  version.
  [(#1769)](https://github.com/PennyLaneAI/pennylane/pull/1769)

* Fixes a bug where the GPU cannot be used with `qml.qnn.TorchLayer`.
  [(#1705)](https://github.com/PennyLaneAI/pennylane/pull/1705)

* Fix a bug where the devices cache the same result for different observables
  return types.
  [(#1719)](https://github.com/PennyLaneAI/pennylane/pull/1719)

* Fixed a bug of the default circuit drawer where having more measurements
  compared to the number of measurements on any wire raised a `KeyError`.
  [(#1702)](https://github.com/PennyLaneAI/pennylane/pull/1702)

* Fix a bug where it was not possible to use `jax.jit` on a `QNode` when using
  `QubitStateVector`.
  [(#1683)](https://github.com/PennyLaneAI/pennylane/pull/1683)

* The device suite tests can now execute successfully if no shots configuration
  variable is given.
  [(#1641)](https://github.com/PennyLaneAI/pennylane/pull/1641)

* Fixes a bug where the `qml.gradients.param_shift` transform would raise an
  error while attempting to compute the variance of a QNode with ragged output.
  [(#1646)](https://github.com/PennyLaneAI/pennylane/pull/1646)

* Fixes a bug in `default.mixed`, to ensure that returned probabilities are
  always non-negative.
  [(#1680)](https://github.com/PennyLaneAI/pennylane/pull/1680)

* Fixes a bug where gradient transforms would fail to apply to QNodes
  containing classical processing.
  [(#1699)](https://github.com/PennyLaneAI/pennylane/pull/1699)

* Fixes a bug where the the parameter-shift method was not correctly using the
  fallback gradient function when *all* circuit parameters required the fallback.
  [(#1782)](https://github.com/PennyLaneAI/pennylane/pull/1782)

<h3>Documentation</h3>

* Adds a link to https://pennylane.ai/qml/demonstrations in the navbar.
  [(#1624)](https://github.com/PennyLaneAI/pennylane/pull/1624)

* Corrects the docstring of `ExpvalCost` by adding `wires` to the signature of
  the `ansatz` argument.
  [(#1715)](https://github.com/PennyLaneAI/pennylane/pull/1715)

* Updated docstring examples using the `qchem.molecular_hamiltonian` function.
  [(#1724)](https://github.com/PennyLaneAI/pennylane/pull/1724)

* Updates the 'Gradients and training' quickstart guide to provide information
  on gradient transforms.
  [(#1751)](https://github.com/PennyLaneAI/pennylane/pull/1751)

* All instances of `qnode.draw()` have been updated to instead use the
  transform `qml.draw(qnode)`.
  [(#1750)](https://github.com/PennyLaneAI/pennylane/pull/1750)

* Add the `jax` interface in QNode Documentation.
  [(#1755)](https://github.com/PennyLaneAI/pennylane/pull/1755)

* Reorganized all the templates related to quantum chemistry under a common
  header `Quantum Chemistry templates`.
  [(#1822)](https://github.com/PennyLaneAI/pennylane/pull/1822)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Catalina Albornoz, Juan Miguel Arrazola, Utkarsh Azad, Akash Narayanan B, Sam
Banning, Thomas Bromley, Jack Ceroni, Alain Delgado, Olivia Di Matteo, Andrew
Gardhouse, Anthony Hayes, Theodor Isacsson, David Ittah, Josh Izaac, Soran
Jahangiri, Nathan Killoran, Christina Lee, Guillermo Alonso-Linaje, Romain
Moyard, Lee James O'Riordan, Carrie-Anne Rubidge, Maria Schuld, Rishabh Singh,
Jay Soni, Ingrid Strandberg, Antal Száva, Teresa Tamayo-Mendoza, Rodrigo
Vargas, Cody Wang, David Wierichs, Moritz Willmann.
