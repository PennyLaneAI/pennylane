:orphan:

# Release 0.21.0-dev (development release)

<h3>New features since last release</h3>

* A tensor network template has been added. Quantum circuits with the shape of a matrix product state tensor network can now be easily implemented. Motivation and theory can be found in [arXiv:1803.11537](https://arxiv.org/abs/1803.11537). [(#1871)](https://github.com/PennyLaneAI/pennylane/pull/1871)

  An example circuit that uses the `MPS` template is:
  ```python
  import pennylane as qml
  import numpy as np

  def block(weights, wires):
      qml.CNOT(wires=[wires[0],wires[1]])
      qml.RY(weights[0], wires=wires[0])
      qml.RY(weights[1], wires=wires[1])

  n_wires = 4
  n_block_wires = 2
  n_params_block = 2
  template_weights = [[0.1,-0.3],[0.4,0.2],[-0.15,0.5]]

  dev= qml.device('default.qubit',wires=range(n_wires))
  @qml.qnode(dev)
  def circuit(weights):
      qml.MPS(range(n_wires),n_block_wires,block, n_params_block, weights)
      return qml.expval(qml.PauliZ(wires=n_wires-1))
  ```

  The resulting circuit is:
  ```pycon
  >>> print(qml.draw(circuit,expansion_strategy='device')(template_weights))
  0: ──╭C──RY(0.1)───────────────────────────────┤
  1: ──╰X──RY(-0.3)──╭C──RY(0.4)─────────────────┤
  2: ────────────────╰X──RY(0.2)──╭C──RY(-0.15)──┤
  3: ─────────────────────────────╰X──RY(0.5)────┤ ⟨Z⟩
  ```
  
* Functions for tapering qubits based on molecular symmetries is added.
  [(#1966)](https://github.com/PennyLaneAI/pennylane/pull/1966)
* [(#1974)](https://github.com/PennyLaneAI/pennylane/pull/1974)

  With this functionality, a molecular Hamiltonian can be transformed to a new Hamiltonian that acts
  on a reduced number of qubits.

  ```python
  symbols = ["H", "H"]
  geometry = np.array([[0.0, 0.0, -0.69440367], [0.0, 0.0, 0.69440367]])
  mol = qml.hf.Molecule(symbols, geometry)
  H = qml.hf.generate_hamiltonian(mol)(geometry)
  generators, paulix_ops = qml.hf.generate_symmetries(H, len(H.wires))
  paulix_sector = [1, -1, -1]
  H_tapered = qml.hf.transform_hamiltonian(H, generators, paulix_ops, paulix_sector)
  ```

  ```pycon
  >>> print(H_tapered)
    ((-0.321034397355719+0j)) [I0]
  + ((0.1809270275619743+0j)) [X0]
  + ((0.7959678503870796+0j)) [Z0]
  ```

* Added the adjoint method for the metric tensor.
  [(#1992)](https://github.com/PennyLaneAI/pennylane/pull/1992)

  This method, detailed in [Jones 2020](https://arxiv.org/abs/2011.02991),
  computes the metric tensor using four copies of the state vector and
  a number of operations that scales quadratically in the number of trainable
  parameters (see below for details).
  
  Note that as it makes use of state cloning, it is inherently classical
  and can only be used with statevector simulators and `shots=None`.

  It is particular useful for larger circuits for which backpropagation requires
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

<h3>Improvements</h3>

* Insert transform now supports adding operation after or before certain specific gates.
  [(#1980)](https://github.com/PennyLaneAI/pennylane/pull/1980)

* Interferometer is now a class with `shape` method.
  [(#1946)](https://github.com/PennyLaneAI/pennylane/pull/1946)

* The `CircuitGraph`, used to represent circuits via directed acyclic graphs, now
  uses RetworkX for its internal representation. This results in significant speedup
  for algorithms that rely on a directed acyclic graph representation.
  [(#1791)](https://github.com/PennyLaneAI/pennylane/pull/1791)
  
* The QAOA module now accepts both NetworkX and RetworkX graphs as function inputs.
  [(#1791)](https://github.com/PennyLaneAI/pennylane/pull/1791)

<h3>Breaking changes</h3>

<h3>Bug fixes</h3>

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

<h3>Documentation</h3>

* Extended the interfaces description page to explicitly mention device
  compatibility.
  [(#2031)](https://github.com/PennyLaneAI/pennylane/pull/2031)

<h3>Operator class refactor</h3>

The Operator class has undergone a major refactor with the following changes:

* The static `compute_decomposition` method defines the decomposition
  of an operator into a product of simpler operators, and the instance method
  `decomposition()` computes this for a given instance. When a custom 
  decomposition does not exist, the code now raises a custom `NoDecompositionError`
  instead of `NotImplementedError`.
  [(#2024)](https://github.com/PennyLaneAI/pennylane/pull/2024)

* The `diagonalizing_gates()` representation has been moved to the highest-level
  `Operator` class and is therefore available to all subclasses. A condition
  `qml.operation.defines_diagonalizing_gates` has been added, which can be used
  in tape contexts without queueing.
  [(#1985)](https://github.com/PennyLaneAI/pennylane/pull/1985)

* A static `compute_diagonalizing_gates` method has been added, which is called 
  by default in `diagonalizing_gates()`.
  [(#1993)](https://github.com/PennyLaneAI/pennylane/pull/1993)

* A `hyperparameters` attribute was added to the operator class.
  [(#2017)](https://github.com/PennyLaneAI/pennylane/pull/2017)
  
* The representation of an operator as a matrix has been overhauled. 
  
  The `matrix()` method now accepts a 
  `wire_order` argument and calculates the correct numerical representation 
  with respect to that ordering. 
    
  ```pycon
  >>> op = qml.RX(0.5, wires="b")
  >>> op.matrix()
  [[0.96891242+0.j         0.        -0.24740396j]
   [0.        -0.24740396j 0.96891242+0.j        ]]
  >>> op.matrix(wire_order=["a", "b"])
  [[0.9689+0.j  0.-0.2474j 0.+0.j         0.+0.j]
   [0.-0.2474j  0.9689+0.j 0.+0.j         0.+0.j]
   [0.+0.j          0.+0.j 0.9689+0.j 0.-0.2474j]
   [0.+0.j          0.+0.j 0.-0.2474j 0.9689+0.j]]
  ```
    
  The "canonical matrix", which is independent of wires,
  is now defined in the static method `compute_matrix()` instead of `_matrix`.
  By default, this method is assumed to take all parameters and non-trainable 
  hyperparameters that define the operation. 
    
  ```pycon
  >>> qml.RX.compute_matrix(0.5)
  [[0.96891242+0.j         0.        -0.24740396j]
   [0.        -0.24740396j 0.96891242+0.j        ]]
  ```
       
  If no canonical matrix is specified for a gate, `compute_matrix()` 
  raises a `NotImplementedError`.
  
  The new `matrix()` method is now used in the 
  `pennylane.transforms.get_qubit_unitary()` transform.
  [(#1996)](https://github.com/PennyLaneAI/pennylane/pull/1996)

* The `string_for_inverse` attribute is removed.
  [(#2021)](https://github.com/PennyLaneAI/pennylane/pull/2021)

* A `terms()` method and a `compute_terms()` static method were added to `Operator`. 
  Currently, only the `Hamiltonian` class overwrites `compute_terms` to store 
  coefficients and operators. The `Hamiltonian.terms` property hence becomes 
  a proper method called by `Hamiltonian.terms()`.

* The generator property has been updated to an instance method,
  `Operation.generator()`. It now returns an instantiated operation,
  representing the generator of the instantiated operator.
  [(#2030)](https://github.com/PennyLaneAI/pennylane/pull/2030)

  Various operators have been updated to specify the generator as either
  an `Observable`, `Tensor`, `Hamiltonian`, `SparseHamiltonian`, or `Hermitian`
  operator.

  In addition, a temporary utility function get_generator has been added
  to the utils module, to automate:

  - Extracting the matrix representation
  - Extracting the 'coefficient' if possible (only occurs if the generator is a single Pauli word)
  - Converting a Hamiltonian to a sparse matrix if there are more than 1 Pauli word present.
  - Negating the coefficient/taking the adjoint of the matrix if the operation was inverted

  This utility logic is currently needed because:

  - Extracting the matrix representation is not supported natively on
    Hamiltonians and SparseHamiltonians.
  - By default, calling `op.generator()` does not take into account `op.inverse()`.
  - If the generator is a single Pauli word, it is convenient to have access to
    both the coefficient and the observable separately.

* A `sparse_matrix` method and a `compute_sparse_matrix` static method were added 
    to the `Operator` class. The sparse representation of `SparseHamiltonian`
    is moved to this method, so that its `matrix` method now returns a dense matrix.
    [(#2050)](https://github.com/PennyLaneAI/pennylane/pull/2050)

* The argument `wires` in `heisenberg_obs`, `heisenberg_expand` and `heisenberg_tr`
  was renamed to `wire_order` to be consistent with other matrix representations.
  [(#2051)](https://github.com/PennyLaneAI/pennylane/pull/2051)

* The property `kraus_matrices` has been changed to a method, and `_kraus_matrices` renamed to 
  `compute_kraus_matrices`, which is now a static method.
  [(#2055)](https://github.com/PennyLaneAI/pennylane/pull/2055)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Ali Asadi, Esther Cruz, Olivia Di Matteo, Diego Guala, Josh Izaac, Ankit Khandelwal, 
Christina Lee, Maria Schuld, Antal Száva, David Wierichs, Shaoming Zhang
