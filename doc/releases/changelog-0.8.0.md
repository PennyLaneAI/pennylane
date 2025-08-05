
# Release 0.8.0

<h3>New features since last release</h3>

* Added a quantum chemistry package, `pennylane.qchem`, which supports
  integration with OpenFermion, Psi4, PySCF, and OpenBabel.
  [(#453)](https://github.com/XanaduAI/pennylane/pull/453)

  Features include:

  - Generate the qubit Hamiltonians directly starting with the atomic structure of the molecule.
  - Calculate the mean-field (Hartree-Fock) electronic structure of molecules.
  - Allow to define an active space based on the number of active electrons and active orbitals.
  - Perform the fermionic-to-qubit transformation of the electronic Hamiltonian by
    using different functions implemented in OpenFermion.
  - Convert OpenFermion's QubitOperator to a Pennylane `Hamiltonian` class.
  - Perform a Variational Quantum Eigensolver (VQE) computation with this Hamiltonian in PennyLane.

  Check out the [quantum chemistry quickstart](https://pennylane.readthedocs.io/en/latest/introduction/chemistry.html), as well the quantum chemistry and VQE tutorials.

* PennyLane now has some functions and classes for creating and solving VQE
  problems. [(#467)](https://github.com/XanaduAI/pennylane/pull/467)

  - `qml.Hamiltonian`: a lightweight class for representing qubit Hamiltonians
  - `qml.VQECost`: a class for quickly constructing a differentiable cost function
    given a circuit ansatz, Hamiltonian, and one or more devices

    ```python
    >>> H = qml.vqe.Hamiltonian(coeffs, obs)
    >>> cost = qml.VQECost(ansatz, hamiltonian, dev, interface="torch")
    >>> params = torch.rand([4, 3])
    >>> cost(params)
    tensor(0.0245, dtype=torch.float64)
    ```

* Added a circuit drawing feature that provides a text-based representation
  of a QNode instance. It can be invoked via `qnode.draw()`. The user can specify
  to display variable names instead of variable values and choose either an ASCII
  or Unicode charset.
  [(#446)](https://github.com/XanaduAI/pennylane/pull/446)

  Consider the following circuit as an example:
  ```python3
  @qml.qnode(dev)
  def qfunc(a, w):
      qml.Hadamard(0)
      qml.CRX(a, wires=[0, 1])
      qml.Rot(w[0], w[1], w[2], wires=[1])
      qml.CRX(-a, wires=[0, 1])

      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  ```

  We can draw the circuit after it has been executed:

  ```python
  >>> result = qfunc(2.3, [1.2, 3.2, 0.7])
  >>> print(qfunc.draw())
   0: ──H──╭C────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩
   1: ─────╰RX(2.3)──Rot(1.2, 3.2, 0.7)──╰RX(-2.3)──╰┤ ⟨Z ⊗ Z⟩
  >>> print(qfunc.draw(charset="ascii"))
   0: --H--+C----------------------------+C---------+| <Z @ Z>
   1: -----+RX(2.3)--Rot(1.2, 3.2, 0.7)--+RX(-2.3)--+| <Z @ Z>
  >>> print(qfunc.draw(show_variable_names=True))
   0: ──H──╭C─────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩
   1: ─────╰RX(a)──Rot(w[0], w[1], w[2])──╰RX(-1*a)──╰┤ ⟨Z ⊗ Z⟩
  ```

* Added `QAOAEmbedding` and its parameter initialization
  as a new trainable template.
  [(#442)](https://github.com/XanaduAI/pennylane/pull/442)

  <img src="https://pennylane.readthedocs.io/en/latest/_images/qaoa_layers.png"
  width=70%></img>

* Added the `qml.probs()` measurement function, allowing QNodes
  to differentiate variational circuit probabilities
  on simulators and hardware.
  [(#432)](https://github.com/XanaduAI/pennylane/pull/432)

  ```python
  @qml.qnode(dev)
  def circuit(x):
      qml.Hadamard(wires=0)
      qml.RY(x, wires=0)
      qml.RX(x, wires=1)
      qml.CNOT(wires=[0, 1])
      return qml.probs(wires=[0])
  ```
  Executing this circuit gives the marginal probability of wire 1:
  ```python
  >>> circuit(0.2)
  [0.40066533 0.59933467]
  ```
  QNodes that return probabilities fully support autodifferentiation.

* Added the convenience load functions `qml.from_pyquil`, `qml.from_quil` and
  `qml.from_quil_file` that convert pyQuil objects and Quil code to PennyLane
  templates. This feature requires version 0.8 or above of the PennyLane-Forest
  plugin.
  [(#459)](https://github.com/XanaduAI/pennylane/pull/459)

* Added a `qml.inv` method that inverts templates and sequences of Operations.
  Added a `@qml.template` decorator that makes templates return the queued Operations.
  [(#462)](https://github.com/XanaduAI/pennylane/pull/462)

  For example, using this function to invert a template inside a QNode:

  ```python3
      @qml.template
      def ansatz(weights, wires):
          for idx, wire in enumerate(wires):
              qml.RX(weights[idx], wires=[wire])

          for idx in range(len(wires) - 1):
              qml.CNOT(wires=[wires[idx], wires[idx + 1]])

      dev = qml.device('default.qubit', wires=2)

      @qml.qnode(dev)
      def circuit(weights):
          qml.inv(ansatz(weights, wires=[0, 1]))
          return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    ```

* Added the `QNodeCollection` container class, that allows independent
  QNodes to be stored and evaluated simultaneously. Experimental support
  for asynchronous evaluation of contained QNodes is provided with the
  `parallel=True` keyword argument.
  [(#466)](https://github.com/XanaduAI/pennylane/pull/466)

* Added a high level `qml.map` function, that maps a quantum
  circuit template over a list of observables or devices, returning
  a `QNodeCollection`.
  [(#466)](https://github.com/XanaduAI/pennylane/pull/466)

  For example:

  ```python3
  >>> def my_template(params, wires, **kwargs):
  >>>    qml.RX(params[0], wires=wires[0])
  >>>    qml.RX(params[1], wires=wires[1])
  >>>    qml.CNOT(wires=wires)

  >>> obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliX(1)]
  >>> dev = qml.device("default.qubit", wires=2)
  >>> qnodes = qml.map(my_template, obs_list, dev, measure="expval")
  >>> qnodes([0.54, 0.12])
  array([-0.06154835  0.99280864])
  ```

* Added high level `qml.sum`, `qml.dot`, `qml.apply` functions
  that act on QNode collections.
  [(#466)](https://github.com/XanaduAI/pennylane/pull/466)

  `qml.apply` allows vectorized functions to act over the entire QNode
  collection:
  ```python
  >>> qnodes = qml.map(my_template, obs_list, dev, measure="expval")
  >>> cost = qml.apply(np.sin, qnodes)
  >>> cost([0.54, 0.12])
  array([-0.0615095  0.83756375])
  ```

  `qml.sum` and `qml.dot` take the sum of a QNode collection, and a
  dot product of tensors/arrays/QNode collections, respectively.

<h3>Breaking changes</h3>

* Deprecated the old-style `QNode` such that only the new-style `QNode` and its syntax can be used,
  moved all related files from the `pennylane/beta` folder to `pennylane`.
  [(#440)](https://github.com/XanaduAI/pennylane/pull/440)

<h3>Improvements</h3>

* Added the `Tensor.prune()` method and the `Tensor.non_identity_obs` property for extracting
  non-identity instances from the observables making up a `Tensor` instance.
  [(#498)](https://github.com/XanaduAI/pennylane/pull/498)

* Renamed the `expt.tensornet` and `expt.tensornet.tf` devices to `default.tensor` and
  `default.tensor.tf`.
  [(#495)](https://github.com/XanaduAI/pennylane/pull/495)

* Added a serialization method to the `CircuitGraph` class that is used to create a unique
  hash for each quantum circuit graph.
  [(#470)](https://github.com/XanaduAI/pennylane/pull/470)

* Added the `Observable.eigvals` method to return the eigenvalues of observables.
  [(#449)](https://github.com/XanaduAI/pennylane/pull/449)

* Added the `Observable.diagonalizing_gates` method to return the gates
  that diagonalize an observable in the computational basis.
  [(#454)](https://github.com/XanaduAI/pennylane/pull/454)

* Added the `Operator.matrix` method to return the matrix representation
  of an operator in the computational basis.
  [(#454)](https://github.com/XanaduAI/pennylane/pull/454)

* Added a `QubitDevice` class which implements common functionalities of plugin devices such that
  plugin devices can rely on these implementations. The new `QubitDevice` also includes
  a new `execute` method, which allows for more convenient plugin design. In addition, `QubitDevice`
  also unifies the way samples are generated on qubit-based devices.
  [(#452)](https://github.com/XanaduAI/pennylane/pull/452)
  [(#473)](https://github.com/XanaduAI/pennylane/pull/473)

* Improved documentation of `AmplitudeEmbedding` and `BasisEmbedding` templates.
  [(#441)](https://github.com/XanaduAI/pennylane/pull/441)
  [(#439)](https://github.com/XanaduAI/pennylane/pull/439)

* Codeblocks in the documentation now have a 'copy' button for easily
  copying examples.
  [(#437)](https://github.com/XanaduAI/pennylane/pull/437)

<h3>Documentation</h3>

* Update the developers plugin guide to use QubitDevice.
  [(#483)](https://github.com/XanaduAI/pennylane/pull/483)

<h3>Bug fixes</h3>

* Fixed a bug in `CVQNode._pd_analytic`, where non-descendant observables were not
  Heisenberg-transformed before evaluating the partial derivatives when using the
  order-2 parameter-shift method, resulting in an erroneous Jacobian for some circuits.
  [(#433)](https://github.com/XanaduAI/pennylane/pull/433)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Ville Bergholm, Alain Delgado Gran, Olivia Di Matteo,
Theodor Isacsson, Josh Izaac, Soran Jahangiri, Nathan Killoran, Johannes Jakob Meyer,
Zeyue Niu, Maria Schuld, Antal Száva.
