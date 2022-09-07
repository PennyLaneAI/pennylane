:orphan:

# Release 0.26.0-dev (development release)

<h3>New features since last release</h3>

<h4>Qutrits: quantum circuits for tertiary degrees of freedom ☘️</h4>

* An entirely new framework for quantum computing is now simulatable with the addition of qutrit functionalities.
  [(#2699)](https://github.com/PennyLaneAI/pennylane/pull/2699)
  [(#2781)](https://github.com/PennyLaneAI/pennylane/pull/2781)
  [(#2782)](https://github.com/PennyLaneAI/pennylane/pull/2782)
  [(#2783)](https://github.com/PennyLaneAI/pennylane/pull/2783)
  ([#2784](https://github.com/PennyLaneAI/pennylane/pull/2784))
  ([#2841](https://github.com/PennyLaneAI/pennylane/pull/2841))

  [Qutrits](https://en.wikipedia.org/wiki/Qutrit) are like qubits, but instead live in a *three*-dimensional Hilbert space; they are not binary degrees of freedom, they are *tertiary*. 
  The advent of qutrits allows for all sorts of interesting theoretical, practical, and algorithmic capabilities that have yet to be discovered.
  
  To facilitate qutrit circuits requires a new device: `"default.qutrit"`.
  The `"default.qutrit"` device is a Python-based simulator, akin to `"default.qubit"`, and is defined as per usual:

  ```pycon
  >>> dev = qml.device("default.qutrit", wires=1)
  >>> type(dev)
  <class 'pennylane.devices.default_qutrit.DefaultQutrit'>
  ```

  The following operations are supported on `"default.qutrit"` devices:  

  - The qutrit shift operator, `qml.TShift`, and the ternary clock operator, `qml.TClock`, as defined in this paper by [Yeh et al. (2022)](https://arxiv.org/abs/2204.00552),
  which are the qutrit analogs of the Pauli X and Pauli Z operations, respectively.

  - Custom unitary operations via `qml.QutritUnitary`.

  - `qml.state` and `qml.probs` measurements.

  - Measuring user-specified Hermitian matrix observables via `qml.THermitian`.

    ```python
    import pennylane as qml
    from pennylane import numpy as np

    dev = qml.device("default.qutrit", wires=2)

    U = np.array([
            [1, 1, 1], 
            [1, 1, 1], 
            [1, 1, 1]
        ]
    ) / np.sqrt(3) 

    obs = np.array([
            [1, 1, 0], 
            [1, -1, 0], 
            [0, 0, np.sqrt(2)]
        ]
    ) / np.sqrt(2)

    def qutrit_function(U):
        qml.TShift(0)
        qml.TClock(0)
        qml.QutritUnitary(U, wires=0)

    @qml.qnode(dev)
    def qutrit_state(U, obs)
        qutrit_function(U, obs)
        return qml.state()

    @qml.qnode(dev)
    def qutrit_expval(U, obs)
        qutrit_function(U, obs)
        return qml.expval(qml.THermitian(obs, wires=0))
    ```

    ```pycon
    >>> qutrit_state(U, obs)
    tensor([-0.28867513+0.5j, -0.28867513+0.5j, -0.28867513+0.5j], requires_grad=True) 
    >>> qutrit_expval(U, obs)
    tensor(0.80473785, requires_grad=True)
    ```
  
    We will continue to add more and more support for qutrits in future releases!

<h4>Classical shadows 👤</h4>

* Measurements required for the classical-shadows protocol are now available. 
  [(#2820)](https://github.com/PennyLaneAI/pennylane/pull/2820)
  [(#2821)](https://github.com/PennyLaneAI/pennylane/pull/2821)
  [(#2871)](https://github.com/PennyLaneAI/pennylane/pull/2871)

  The classical-shadow measurement protocol is described in detail in the
  [classical shadows paper](https://arxiv.org/abs/2002.08953).
  As part of the support for classical shadows in this release, two new finite-shot and fully-differentiable measurements are available: 

  - QNodes returning `qml.classical_shadow` will return two entities: 

    + `bits`: `0` or `1` if the 1 or -1 eigenvalue is sampled, respectively
    + `recipes`: the randomized Pauli measurements that are performed for each qubit, identified as a unique integer:

      = 0 for Pauli X
      = 1 for Pauli Y
      = 2 for Pauli Z

    ```python
    dev = qml.device("default.qubit", wires=2, shots=3)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.classical_shadow(wires=[0, 1])
    ```

    ```pycon
    >>> bits, recipes = circuit()
    >>> bits
    tensor([[0, 0],
            [1, 0],
            [0, 1]], dtype=uint8, requires_grad=True)
    >>> recipes
    tensor([[2, 2],
            [0, 2],
            [0, 2]], dtype=uint8, requires_grad=True)
    ```

  - QNodes returning `qml.shadow_expval` yield the expectation value estimation using classical shadows:

    ```python
    dev = qml.device("default.qubit", wires=range(2), shots=10000)

    @qml.qnode(dev)
    def circuit(x, H):
        qml.Hadamard(0)
        qml.CNOT((0,1))
        qml.RX(x, wires=0)
        return qml.shadow_expval(H)

    x = np.array(0.5, requires_grad=True) 
    H = qml.Hamiltonian(
            [1., 1.], 
            [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)]
        )  
    ```

    ```pycon
    >>> circuit(x, H), 
    tensor(1.8486, requires_grad=True) 
    >>> qml.grad(circuit)(x, H))
    -0.4797000000000001
    ```

<h4>Simplifying just got... simpler 🫰</h4>

* The `qml.simplify` module has several intuitive additions with this release.
  [(#2978)](https://github.com/PennyLaneAI/pennylane/pull/2978)
  [(#2982)](https://github.com/PennyLaneAI/pennylane/pull/2982)
  [(#2922)](https://github.com/PennyLaneAI/pennylane/pull/2922)
  [(#3012)](https://github.com/PennyLaneAI/pennylane/pull/3012)

  Enjoy these easy-to-use and new functionalities built into `qml.simplify`:

  - Parametrized operations:

    ```pycon
    >>> op1 = qml.RX(30.0, wires=0)
    >>> qml.simplify(op1)
    RX(4.867258771281655, wires=[0])
    >>> op2 = qml.RX(4 * np.pi, wires=0)
    >>> qml.simplify(op2)
    Identity(wires=[0])
    ```

  - The adjoint and power of specific operators:

    ```pycon
    >>> adj_op = qml.adjoint(qml.RX(1, 0))
    >>> qml.simplify(adj_op)
    RX(-1, wires=[0])
    ```

  - Grouping of like terms in a sum:

    ```pycon
    >>> qml.simplify(qml.op_sum(qml.PauliX(0), qml.PauliY(1), qml.PauliX(0), qml.PauliY(1)))
    2*(PauliX(wires=[0])) + 2*(PauliY(wires=[1]))
    ```

  - Resolving products of Pauli operators:
  
    ```pycon
    >>> qml.simplify(qml.prod(qml.PauliX(0), qml.PauliY(1), qml.PauliX(0), qml.PauliY(1)))
    Identity(wires=[0]) @ Identity(wires=[1])
    ```
  
  - Combining rotation angles of identical rotation gates:

    ```pycon
    >>> qml.simplify(qml.prod(qml.RZ(1, 0), qml.RZ(1, 0)))
    RZ(2, wires=[0])
    ```

  All of these simplification features can be applied directly to quantum functions, QNodes, and tapes via decorating with `@qml.simplify`, as well:

  ```python
  dev = qml.device("default.qubit", wires=2)
  @qml.simplify
  @qml.qnode(dev)
  def circuit():
      qml.adjoint(qml.prod(qml.RX(1, 0) ** 1, qml.RY(1, 0), qml.RZ(1, 0)))
      return qml.probs(wires=0)
  ```

  ```pycon
  >>> circuit()
  >>> list(circuit.tape)
  [RZ(-1, wires=[0]) @ RY(-1, wires=[0]) @ RX(-1, wires=[0]), probs(wires=[0])]
  ```

<h4>Operator and parameter broadcasting supplements 📈</h4>

* Represent the exponentiation of operators via `qml.exp`.
  [(#2799)](https://github.com/PennyLaneAI/pennylane/pull/2799)

  v0.25 of PennyLane saw the addition of a whole host of additions that make operator arithmetic more intuitive. With this release, we've included the ability to exponentiate operators.
  The `qml.exp` function can be used to create observables or generic rotation gates:

  ```pycon
  >>> obs = qml.exp(qml.PauliX(0), 3)
  >>> qml.is_hermitian(obs)
  True
  >>> x = 1.234
  >>> t = qml.PauliX(0) @ qml.PauliX(1) + qml.PauliY(0) @ qml.PauliY(1)
  >>> isingxy = qml.exp(t, 0.25j * x)
  >>> qml.is_unitary(isingxy)
  True
  ```

* An operator called `qml.PSWAP` is now available.
  [(#2667)](https://github.com/PennyLaneAI/pennylane/pull/2667)

  The `qml.PSWAP` gate -- or phase-SWAP gate -- was previously available within the PennyLane-Braket plugin only. Enjoy it natively in PennyLane with v0.26.

* Added the `qml.is_hermitian` and `qml.is_unitary` function checks.
  [(#2960)](https://github.com/PennyLaneAI/pennylane/pull/2960)

  ```pycon
  >>> op = qml.PauliX(wires=0)
  >>> qml.is_hermitian(op)
  True
  >>> op2 = qml.RX(0.54, wires=0)
  >>> qml.is_hermitian(op2)
  False
  ```
* Embedding templates now support parameter broadcasting.
  [(#2810)](https://github.com/PennyLaneAI/pennylane/pull/2810)

  Embedding templates like `AmplitudeEmbedding` or `IQPEmbedding` now support
  parameter broadcasting with a leading broadcasting dimension in their variational
  parameters. `AmplitudeEmbedding`, for example, would usually use a one-dimensional input
  vector of features. With broadcasting, we can now compute

  ```pycon
  >>> features = np.array([
  ...     [0.5, 0.5, 0., 0., 0.5, 0., 0.5, 0.],
  ...     [1., 0., 0., 0., 0., 0., 0., 0.],
  ...     [0.5, 0.5, 0., 0., 0., 0., 0.5, 0.5],
  ... ])
  >>> op = qml.AmplitudeEmbedding(features, wires=[1, 5, 2])
  >>> op.batch_size
  3
  ```

  An exception is `BasisEmbedding`, which is not broadcastable.

<h3>Improvements</h3>

* `qml.math.expand_matrix()` method now allows the sparse matrix representation of an operator to be extended to
  a larger hilbert space.
  [(#2998)](https://github.com/PennyLaneAI/pennylane/pull/2998)

  ```pycon
  >>> from scipy import sparse
  >>> mat = sparse.csr_matrix([[0, 1], [1, 0]])
  >>> qml.math.expand_matrix(mat, wires=[1], wire_order=[0,1]).toarray()
  array([[0., 1., 0., 0.],
         [1., 0., 0., 0.],
         [0., 0., 0., 1.],
         [0., 0., 1., 0.]])
  ```

* Lists of operators are now internally sorted by their respective wires while also taking into account their commutativity property.
  [(#2995)](https://github.com/PennyLaneAI/pennylane/pull/2995)

* Some methods of the `QuantumTape` class have been simplified and reordered to
  improve both readability and performance. The `Wires.all_wires` method has been rewritten
  to improve performance.
  [(#2963)](https://github.com/PennyLaneAI/pennylane/pull/2963)

* The `qml.qchem.molecular_hamiltonian` function is modified to support observable grouping.
  [(#2997)](https://github.com/PennyLaneAI/pennylane/pull/2997)

* `qml.ops.op_math.Controlled` now has basic decomposition functionality.
  [(#2938)](https://github.com/PennyLaneAI/pennylane/pull/2938)

* Automatic circuit cutting is improved by making better partition imbalance derivations.
  Now it is more likely to generate optimal cuts for larger circuits.
  [(#2517)](https://github.com/PennyLaneAI/pennylane/pull/2517)


* Added `sparse_matrix()` support for single qubit observables.
  [(#2964)](https://github.com/PennyLaneAI/pennylane/pull/2964)

* Internal use of in-place inversion is eliminated in preparation for its deprecation.
  [(#2965)](https://github.com/PennyLaneAI/pennylane/pull/2965)

* `qml.is_commuting` is moved to `pennylane/ops/functions` from `pennylane/transforms/commutation_dag.py`.
  [(#2991)](https://github.com/PennyLaneAI/pennylane/pull/2991)

* `Controlled` operators now work with `qml.is_commuting`.
  [(#2994)](https://github.com/PennyLaneAI/pennylane/pull/2994)

* `Prod` and `Sum` class now support the `sparse_matrix()` method. 
  [(#3006)](https://github.com/PennyLaneAI/pennylane/pull/3006)
  
  ```pycon
  >>> xy = qml.prod(qml.PauliX(1), qml.PauliY(1))
  >>> op = qml.op_sum(xy, qml.Identity(0))
  >>>
  >>> sparse_mat = op.sparse_matrix(wire_order=[0,1])
  >>> type(sparse_mat)
  <class 'scipy.sparse.csr.csr_matrix'>
  >>> print(sparse_mat.toarray())
  [[1.+1.j 0.+0.j 0.+0.j 0.+0.j]
  [0.+0.j 1.-1.j 0.+0.j 0.+0.j]
  [0.+0.j 0.+0.j 1.+1.j 0.+0.j]
  [0.+0.j 0.+0.j 0.+0.j 1.-1.j]]
  ```

* `qml.Barrier` with `only_visual=True` now simplifies, via `op.simplify()` to the identity
  or a product of identities.
  [(#3016)](https://github.com/PennyLaneAI/pennylane/pull/3016)

* `__repr__` and `label` methods are more correct and meaningful for Operators with an arithmetic
  depth greater than 0. The `__repr__` for `Controlled` show `control_wires` instead of `wires`.
  [(#3013)](https://github.com/PennyLaneAI/pennylane/pull/3013)

<h3>Breaking changes</h3>

* Measuring an operator that might not be hermitian as an observable now raises a warning instead of an
  error. To definitively determine whether or not an operator is hermitian, use `qml.is_hermitian`.
  [(#2960)](https://github.com/PennyLaneAI/pennylane/pull/2960)

* The default `execute` method for the `QubitDevice` base class now calls `self.statistics`
  with an additional keyword argument `circuit`, which represents the quantum tape
  being executed.

  Any device that overrides `statistics` should edit the signature of the method to include
  the new `circuit` keyword argument.
  [(#2820)](https://github.com/PennyLaneAI/pennylane/pull/2820)

* The `expand_matrix()` has been moved from `~/operation.py` to
  `~/math/matrix_manipulation.py`
  [(#3008)](https://github.com/PennyLaneAI/pennylane/pull/3008)

<h3>Deprecations</h3>

* The `supports_reversible_diff` device capability is unused and has been removed.
  [(#2993)](https://github.com/PennyLaneAI/pennylane/pull/2993)

<h3>Documentation</h3>

* Fix fourier docs to use `circuit_spectrum`.
  [(#3018)](https://github.com/PennyLaneAI/pennylane/pull/3018)
  
* Corrects the docstrings for diagonalizing gates for all relevant operations. The docstrings used 
  to say that the diagonalizing gates implemented $U$, the unitary such that $O = U \Sigma U^{\dagger}$, where $O$ is 
  the original observable and $\Sigma$ a diagonal matrix. However, the diagonalizing gates actually implement 
  $U^{\dagger}$, since $\langle \psi | O | \psi \rangle = \langle \psi | U \Sigma U^{\dagger} | \psi \rangle$, 
  making $U^{\dagger} | \psi \rangle$ the actual state being measured in the $Z$-basis.
  [(#2981)](https://github.com/PennyLaneAI/pennylane/pull/2981)

<h3>Bug fixes</h3>

* Jax gradients now work with a QNode when the quantum function was transformed by `qml.simplify`.
  [(#3017)](https://github.com/PennyLaneAI/pennylane/pull/3017)

* Operators that have `num_wires = AnyWires` or `num_wires = AnyWires` raise an error, with
  certain exceptions, when instantiated with `wires=[]`.
  [(#2979)](https://github.com/PennyLaneAI/pennylane/pull/2979)

* Fixes a bug where printing `qml.Hamiltonian` with complex coefficients raises `TypeError` in some cases.
  [(#3005)](https://github.com/PennyLaneAI/pennylane/pull/3004)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola,
Utkarsh Azad,
Tom Bromley,
Olivia Di Matteo,
Isaac De Vlugt,
Josh Izaac,
Soran Jahangiri,
Edward Jiang,
Ankit Khandelwal,
Korbinian Kottmann,
Meenu Kumari,
Christina Lee,
Albert Mitjans Coma,
Romain Moyard,
Rashid N H M,
Zeyue Niu,
Mudit Pandey,
Matthew Silverman,
Jay Soni,
Antal Száva,
Cody Wang,
David Wierichs.
