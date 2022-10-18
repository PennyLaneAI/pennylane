:orphan:

# Release 0.26.0 (current release)

<h3>New features since last release</h3>

<h4>Classical shadows 👤</h4>

* PennyLane now provides built-in support for implementing the classical-shadows measurement protocol. 
  [(#2820)](https://github.com/PennyLaneAI/pennylane/pull/2820)
  [(#2821)](https://github.com/PennyLaneAI/pennylane/pull/2821)
  [(#2871)](https://github.com/PennyLaneAI/pennylane/pull/2871)
  [(#2968)](https://github.com/PennyLaneAI/pennylane/pull/2968)
  [(#2959)](https://github.com/PennyLaneAI/pennylane/pull/2959)
  [(#2968)](https://github.com/PennyLaneAI/pennylane/pull/2968)

  The classical-shadow measurement protocol is described in detail in the paper
  [Predicting Many Properties of a Quantum System from Very Few Measurements](https://arxiv.org/abs/2002.08953).
  As part of the support for classical shadows in this release, two new finite-shot and fully-differentiable measurements are available: 

  - QNodes returning the new measurement `qml.classical_shadow()` will return two entities;
  `bits` (0 or 1 if the 1 or -1 eigenvalue is sampled, respectively) and `recipes`
  (the randomized Pauli measurements that are performed for each qubit, labelled by integer):
    
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

  - QNodes returning `qml.shadow_expval()` yield the expectation value estimation using classical shadows:

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
    >>> circuit(x, H)
    tensor(1.8486, requires_grad=True) 
    >>> qml.grad(circuit)(x, H)
    -0.4797000000000001
    ```

  Fully-differentiable QNode transforms for both new classical-shadows measurements are also available via
  `qml.shadows.shadow_state` and `qml.shadows.shadow_expval`, respectively.
  
  For convenient post-processing, we've also added the ability to calculate general Renyi entropies 
  by way of the `ClassicalShadow` class' `entropy` method, which requires the wires of the subsystem of interest
  and the Renyi entropy order:

  ```pycon
  >>> shadow = qml.ClassicalShadow(bits, recipes)
  >>> vN_entropy = shadow.entropy(wires=[0, 1], alpha=1)
  ``` 

<h4>Qutrits: quantum circuits for tertiary degrees of freedom ☘️</h4>

* An entirely new framework for quantum computing is now simulatable with the addition of qutrit functionalities.
  [(#2699)](https://github.com/PennyLaneAI/pennylane/pull/2699)
  [(#2781)](https://github.com/PennyLaneAI/pennylane/pull/2781)
  [(#2782)](https://github.com/PennyLaneAI/pennylane/pull/2782)
  [(#2783)](https://github.com/PennyLaneAI/pennylane/pull/2783)
  [(#2784)](https://github.com/PennyLaneAI/pennylane/pull/2784)
  [(#2841)](https://github.com/PennyLaneAI/pennylane/pull/2841)
  [(#2843)](https://github.com/PennyLaneAI/pennylane/pull/2843)

  [Qutrits](https://en.wikipedia.org/wiki/Qutrit) are like qubits, but instead live in a *three*-dimensional Hilbert space; they are not binary degrees of freedom, they are *tertiary*. 
  The advent of qutrits allows for all sorts of interesting theoretical, practical, and algorithmic capabilities that have yet to be discovered.
  
  To facilitate qutrit circuits requires a new device: `default.qutrit`.
  The `default.qutrit` device is a Python-based simulator, akin to `default.qubit`, and is defined as per usual:

  ```pycon
  >>> dev = qml.device("default.qutrit", wires=1)
  ```

  The following operations are supported on `default.qutrit` devices:

  - The qutrit shift operator, `qml.TShift`, and the ternary clock operator, `qml.TClock`, as defined in this paper by [Yeh et al. (2022)](https://arxiv.org/abs/2204.00552),
  which are the qutrit analogs of the Pauli X and Pauli Z operations, respectively.
  - The `qml.TAdd` and `qml.TSWAP` operations which are the qutrit analogs of the CNOT and SWAP operations, respectively.
  - Custom unitary operations via `qml.QutritUnitary`.
  - `qml.state` and `qml.probs` measurements.
  - Measuring user-specified Hermitian matrix observables via `qml.THermitian`.

  A comprehensive example of these features is given below:

  ```python
  dev = qml.device("default.qutrit", wires=1)

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

  @qml.qnode(dev)
  def qutrit_state(U, obs):
      qml.TShift(0)
      qml.TClock(0)
      qml.QutritUnitary(U, wires=0)
      return qml.state()

  @qml.qnode(dev)
  def qutrit_expval(U, obs):
      qml.TShift(0)
      qml.TClock(0)
      qml.QutritUnitary(U, wires=0)
      return qml.expval(qml.THermitian(obs, wires=0))
  ```

  ```pycon
  >>> qutrit_state(U, obs)
  tensor([-0.28867513+0.5j, -0.28867513+0.5j, -0.28867513+0.5j], requires_grad=True) 
  >>> qutrit_expval(U, obs)
  tensor(0.80473785, requires_grad=True)
  ```

  We will continue to add more and more support for qutrits in future releases.

<h4>Simplifying just got... simpler 😌</h4>

* The `qml.simplify()` function has several intuitive improvements with this release.
  [(#2978)](https://github.com/PennyLaneAI/pennylane/pull/2978)
  [(#2982)](https://github.com/PennyLaneAI/pennylane/pull/2982)
  [(#2922)](https://github.com/PennyLaneAI/pennylane/pull/2922)
  [(#3012)](https://github.com/PennyLaneAI/pennylane/pull/3012)

  `qml.simplify` can now perform the following:

  - simplify parametrized operations
  - simplify the adjoint and power of specific operators
  - group like terms in a sum
  - resolve products of Pauli operators
  - combine rotation angles of identical rotation gates

  Here is an example of `qml.simplify` in action with parameterized rotation gates. 
  In this case, the angles of rotation are simplified to be modulo :math:`4\pi`.

  ```pycon
  >>> op1 = qml.RX(30.0, wires=0)
  >>> qml.simplify(op1)
  RX(4.867258771281655, wires=[0])
  >>> op2 = qml.RX(4 * np.pi, wires=0)
  >>> qml.simplify(op2)
  Identity(wires=[0])
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
  [RZ(11.566370614359172, wires=[0]) @ RY(11.566370614359172, wires=[0]) @ RX(11.566370614359172, wires=[0]),
   probs(wires=[0])]
  ```

<h4>QNSPSA optimizer 💪</h4>

* A new optimizer called `qml.QNSPSAOptimizer` is available that implements the quantum natural simultaneous
  perturbation stochastic approximation (QNSPSA) method based on 
  [Simultaneous Perturbation Stochastic Approximation of the Quantum Fisher Information](https://quantum-journal.org/papers/q-2021-10-20-567/). 
  [(#2818)](https://github.com/PennyLaneAI/pennylane/pull/2818) 

  `qml.QNSPSAOptimizer` is a second-order [SPSA algorithm](https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation), 
  which combines the convergence power of the quantum-aware Quantum Natural Gradient
  (QNG) optimization method with the reduced quantum evaluations of SPSA methods.

  While the QNSPSA optimizer requires additional circuit executions (10 executions per
  step) compared to standard SPSA optimization (3 executions per step), these additional
  evaluations are used to provide a stochastic estimation of a second-order
  metric tensor, which often helps the optimizer to achieve faster convergence.

  Use `qml.QNSPSAOptimizer` like you would any other optimizer:
  
  ```python
  max_iterations = 50
  opt = qml.QNSPSAOptimizer() 

  for _ in range(max_iterations):
      params, cost = opt.step_and_cost(cost, params)
  ```  

  Check out [our demo](https://pennylane.ai/qml/demos/qnspsa.html) on the QNSPSA optimizer for more information.

<h4>Operator and parameter broadcasting supplements 📈</h4>

* Operator methods for exponentiation and raising to a power have been added.
  [(#2799)](https://github.com/PennyLaneAI/pennylane/pull/2799)
  [(#3029)](https://github.com/PennyLaneAI/pennylane/pull/3029)

  - The `qml.exp` function can be used to create observables or generic rotation gates:

    ```pycon
    >>> x = 1.234
    >>> t = qml.PauliX(0) @ qml.PauliX(1) + qml.PauliY(0) @ qml.PauliY(1)
    >>> isingxy = qml.exp(t, 0.25j * x)
    >>> isingxy.matrix()
    array([[1.       +0.j        , 0.       +0.j        ,
        1.       +0.j        , 0.       +0.j        ],
       [0.       +0.j        , 0.8156179+0.j        ,
        1.       +0.57859091j, 0.       +0.j        ],
       [0.       +0.j        , 0.       +0.57859091j,
        0.8156179+0.j        , 0.       +0.j        ],
       [0.       +0.j        , 0.       +0.j        ,
        1.       +0.j        , 1.       +0.j        ]]) 
    ```

  - The `qml.pow` function raises a given operator to a power:

    ```pycon
    >>> op = qml.pow(qml.PauliX(0), 2)
    >>> op.matrix()
    array([[1, 0], [0, 1]])
    ```

* An operator called `qml.PSWAP` is now available.
  [(#2667)](https://github.com/PennyLaneAI/pennylane/pull/2667)

  The `qml.PSWAP` gate -- or phase-SWAP gate -- was previously available within the PennyLane-Braket plugin only. Enjoy it natively in PennyLane with v0.26.

* Check whether or not an operator is hermitian or unitary with `qml.is_hermitian` and `qml.is_unitary`.
  [(#2960)](https://github.com/PennyLaneAI/pennylane/pull/2960)

  ```pycon
  >>> op1 = qml.PauliX(wires=0)
  >>> qml.is_hermitian(op1)
  True
  >>> op2 = qml.PauliX(0) + qml.RX(np.pi/3, 0) 
  >>> qml.is_unitary(op2)
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

* The `qml.math.expand_matrix()` method now allows the sparse matrix representation of an operator to be extended to
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

* `qml.ctrl` now uses `Controlled` instead of `ControlledOperation`.  The new `Controlled` class
  wraps individual `Operator`'s instead of a tape.  It provides improved representations and integration.
  [(#2990)](https://github.com/PennyLaneAI/pennylane/pull/2990)

* `qml.matrix` can now compute the matrix of tapes and QNodes that contain multiple 
  broadcasted operations or non-broadcasted operations after broadcasted ones.
  [(#3025)](https://github.com/PennyLaneAI/pennylane/pull/3025)

  A common scenario in which this becomes relevant is the decomposition of broadcasted
  operations: the decomposition in general will contain one or multiple broadcasted
  operations as well as operations with no or fixed parameters that are not broadcasted.

* Lists of operators are now internally sorted by their respective wires while also taking into account their commutativity property.
  [(#2995)](https://github.com/PennyLaneAI/pennylane/pull/2995)

* Some methods of the `QuantumTape` class have been simplified and reordered to
  improve both readability and performance. 
  [(#2963)](https://github.com/PennyLaneAI/pennylane/pull/2963)

* The `qml.qchem.molecular_hamiltonian` function is modified to support observable grouping.
  [(#2997)](https://github.com/PennyLaneAI/pennylane/pull/2997)

* `qml.ops.op_math.Controlled` now has basic decomposition functionality.
  [(#2938)](https://github.com/PennyLaneAI/pennylane/pull/2938)

* Automatic circuit cutting has been improved by making better partition imbalance derivations.
  Now it is more likely to generate optimal cuts for larger circuits.
  [(#2517)](https://github.com/PennyLaneAI/pennylane/pull/2517)

* By default, `qml.counts` only returns the outcomes observed in sampling. Optionally, specifying `qml.counts(all_outcomes=True)`
  will return a dictionary containing all possible outcomes. 
  [(#2889)](https://github.com/PennyLaneAI/pennylane/pull/2889)
  
  ```pycon
  >>> dev = qml.device("default.qubit", wires=2, shots=1000)
  >>>
  >>> @qml.qnode(dev)
  >>> def circuit():
  ...     qml.Hadamard(wires=0)
  ...     qml.CNOT(wires=[0, 1])
  ...     return qml.counts(all_outcomes=True)
  >>> result = circuit()
  >>> result
  {'00': 495, '01': 0, '10': 0,  '11': 505}
  ```
  
* Internal use of in-place inversion is eliminated in preparation for its deprecation.
  [(#2965)](https://github.com/PennyLaneAI/pennylane/pull/2965)

* `Controlled` operators now work with `qml.is_commuting`.
  [(#2994)](https://github.com/PennyLaneAI/pennylane/pull/2994)

* `qml.prod` and `qml.op_sum` now support the `sparse_matrix()` method.
  [(#3006)](https://github.com/PennyLaneAI/pennylane/pull/3006)

  ```pycon
  >>> xy = qml.prod(qml.PauliX(1), qml.PauliY(1))
  >>> op = qml.op_sum(xy, qml.Identity(0))
  >>>
  >>> sparse_mat = op.sparse_matrix(wire_order=[0,1])
  >>> type(sparse_mat)
  <class 'scipy.sparse.csr.csr_matrix'>
  >>> sparse_mat.toarray()
  [[1.+1.j 0.+0.j 0.+0.j 0.+0.j]
  [0.+0.j 1.-1.j 0.+0.j 0.+0.j]
  [0.+0.j 0.+0.j 1.+1.j 0.+0.j]
  [0.+0.j 0.+0.j 0.+0.j 1.-1.j]]
  ```

* Provided `sparse_matrix()` support for single qubit observables.
  [(#2964)](https://github.com/PennyLaneAI/pennylane/pull/2964)

* `qml.Barrier` with `only_visual=True` now simplifies via `op.simplify()` to the identity operator
  or a product of identity operators.
  [(#3016)](https://github.com/PennyLaneAI/pennylane/pull/3016)

* More accurate and intuitive outputs for printing some operators have been added.
  [(#3013)](https://github.com/PennyLaneAI/pennylane/pull/3013)

* Results for the matrix of the sum or product of operators are stored in a more efficient manner.
  [(#3022)](https://github.com/PennyLaneAI/pennylane/pull/3022)

* The computation of the (sparse) matrix for the sum or product of operators is now more efficient.
  [(#3030)](https://github.com/PennyLaneAI/pennylane/pull/3030)

* When the factors of `qml.prod` don't share any wires, the matrix and sparse matrix are computed using
  a kronecker product for improved efficiency.
  [(#3040)](https://github.com/PennyLaneAI/pennylane/pull/3040)
  
* `qml.grouping.is_pauli_word` now returns `False` for operators that don't inherit from `qml.Observable` instead of raising an error.
  [(#3039)](https://github.com/PennyLaneAI/pennylane/pull/3039)

* Added functionality to iterate over operators created from `qml.op_sum` and `qml.prod`.
  [(#3028)](https://github.com/PennyLaneAI/pennylane/pull/3028)

  ```pycon
  >>> op = qml.op_sum(qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2))
  >>> len(op)
  3
  >>> op[1]
  PauliY(wires=[1])
  >>> [o.name for o in op]
  ['PauliX', 'PauliY', 'PauliZ']
  ```

<h3>Deprecations</h3>

* In-place inversion is now deprecated. This includes `op.inv()` and `op.inverse=value`. Please
  use `qml.adjoint` or `qml.pow` instead. Support for these methods will remain till v0.28.
  [(#2988)](https://github.com/PennyLaneAI/pennylane/pull/2988)

  Don't use:

  ```pycon
  >>> v1 = qml.PauliX(0).inv()
  >>> v2 = qml.PauliX(0)
  >>> v2.inverse = True
  ```

  Instead use:

  ```pycon
  >>> qml.adjoint(qml.PauliX(0))
  Adjoint(PauliX(wires=[0]))
  >>> qml.pow(qml.PauliX(0), -1)
  PauliX(wires=[0])**-1
  >>> qml.pow(qml.PauliX(0), -1, lazy=False)
  PauliX(wires=[0])
  >>> qml.PauliX(0) ** -1
  PauliX(wires=[0])**-1
  ```

  `qml.adjoint` takes the conjugate transpose of an operator, while `qml.pow(op, -1)` indicates matrix
  inversion. For unitary operators, `adjoint` will be more efficient than `qml.pow(op, -1)`, even
  though they represent the same thing.

* The `supports_reversible_diff` device capability is unused and has been removed.
  [(#2993)](https://github.com/PennyLaneAI/pennylane/pull/2993)

<h3>Breaking changes</h3>

* Measuring an operator that might not be hermitian now raises a warning instead of an
  error. To definitively determine whether or not an operator is hermitian, use `qml.is_hermitian`.
  [(#2960)](https://github.com/PennyLaneAI/pennylane/pull/2960)

* The `ControlledOperation` class has been removed. This was a developer-only class, so the change should not be evident to
  any users. It is replaced by `Controlled`.
  [(#2990)](https://github.com/PennyLaneAI/pennylane/pull/2990)

* The default `execute` method for the `QubitDevice` base class now calls `self.statistics`
  with an additional keyword argument `circuit`, which represents the quantum tape
  being executed.
  Any device that overrides `statistics` should edit the signature of the method to include
  the new `circuit` keyword argument.
  [(#2820)](https://github.com/PennyLaneAI/pennylane/pull/2820)

* The `expand_matrix()` has been moved from `pennylane.operation` to
  `pennylane.math.matrix_manipulation`
  [(#3008)](https://github.com/PennyLaneAI/pennylane/pull/3008)

* `qml.grouping.utils.is_commuting` has been removed, and its Pauli word logic is now part of `qml.is_commuting`.
  [(#3033)](https://github.com/PennyLaneAI/pennylane/pull/3033)

* `qml.is_commuting` has been moved from `pennylane.transforms.commutation_dag` to `pennylane.ops.functions`.
  [(#2991)](https://github.com/PennyLaneAI/pennylane/pull/2991)

<h3>Documentation</h3>

* Updated the Fourier transform docs to use `circuit_spectrum` instead of `spectrum`, which has been deprecated.
  [(#3018)](https://github.com/PennyLaneAI/pennylane/pull/3018)
  
* Corrected the docstrings for diagonalizing gates for all relevant operations. The docstrings used 
  to say that the diagonalizing gates implemented :math:`U`, the unitary such that :math:`O = U \Sigma U^{\dagger}`, where :math:`O` is 
  the original observable and :math:`\Sigma` a diagonal matrix. However, the diagonalizing gates actually implement 
  :math:`U^{\dagger}`, since :math:`\langle \psi | O | \psi \rangle = \langle \psi | U \Sigma U^{\dagger} | \psi \rangle`, 
  making :math:`U^{\dagger} | \psi \rangle` the actual state being measured in the Z-basis.
  [(#2981)](https://github.com/PennyLaneAI/pennylane/pull/2981)

<h3>Bug fixes</h3>

* Fixed a bug with `qml.ops.Exp` operators when the coefficient is autograd but the diagonalizing gates
  don't act on all wires.
  [(#3057)](https://github.com/PennyLaneAI/pennylane/pull/3057)

* Fixed a bug where the tape transform `single_qubit_fusion` computed wrong rotation angles
  for specific combinations of rotations.
  [(#3024)](https://github.com/PennyLaneAI/pennylane/pull/3024)

* Jax gradients now work with a QNode when the quantum function was transformed by `qml.simplify`.
  [(#3017)](https://github.com/PennyLaneAI/pennylane/pull/3017)

* Operators that have `num_wires = AnyWires` or `num_wires = AnyWires` now raise an error, with
  certain exceptions, when instantiated with `wires=[]`.
  [(#2979)](https://github.com/PennyLaneAI/pennylane/pull/2979)

* Fixed a bug where printing `qml.Hamiltonian` with complex coefficients raises `TypeError` in some cases.
  [(#3005)](https://github.com/PennyLaneAI/pennylane/pull/3004)

* Added a more descriptive error message when measuring non-commuting observables at the end of a circuit with
  `probs`, `samples`, `counts` and `allcounts`.
  [(#3065)](https://github.com/PennyLaneAI/pennylane/pull/3065)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Utkarsh Azad, Tom Bromley, Olivia Di Matteo, Isaac De Vlugt, Yiheng Duan, 
Lillian Marie Austin Frederiksen, Josh Izaac, Soran Jahangiri, Edward Jiang, Ankit Khandelwal, 
Korbinian Kottmann, Meenu Kumari, Christina Lee, Albert Mitjans Coma, Romain Moyard, Rashid N H M,
Zeyue Niu, Mudit Pandey, Matthew Silverman, Jay Soni, Antal Száva, Cody Wang, David Wierichs.
