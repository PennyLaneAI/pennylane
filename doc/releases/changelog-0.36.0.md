
# Release 0.36.0

<h3>New features since last release</h3>

<h4>Estimate errors in a quantum circuit 🧮</h4>

* This version of PennyLane lays the foundation for estimating the total error
  in a quantum circuit from the combination of individual gate errors.
  [(#5154)](https://github.com/PennyLaneAI/pennylane/pull/5154)
  [(#5464)](https://github.com/PennyLaneAI/pennylane/pull/5464)
  [(#5465)](https://github.com/PennyLaneAI/pennylane/pull/5465)
  [(#5278)](https://github.com/PennyLaneAI/pennylane/pull/5278)
  [(#5384)](https://github.com/PennyLaneAI/pennylane/pull/5384)

  Two new user-facing classes enable calculating and propagating 
  gate errors in PennyLane:

  * `qml.resource.SpectralNormError`: the spectral norm error is defined as the 
    distance, in [spectral norm](https://en.wikipedia.org/wiki/Matrix_norm), between 
    the true unitary we intend to apply and the approximate unitary that is actually applied.

  * `qml.resource.ErrorOperation`: a base class that inherits from `qml.operation.Operation` 
    and represents quantum operations which carry some form of algorithmic error.

  `SpectralNormError` can be used for back-of-the-envelope type calculations like obtaining the
  spectral norm error between two unitaries via `get_error`:

  ```python
  import pennylane as qp
  from pennylane.resource import ErrorOperation, SpectralNormError

  intended_op = qml.RY(0.40, 0)
  actual_op = qml.RY(0.41, 0) # angle of rotation is slightly off
  ```
  
  ```pycon
  >>> SpectralNormError.get_error(intended_op, actual_op)
  0.004999994791668309
  ```
  
  `SpectralNormError` is also a key tool to specify errors in larger quantum circuits:

  * For operations representing a major building block of an algorithm, we can create a custom operation
    that inherits from `ErrorOperation`. This child class must override the `error` method and 
    should return a `SpectralNormError` instance:

    ```python
    class MyErrorOperation(ErrorOperation):
        def __init__(self, error_val, wires):
            self.error_val = error_val
            super().__init__(wires=wires)

        def error(self):
            return SpectralNormError(self.error_val)
    ```

    In this toy example, `MyErrorOperation` introduces an arbitrary `SpectralNormError`
    when called in a QNode. It does not require a decomposition or matrix representation
    when used with `null.qubit` (suggested for use with resource and error estimation since circuit executions are
    not required to calculate resources or errors).
  
    ```python
    dev = qml.device("null.qubit")

    @qml.qnode(dev)
    def circuit():
        MyErrorOperation(0.1, wires=0)
        MyErrorOperation(0.2, wires=1)
        return qml.state()
    ```

    The total spectral norm error of the circuit can be calculated using `qml.specs`:

    ```pycon
    >>> qml.specs(circuit)()['errors']
    {'SpectralNormError': SpectralNormError(0.30000000000000004)}
    ```

  * PennyLane already includes a number of built-in building blocks for algorithms like
    `QuantumPhaseEstimation` and `TrotterProduct`. `TrotterProduct` now propagates errors 
    based on the number of steps performed in the Trotter product. `QuantumPhaseEstimation` 
    now propagates errors based on the error of its input unitary.

    ```python
    dev = qml.device('null.qubit')
    hamiltonian = qml.dot([1.0, 0.5, -0.25], [qml.X(0), qml.Y(0), qml.Z(0)])

    @qml.qnode(dev)
    def circuit():
        qml.TrotterProduct(hamiltonian, time=0.1, order=2)
        qml.QuantumPhaseEstimation(MyErrorOperation(0.01, wires=0), estimation_wires=[1, 2, 3])
        return qml.state()
    ```

    Again, the total spectral norm error of the circuit can be calculated using `qml.specs`:

    ```pycon
    >>> qml.specs(circuit)()["errors"]
    {'SpectralNormError': SpectralNormError(0.07616666666666666)}
    ```

  Check out our [error propagation demo](https://pennylane.ai/qml/demos/tutorial_error_prop/) to see how to use these new features in a real-world example!

<h4>Access an extended arsenal of quantum algorithms 🏹</h4>

* The Fast Approximate BLock-Encodings (FABLE) algorithm for embedding
  a matrix into a quantum circuit as outlined in
  [arXiv:2205.00081](https://arxiv.org/abs/2205.00081) is now accessible via the `qml.FABLE` 
  template.
  [(#5107)](https://github.com/PennyLaneAI/pennylane/pull/5107)

  The usage of `qml.FABLE` is similar to `qml.BlockEncode` but provides a more
  efficient circuit construction at the cost of a user-defined approximation 
  level, `tol`. The number of wires that `qml.FABLE` operates on is `2*n + 1`, 
  where `n` defines the dimension of the :math:`2^n \times 2^n` matrix that we
  want to block-encode.

  ```python
  import numpy as np

  A = np.array([[0.1, 0.2], [0.3, 0.4]])
  dev = qml.device('default.qubit', wires=3)

  @qml.qnode(dev)
  def circuit():
      qml.FABLE(A, tol = 0.001, wires=range(3))  
      return qml.state()
  ```

  ```pycon
  >>> mat = qml.matrix(circuit)()
  >>> 2 * mat[0:2, 0:2]
  array([[0.1+0.j, 0.2+0.j],
         [0.3+0.j, 0.4+0.j]])
  ```

* A high-level interface for amplitude amplification and its variants is now 
  available via the new `qml.AmplitudeAmplification` template.
  [(#5160)](https://github.com/PennyLaneAI/pennylane/pull/5160)

  Based on [arXiv:quant-ph/0005055](https://arxiv.org/abs/quant-ph/0005055), 
  given a state :math:`\vert \Psi \rangle = \alpha \vert \phi \rangle + \beta \vert \phi^{\perp} \rangle`, 
  `qml.AmplitudeAmplification` amplifies the amplitude of :math:`\vert \phi \rangle`.

  Here's an example with a target state
  :math:`\vert \phi \rangle = \vert 2 \rangle = \vert 010 \rangle`,
  an input state :math:`\vert \Psi \rangle = H^{\otimes 3} \vert 000 \rangle`, as well as an
  oracle that flips the sign of :math:`\vert \phi \rangle` and does nothing to
  :math:`\vert \phi^{\perp} \rangle`, which can be achieved in this case through
  `qml.FlipSign`.

  ```python
  @qml.prod
  def generator(wires):
      for wire in wires:
          qml.Hadamard(wires=wire)

  U = generator(wires=range(3))
  O = qml.FlipSign(2, wires=range(3))
  ```

  Here, `U` is a quantum operation that is created by decorating a quantum 
  function with `@qml.prod`. This could alternatively be done by creating a 
  user-defined 
  [custom operation](https://docs.pennylane.ai/en/stable/development/adding_operators.html) 
  with a decomposition. Amplitude amplification can then be set up within a 
  circuit: 
  
  ```python
  dev = qml.device("default.qubit")

  @qml.qnode(dev)
  def circuit():
      generator(wires=range(3)) # prepares |Psi> = U|0>
      qml.AmplitudeAmplification(U, O, iters=10)

      return qml.probs(wires=range(3))
  ```
  
  ```pycon
  >>> print(np.round(circuit(), 3))
  [0.01  0.01  0.931 0.01  0.01  0.01  0.01  0.01 ]
  ```

  As expected, we amplify the :math:`\vert 2 \rangle` state.

* Reflecting about a given quantum state is now available via `qml.Reflection`.
  This operation is very useful in the amplitude amplification algorithm and offers a generalization
  of `qml.FlipSign`, which operates on basis states.
  [(#5159)](https://github.com/PennyLaneAI/pennylane/pull/5159)

  `qml.Reflection` works by providing an operation, :math:`U`, that *prepares* the 
  desired state, :math:`\vert \psi \rangle`, that we want to reflect about. In other 
  words, :math:`U` is such that :math:`U \vert 0 \rangle = \vert \psi \rangle`. In 
  PennyLane, :math:`U` must be an `Operator`.
  
  For example, if we want to reflect about 
  :math:`\vert \psi \rangle = \vert + \rangle`, then :math:`U = H`:

  ```python
  U = qml.Hadamard(wires=0)

  dev = qml.device('default.qubit')
  @qml.qnode(dev)
  def circuit():
        qml.Reflection(U)
        return qml.state()
  ```

  ```pycon
  >>> circuit()
  tensor([0.-6.123234e-17j, 1.+6.123234e-17j], requires_grad=True)
  ```

* Performing qubitization is now easily accessible with the new 
  `qml.Qubitization` operator.
  [(#5500)](https://github.com/PennyLaneAI/pennylane/pull/5500)

  `qml.Qubitization` encodes a Hamiltonian into a suitable unitary operator. 
  When applied in conjunction with quantum phase estimation (QPE), it allows 
  for computing the eigenvalue of an eigenvector of the given Hamiltonian. 

  ```python
  H = qml.dot([0.1, 0.3, -0.3], [qml.Z(0), qml.Z(1), qml.Z(0) @ qml.Z(2)])
  @qml.qnode(qml.device("default.qubit"))
  def circuit():
      # initialize the eigenvector
      qml.PauliX(2)
      # apply QPE
      measurements = qml.iterative_qpe(
          qml.Qubitization(H, control = [3,4]), ancilla = 5, iters = 3
      )
      return qml.probs(op = measurements)
  ```

<h4>Make use of more methods to map from molecules 🗺️</h4>

* A new function called `qml.bravyi_kitaev` has been added to perform the 
  Bravyi-Kitaev mapping of fermionic Hamiltonians to qubit Hamiltonians.
  [(#5390)](https://github.com/PennyLaneAI/pennylane/pull/5390)

  This function presents an alternative mapping to `qml.jordan_wigner` or
  `qml.parity_transform` which can help us measure expectation values more
  efficiently on hardware. Simply provide a fermionic 
  Hamiltonian (created from `from_string`, `FermiA`, `FermiC`, `FermiSentence`, 
  or `FermiWord`) and the number of qubits / spin orbitals in the system, `n`:

  ```pycon
  >>> fermi_ham = qml.fermi.from_string('0+ 1+ 1- 0-')
  >>> qubit_ham = qml.bravyi_kitaev(fermi_ham, n=6, tol=0.0)
  >>> print(qubit_ham)
  0.25 * I(0) + -0.25 * Z(0) + -0.25 * (Z(0) @ Z(1)) + 0.25 * Z(1)
  ```

* The `qml.qchem.hf_state` function has been upgraded to be compatible with
  `qml.parity_transform` and the new Bravyi-Kitaev mapping 
  (`qml.bravyi_kitaev`).
  [(#5472)](https://github.com/PennyLaneAI/pennylane/pull/5472)
  [(#5472)](https://github.com/PennyLaneAI/pennylane/pull/5472)

  ```pycon
  >>> state_bk = qml.qchem.hf_state(2, 6, basis="bravyi_kitaev")
  >>> print(state_bk)
  [1 0 0 0 0 0]
  >>> state_parity = qml.qchem.hf_state(2, 6, basis="parity")
  >>> print(state_parity)
  [1 0 0 0 0 0]
  ```

<h4>Calculate dynamical Lie algebras 👾</h4>

* The dynamical Lie algebra (DLA) of a set of operators captures the range of unitary evolutions
  that the operators can generate. In v0.36 of PennyLane, we have added support for calculating
  important DLA concepts including:

  * A new `qml.lie_closure` function to compute the Lie closure of a list of operators, providing
    one way to obtain the DLA.
    [(#5161)](https://github.com/PennyLaneAI/pennylane/pull/5161)
    [(#5169)](https://github.com/PennyLaneAI/pennylane/pull/5169)
    [(#5627)](https://github.com/PennyLaneAI/pennylane/pull/5627)

    For a list of operators `ops = [op1, op2, op3, ..]`, one computes all nested commutators between `ops` until no new operators are generated from commutation.
    All these operators together form the DLA, see e.g. section IIB of [arXiv:2308.01432](https://arxiv.org/abs/2308.01432).

    Take for example the following operators:

    ```python
    from pennylane import X, Y, Z
    ops = [X(0) @ X(1), Z(0), Z(1)]
    ```

    A first round of commutators between all elements yields the new operators `Y(0) @ X(1)` and `X(0) @ Y(1)` (omitting scalar prefactors).

    ```python
    >>> qml.commutator(X(0) @ X(1), Z(0))
    -2j * (Y(0) @ X(1))
    >>> qml.commutator(X(0) @ X(1), Z(1))
    -2j * (X(0) @ Y(1))
    ```

    A next round of commutators between all elements further yields the new operator `Y(0) @ Y(1)`.

    ```python
    >>> qml.commutator(X(0) @ Y(1), Z(0))
    -2j * (Y(0) @ Y(1))
    ```

    After that, no new operators emerge from taking nested commutators and we have the resulting DLA.
    This can now be done in short via `qml.lie_closure` as follows.

    ```python
    >>> ops = [X(0) @ X(1), Z(0), Z(1)]
    >>> dla = qml.lie_closure(ops)
    >>> dla
    [X(0) @ X(1),
     Z(0),
     Z(1),
     -1.0 * (Y(0) @ X(1)),
     -1.0 * (X(0) @ Y(1)),
     -1.0 * (Y(0) @ Y(1))]
    ```

  * Computing the structure constants (the adjoint representation) of a dynamical Lie algebra.
    [(5406)](https://github.com/PennyLaneAI/pennylane/pull/5406)

    For example, we can compute the adjoint representation of the transverse field Ising model DLA.

    ```pycon
    >>> dla = [X(0) @ X(1), Z(0), Z(1), Y(0) @ X(1), X(0) @ Y(1), Y(0) @ Y(1)]
    >>> structure_const = qml.structure_constants(dla)
    >>> structure_const.shape
    (6, 6, 6)
    ```
    Visit the [documentation of qml.structure_constants](https://docs.pennylane.ai/en/stable/code/api/pennylane.structure_constants.html)
    to understand how structure constants are a useful way to represent a DLA.

  * Computing the center of a dynamical Lie algebra.
    [(#5477)](https://github.com/PennyLaneAI/pennylane/pull/5477)

    Given a DLA `g`, we can now compute its centre. The `center` is the collection of operators that commute with _all_ other operators in the DLA.

    ```pycon
    >>> g = [X(0), X(1) @ X(0), Y(1), Z(1) @ X(0)]
    >>> qml.center(g)
    [X(0)]
    ```

  To help explain these concepts, check out the
  [dynamical Lie algebras demo](https://pennylane.ai/qml/demos/tutorial_liealgebra).

<h3>Improvements 🛠</h3>

<h4>Simulate mixed-state qutrit systems</h4>

* Mixed qutrit states can now be simulated with the `default.qutrit.mixed` device.
  [(#5495)](https://github.com/PennyLaneAI/pennylane/pull/5495)
  [(#5451)](https://github.com/PennyLaneAI/pennylane/pull/5451)
  [(#5186)](https://github.com/PennyLaneAI/pennylane/pull/5186)
  [(#5082)](https://github.com/PennyLaneAI/pennylane/pull/5082)
  [(#5213)](https://github.com/PennyLaneAI/pennylane/pull/5213)

  Thanks to contributors from the University of British Columbia, a mixed-state
  qutrit device is now available for simulation, providing a noise-capable
  equivalent to `default.qutrit`.

  ```python
  dev = qml.device("default.qutrit.mixed")

  def circuit():
      qml.TRY(0.1, wires=0)

  @qml.qnode(dev)
  def shots_circuit():
      circuit()
      return qml.sample(), qml.expval(qml.GellMann(wires=0, index=1))

  @qml.qnode(dev)
  def density_matrix_circuit():
      circuit()
      return qml.state()
  ```

  ```pycon
  >>> shots_circuit(shots=5)
  (array([0, 0, 0, 0, 0]), 0.19999999999999996)
  >>> density_matrix_circuit()
  tensor([[0.99750208+0.j, 0.04991671+0.j, 0.        +0.j],
         [0.04991671+0.j, 0.00249792+0.j, 0.        +0.j],
         [0.        +0.j, 0.        +0.j, 0.        +0.j]], requires_grad=True)
  ```

  However, there's one crucial ingredient that we still need to add: support for qutrit noise
  operations. Keep your eyes peeled for this to arrive in the coming releases!

<h4>Work easily and efficiently with operators</h4>

* This release completes the main phase of PennyLane's switchover to an updated approach for
  handling arithmetic operations between operators. The new approach is now enabled by default and
  is intended to realize a few objectives:

  1. To make it as easy to work with PennyLane operators as it would be with pen and paper.
  2. To improve the efficiency of operator arithmetic.

  In many cases, this update should not break code. If issues do arise, check out the
  [updated operator troubleshooting page](https://docs.pennylane.ai/en/stable/news/new_opmath.html)
  and don't hesitate to reach out to us on the
  [PennyLane discussion forum](https://discuss.pennylane.ai/). As a last resort the old behaviour
  can be enabled by calling `qml.operation.disable_new_opmath()`, but this is not recommended
  because support will not continue in future PennyLane versions (v0.36 and higher).
  [(#5269)](https://github.com/PennyLaneAI/pennylane/pull/5269)

* A new class called `qml.ops.LinearCombination` has been introduced. In essence, this class is an updated equivalent of the now-deprecated `qml.ops.Hamiltonian`
  but for usage with the new operator arithmetic.
  [(#5216)](https://github.com/PennyLaneAI/pennylane/pull/5216)

* `qml.ops.Sum` now supports storing grouping information. Grouping type and method can be
  specified during construction using the `grouping_type` and `method` keyword arguments of
  `qml.dot`, `qml.sum`, or `qml.ops.Sum`. The grouping indices are stored in `Sum.grouping_indices`.
  [(#5179)](https://github.com/PennyLaneAI/pennylane/pull/5179)

  ```python
  a = qml.X(0)
  b = qml.prod(qml.X(0), qml.X(1))
  c = qml.Z(0)
  obs = [a, b, c]
  coeffs = [1.0, 2.0, 3.0]

  op = qml.dot(coeffs, obs, grouping_type="qwc")
  ```

  ```pycon
  >>> op.grouping_indices
  ((2,), (0, 1))
  ```

  Additionally, `grouping_type` and `method` can be set or changed after construction using
  `Sum.compute_grouping()`:

  ```python
  a = qml.X(0)
  b = qml.prod(qml.X(0), qml.X(1))
  c = qml.Z(0)
  obs = [a, b, c]
  coeffs = [1.0, 2.0, 3.0]

  op = qml.dot(coeffs, obs)
  ```

  ```pycon
  >>> op.grouping_indices is None
  True
  >>> op.compute_grouping(grouping_type="qwc")
  >>> op.grouping_indices
  ((2,), (0, 1))
  ```

  Note that the grouping indices refer to the lists returned by `Sum.terms()`, not `Sum.operands`.

* A new function called `qml.operation.convert_to_legacy_H` that converts `Sum`, `SProd`, and `Prod` to `Hamiltonian` instances has been added.
  This function is intended for developers and will be removed in a future release without a
  deprecation cycle.
  [(#5309)](https://github.com/PennyLaneAI/pennylane/pull/5309)

* The `qml.is_commuting` function now accepts `Sum`, `SProd`, and `Prod` instances.
  [(#5351)](https://github.com/PennyLaneAI/pennylane/pull/5351)

* Operators can now be left-multiplied by NumPy arrays (i.e., `arr * op`).
  [(#5361)](https://github.com/PennyLaneAI/pennylane/pull/5361)

* `op.generator()`, where `op` is an `Operator` instance, now returns operators 
  consistent with the global setting for `qml.operator.active_new_opmath()` wherever possible. 
  `Sum`, `SProd` and `Prod` instances will be returned even after disabling the 
  new operator arithmetic in cases where they offer additional functionality not 
  available using legacy operators.
  [(#5253)](https://github.com/PennyLaneAI/pennylane/pull/5253)
  [(#5410)](https://github.com/PennyLaneAI/pennylane/pull/5410)
  [(#5411)](https://github.com/PennyLaneAI/pennylane/pull/5411)
  [(#5421)](https://github.com/PennyLaneAI/pennylane/pull/5421)

* `Prod` instances temporarily have a new `obs` property, which helps smoothen the 
  transition of the new operator arithmetic system. In particular, this is aimed at preventing 
  breaking code that uses `Tensor.obs`. The property has been immediately deprecated. Moving 
  forward, we recommend using `op.operands`.
  [(#5539)](https://github.com/PennyLaneAI/pennylane/pull/5539)
  
* `qml.ApproxTimeEvolution` is now compatible with any operator that has a defined `pauli_rep`.
  [(#5362)](https://github.com/PennyLaneAI/pennylane/pull/5362)

* `Hamiltonian.pauli_rep` is now defined if the Hamiltonian is a linear combination of Pauli operators.
  [(#5377)](https://github.com/PennyLaneAI/pennylane/pull/5377)

* `Prod` instances created with qutrit operators now have a defined `eigvals()` method.
  [(#5400)](https://github.com/PennyLaneAI/pennylane/pull/5400)

* `qml.transforms.hamiltonian_expand` and `qml.transforms.sum_expand` can now 
  handle multi-term observables with a constant offset (i.e., terms like 
  `qml.I()`).
  [(#5414)](https://github.com/PennyLaneAI/pennylane/pull/5414)
  [(#5543)](https://github.com/PennyLaneAI/pennylane/pull/5543)

* `qml.qchem.taper_operation` is now compatible with the new operator arithmetic.
  [(#5326)](https://github.com/PennyLaneAI/pennylane/pull/5326)

* The warning for an observable that might not be hermitian in QNode executions has been removed. This enables jit-compilation.
  [(#5506)](https://github.com/PennyLaneAI/pennylane/pull/5506)

* `qml.transforms.split_non_commuting` will now work with single-term operator arithmetic.
  [(#5314)](https://github.com/PennyLaneAI/pennylane/pull/5314)

* `LinearCombination` and `Sum` now accept `_grouping_indices` on initialization. This addition is relevant to developers only. 
  [(#5524)](https://github.com/PennyLaneAI/pennylane/pull/5524)

* Calculating the dense, differentiable matrix for `PauliSentence` and operators with Pauli sentences
  is now faster.
  [(#5578)](https://github.com/PennyLaneAI/pennylane/pull/5578)

<h4>Community contributions 🥳</h4>

* `ExpectationMP`, `VarianceMP`, `CountsMP`, and `SampleMP` now have a 
  `process_counts` method (similar to `process_samples`). This allows for 
  calculating measurements given a `counts` dictionary.
  [(#5256)](https://github.com/PennyLaneAI/pennylane/pull/5256)
  [(#5395)](https://github.com/PennyLaneAI/pennylane/pull/5395)

* Type-hinting has been added in the `Operator` class for better interpretability.
  [(#5490)](https://github.com/PennyLaneAI/pennylane/pull/5490)

* An alternate strategy for sampling with multiple different `shots` values has 
  been implemented via the `shots.bins()` method, which samples all shots at once and 
  then processes each separately.
  [(#5476)](https://github.com/PennyLaneAI/pennylane/pull/5476)

<h4>Mid-circuit measurements and dynamic circuits</h4>

* A new module called `qml.capture` that will contain PennyLane's own capturing mechanism for hybrid
  quantum-classical programs has been added.
  [(#5509)](https://github.com/PennyLaneAI/pennylane/pull/5509)

* The `dynamic_one_shot` transform has been introduced, enabling dynamic circuit 
  execution on circuits with finite `shots` and devices that natively support 
  mid-circuit measurements.
  [(#5266)](https://github.com/PennyLaneAI/pennylane/pull/5266)

* The `QubitDevice` class and children classes support the `dynamic_one_shot` transform provided 
  that they support mid-circuit measurement operations natively.
  [(#5317)](https://github.com/PennyLaneAI/pennylane/pull/5317)

* `default.qubit` can now be provided a random seed for sampling mid-circuit 
  measurements with finite shots. This (1) ensures that random behaviour is more 
  consistent with `dynamic_one_shot` and `defer_measurements` and (2) makes our 
  continuous-integration (CI) have less failures due to stochasticity.
  [(#5337)](https://github.com/PennyLaneAI/pennylane/pull/5337)

<h4>Performance and broadcasting</h4>

* Gradient transforms may now be applied to batched/broadcasted QNodes as long as the
  broadcasting is in non-trainable parameters.
  [(#5452)](https://github.com/PennyLaneAI/pennylane/pull/5452)

* The performance of computing the matrix of `qml.QFT` has been improved.
  [(#5351)](https://github.com/PennyLaneAI/pennylane/pull/5351)

* `qml.transforms.broadcast_expand` now supports shot vectors when returning `qml.sample()`.
  [(#5473)](https://github.com/PennyLaneAI/pennylane/pull/5473)

* `LightningVJPs` is now compatible with Lightning devices using the new device API.
  [(#5469)](https://github.com/PennyLaneAI/pennylane/pull/5469)

<h4>Device capabilities</h4>

* Obtaining classical shadows using the `default.clifford` device is now compatible with
  [stim](https://github.com/quantumlib/Stim) `v1.13.0`.
  [(#5409)](https://github.com/PennyLaneAI/pennylane/pull/5409)

* `default.mixed` has improved support for sampling-based measurements with non-NumPy interfaces.
  [(#5514)](https://github.com/PennyLaneAI/pennylane/pull/5514)
  [(#5530)](https://github.com/PennyLaneAI/pennylane/pull/5530)

* `default.mixed` now supports arbitrary state-based measurements with `qml.Snapshot`.
  [(#5552)](https://github.com/PennyLaneAI/pennylane/pull/5552)

* `null.qubit` has been upgraded to the new device API and has support for all 
  measurements and various modes of differentiation.
  [(#5211)](https://github.com/PennyLaneAI/pennylane/pull/5211)

<h4>Other improvements</h4>

* Entanglement entropy can now be calculated with 
  `qml.math.vn_entanglement_entropy`, which computes the von Neumann 
  entanglement entropy from a density matrix. A corresponding QNode transform,
  `qml.qinfo.vn_entanglement_entropy`, has also been added.
  [(#5306)](https://github.com/PennyLaneAI/pennylane/pull/5306)

* `qml.draw` and `qml.draw_mpl` will now attempt to sort the wires if no wire order
  is provided by the user or the device.
  [(#5576)](https://github.com/PennyLaneAI/pennylane/pull/5576)

* A clear error message is added in `KerasLayer` when using the newest version of TensorFlow with Keras 3 
  (which is not currently compatible with `KerasLayer`), linking to instructions to enable Keras 2.
  [(#5488)](https://github.com/PennyLaneAI/pennylane/pull/5488)

* `qml.ops.Conditional` now stores the `data`, `num_params`, and `ndim_param` attributes of
  the operator it wraps.
  [(#5473)](https://github.com/PennyLaneAI/pennylane/pull/5473)

* The `molecular_hamiltonian` function calls `PySCF` directly when `method='pyscf'` is selected.
  [(#5118)](https://github.com/PennyLaneAI/pennylane/pull/5118)

* `cache_execute` has been replaced with an alternate implementation based on 
  `@transform`.
  [(#5318)](https://github.com/PennyLaneAI/pennylane/pull/5318)

* QNodes now defer `diff_method` validation to the device under the new device 
  API.
  [(#5176)](https://github.com/PennyLaneAI/pennylane/pull/5176)

* The device test suite has been extended to cover gradient methods, templates 
  and arithmetic observables.
  [(#5273)](https://github.com/PennyLaneAI/pennylane/pull/5273)
  [(#5518)](https://github.com/PennyLaneAI/pennylane/pull/5518)

* A typo and string formatting mistake have been fixed in the error message for 
  `ClassicalShadow._convert_to_pauli_words` when the input is not a valid 
  `pauli_rep`.
  [(#5572)](https://github.com/PennyLaneAI/pennylane/pull/5572)

* Circuits running on `lightning.qubit` and that return `qml.state()` now preserve 
  the `dtype` when specified.
  [(#5547)](https://github.com/PennyLaneAI/pennylane/pull/5547)

<h3>Breaking changes 💔</h3>

* `qml.matrix()` called on the following will now raise an error if `wire_order` 
  is not specified:
  * tapes with more than one wire
  * quantum functions
  * `Operator` classes where `num_wires` does not equal to 1
  * QNodes if the device does not have wires specified.
  * `PauliWord`s and `PauliSentence`s with more than one wire.
  [(#5328)](https://github.com/PennyLaneAI/pennylane/pull/5328)
  [(#5359)](https://github.com/PennyLaneAI/pennylane/pull/5359)

* `single_tape_transform`, `batch_transform`, `qfunc_transform`, `op_transform`, `gradient_transform`
  and `hessian_transform` have been removed. Instead, switch to using the new `qml.transform` function. Please refer to
  `the transform docs <https://docs.pennylane.ai/en/stable/code/qml_transforms.html#custom-transforms>`_
  to see how this can be done.
  [(#5339)](https://github.com/PennyLaneAI/pennylane/pull/5339)

* Attempting to multiply `PauliWord` and `PauliSentence` with `*` will raise an error. Instead, use `@` to conform with the PennyLane convention.
  [(#5341)](https://github.com/PennyLaneAI/pennylane/pull/5341)

* `DefaultQubit` now uses a pre-emptive key-splitting strategy to avoid reusing JAX PRNG keys throughout a single `execute` call. 
  [(#5515)](https://github.com/PennyLaneAI/pennylane/pull/5515)

* `qml.pauli.pauli_mult` and `qml.pauli.pauli_mult_with_phase` have been removed. Instead, use `qml.simplify(qml.prod(pauli_1, pauli_2))` to get the reduced operator.
  [(#5324)](https://github.com/PennyLaneAI/pennylane/pull/5324)

  ```pycon
  >>> op = qml.simplify(qml.prod(qml.PauliX(0), qml.PauliZ(0)))
  >>> op
  -1j*(PauliY(wires=[0]))
  >>> [phase], [base] = op.terms()
  >>> phase, base
  (-1j, PauliY(wires=[0]))
  ```

* The `dynamic_one_shot` transform now uses sampling (`SampleMP`) to get back the values of the mid-circuit measurements.
  [(#5486)](https://github.com/PennyLaneAI/pennylane/pull/5486)

* `Operator` dunder methods now combine like-operator arithmetic classes via `lazy=False`. This reduces the chances of getting a `RecursionError` and makes nested
  operators easier to work with.
  [(#5478)](https://github.com/PennyLaneAI/pennylane/pull/5478)

* The private functions `_pauli_mult`, `_binary_matrix` and `_get_pauli_map` from the `pauli` module have been removed. The same functionality can be achieved using newer features in the `pauli` module.
  [(#5323)](https://github.com/PennyLaneAI/pennylane/pull/5323)

* `MeasurementProcess.name` and `MeasurementProcess.data` have been removed. Use `MeasurementProcess.obs.name` and `MeasurementProcess.obs.data` instead.
  [(#5321)](https://github.com/PennyLaneAI/pennylane/pull/5321)

* `Operator.validate_subspace(subspace)` has been removed. Instead, use `qml.ops.qutrit.validate_subspace(subspace)`.
  [(#5311)](https://github.com/PennyLaneAI/pennylane/pull/5311)

* The contents of `qml.interfaces` has been moved inside `qml.workflow`. The old import path no longer exists.
  [(#5329)](https://github.com/PennyLaneAI/pennylane/pull/5329)

* Since `default.mixed` does not support snapshots with measurements, attempting to do so will result in a `DeviceError` instead of getting the density matrix.
  [(#5416)](https://github.com/PennyLaneAI/pennylane/pull/5416)

* `LinearCombination._obs_data` has been removed. You can still use `LinearCombination.compare` to check mathematical equivalence between a `LinearCombination` and another operator.
  [(#5504)](https://github.com/PennyLaneAI/pennylane/pull/5504)

<h3>Deprecations 👋</h3>

* Accessing `qml.ops.Hamiltonian` is deprecated because it points to the old version of the class
  that may not be compatible with the new approach to operator arithmetic. Instead, using
  `qml.Hamiltonian` is recommended because it dispatches to the :class:`~.LinearCombination` class
  when the new approach to operator arithmetic is enabled. This will allow you to continue to use
  `qml.Hamiltonian` with existing code without needing to make any changes.
  [(#5393)](https://github.com/PennyLaneAI/pennylane/pull/5393)

* `qml.load` has been deprecated. Instead, please use the functions outlined in the [Importing workflows quickstart guide](https://docs.pennylane.ai/en/latest/introduction/importing_workflows.html).
  [(#5312)](https://github.com/PennyLaneAI/pennylane/pull/5312)

* Specifying `control_values` with a bit string in `qml.MultiControlledX` has been deprecated. Instead, use a list of booleans or 1s and 0s.
  [(#5352)](https://github.com/PennyLaneAI/pennylane/pull/5352)

* `qml.from_qasm_file` has been deprecated. Instead, please open the file and then load its content using `qml.from_qasm`.
  [(#5331)](https://github.com/PennyLaneAI/pennylane/pull/5331)

  ```pycon
  >>> with open("test.qasm", "r") as f:
  ...     circuit = qml.from_qasm(f.read())
  ```

<h3>Documentation 📝</h3>

* [A new page](https://docs.pennylane.ai/en/latest/code/qml_workflow.html#return-type-specification) 
  explaining the shapes and nesting of return types has been added.
  [(#5418)](https://github.com/PennyLaneAI/pennylane/pull/5418)

* Redundant documentation for the `evolve` function has been removed.
  [(#5347)](https://github.com/PennyLaneAI/pennylane/pull/5347)

* The final example in the `compile` docstring has been updated to use transforms correctly.
  [(#5348)](https://github.com/PennyLaneAI/pennylane/pull/5348)

* A link to the demos for using `qml.SpecialUnitary` and `qml.QNGOptimizer` has been added to their respective docstrings.
  [(#5376)](https://github.com/PennyLaneAI/pennylane/pull/5376)

* A code example in the `qml.measure` docstring has been added that showcases returning mid-circuit measurement statistics from QNodes.
  [(#5441)](https://github.com/PennyLaneAI/pennylane/pull/5441)

* The computational basis convention used for `qml.measure` — 0 and 1 rather than ±1 — has been clarified in its docstring.
  [(#5474)](https://github.com/PennyLaneAI/pennylane/pull/5474)

* A new *Release news* section has been added to the table of contents, containing release notes,
  deprecations, and other pages focusing on recent changes.
  [(#5548)](https://github.com/PennyLaneAI/pennylane/pull/5548)

* A summary of all changes has been added in the "Updated Operators" page in the new "Release news" section in the docs.
  [(#5483)](https://github.com/PennyLaneAI/pennylane/pull/5483)
  [(#5636)](https://github.com/PennyLaneAI/pennylane/pull/5636)

<h3>Bug fixes 🐛</h3>

* Patches the QNode so that parameter-shift will be considered best with lightning if
  `qml.metric_tensor` is in the transform program.
  [(#5624)](https://github.com/PennyLaneAI/pennylane/pull/5624)

* Stopped printing the ID of `qcut.MeasureNode` and `qcut.PrepareNode` in tape drawing.
  [(#5613)](https://github.com/PennyLaneAI/pennylane/pull/5613)
 
* Improves the error message for setting shots on the new device interface, or trying to access a property
  that no longer exists.
  [(#5616)](https://github.com/PennyLaneAI/pennylane/pull/5616)

* Fixed a bug where `qml.draw` and `qml.draw_mpl` incorrectly raised errors for circuits collecting statistics on mid-circuit measurements
  while using `qml.defer_measurements`.
  [(#5610)](https://github.com/PennyLaneAI/pennylane/pull/5610)

* Using shot vectors with `param_shift(... broadcast=True)` caused a bug. This combination is no longer supported
  and will be added again in the next release. Fixed a bug with custom gradient recipes that only consist of unshifted terms.
  [(#5612)](https://github.com/PennyLaneAI/pennylane/pull/5612)
  [(#5623)](https://github.com/PennyLaneAI/pennylane/pull/5623)

* `qml.counts` now returns the same keys with `dynamic_one_shot` and `defer_measurements`.
  [(#5587)](https://github.com/PennyLaneAI/pennylane/pull/5587)

* `null.qubit` now automatically supports any operation without a decomposition.
  [(#5582)](https://github.com/PennyLaneAI/pennylane/pull/5582)

* Fixed a bug where the shape and type of derivatives obtained by applying a gradient transform to
  a QNode differed based on whether the QNode uses classical coprocessing.
  [(#4945)](https://github.com/PennyLaneAI/pennylane/pull/4945)

* `ApproxTimeEvolution`, `CommutingEvolution`, `QDrift`, and `TrotterProduct` 
  now de-queue their input observable.
  [(#5524)](https://github.com/PennyLaneAI/pennylane/pull/5524)

* (In)equality of `qml.HilbertSchmidt` instances is now reported correctly by `qml.equal`.
  [(#5538)](https://github.com/PennyLaneAI/pennylane/pull/5538)

* `qml.ParticleConservingU1` and `qml.ParticleConservingU2` no longer raise an error when the initial state is not specified but default to the all-zeros state.
  [(#5535)](https://github.com/PennyLaneAI/pennylane/pull/5535)

* `qml.counts` no longer returns negative samples when measuring 8 or more wires.
  [(#5544)](https://github.com/PennyLaneAI/pennylane/pull/5544)
  [(#5556)](https://github.com/PennyLaneAI/pennylane/pull/5556)

* The `dynamic_one_shot` transform now works with broadcasting.
  [(#5473)](https://github.com/PennyLaneAI/pennylane/pull/5473)

* Diagonalizing gates are now applied when measuring `qml.probs` on non-computational basis states on a Lightning device.
  [(#5529)](https://github.com/PennyLaneAI/pennylane/pull/5529)

* `two_qubit_decomposition` no longer diverges at a special case of a unitary matrix.
  [(#5448)](https://github.com/PennyLaneAI/pennylane/pull/5448)

* The `qml.QNSPSAOptimizer` now correctly handles optimization for legacy devices that do not follow the new device API.
  [(#5497)](https://github.com/PennyLaneAI/pennylane/pull/5497)

* Operators applied to all wires are now drawn correctly in a circuit with mid-circuit measurements.
  [(#5501)](https://github.com/PennyLaneAI/pennylane/pull/5501)

* Fixed a bug where certain unary mid-circuit measurement expressions would raise an uncaught error.
  [(#5480)](https://github.com/PennyLaneAI/pennylane/pull/5480)

* Probabilities now sum to 1 when using the `torch` interface with `default_dtype` set to `torch.float32`. 
  [(#5462)](https://github.com/PennyLaneAI/pennylane/pull/5462)

* Tensorflow can now handle devices with `float32` results but `float64` input parameters.
  [(#5446)](https://github.com/PennyLaneAI/pennylane/pull/5446)

* Fixed a bug where the `argnum` keyword argument of `qml.gradients.stoch_pulse_grad` references the wrong parameters in a tape,
  creating an inconsistency with other differentiation methods and preventing some use cases.
  [(#5458)](https://github.com/PennyLaneAI/pennylane/pull/5458)

* Bounded value failures due to numerical noise with calls to `np.random.binomial` is now avoided.
  [(#5447)](https://github.com/PennyLaneAI/pennylane/pull/5447)

* Using `@` with legacy Hamiltonian instances now properly de-queues the previously existing operations.
  [(#5455)](https://github.com/PennyLaneAI/pennylane/pull/5455)

* The `QNSPSAOptimizer` now properly handles differentiable parameters, resulting in being able to use it for more than one optimization step.
  [(#5439)](https://github.com/PennyLaneAI/pennylane/pull/5439)

* The QNode interface now resets if an error occurs during execution.
  [(#5449)](https://github.com/PennyLaneAI/pennylane/pull/5449)

* Failing tests due to changes with Lightning's adjoint diff pipeline have been fixed.
  [(#5450)](https://github.com/PennyLaneAI/pennylane/pull/5450)

* Failures occurring when making autoray-dispatched calls to Torch with paired CPU data have been fixed.
  [(#5438)](https://github.com/PennyLaneAI/pennylane/pull/5438)

* `jax.jit` now works with `qml.sample` with a multi-wire observable.
  [(#5422)](https://github.com/PennyLaneAI/pennylane/pull/5422)

* `qml.qinfo.quantum_fisher` now works with non-`default.qubit` devices.
  [(#5423)](https://github.com/PennyLaneAI/pennylane/pull/5423)

* We no longer perform unwanted `dtype` promotion in the `pauli_rep` of `SProd` instances when using Tensorflow.
  [(#5246)](https://github.com/PennyLaneAI/pennylane/pull/5246)

* Fixed `TestQubitIntegration.test_counts` in `tests/interfaces/test_jax_qnode.py` to always produce counts for all
  outcomes.
  [(#5336)](https://github.com/PennyLaneAI/pennylane/pull/5336)

* Fixed `PauliSentence.to_mat(wire_order)` to support identities with wires.
  [(#5407)](https://github.com/PennyLaneAI/pennylane/pull/5407)

* `CompositeOp.map_wires` now correctly maps the `overlapping_ops` property.
  [(#5430)](https://github.com/PennyLaneAI/pennylane/pull/5430)

* `DefaultQubit.supports_derivatives` has been updated to correctly handle circuits containing mid-circuit measurements and adjoint
  differentiation.
  [(#5434)](https://github.com/PennyLaneAI/pennylane/pull/5434)

* `SampleMP`, `ExpectationMP`, `CountsMP`, and `VarianceMP` constructed with `eigvals` can now properly process samples.
  [(#5463)](https://github.com/PennyLaneAI/pennylane/pull/5463)

* Fixed a bug in `hamiltonian_expand` that produces incorrect output dimensions when shot vectors are combined with parameter broadcasting.
  [(#5494)](https://github.com/PennyLaneAI/pennylane/pull/5494)

* `default.qubit` now allows measuring `Identity` on no wires and observables containing `Identity` on
  no wires.
  [(#5570)](https://github.com/PennyLaneAI/pennylane/pull/5570/)

* Fixed a bug where `TorchLayer` does not work with shot vectors.
  [(#5492)](https://github.com/PennyLaneAI/pennylane/pull/5492)

* Fixed a bug where the output shape of a QNode returning a list containing a single measurement is incorrect when combined with shot vectors.
  [(#5492)](https://github.com/PennyLaneAI/pennylane/pull/5492)

* Fixed a bug in `qml.math.kron` that makes Torch incompatible with NumPy.
  [(#5540)](https://github.com/PennyLaneAI/pennylane/pull/5540)

* Fixed a bug in `_group_measurements` that fails to group measurements with commuting observables when they are operands of `Prod`.
  [(#5525)](https://github.com/PennyLaneAI/pennylane/pull/5525)

* `qml.equal` can now be used with sums and products that contain operators on no wires like `I` and `GlobalPhase`.
  [(#5562)](https://github.com/PennyLaneAI/pennylane/pull/5562)

* `CompositeOp.has_diagonalizing_gates` now does a more complete check of the base operators to ensure consistency 
  between `op.has_diagonalzing_gates` and `op.diagonalizing_gates()`
  [(#5603)](https://github.com/PennyLaneAI/pennylane/pull/5603)

* Updated the `method` kwarg of `qml.TrotterProduct().error()` to be more clear that we are computing upper-bounds.
  [(#5637)](https://github.com/PennyLaneAI/pennylane/pull/5637)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Tarun Kumar Allamsetty,
Guillermo Alonso,
Mikhail Andrenkov,
Utkarsh Azad,
Gabriel Bottrill,
Thomas Bromley,
Astral Cai,
Diksha Dhawan,
Isaac De Vlugt,
Amintor Dusko,
Pietropaolo Frisoni,
Lillian M. A. Frederiksen,
Diego Guala,
Austin Huang,
Soran Jahangiri,
Korbinian Kottmann,
Christina Lee,
Vincent Michaud-Rioux,
Mudit Pandey,
Kenya Sakka,
Jay Soni,
Matthew Silverman,
David Wierichs.
