:orphan:

# Release 0.28.0-dev (development release)

<h3>New features since last release</h3>

* Support for getting the ZX calculus graph of a circuit with the PyZX framework and converting
  a PyZX graph back into a PennyLane circuit.
  [#3446](https://github.com/PennyLaneAI/pennylane/pull/3446)
  
  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.transforms.to_zx
  @qml.qnode(device=dev)
  def circuit(p):
      qml.RZ(p[0], wires=1),
      qml.RZ(p[1], wires=1),
      qml.RX(p[2], wires=0),
      qml.PauliZ(wires=0),
      qml.RZ(p[3], wires=1),
      qml.PauliX(wires=1),
      qml.CNOT(wires=[0, 1]),
      qml.CNOT(wires=[1, 0]),
      qml.SWAP(wires=[0, 1]),
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  ```
  ```pycon
  >>> params = [5 / 4 * np.pi, 3 / 4 * np.pi, 0.1, 0.3]
  >>> circuit(params)
  Graph(20 vertices, 23 edges)
  ```
  
* Added ability to create expressions from mid-circuit measurements.
  [#3159](https://github.com/PennyLaneAI/pennylane/pull/3159)

  ```python
  def circuit():
      qml.Hadamard(wires=0)
      qml.Hadamard(wires=1)
      m0 = qml.measure(wires=0)
      m1 = qml.measure(wires=1)
      combined = 2 * m1 + m0
      qml.cond(combined == 2, qml.RX)(1.3, wires=2)
      return qml.probs(wires=2)
  ```
  ```pycon 
  >>> circuit()
  [0.90843735 0.09156265]  
  ```

* New gradient transform `qml.gradients.spsa_grad` based on the idea of SPSA.
  [#3366](https://github.com/PennyLaneAI/pennylane/pull/3366)

  This new transform allows users to compute a single estimate of a quantum gradient
  using simultaneous perturbation of parameters and a stochastic approximation.
  Given some QNode `circuit` that takes, say, an argument `x`, the approximate
  gradient can be computed via

  ```pycon
  >>> dev = qml.device("default.qubit", wires=2)
  >>> x = pnp.array(0.4, requires_grad=True)
  >>> @qml.qnode(dev)
  ... def circuit(x):
  ...     qml.RX(x, 0)
  ...     qml.RX(x, 1)
  ...     return qml.expval(qml.PauliZ(0))
  >>> grad_fn = qml.gradients.spsa_grad(circuit, h=0.1, num_directions=1)
  >>> grad_fn(x)
  array(-0.38876964)
  ```

  The argument `num_directions` determines how many directions of simultaneous
  perturbation are used and therefore the number of circuit evaluations, up
  to a prefactor. See the
  [SPSA gradient transform documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.gradients.spsa_grad.html) for details.
  Note: The full SPSA optimization method is already available as `SPSAOptimizer`.

* Add the controlled CZ gate: CCZ.

  ```pycon
  >>> ccz = qml.CCZ(wires=[0, 1, 2])
  >>> matrix = ccz.compute_matrix()
  [[ 1  0  0  0  0  0  0  0]
   [ 0  1  0  0  0  0  0  0]
   [ 0  0  1  0  0  0  0  0]
   [ 0  0  0  1  0  0  0  0]
   [ 0  0  0  0  1  0  0  0]
   [ 0  0  0  0  0  1  0  0]
   [ 0  0  0  0  0  0  1  0]
   [ 0  0  0  0  0  0  0 -1]]
  ```

  [(#3408)](https://github.com/PennyLaneAI/pennylane/pull/3408)

* Add the controlled Hadamard gate.

  ```pycon
  >>> ch = qml.CH(wires=[0, 1])
  >>> matrix = ch.compute_matrix()
  [[ 1.          0.          0.          0.        ]
   [ 0.          1.          0.          0.        ]
   [ 0.          0.          0.70710678  0.70710678]
   [ 0.          0.          0.70710678 -0.70710678]]
  ```

  [(#3408)](https://github.com/PennyLaneAI/pennylane/pull/3408)

* Support custom measurement processes:
  * `SampleMeasurement`, `StateMeasurement` and `MeasurementTransform` classes have been added.
    They contain an abstract method to process samples/quantum state/quantum script.

  * Add `ExpectationMP`, `SampleMP`, `VarianceMP`, `ProbabilityMP`, `CountsMP`, `StateMP`,
    `VnEntropyMP`, `MutualInfoMP`, `ClassicalShadowMP` and `ShadowExpvalMP` classes.

  * Allow the execution of `SampleMeasurement`, `StateMeasurement` and `MeasurementTransform`
    measurement processes in `QubitDevice`. Also allow devices to override measurement processes by
    adding a `measurement_map` attribute, which should contain the measurement class as key and
    the method name as value.
    [(#3286)](https://github.com/PennyLaneAI/pennylane/pull/3286)
    [(#3388)](https://github.com/PennyLaneAI/pennylane/pull/3388)
    [(#3343)](https://github.com/PennyLaneAI/pennylane/pull/3343)
    [(#3288)](https://github.com/PennyLaneAI/pennylane/pull/3288)
    [(#3312)](https://github.com/PennyLaneAI/pennylane/pull/3312)
    [(#3287)](https://github.com/PennyLaneAI/pennylane/pull/3287)
    [(#3292)](https://github.com/PennyLaneAI/pennylane/pull/3292)
    [(#3287)](https://github.com/PennyLaneAI/pennylane/pull/3287)
    [(#3326)](https://github.com/PennyLaneAI/pennylane/pull/3326)
    [(#3327)](https://github.com/PennyLaneAI/pennylane/pull/3327)
    [(#3388)](https://github.com/PennyLaneAI/pennylane/pull/3388)
    [(#3388)](https://github.com/PennyLaneAI/pennylane/pull/3388)
    [(#3439)](https://github.com/PennyLaneAI/pennylane/pull/3439)
    [(#3466)](https://github.com/PennyLaneAI/pennylane/pull/3466)
    [(#3503)](https://github.com/PennyLaneAI/pennylane/pull/3503)

* Functionality for fetching symbols and geometry of a compound from the PubChem Database using `qchem.mol_data`.
  [(#3289)](https://github.com/PennyLaneAI/pennylane/pull/3289)
  [(#3378)](https://github.com/PennyLaneAI/pennylane/pull/3378)

  ```pycon
  >>> mol_data("BeH2")
  (['Be', 'H', 'H'],
  array([[ 4.79405604,  0.29290815,  0.        ],
         [ 3.77946   , -0.29290815,  0.        ],
         [ 5.80884105, -0.29290815,  0.        ]]))

  >>> mol_data(223, "CID")
  (['N', 'H', 'H', 'H', 'H'],
  array([[ 4.79404621,  0.        ,  0.        ],
         [ 5.80882913,  0.5858151 ,  0.        ],
         [ 3.77945225, -0.5858151 ,  0.        ],
         [ 4.20823111,  1.01459396,  0.        ],
         [ 5.3798613 , -1.01459396,  0.        ]]))
  ```

* New basis sets, `6-311g` and `CC-PVDZ`, are added to the qchem basis set repo.
  [#3279](https://github.com/PennyLaneAI/pennylane/pull/3279)

* Add `sum_expand` function, which splits a tape measuring a `Sum` expectation into mutliple tapes
  of summand expectations, and provides a function to recombine the results.
  [#3230](https://github.com/PennyLaneAI/pennylane/pull/3230)

* Added a `pauli_decompose()` which takes a hermitian matrix and decomposes it in the
  Pauli basis, returning it either as a `Hamiltonian` or `PauliSentence` instance.
  [(#3384)](https://github.com/PennyLaneAI/pennylane/pull/3384)

  ```pycon
  >>> mat = np.array([[1, 1], [1, -1]])
  >>> h = qml.pauli_decompose(mat)
  >>> print(h)
    (1.0) [X0]
  + (1.0) [Z0]
  >>> ps = qml.pauli_decompose(mat, pauli=True, wire_order=["a"])
  >>> print(ps)
  1.0 * X(a)
  + 1.0 * Z(a)
  ```

* New `pauli_sentence()` function which takes native `Operator` or `Hamiltonian`
  instances representing a linear combination of Pauli words and returns
  the equivalent `PauliSentence`.
  [(#3389)](https://github.com/PennyLaneAI/pennylane/pull/3389)

  ```pycon
  >>> op = 1.23 * qml.prod(qml.PauliX(wires=0), qml.PauliZ(wires=1))
  >>> op
  1.23*(PauliX(wires=[0]) @ PauliZ(wires=[1]))
  >>> h = qml.Hamiltonian([1.23], [qml.PauliX(wires=0) @ qml.PauliZ(wires=1)])
  >>> print(h)
    (1.23) [X0 Z1]
  >>> qml.pauli.pauli_sentence(op)
  1.23 * Z(1) @ X(0)
  >>> qml.pauli.pauli_sentence(h)
  1.23 * X(0) @ Z(1)
  ```

* Added two new methods `operation()`, `hamiltonian()` for both `PauliSentence` and `PauliWord` classes to generate an equivalent PennyLane
  `Operation` or `Hamiltonian` instance from a `PauliSentence` or `PauliWord` one.
  [(#3391)](https://github.com/PennyLaneAI/pennylane/pull/3391)

  ```pycon
  >>> pw = qml.pauli.PauliWord({0: 'X', 1: 'Y'})
  >>> print(pw.operation())
  PauliX(wires=[0]) @ PauliY(wires=[1])
  >>> print(pw.hamiltonian())
    (1) [X0 Y1]
  >>>
  >>> ps = qml.pauli.PauliSentence({pw: -1.23})
  >>> print(ps.operation())
  -1.23*(PauliX(wires=[0]) @ PauliY(wires=[1]))
  >>> print(ps.hamiltonian())
    (-1.23) [X0 Y1]
  ```

* Added a new gate operation `FermionicSWAP`, which implements the exchange of spin orbitals
  representing fermionic-modes while maintaining proper anti-symmetrization.
  [(#3380)](https://github.com/PennyLaneAI/pennylane/pull/3380)

  An example circuit that uses `FermionicSWAP` operation is:

  ```python
  dev = qml.device('default.qubit', wires=2)

  @qml.qnode(dev)
  def circuit(phi):
      qml.BasisState(np.array([0, 1]), wires=[0, 1])
      qml.FermionicSWAP(phi, wires=[0, 1])
      return qml.state()
  ```

  If we run this circuit, we will get the following output

  ```pycon
  >>> circuit(0.1)
  array([0.+0.j, 0.9975+0.04992j, 0.0025-0.04992j, 0.+0.j])
  ```

* New parametric qubit ops `qml.CPhaseShift00`, `qml.CPhaseShift01` and `qml.CPhaseShift10` which perform a phaseshift, similar to `qml.ControlledPhaseShift` but on different positions of the state vector.
  [(#2715)](https://github.com/PennyLaneAI/pennylane/pull/2715)

* Support for purity computation is added. The `qml.math.purity` function computes the purity from a state vector or a density matrix:

  [(#3290)](https://github.com/PennyLaneAI/pennylane/pull/3290)

  ```pycon
  >>> x = [1, 0, 0, 1] / np.sqrt(2)
  >>> qml.math.purity(x, [0, 1])
  1.0
  >>> qml.math.purity(x, [0])
  0.5

  >>> x = [[1 / 2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1 / 2]]
  >>> qml.math.purity(x, [0, 1])
  0.5
  ```

  The `qml.qinfo.purity` can be used to transform a QNode returning a state to a function that returns the purity:

  ```python3
  dev = qml.device("default.mixed", wires=2)

  @qml.qnode(dev)
  def circuit(x):
    qml.IsingXX(x, wires=[0, 1])
    return qml.state()
  ```

  ```pycon
  >>> qml.qinfo.purity(circuit, wires=[0])(np.pi / 2)
  0.5
  >>> qml.qinfo.purity(circuit, wires=[0, 1])(np.pi / 2)
  1.0
  ```

  Taking the gradient is also supported:

  ```pycon
  >>> param = np.array(np.pi / 4, requires_grad=True)
  >>> qml.grad(qml.qinfo.purity(circuit, wires=[0]))(param)
  -0.5
  ```
  
* Added operation `qml.THadamard`, which is the qutrit Hadamard gate. The operation accepts a `subspace`
  keyword argument which determines which variant of the qutrit Hadamard to use.
  [#3340](https://github.com/PennyLaneAI/pennylane/pull/3340)

* New operation `Evolution` defines the exponential of an operator $\hat{O}$ of the form $e^{ix\hat{O}}$, with a single 
  trainable parameter, x. Limiting to a single trainable parameter allows the use of `qml.gradient.param_shift` to 
  find the gradient with respect to the parameter x.
  [(#3375)](https://github.com/PennyLaneAI/pennylane/pull/3375)

  This example circuit uses the `Evolution` operation to define $e^{-\frac{i}{2}\phi\hat{\sigma}_x}$ and finds a 
  gradient using parameter shift:

  ```python
  dev = qml.device('default.qubit', wires=2)
  
  @qml.qnode(dev, diff_method=qml.gradients.param_shift)
  def circuit(phi):
      Evolution(qml.PauliX(0), -.5 * phi)
      return qml.expval(qml.PauliZ(0))
  ```
  
  If we run this circuit, we will get the following output

  ```pycon
  >>> phi = np.array(1.2)
  >>> circuit(phi)
  tensor(0.36235775, requires_grad=True)
  >>> qml.grad(circuit)(phi)
  -0.9320390495504149
  ```

<h3>Improvements</h3>

* The `qml.is_pauli_word` now supports instances of `Hamiltonian`.
  [(#3389)](https://github.com/PennyLaneAI/pennylane/pull/3389)

* Support calling `qml.probs()`, `qml.counts()` and `qml.sample()` with no arguments to measure all
  wires. Calling any measurement with an empty wire list will raise an error.
  [#3299](https://github.com/PennyLaneAI/pennylane/pull/3299)

* Made `gradients.finite_diff` more convenient to use with custom data type observables/devices.
  [(#3426)](https://github.com/PennyLaneAI/pennylane/pull/3426)

* The `qml.ISWAP` gate is now natively supported on `default.mixed`, improving on its efficiency.
  [(#3284)](https://github.com/PennyLaneAI/pennylane/pull/3284)

* Added more input validation to `hamiltonian_expand` such that Hamiltonian objects with no terms raise an error.
  [(#3339)](https://github.com/PennyLaneAI/pennylane/pull/3339)

* Continuous integration checks are now performed for Python 3.11 and Torch v1.13. Python 3.7 is dropped.
  [(#3276)](https://github.com/PennyLaneAI/pennylane/pull/3276)

* `qml.Tracker` now also logs results in `tracker.history` when tracking execution of a circuit.
   [(#3306)](https://github.com/PennyLaneAI/pennylane/pull/3306)

* Improve performance of `Wires.all_wires`.
  [(#3302)](https://github.com/PennyLaneAI/pennylane/pull/3302)

* A representation has been added to the `Molecule` class.
  [(#3364)](https://github.com/PennyLaneAI/pennylane/pull/3364)

* Add detail to the error message when the `insert` transform
  fails to diagonalize non-qubit-wise-commuting observables.
  [(#3381)](https://github.com/PennyLaneAI/pennylane/pull/3381)

* Extended the `qml.equal` function to `Hamiltonian` and `Tensor` objects.
  [(#3390)](https://github.com/PennyLaneAI/pennylane/pull/3390)

* Remove private `_wires` setter from the `Controlled.map_wires` method.
  [(#3405)](https://github.com/PennyLaneAI/pennylane/pull/3405)

* `QuantumTape._process_queue` has been moved to `qml.queuing.process_queue` to disentangle
  its functionality from the `QuantumTape` class.
  [(#3401)](https://github.com/PennyLaneAI/pennylane/pull/3401)

* Adds `qml.tape.make_qscript` for converting a quantum function into a quantum script.
  Replaces `qml.transforms.make_tape` with `make_qscript`.
  [(#3429)](https://github.com/PennyLaneAI/pennylane/pull/3429)

* Extended the functionality of `qml.matrix` to qutrits.
  [(#3460)](https://github.com/PennyLaneAI/pennylane/pull/3460)

* File `qcut.py` in `qml.transforms` reorganized into multiple files in `qml.transforms.qcut`
  [3413](https://github.com/PennyLaneAI/pennylane/pull/3413)

* Add a UserWarning when creating a `Tensor` object with overlapping wires,
  informing that this can in some cases lead to undefined behaviour.
  [(#3459)](https://github.com/PennyLaneAI/pennylane/pull/3459)

* Extended the `qml.equal` function to `Controlled` and `ControlledOp` objects.
  [(#3463)](https://github.com/PennyLaneAI/pennylane/pull/3463)

* Replace (almost) all instances of `with QuantumTape()` with `QuantumScript` construction.
  [(#3454)](https://github.com/PennyLaneAI/pennylane/pull/3454)
  
* Added `validate_subspace` static method to `qml.Operator` to check the validity of the subspace of certain
  qutrit operations.
  [#3340](https://github.com/PennyLaneAI/pennylane/pull/3340)

* Extended the `qml.equal` function to `Pow`, `SProd`, `Exp` and `Adjoint` objects.
  [(#3471)](https://github.com/PennyLaneAI/pennylane/pull/3471)

* Adds support for devices disregarding observable grouping indices in Hamiltonians through
  the optional `use_grouping` attribute.
  [(#3456)](https://github.com/PennyLaneAI/pennylane/pull/3456)

* Updated `zyz_decomposition` function such that it now supports broadcast operators. This
  means that single-qubit `QubitUnitary` operators, instantiated from a batch of unitaries,
  can now be decomposed.
  [(#3477)](https://github.com/PennyLaneAI/pennylane/pull/3477)

* Reduce usage of `MeasurementProcess.return_type`. Use `isinstance` checks instead.
  [(#3399)](https://github.com/PennyLaneAI/pennylane/pull/3399)

* Improved the performance of executing circuits under the `jax.vmap` transformation, which can now leverage the batch-execution capabilities of some devices. [(#3452)](https://github.com/PennyLaneAI/pennylane/pull/3452)

* The tolerance for converting openfermion Hamiltonian complex coefficients to real is modified to
  prevent conversion errors.
  [(#3363)](https://github.com/PennyLaneAI/pennylane/pull/3363)

<h4>Return types project</h4>

* The autograd interface for the new return types now supports devices with shot vectors.
  [(#3374)](https://github.com/PennyLaneAI/pennylane/pull/3374)

  Example with a single measurement:

  ```python
  dev = qml.device("default.qubit", wires=1, shots=[1000, 2000, 3000])

  @qml.qnode(dev, diff_method="parameter-shift")
  def circuit(a):
      qml.RY(a, wires=0)
      qml.RX(0.2, wires=0)
      return qml.expval(qml.PauliZ(0))

  def cost(a):
      return qml.math.stack(circuit(a))
  ```

  ```pycon
  >>> qml.enable_return()
  >>> a = np.array(0.4)
  >>> circuit(a)
  (array(0.902), array(0.922), array(0.896))
  >>> cost(a)
  array([0.9       , 0.907     , 0.89733333])
  >>> qml.jacobian(cost)(a)
  array([-0.391     , -0.389     , -0.38433333])
  ```

  Example with multiple measurements:

  ```python
  dev = qml.device("default.qubit", wires=2, shots=[1000, 2000, 3000])

  @qml.qnode(dev, diff_method="parameter-shift")
  def circuit(a):
      qml.RY(a, wires=0)
      qml.RX(0.2, wires=0)
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.PauliZ(0)), qml.probs([0, 1])

  def cost(a):
      res = circuit(a)
      return qml.math.stack([qml.math.hstack(r) for r in res])
  ```

  ```pycon
  >>> circuit(a)
  ((array(0.904), array([0.952, 0.   , 0.   , 0.048])),
   (array(0.915), array([0.9575, 0.    , 0.    , 0.0425])),
   (array(0.902), array([0.951, 0.   , 0.   , 0.049])))
  >>> cost(a)
  array([[0.91      , 0.955     , 0.        , 0.        , 0.045     ],
         [0.895     , 0.9475    , 0.        , 0.        , 0.0525    ],
         [0.90666667, 0.95333333, 0.        , 0.        , 0.04666667]])
  >>> qml.jacobian(cost)(a)
  array([[-0.37      , -0.185     ,  0.        ,  0.        ,  0.185     ],
         [-0.409     , -0.2045    ,  0.        ,  0.        ,  0.2045    ],
         [-0.37133333, -0.18566667,  0.        ,  0.        ,  0.18566667]])
  ```

* The TensorFlow interface for the new return types now supports devices with shot vectors.
  [(#3400)](https://github.com/PennyLaneAI/pennylane/pull/3400)

  Example with a single measurement:

  ```python
  dev = qml.device("default.qubit", wires=1, shots=[1000, 2000, 3000])

  @qml.qnode(dev, diff_method="parameter-shift", interface="tf")
  def circuit(a):
      qml.RY(a, wires=0)
      qml.RX(0.2, wires=0)
      return qml.expval(qml.PauliZ(0))
  ```

  ```
  >>> qml.enable_return()
  >>> a = tf.Variable(0.4)
  >>> with tf.GradientTape() as tape:
  ...     res = circuit(a)
  ...     res = tf.stack(res)
  ...
  >>> res
  <tf.Tensor: shape=(3,), dtype=float64, numpy=array([0.902     , 0.904     , 0.89533333])>
  >>> tape.jacobian(res, a)
  <tf.Tensor: shape=(3,), dtype=float64, numpy=array([-0.365     , -0.3765    , -0.37533333])>
  ```

  Example with multiple measurements:

  ```python
  dev = qml.device("default.qubit", wires=2, shots=[1000, 2000, 3000])

  @qml.qnode(dev, diff_method="parameter-shift", interface="tf")
  def circuit(a):
      qml.RY(a, wires=0)
      qml.RX(0.2, wires=0)
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.PauliZ(0)), qml.probs([0, 1])
  ```

  ```
  >>> with tf.GradientTape() as tape:
  ...     res = circuit(a)
  ...     res = tf.stack([tf.experimental.numpy.hstack(r) for r in res])
  ...
  >>> res
  <tf.Tensor: shape=(3, 5), dtype=float64, numpy=
  array([[0.902, 0.951, 0.   , 0.   , 0.049],
         [0.898, 0.949, 0.   , 0.   , 0.051],
         [0.892, 0.946, 0.   , 0.   , 0.054]])>
  >>> tape.jacobian(res, a)
  <tf.Tensor: shape=(3, 5), dtype=float64, numpy=
  array([[-0.345     , -0.1725    ,  0.        ,  0.        ,  0.1725    ],
         [-0.383     , -0.1915    ,  0.        ,  0.        ,  0.1915    ],
         [-0.38466667, -0.19233333,  0.        ,  0.        ,  0.19233333]])>
  ```

* Thy PyTorch interface supports the new return system and users can use jacobian and hessian using custom differentiation
  methods (e.g., parameter-shift, finite difference or adjoint).
  [(#3416)](https://github.com/PennyLaneAI/pennylane/pull/3414)
  
  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev, diff_method="parameter-shift", interface="torch")
  def circuit(a, b):
      qml.RY(a, wires=0)
      qml.RX(b, wires=1)
      qml.CNOT(wires=[0, 1])>
      return qml.expval(qml.PauliZ(0)), qml.probs([0, 1])
  ```

  ```pycon
  >>> a = torch.tensor(0.1, requires_grad=True)
  >>> b = torch.tensor(0.2, requires_grad=True)
  >>> torch.autograd.functional.jacobian(circuit, (a, b))
  ((tensor(-0.0998), tensor(0.)), (tensor([-0.0494, -0.0005,  0.0005,  0.0494]), tensor([-0.0991,  0.0991,  0.0002, -0.0002])))
  ```

* The JAX-JIT interface now supports first-order gradient computation with the new return types system.
  [(#3235)](https://github.com/PennyLaneAI/pennylane/pull/3235)
  [(#3445)](https://github.com/PennyLaneAI/pennylane/pull/3445)

  ```python
  import pennylane as qml
  import jax
  from jax import numpy as jnp

  jax.config.update("jax_enable_x64", True)

  qml.enable_return()

  dev = qml.device("lightning.qubit", wires=2)

  @jax.jit
  @qml.qnode(dev, interface="jax-jit", diff_method="parameter-shift")
  def circuit(a, b):
      qml.RY(a, wires=0)
      qml.RX(b, wires=0)
      return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

  a, b = jnp.array(1.0), jnp.array(2.0)
  ```

  ```pycon
  >>> jax.jacobian(circuit, argnums=[0, 1])(a, b)
  ((DeviceArray(0.35017549, dtype=float64, weak_type=True),
  DeviceArray(-0.4912955, dtype=float64, weak_type=True)),
  (DeviceArray(5.55111512e-17, dtype=float64, weak_type=True),
  DeviceArray(0., dtype=float64, weak_type=True)))
  ```

* Updated `qml.transforms.split_non_commuting` to support the new return types.
  [(#3414)](https://github.com/PennyLaneAI/pennylane/pull/3414)

* Updated `qml.transforms.mitigate_with_zne` to support the new return types.
  [(#3415)](https://github.com/PennyLaneAI/pennylane/pull/3415)

* Updated `qml.transforms.metric_tensor`, `qml.transforms.adjoint_metric_tensor`,
  `qml.qinfo.classical_fisher`, and `qml.qinfo.quantum_fisher` to support the new return types.
  [(#3449)](https://github.com/PennyLaneAI/pennylane/pull/3449)

* Updated `qml.transforms.batch_params` and `qml.transforms.batch_input` to support the new return types
  [(#3431)](https://github.com/PennyLaneAI/pennylane/pull/3431)

* Updated `qml.transforms.cut_circuit` and `qml.transforms.cut_circuit_mc` to
  support the new return types.
  [(#3346)](https://github.com/PennyLaneAI/pennylane/pull/3346)

* Update `OperationRecorder` to inherit from `AnnotatedQueue` and `QuantumScript` instead of `QuantumTape`.
  [(#3496)](https://github.com/PennyLaneAI/pennylane/pull/3496)

<h3>Breaking changes</h3>

* The `log_base` attribute has been moved from `MeasurementProcess` to the new `VnEntropyMP` and
  `MutualInfoMP` classes, which inherit from `MeasurementProcess`.
  [(#3326)](https://github.com/PennyLaneAI/pennylane/pull/3326)

* Python 3.7 support is no longer maintained.
  [(#3276)](https://github.com/PennyLaneAI/pennylane/pull/3276)

* Removed `qml.utils.decompose_hamiltonian()`, please use `qml.pauli_decompose()` instead.
  [(#3384)](https://github.com/PennyLaneAI/pennylane/pull/3384)

* Instead of having an `OrderedDict` attribute called `_queue`, `AnnotatedQueue` now inherits from
  `OrderedDict` and encapsulates the queue. Consequentially, this also applies to the `QuantumTape`
  class which inherits from `AnnotatedQueue`.
  [(#3401)](https://github.com/PennyLaneAI/pennylane/pull/3401)

* Change class name `ShadowMeasurementProcess` to `ClassicalShadowMP`
  [(#3388)](https://github.com/PennyLaneAI/pennylane/pull/3388)

* The method `qml.Operation.get_parameter_shift` is removed. The `gradients` module should be used
  for general parameter-shift rules instead.
  [(#3419)](https://github.com/PennyLaneAI/pennylane/pull/3419)

* Changed the signature of the `QubitDevice.statistics` method from

  ```python
  def statistics(self, observables, shot_range=None, bin_size=None, circuit=None):
  ```

  to

  ```python
  def statistics(self, circuit: QuantumScript, shot_range=None, bin_size=None):
  ```

  [(#3421)](https://github.com/PennyLaneAI/pennylane/pull/3421)

* The `MeasurementProcess.return_type` argument has been removed from the `__init__` method. Now
  it is a property of the class.
  [(#3434)](https://github.com/PennyLaneAI/pennylane/pull/3434)

* The `MeasurementProcess` class is now an abstract class.
  [(#3434)](https://github.com/PennyLaneAI/pennylane/pull/3434)

<h3>Deprecations</h3>

Deprecations cycles are tracked at [doc/developement/deprecations.rst](https://docs.pennylane.ai/en/latest/development/deprecations.html).

* The following deprecated methods are removed:
  [(#3281)](https://github.com/PennyLaneAI/pennylane/pull/3281/)

  * `qml.tape.get_active_tape`: Use `qml.QueuingManager.active_context()`
  * `qml.transforms.qcut.remap_tape_wires`: Use `qml.map_wires`
  * `qml.tape.QuantumTape.inv()`: Use `qml.tape.QuantumTape.adjoint()`
  * `qml.tape.stop_recording()`: Use `qml.QueuingManager.stop_recording()`
  * `qml.tape.QuantumTape.stop_recording()`: Use `qml.QueuingManager.stop_recording()`
  * `qml.QueuingContext` is now `qml.QueuingManager`
  * `QueuingManager.safe_update_info` and `AnnotatedQueue.safe_update_info`: Use plain `update_info`

* `qml.transforms.measurement_grouping` has been deprecated. Use `qml.transforms.hamiltonian_expand` instead.
  [(#3417)](https://github.com/PennyLaneAI/pennylane/pull/3417)

* The `observables` argument in `QubitDevice.statistics` is deprecated. Please use `circuit`
  instead.
  [(#3433)](https://github.com/PennyLaneAI/pennylane/pull/3433)

* The `seed_recipes` argument in `qml.classical_shadow` and `qml.shadow_expval` is deprecated.
  A new argument `seed` has been added, which defaults to None and can contain an integer with the
  wanted seed.
  [(#3388)](https://github.com/PennyLaneAI/pennylane/pull/3388)

* `make_tape` is deprecated. Please use `qml.tape.make_qscript` instead.
  [(#3478)](https://github.com/PennyLaneAI/pennylane/pull/3478)

<h3>Documentation</h3>

* Added documentation on parameter broadcasting regarding both its usage and technical aspects
  [#3356](https://github.com/PennyLaneAI/pennylane/pull/3356)

  The [quickstart guide on circuits](https://docs.pennylane.ai/en/stable/introduction/circuits.html#parameter-broadcasting-in-qnodes)
  as well as the the documentation of
  [QNodes](https://docs.pennylane.ai/en/stable/code/api/pennylane.QNode.html) and
  [Operators](https://docs.pennylane.ai/en/stable/code/api/pennylane.operation.Operator.html)
  now contain introductions and details on parameter broadcasting. The QNode documentation
  mostly contains usage details, the Operator documentation is concerned with implementation
  details and a guide to support broadcasting in custom operators.

* Corrects the return type statements of gradient and Hessian transforms, as well as a series
  of other functions that are a `batch_transform`.
  [(#3476)](https://github.com/PennyLaneAI/pennylane/pull/3476)

* Adds developer documentation for the queuing module.
  [(#3268)](https://github.com/PennyLaneAI/pennylane/pull/3268)

* Corrects more mentions for diagonalizing gates for all relevant operations. The docstrings for `compute_eigvals` used
  to say that the diagonalizing gates implemented $U$, the unitary such that $O = U \Sigma U^{\dagger}$, where $O$ is
  the original observable and $\Sigma$ a diagonal matrix. However, the diagonalizing gates actually implement
  $U^{\dagger}$, since $\langle \psi | O | \psi \rangle = \langle \psi | U \Sigma U^{\dagger} | \psi \rangle$, making
  $U^{\dagger} | \psi \rangle$ the actual state being measured in the $Z$-basis.
  [(#3409)](https://github.com/PennyLaneAI/pennylane/pull/3409)

* Adds warnings about using ``dill`` to pickle and unpickle datasets. 
  [#3505](https://github.com/PennyLaneAI/pennylane/pull/3505)

<h3>Bug fixes</h3>

* Fixed a bug where `hamiltonian_expand` didn't preserve the type of the inputted results in its output.
  [(#3339)](https://github.com/PennyLaneAI/pennylane/pull/3339)

* Fixed a bug that made `gradients.param_shift` raise an error when used with unshifted terms only
  in a custom recipe, and when using any unshifted terms at all under the new return type system.
  [(#3177)](https://github.com/PennyLaneAI/pennylane/pull/3177)

* Original tape `_obs_sharing_wires` attribute is updated during its expansion.
  [(#3293)](https://github.com/PennyLaneAI/pennylane/pull/3293)

* Small fix of `MeasurementProcess.map_wires`, where both the `self.obs` and `self._wires`
  attributes were modified.
  [(#3292)](https://github.com/PennyLaneAI/pennylane/pull/3292)

* An issue with `drain=False` in the adaptive optimizer is fixed. Before the fix, the operator pool
  needed to be re-constructed inside the optimization pool when `drain=False`. With the new fix,
  this reconstruction is not needed.
  [(#3361)](https://github.com/PennyLaneAI/pennylane/pull/3361)

* If the device originally has no shots but finite shots are dynamically specified, Hamiltonian
  expansion now occurs.
  [(#3369)](https://github.com/PennyLaneAI/pennylane/pull/3369)

* `qml.matrix(op)` now fails if the operator truly has no matrix (eg. `Barrier`) to match `op.matrix()`
  [(#3386)](https://github.com/PennyLaneAI/pennylane/pull/3386)

* The `pad_with` argument in the `AmplitudeEmbedding` template is now compatible
  with all interfaces
  [(#3392)](https://github.com/PennyLaneAI/pennylane/pull/3392)

* Fixed a bug where a QNode returning `qml.sample` would produce incorrect results when
  run on a device defined with a shot vector.
  [(#3422)](https://github.com/PennyLaneAI/pennylane/pull/3422)

* The `qml.data` module now works as expected on Windows.
  [(#3504)](https://github.com/PennyLaneAI/pennylane/pull/3504)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso
Juan Miguel Arrazola
Utkarsh Azad
Samuel Banning
Astral Cai
Ahmed Darwish
Isaac De Vlugt
Pieter Eendebak
Lillian M. A. Frederiksen
Katharine Hyatt
Soran Jahangiri
Edward Jiang
Christina Lee
Albert Mitjans Coma
Romain Moyard
Mudit Pandey
Matthew Silverman
Jay Soni
Antal Száva
David Wierichs
Moritz Willmann
Filippo Vicentini
