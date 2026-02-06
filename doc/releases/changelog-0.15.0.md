
# Release 0.15.0

<h3>New features since last release</h3>

<h4>Better and more flexible shot control</h4>

* Adds a new optimizer `qp.ShotAdaptiveOptimizer`, a gradient-descent optimizer where
  the shot rate is adaptively calculated using the variances of the parameter-shift gradient.
  [(#1139)](https://github.com/PennyLaneAI/pennylane/pull/1139)

  By keeping a running average of the parameter-shift gradient and the *variance* of the
  parameter-shift gradient, this optimizer frugally distributes a shot budget across the partial
  derivatives of each parameter.

  In addition, if computing the expectation value of a Hamiltonian, weighted random sampling can be
  used to further distribute the shot budget across the local terms from which the Hamiltonian is
  constructed.

  This optimizer is based on both the [iCANS1](https://quantum-journal.org/papers/q-2020-05-11-263)
  and [Rosalin](https://arxiv.org/abs/2004.06252) shot-adaptive optimizers.

  Once constructed, the cost function can be passed directly to the optimizer's `step` method.  The
  attribute `opt.total_shots_used` can be used to track the number of shots per iteration.

  ```pycon
  >>> coeffs = [2, 4, -1, 5, 2]
  >>> obs = [
  ...   qp.PauliX(1),
  ...   qp.PauliZ(1),
  ...   qp.PauliX(0) @ qp.PauliX(1),
  ...   qp.PauliY(0) @ qp.PauliY(1),
  ...   qp.PauliZ(0) @ qp.PauliZ(1)
  ... ]
  >>> H = qp.Hamiltonian(coeffs, obs)
  >>> dev = qp.device("default.qubit", wires=2, shots=100)
  >>> cost = qp.ExpvalCost(qp.templates.StronglyEntanglingLayers, H, dev)
  >>> params = qp.init.strong_ent_layers_uniform(n_layers=2, n_wires=2)
  >>> opt = qp.ShotAdaptiveOptimizer(min_shots=10)
  >>> for i in range(5):
  ...    params = opt.step(cost, params)
  ...    print(f"Step {i}: cost = {cost(params):.2f}, shots_used = {opt.total_shots_used}")
  Step 0: cost = -5.68, shots_used = 240
  Step 1: cost = -2.98, shots_used = 336
  Step 2: cost = -4.97, shots_used = 624
  Step 3: cost = -5.53, shots_used = 1054
  Step 4: cost = -6.50, shots_used = 1798
  ```

* Batches of shots can now be specified as a list, allowing measurement statistics
  to be course-grained with a single QNode evaluation.
  [(#1103)](https://github.com/PennyLaneAI/pennylane/pull/1103)

  ```pycon
  >>> shots_list = [5, 10, 1000]
  >>> dev = qp.device("default.qubit", wires=2, shots=shots_list)
  ```

  When QNodes are executed on this device, a single execution of 1015 shots will be submitted.
  However, three sets of measurement statistics will be returned; using the first 5 shots,
  second set of 10 shots, and final 1000 shots, separately.

  For example, executing a circuit with two outputs will lead to a result of shape `(3, 2)`:

  ```pycon
  >>> @qp.qnode(dev)
  ... def circuit(x):
  ...     qp.RX(x, wires=0)
  ...     qp.CNOT(wires=[0, 1])
  ...     return qp.expval(qp.PauliZ(0) @ qp.PauliX(1)), qp.expval(qp.PauliZ(0))
  >>> circuit(0.5)
  [[0.33333333 1.        ]
   [0.2        1.        ]
   [0.012      0.868     ]]
  ```

  This output remains fully differentiable.

- The number of shots can now be specified on a per-call basis when evaluating a QNode.
  [(#1075)](https://github.com/PennyLaneAI/pennylane/pull/1075).

  For this, the qnode should be called with an additional `shots` keyword argument:

  ```pycon
  >>> dev = qp.device('default.qubit', wires=1, shots=10) # default is 10
  >>> @qp.qnode(dev)
  ... def circuit(a):
  ...     qp.RX(a, wires=0)
  ...     return qp.sample(qp.PauliZ(wires=0))
  >>> circuit(0.8)
  [ 1  1  1 -1 -1  1  1  1  1  1]
  >>> circuit(0.8, shots=3)
  [ 1  1  1]
  >>> circuit(0.8)
  [ 1  1  1 -1 -1  1  1  1  1  1]
  ```

<h4>New differentiable quantum transforms</h4>

A new module is available,
[qp.transforms](https://pennylane.rtfd.io/en/stable/code/qml_transforms.html),
which contains *differentiable quantum transforms*. These are functions that act
on QNodes, quantum functions, devices, and tapes, transforming them while remaining
fully differentiable.

* A new adjoint transform has been added.
  [(#1111)](https://github.com/PennyLaneAI/pennylane/pull/1111)
  [(#1135)](https://github.com/PennyLaneAI/pennylane/pull/1135)

  This new method allows users to apply the adjoint of an arbitrary sequence of operations.

  ```python
  def subroutine(wire):
      qp.RX(0.123, wires=wire)
      qp.RY(0.456, wires=wire)

  dev = qp.device('default.qubit', wires=1)
  @qp.qnode(dev)
  def circuit():
      subroutine(0)
      qp.adjoint(subroutine)(0)
      return qp.expval(qp.PauliZ(0))
  ```

  This creates the following circuit:

  ```pycon
  >>> print(qp.draw(circuit)())
  0: --RX(0.123)--RY(0.456)--RY(-0.456)--RX(-0.123)--| <Z>
  ```

  Directly applying to a gate also works as expected.

  ```python
  qp.adjoint(qp.RX)(0.123, wires=0) # applies RX(-0.123)
  ```

* A new transform `qp.ctrl` is now available that adds control wires to subroutines.
  [(#1157)](https://github.com/PennyLaneAI/pennylane/pull/1157)

  ```python
  def my_ansatz(params):
     qp.RX(params[0], wires=0)
     qp.RZ(params[1], wires=1)

  # Create a new operation that applies `my_ansatz`
  # controlled by the "2" wire.
  my_ansatz2 = qp.ctrl(my_ansatz, control=2)

  @qp.qnode(dev)
  def circuit(params):
      my_ansatz2(params)
      return qp.state()
  ```

  This is equivalent to:

  ```python
  @qp.qnode(...)
  def circuit(params):
      qp.CRX(params[0], wires=[2, 0])
      qp.CRZ(params[1], wires=[2, 1])
      return qp.state()
  ```

* The `qp.transforms.classical_jacobian` transform has been added.
  [(#1186)](https://github.com/PennyLaneAI/pennylane/pull/1186)

  This transform returns a function to extract the Jacobian matrix of the classical part of a
  QNode, allowing the classical dependence between the QNode arguments and the quantum gate
  arguments to be extracted.

  For example, given the following QNode:

  ```pycon
  >>> @qp.qnode(dev)
  ... def circuit(weights):
  ...     qp.RX(weights[0], wires=0)
  ...     qp.RY(weights[0], wires=1)
  ...     qp.RZ(weights[2] ** 2, wires=1)
  ...     return qp.expval(qp.PauliZ(0))
  ```

  We can use this transform to extract the relationship
  :math:`f: \mathbb{R}^n \rightarrow\mathbb{R}^m` between the input QNode
  arguments :math:`w` and the gate arguments :math:`g`, for
  a given value of the QNode arguments:

  ```pycon
  >>> cjac_fn = qp.transforms.classical_jacobian(circuit)
  >>> weights = np.array([1., 1., 1.], requires_grad=True)
  >>> cjac = cjac_fn(weights)
  >>> print(cjac)
  [[1. 0. 0.]
   [1. 0. 0.]
   [0. 0. 2.]]
  ```

  The returned Jacobian has rows corresponding to gate arguments, and columns corresponding to
  QNode arguments; that is, :math:`J_{ij} = \frac{\partial}{\partial g_i} f(w_j)`.

<h4>More operations and templates</h4>

* Added the `SingleExcitation` two-qubit operation, which is useful for quantum
  chemistry applications.
  [(#1121)](https://github.com/PennyLaneAI/pennylane/pull/1121)

  It can be used to perform an SO(2) rotation in the subspace
  spanned by the states :math:`|01\rangle` and :math:`|10\rangle`.
  For example, the following circuit performs the transformation
  :math:`|10\rangle \rightarrow \cos(\phi/2)|10\rangle - \sin(\phi/2)|01\rangle`:

  ```python
  dev = qp.device('default.qubit', wires=2)

  @qp.qnode(dev)
  def circuit(phi):
      qp.PauliX(wires=0)
      qp.SingleExcitation(phi, wires=[0, 1])
  ```

  The `SingleExcitation` operation supports analytic gradients on hardware
  using only four expectation value calculations, following results from
  [Kottmann et al.](https://arxiv.org/abs/2011.05938)

* Added the `DoubleExcitation` four-qubit operation, which is useful for quantum
  chemistry applications.
  [(#1123)](https://github.com/PennyLaneAI/pennylane/pull/1123)

  It can be used to perform an SO(2) rotation in the subspace
  spanned by the states :math:`|1100\rangle` and :math:`|0011\rangle`.
  For example, the following circuit performs the transformation
  :math:`|1100\rangle\rightarrow \cos(\phi/2)|1100\rangle - \sin(\phi/2)|0011\rangle`:

  ```python
  dev = qp.device('default.qubit', wires=2)

  @qp.qnode(dev)
  def circuit(phi):
      qp.PauliX(wires=0)
      qp.PauliX(wires=1)
      qp.DoubleExcitation(phi, wires=[0, 1, 2, 3])
  ```

  The `DoubleExcitation` operation supports analytic gradients on hardware using only
  four expectation value calculations, following results from
  [Kottmann et al.](https://arxiv.org/abs/2011.05938).

* Added the `QuantumMonteCarlo` template for performing quantum Monte Carlo estimation of an
  expectation value on simulator.
  [(#1130)](https://github.com/PennyLaneAI/pennylane/pull/1130)

  The following example shows how the expectation value of sine squared over a standard normal
  distribution can be approximated:

  ```python
  from scipy.stats import norm

  m = 5
  M = 2 ** m
  n = 10
  N = 2 ** n
  target_wires = range(m + 1)
  estimation_wires = range(m + 1, n + m + 1)

  xmax = np.pi  # bound to region [-pi, pi]
  xs = np.linspace(-xmax, xmax, M)

  probs = np.array([norm().pdf(x) for x in xs])
  probs /= np.sum(probs)

  func = lambda i: np.sin(xs[i]) ** 2

  dev = qp.device("default.qubit", wires=(n + m + 1))

  @qp.qnode(dev)
  def circuit():
      qp.templates.QuantumMonteCarlo(
          probs,
          func,
          target_wires=target_wires,
          estimation_wires=estimation_wires,
      )
      return qp.probs(estimation_wires)

  phase_estimated = np.argmax(circuit()[:int(N / 2)]) / N
  expectation_estimated = (1 - np.cos(np.pi * phase_estimated)) / 2
  ```

* Added the `QuantumPhaseEstimation` template for performing quantum phase estimation for an input
  unitary matrix.
  [(#1095)](https://github.com/PennyLaneAI/pennylane/pull/1095)

  Consider the matrix corresponding to a rotation from an `RX` gate:

  ```pycon
  >>> phase = 5
  >>> target_wires = [0]
  >>> unitary = qp.RX(phase, wires=0).matrix
  ```

  The ``phase`` parameter can be estimated using ``QuantumPhaseEstimation``. For example, using five
  phase-estimation qubits:

  ```python
  n_estimation_wires = 5
  estimation_wires = range(1, n_estimation_wires + 1)

  dev = qp.device("default.qubit", wires=n_estimation_wires + 1)

  @qp.qnode(dev)
  def circuit():
      # Start in the |+> eigenstate of the unitary
      qp.Hadamard(wires=target_wires)

      QuantumPhaseEstimation(
          unitary,
          target_wires=target_wires,
          estimation_wires=estimation_wires,
      )

      return qp.probs(estimation_wires)

  phase_estimated = np.argmax(circuit()) / 2 ** n_estimation_wires

  # Need to rescale phase due to convention of RX gate
  phase_estimated = 4 * np.pi * (1 - phase)
  ```

- Added the `ControlledPhaseShift` gate as well as the `QFT` operation for applying quantum Fourier
  transforms.
  [(#1064)](https://github.com/PennyLaneAI/pennylane/pull/1064)

  ```python
  @qp.qnode(dev)
  def circuit_qft(basis_state):
      qp.BasisState(basis_state, wires=range(3))
      qp.templates.QFT(wires=range(3))
      return qp.state()
  ```

- Added the `ControlledQubitUnitary` operation. This
  enables implementation of multi-qubit gates with a variable number of
  control qubits. It is also possible to specify a different state for the
  control qubits using the `control_values` argument (also known as a
  mixed-polarity multi-controlled operation).
  [(#1069)](https://github.com/PennyLaneAI/pennylane/pull/1069)
  [(#1104)](https://github.com/PennyLaneAI/pennylane/pull/1104)

  For example, we can  create a multi-controlled T gate using:

  ```python
  T = qp.T._matrix()
  qp.ControlledQubitUnitary(T, control_wires=[0, 1, 3], wires=2, control_values="110")
  ```

  Here, the T gate will be applied to wire `2` if control wires `0` and `1` are in
  state `1`, and control wire `3` is in state `0`. If no value is passed to
  `control_values`, the gate will be applied if all control wires are in
  the `1` state.

- Added `MultiControlledX` for multi-controlled `NOT` gates.
  This is a special case of `ControlledQubitUnitary` that applies a
  Pauli X gate conditioned on the state of an arbitrary number of
  control qubits.
  [(#1104)](https://github.com/PennyLaneAI/pennylane/pull/1104)

<h4>Support for higher-order derivatives on hardware</h4>

* Computing second derivatives and Hessians of QNodes is now supported with
  the parameter-shift differentiation method, on all machine learning interfaces.
  [(#1130)](https://github.com/PennyLaneAI/pennylane/pull/1130)
  [(#1129)](https://github.com/PennyLaneAI/pennylane/pull/1129)
  [(#1110)](https://github.com/PennyLaneAI/pennylane/pull/1110)

  Hessians are computed using the parameter-shift rule, and can be
  evaluated on both hardware and simulator devices.

  ```python
  dev = qp.device('default.qubit', wires=1)

  @qp.qnode(dev, diff_method="parameter-shift")
  def circuit(p):
      qp.RY(p[0], wires=0)
      qp.RX(p[1], wires=0)
      return qp.expval(qp.PauliZ(0))

  x = np.array([1.0, 2.0], requires_grad=True)
  ```

  ```python
  >>> hessian_fn = qp.jacobian(qp.grad(circuit))
  >>> hessian_fn(x)
  [[0.2248451 0.7651474]
   [0.7651474 0.2248451]]
  ```

* Added the function `finite_diff()` to compute finite-difference
  approximations to the gradient and the second-order derivatives of
  arbitrary callable functions.
  [(#1090)](https://github.com/PennyLaneAI/pennylane/pull/1090)

  This is useful to compute the derivative of parametrized
  `pennylane.Hamiltonian` observables with respect to their parameters.

  For example, in quantum chemistry simulations it can be used to evaluate
  the derivatives of the electronic Hamiltonian with respect to the nuclear
  coordinates:

  ```pycon
  >>> def H(x):
  ...    return qp.qchem.molecular_hamiltonian(['H', 'H'], x)[0]
  >>> x = np.array([0., 0., -0.66140414, 0., 0., 0.66140414])
  >>> grad_fn = qp.finite_diff(H, N=1)
  >>> grad = grad_fn(x)
  >>> deriv2_fn = qp.finite_diff(H, N=2, idx=[0, 1])
  >>> deriv2_fn(x)
  ```

* The JAX interface now supports all devices, including hardware devices,
  via the parameter-shift differentiation method.
  [(#1076)](https://github.com/PennyLaneAI/pennylane/pull/1076)

  For example, using the JAX interface with Cirq:

  ```python
  dev = qp.device('cirq.simulator', wires=1)
  @qp.qnode(dev, interface="jax", diff_method="parameter-shift")
  def circuit(x):
      qp.RX(x[1], wires=0)
      qp.Rot(x[0], x[1], x[2], wires=0)
      return qp.expval(qp.PauliZ(0))
  weights = jnp.array([0.2, 0.5, 0.1])
  print(circuit(weights))
  ```

  Currently, when used with the parameter-shift differentiation method,
  only a single returned expectation value or variance is supported.
  Multiple expectations/variances, as well as probability and state returns,
  are not currently allowed.

<h3>Improvements</h3>

  ```python
  dev = qp.device("default.qubit", wires=2)

  inputstate = [np.sqrt(0.2), np.sqrt(0.3), np.sqrt(0.4), np.sqrt(0.1)]

  @qp.qnode(dev)
  def circuit():
      mottonen.MottonenStatePreparation(inputstate,wires=[0, 1])
      return qp.expval(qp.PauliZ(0))
  ```

  Previously returned:

  ```pycon
  >>> print(qp.draw(circuit)())
  0: ──RY(1.57)──╭C─────────────╭C──╭C──╭C──┤ ⟨Z⟩
  1: ──RY(1.35)──╰X──RY(0.422)──╰X──╰X──╰X──┤
  ```

  In this release, it now returns:

  ```pycon
  >>> print(qp.draw(circuit)())
  0: ──RY(1.57)──╭C─────────────╭C──┤ ⟨Z⟩
  1: ──RY(1.35)──╰X──RY(0.422)──╰X──┤
  ```

- The templates are now classes inheriting
  from `Operation`, and define the ansatz in their `expand()` method. This
  change does not affect the user interface.
  [(#1138)](https://github.com/PennyLaneAI/pennylane/pull/1138)
  [(#1156)](https://github.com/PennyLaneAI/pennylane/pull/1156)
  [(#1163)](https://github.com/PennyLaneAI/pennylane/pull/1163)
  [(#1192)](https://github.com/PennyLaneAI/pennylane/pull/1192)

  For convenience, some templates have a new method that returns the expected
  shape of the trainable parameter tensor, which can be used to create
  random tensors.

  ```python
  shape = qp.templates.BasicEntanglerLayers.shape(n_layers=2, n_wires=4)
  weights = np.random.random(shape)
  qp.templates.BasicEntanglerLayers(weights, wires=range(4))
  ```

- `QubitUnitary` now validates to ensure the input matrix is two dimensional.
  [(#1128)](https://github.com/PennyLaneAI/pennylane/pull/1128)

* Most layers in Pytorch or Keras accept arbitrary dimension inputs, where each dimension barring
  the last (in the case where the actual weight function of the layer operates on one-dimensional
  vectors) is broadcast over. This is now also supported by KerasLayer and TorchLayer.
  [(#1062)](https://github.com/PennyLaneAI/pennylane/pull/1062).

  Example use:

  ```python
  dev = qp.device("default.qubit", wires=4)
  x = tf.ones((5, 4, 4))

  @qp.qnode(dev)
  def layer(weights, inputs):
      qp.templates.AngleEmbedding(inputs, wires=range(4))
      qp.templates.StronglyEntanglingLayers(weights, wires=range(4))
      return [qp.expval(qp.PauliZ(i)) for i in range(4)]

  qlayer = qp.qnn.KerasLayer(layer, {"weights": (4, 4, 3)}, output_dim=4)
  out = qlayer(x)
  ```

  The output tensor has the following shape:
  ```pycon
  >>> out.shape
  (5, 4, 4)
  ```

* If only one argument to the function `qp.grad` has the `requires_grad` attribute
  set to True, then the returned gradient will be a NumPy array, rather than a
  tuple of length 1.
  [(#1067)](https://github.com/PennyLaneAI/pennylane/pull/1067)
  [(#1081)](https://github.com/PennyLaneAI/pennylane/pull/1081)

* An improvement has been made to how `QubitDevice` generates and post-processess samples,
  allowing QNode measurement statistics to work on devices with more than 32 qubits.
  [(#1088)](https://github.com/PennyLaneAI/pennylane/pull/1088)

* Due to the addition of `density_matrix()` as a return type from a QNode, tuples are now supported
  by the `output_dim` parameter in `qnn.KerasLayer`.
  [(#1070)](https://github.com/PennyLaneAI/pennylane/pull/1070)

* Two new utility methods are provided for working with quantum tapes.
  [(#1175)](https://github.com/PennyLaneAI/pennylane/pull/1175)

  - `qp.tape.get_active_tape()` gets the currently recording tape.

  - `tape.stop_recording()` is a context manager that temporarily
    stops the currently recording tape from recording additional
    tapes or quantum operations.

  For example:

  ```pycon
  >>> with qp.tape.QuantumTape():
  ...     qp.RX(0, wires=0)
  ...     current_tape = qp.tape.get_active_tape()
  ...     with current_tape.stop_recording():
  ...         qp.RY(1.0, wires=1)
  ...     qp.RZ(2, wires=1)
  >>> current_tape.operations
  [RX(0, wires=[0]), RZ(2, wires=[1])]
  ```

* When printing `qp.Hamiltonian` objects, the terms are sorted by number of wires followed by coefficients.
  [(#981)](https://github.com/PennyLaneAI/pennylane/pull/981)

* Adds `qp.math.conj` to the PennyLane math module.
  [(#1143)](https://github.com/PennyLaneAI/pennylane/pull/1143)

  This new method will do elementwise conjugation to the given tensor-like object,
  correctly dispatching to the required tensor-manipulation framework
  to preserve differentiability.

  ```python
  >>> a = np.array([1.0 + 2.0j])
  >>> qp.math.conj(a)
  array([1.0 - 2.0j])
  ```

* The four-term parameter-shift rule, as used by the controlled rotation operations,
  has been updated to use coefficients that minimize the variance as per
  https://arxiv.org/abs/2104.05695.
  [(#1206)](https://github.com/PennyLaneAI/pennylane/pull/1206)

* A new transform `qp.transforms.invisible` has been added, to make it easier
  to transform QNodes.
  [(#1175)](https://github.com/PennyLaneAI/pennylane/pull/1175)

<h3>Breaking changes</h3>

* Devices do not have an `analytic` argument or attribute anymore.
  Instead, `shots` is the source of truth for whether a simulator
  estimates return values from a finite number of shots, or whether
  it returns analytic results (`shots=None`).
  [(#1079)](https://github.com/PennyLaneAI/pennylane/pull/1079)
  [(#1196)](https://github.com/PennyLaneAI/pennylane/pull/1196)

  ```python
  dev_analytic = qp.device('default.qubit', wires=1, shots=None)
  dev_finite_shots = qp.device('default.qubit', wires=1, shots=1000)

  def circuit():
      qp.Hadamard(wires=0)
      return qp.expval(qp.PauliZ(wires=0))

  circuit_analytic = qp.QNode(circuit, dev_analytic)
  circuit_finite_shots = qp.QNode(circuit, dev_finite_shots)
  ```

  Devices with `shots=None` return deterministic, exact results:

  ```pycon
  >>> circuit_analytic()
  0.0
  >>> circuit_analytic()
  0.0
  ```
  Devices with `shots > 0` return stochastic results estimated from
  samples in each run:

  ```pycon
  >>> circuit_finite_shots()
  -0.062
  >>> circuit_finite_shots()
  0.034
  ```

  The `qp.sample()` measurement can only be used on devices on which the number
  of shots is set explicitly.

* If creating a QNode from a quantum function with an argument named `shots`,
  a `UserWarning` is raised, warning the user that this is a reserved
  argument to change the number of shots on a per-call basis.
  [(#1075)](https://github.com/PennyLaneAI/pennylane/pull/1075)

* For devices inheriting from `QubitDevice`, the methods `expval`, `var`, `sample`
  accept two new keyword arguments --- `shot_range` and `bin_size`.
  [(#1103)](https://github.com/PennyLaneAI/pennylane/pull/1103)

  These new arguments allow for the statistics to be performed on only a subset of device samples.
  This finer level of control is accessible from the main UI by instantiating a device with a batch
  of shots.

  For example, consider the following device:

  ```pycon
  >>> dev = qp.device("my_device", shots=[5, (10, 3), 100])
  ```

  This device will execute QNodes using 135 shots, however
  measurement statistics will be **course grained** across these 135
  shots:

  * All measurement statistics will first be computed using the
    first 5 shots --- that is, `shots_range=[0, 5]`, `bin_size=5`.

  * Next, the tuple `(10, 3)` indicates 10 shots, repeated 3 times. This will use
    `shot_range=[5, 35]`, performing the expectation value in bins of size 10
    (`bin_size=10`).

  * Finally, we repeat the measurement statistics for the final 100 shots,
    `shot_range=[35, 135]`, `bin_size=100`.


* The old PennyLane core has been removed, including the following modules:
  [(#1100)](https://github.com/PennyLaneAI/pennylane/pull/1100)

  - `pennylane.variables`
  - `pennylane.qnodes`

  As part of this change, the location of the new core within the Python
  module has been moved:

  - Moves `pennylane.tape.interfaces` → `pennylane.interfaces`
  - Merges `pennylane.CircuitGraph` and `pennylane.TapeCircuitGraph`  → `pennylane.CircuitGraph`
  - Merges `pennylane.OperationRecorder` and `pennylane.TapeOperationRecorder`  →
  - `pennylane.tape.operation_recorder`
  - Merges `pennylane.measure` and `pennylane.tape.measure` → `pennylane.measure`
  - Merges `pennylane.operation` and `pennylane.tape.operation` → `pennylane.operation`
  - Merges `pennylane._queuing` and `pennylane.tape.queuing` → `pennylane.queuing`

  This has no affect on import location.

  In addition,

  - All tape-mode functions have been removed (`qp.enable_tape()`, `qp.tape_mode_active()`),
  - All tape fixtures have been deleted,
  - Tests specifically for non-tape mode have been deleted.

* The device test suite no longer accepts the `analytic` keyword.
  [(#1216)](https://github.com/PennyLaneAI/pennylane/pull/1216)

<h3>Bug fixes</h3>

* Fixes a bug where using the circuit drawer with a `ControlledQubitUnitary`
  operation raised an error.
  [(#1174)](https://github.com/PennyLaneAI/pennylane/pull/1174)

* Fixes a bug and a test where the ``QuantumTape.is_sampled`` attribute was not
  being updated.
  [(#1126)](https://github.com/PennyLaneAI/pennylane/pull/1126)

* Fixes a bug where `BasisEmbedding` would not accept inputs whose bits are all ones
  or all zeros.
  [(#1114)](https://github.com/PennyLaneAI/pennylane/pull/1114)

* The `ExpvalCost` class raises an error if instantiated
  with non-expectation measurement statistics.
  [(#1106)](https://github.com/PennyLaneAI/pennylane/pull/1106)

* Fixes a bug where decompositions would reset the differentiation method
  of a QNode.
  [(#1117)](https://github.com/PennyLaneAI/pennylane/pull/1117)

* Fixes a bug where the second-order CV parameter-shift rule would error
  if attempting to compute the gradient of a QNode with more than one
  second-order observable.
  [(#1197)](https://github.com/PennyLaneAI/pennylane/pull/1197)

* Fixes a bug where repeated Torch interface applications after expansion caused an error.
  [(#1223)](https://github.com/PennyLaneAI/pennylane/pull/1223)

* Sampling works correctly with batches of shots specified as a list.
  [(#1232)](https://github.com/PennyLaneAI/pennylane/pull/1232)

<h3>Documentation</h3>

- Updated the diagram used in the Architectural overview page of the
  Development guide such that it doesn't mention Variables.
  [(#1235)](https://github.com/PennyLaneAI/pennylane/pull/1235)

- Typos addressed in templates documentation.
  [(#1094)](https://github.com/PennyLaneAI/pennylane/pull/1094)

- Upgraded the documentation to use Sphinx 3.5.3 and the new m2r2 package.
  [(#1186)](https://github.com/PennyLaneAI/pennylane/pull/1186)

- Added `flaky` as dependency for running tests in the documentation.
  [(#1113)](https://github.com/PennyLaneAI/pennylane/pull/1113)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Shahnawaz Ahmed, Juan Miguel Arrazola, Thomas Bromley, Olivia Di Matteo, Alain Delgado Gran, Kyle
Godbey, Diego Guala, Theodor Isacsson, Josh Izaac, Soran Jahangiri, Nathan Killoran, Christina Lee,
Daniel Polatajko, Chase Roberts, Sankalp Sanand, Pritish Sehzpaul, Maria Schuld, Antal Száva, David Wierichs.

