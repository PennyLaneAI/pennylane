
# Release 0.20.0

<h3>New features since last release</h3>

<h4>Shiny new circuit drawer!🎨🖌️ </h4>

* PennyLane now supports drawing a QNode with matplotlib!
  [(#1803)](https://github.com/PennyLaneAI/pennylane/pull/1803)
  [(#1811)](https://github.com/PennyLaneAI/pennylane/pull/1811)
  [(#1931)](https://github.com/PennyLaneAI/pennylane/pull/1931)
  [(#1954)](https://github.com/PennyLaneAI/pennylane/pull/1954)

  ```python
  dev = qml.device("default.qubit", wires=4)

  @qml.qnode(dev)
  def circuit(x, z):
      qml.QFT(wires=(0,1,2,3))
      qml.Toffoli(wires=(0,1,2))
      qml.CSWAP(wires=(0,2,3))
      qml.RX(x, wires=0)
      qml.CRZ(z, wires=(3,0))
      return qml.expval(qml.PauliZ(0))
  fig, ax = qml.draw_mpl(circuit)(1.2345, 1.2345)
  fig.show()
  ```

  <img src="https://pennylane.readthedocs.io/en/latest/_images/main_example.png" width=70%/>

<h4>New and improved quantum-aware optimizers</h4>

* Added `qml.LieAlgebraOptimizer`, a new quantum-aware Lie Algebra optimizer
  that allows one to perform gradient descent on the special unitary group.
  [(#1911)](https://github.com/PennyLaneAI/pennylane/pull/1911)

  ```python
  dev = qml.device("default.qubit", wires=2)
  H = -1.0 * qml.PauliX(0) - qml.PauliZ(1) - qml.PauliY(0) @ qml.PauliX(1)

  @qml.qnode(dev)
  def circuit():
      qml.RX(0.1, wires=[0])
      qml.RY(0.5, wires=[1])
      qml.CNOT(wires=[0,1])
      qml.RY(0.6, wires=[0])
      return qml.expval(H)
  opt = qml.LieAlgebraOptimizer(circuit=circuit, stepsize=0.1)
  ```

  Note that, unlike other optimizers, the `LieAlgebraOptimizer` accepts a QNode
  with *no* parameters, and instead grows the circuit by appending operations
  during the optimization:

  ```pycon
  >>> circuit()
  tensor(-1.3351865, requires_grad=True)
  >>> circuit1, cost = opt.step_and_cost()
  >>> circuit1()
  tensor(-1.99378872, requires_grad=True)
  ```

  For more details, see the
  [LieAlgebraOptimizer documentation](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.LieAlgebraOptimizer.html).

* The `qml.metric_tensor` transform can now be used to compute the full
  tensor, beyond the block diagonal approximation.
  [(#1725)](https://github.com/PennyLaneAI/pennylane/pull/1725)

  This is performed using Hadamard tests, and requires an additional wire
  on the device to execute the circuits produced by the transform,
  as compared to the number of wires required by the original circuit.
  The transform defaults to computing the full tensor, which can
  be controlled by the `approx` keyword argument.

  As an example, consider the QNode

  ```python
  dev = qml.device("default.qubit", wires=3)

  @qml.qnode(dev)
  def circuit(weights):
      qml.RX(weights[0], wires=0)
      qml.RY(weights[1], wires=0)
      qml.CNOT(wires=[0, 1])
      qml.RZ(weights[2], wires=1)
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

  weights = np.array([0.2, 1.2, -0.9], requires_grad=True)
  ```

  Then we can compute the (block) diagonal metric tensor as before, now using the
  `approx="block-diag"` keyword:

  ```pycon
  >>> qml.metric_tensor(circuit, approx="block-diag")(weights)
  [[0.25       0.         0.        ]
   [0.         0.24013262 0.        ]
   [0.         0.         0.21846983]]
  ```

  Instead, we now can also compute the full metric tensor, using
  Hadamard tests on the additional wire of the device:

  ```pycon
  >>> qml.metric_tensor(circuit)(weights)
  [[ 0.25        0.         -0.23300977]
   [ 0.          0.24013262  0.01763859]
   [-0.23300977  0.01763859  0.21846983]]
  ```

  See the
  [metric tensor documentation](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.transforms.metric_tensor.html).
  for more information and usage details.

<h4>Faster performance with optimized quantum workflows</h4>

* The QNode has been re-written to support batch execution across the board,
  custom gradients, better decomposition strategies, and higher-order derivatives.
  [(#1807)](https://github.com/PennyLaneAI/pennylane/pull/1807)
  [(#1969)](https://github.com/PennyLaneAI/pennylane/pull/1969)

  - Internally, if multiple circuits are generated for simultaneous execution, they
    will be packaged into a single job for execution on the device. This can lead to
    significant performance improvement when executing the QNode on remote
    quantum hardware or simulator devices with parallelization capabilities.

  - Custom gradient transforms can be specified as the differentiation method:

    ```python
    @qml.gradients.gradient_transform
    def my_gradient_transform(tape):
        ...
        return tapes, processing_fn

    @qml.qnode(dev, diff_method=my_gradient_transform)
    def circuit():
    ```

  For breaking changes related to the use of the new QNode, refer to the
  Breaking Changes section.

  Note that the old QNode remains accessible at `@qml.qnode_old.qnode`, however this will
  be removed in the next release.

* Custom decompositions can now be applied to operations at the device level.
  [(#1900)](https://github.com/PennyLaneAI/pennylane/pull/1900)

  For example, suppose we would like to implement the following QNode:

  ```python
  def circuit(weights):
      qml.BasicEntanglerLayers(weights, wires=[0, 1, 2])
      return qml.expval(qml.PauliZ(0))

  original_dev = qml.device("default.qubit", wires=3)
  original_qnode = qml.QNode(circuit, original_dev)
  ```

  ```pycon
  >>> weights = np.array([[0.4, 0.5, 0.6]])
  >>> print(qml.draw(original_qnode, expansion_strategy="device")(weights))
   0: ──RX(0.4)──╭C──────╭X──┤ ⟨Z⟩
   1: ──RX(0.5)──╰X──╭C──│───┤
   2: ──RX(0.6)──────╰X──╰C──┤
  ```

  Now, let's swap out the decomposition of the `CNOT` gate into `CZ`
  and `Hadamard`, and furthermore the decomposition of `Hadamard` into
  `RZ` and `RY` rather than the decomposition already available in PennyLane.
  We define the two decompositions like so, and pass them to a device:

  ```python
  def custom_cnot(wires):
      return [
          qml.Hadamard(wires=wires[1]),
          qml.CZ(wires=[wires[0], wires[1]]),
          qml.Hadamard(wires=wires[1])
      ]

  def custom_hadamard(wires):
      return [
          qml.RZ(np.pi, wires=wires),
          qml.RY(np.pi / 2, wires=wires)
      ]

  # Can pass the operation itself, or a string
  custom_decomps = {qml.CNOT : custom_cnot, "Hadamard" : custom_hadamard}

  decomp_dev = qml.device("default.qubit", wires=3, custom_decomps=custom_decomps)
  decomp_qnode = qml.QNode(circuit, decomp_dev)
  ```

  Now when we draw or run a QNode on this device, the gates will be expanded
  according to our specifications:

  ```pycon
  >>> print(qml.draw(decomp_qnode, expansion_strategy="device")(weights))
   0: ──RX(0.4)──────────────────────╭C──RZ(3.14)──RY(1.57)──────────────────────────╭Z──RZ(3.14)──RY(1.57)──┤ ⟨Z⟩
   1: ──RX(0.5)──RZ(3.14)──RY(1.57)──╰Z──RZ(3.14)──RY(1.57)──╭C──────────────────────│───────────────────────┤
   2: ──RX(0.6)──RZ(3.14)──RY(1.57)──────────────────────────╰Z──RZ(3.14)──RY(1.57)──╰C──────────────────────┤
  ```

  A separate context manager, `set_decomposition`, has also been implemented to enable
  application of custom decompositions on devices that have already been created.

  ```pycon
  >>> with qml.transforms.set_decomposition(custom_decomps, original_dev):
  ...     print(qml.draw(original_qnode, expansion_strategy="device")(weights))
   0: ──RX(0.4)──────────────────────╭C──RZ(3.14)──RY(1.57)──────────────────────────╭Z──RZ(3.14)──RY(1.57)──┤ ⟨Z⟩
   1: ──RX(0.5)──RZ(3.14)──RY(1.57)──╰Z──RZ(3.14)──RY(1.57)──╭C──────────────────────│───────────────────────┤
   2: ──RX(0.6)──RZ(3.14)──RY(1.57)──────────────────────────╰Z──RZ(3.14)──RY(1.57)──╰C──────────────────────┤
  ```

* Given an operator of the form :math:`U=e^{iHt}`, where :math:`H` has
  commuting terms and known eigenvalues,
  `qml.gradients.generate_shift_rule` computes the generalized parameter shift rules for determining
  the gradient of the expectation value :math:`f(t) = \langle 0|U(t)^\dagger \hat{O} U(t)|0\rangle` on
  hardware.
  [(#1788)](https://github.com/PennyLaneAI/pennylane/pull/1788)
  [(#1932)](https://github.com/PennyLaneAI/pennylane/pull/1932)

  Given

  .. math:: H = \sum_i a_i h_i,

  where the eigenvalues of :math:`H` are known and all :math:`h_i` commute, we can compute
  the *frequencies* (the unique positive differences of any two eigenvalues) using
  `qml.gradients.eigvals_to_frequencies`.

  `qml.gradients.generate_shift_rule` can then be used to compute the parameter
  shift rules to compute :math:`f'(t)` using `2R` shifted cost function evaluations.
  This becomes cheaper than the standard application of the chain rule and
  two-term shift rule when `R` is less than the
  number of Pauli words in the generator.

  For example, consider the case where :math:`H` has eigenspectrum `(-1, 0, 1)`:

  ```pycon
  >>> frequencies = qml.gradients.eigvals_to_frequencies((-1, 0, 1))
  >>> frequencies
  (1, 2)
  >>> coeffs, shifts = qml.gradients.generate_shift_rule(frequencies)
  >>> coeffs
  array([ 0.85355339, -0.85355339, -0.14644661,  0.14644661])
  >>> shifts
  array([ 0.78539816, -0.78539816,  2.35619449, -2.35619449])
  ```

  As we can see, `generate_shift_rule` returns four coefficients :math:`c_i` and shifts
  :math:`s_i` corresponding to a four term parameter shift rule. The gradient can then
  be reconstructed via:

  .. math:: \frac{\partial}{\partial\phi}f = \sum_{i} c_i f(\phi + s_i),

  where :math:`f(\phi) = \langle 0|U(\phi)^\dagger \hat{O} U(\phi)|0\rangle`
  for some observable :math:`\hat{O}` and the unitary :math:`U(\phi)=e^{iH\phi}`.

<h4>Support for TensorFlow AutoGraph mode with quantum hardware</h4>

* It is now possible to use TensorFlow's [AutoGraph
  mode](https://www.tensorflow.org/guide/function) with QNodes on all devices and with arbitrary
  differentiation methods. Previously, AutoGraph mode only support `diff_method="backprop"`. This
  will result in significantly more performant model execution, at the cost of a more expensive
  initial compilation. [(#1866)](https://github.com/PennyLaneAI/pennylane/pull/1886)

  Use AutoGraph to convert your QNodes or cost functions into TensorFlow
  graphs by decorating them with `@tf.function`:

  ```python
  dev = qml.device("lightning.qubit", wires=2)

  @qml.qnode(dev, diff_method="adjoint", interface="tf", max_diff=1)
  def circuit(x):
      qml.RX(x[0], wires=0)
      qml.RY(x[1], wires=1)
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), qml.expval(qml.PauliZ(0))

  @tf.function
  def cost(x):
      return tf.reduce_sum(circuit(x))

  x = tf.Variable([0.5, 0.7], dtype=tf.float64)

  with tf.GradientTape() as tape:
      loss = cost(x)

  grad = tape.gradient(loss, x)
  ```

  The initial execution may take slightly longer than when executing the circuit in
  eager mode; this is because TensorFlow is tracing the function to create the graph.
  Subsequent executions will be much more performant.

  Note that using AutoGraph with backprop-enabled devices, such as `default.qubit`,
  will yield the best performance.

  For more details, please see the [TensorFlow AutoGraph
  documentation](https://www.tensorflow.org/guide/function).

<h4>Characterize your quantum models with classical QNode reconstruction</h4>

* The `qml.fourier.reconstruct` function is added. It can be used to
  reconstruct QNodes outputting expectation values along a specified
  parameter dimension, with a minimal number of calls to the
  original QNode. The returned
  reconstruction is exact and purely classical, and can be evaluated
  without any quantum executions.
  [(#1864)](https://github.com/PennyLaneAI/pennylane/pull/1864)

  The reconstruction technique differs for functions with equidistant frequencies
  that are reconstructed using the function value at equidistant sampling points, and
  for functions with arbitrary frequencies reconstructed using arbitrary sampling points.

  As an example, consider the following QNode:

  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def circuit(x, Y, f=1.0):
      qml.RX(f * x, wires=0)
      qml.RY(Y[0], wires=0)
      qml.RY(Y[1], wires=1)
      qml.CNOT(wires=[0, 1])
      qml.RY(3 * Y[1], wires=1)
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  ```

  It has three variational parameters overall: A scalar input `x`
  and an array-valued input `Y` with two entries. Additionally, we can
  tune the dependence on `x` with the frequency `f`.
  We then can reconstruct the QNode output function with respect to `x` via

  ```pycon
  >>> x = 0.3
  >>> Y = np.array([0.1, -0.9])
  >>> rec = qml.fourier.reconstruct(circuit, ids="x", nums_frequency={"x": {0: 1}})(x, Y)
  >>> rec
  {'x': {0: <function pennylane.fourier.reconstruct._reconstruct_equ.<locals>._reconstruction(x)>}}
  ```

  As we can see, we get a nested dictionary in the format of the input `nums_frequency`
  with functions as values. These functions are simple float-to-float callables:

  ```pycon
  >>> univariate = rec["x"][0]
  >>> univariate(x)
  -0.880208251507
  ```

  For more details on usage, reconstruction cost and differentiability support, please see the
  [fourier.reconstruct docstring](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.fourier.reconstruct.html).

<h4>State-of-the-art operations and templates</h4>

* A circuit template for time evolution under a commuting Hamiltonian utilizing generalized
  parameter shift rules for cost function gradients is available as `qml.CommutingEvolution`.
  [(#1788)](https://github.com/PennyLaneAI/pennylane/pull/1788)

  If the template is handed a frequency spectrum during its instantiation, then `generate_shift_rule`
  is internally called to obtain the general parameter shift rules with respect to
  `CommutingEvolution`'s :math:`t` parameter, otherwise the shift rule for a decomposition of
  `CommutingEvolution` will be used.

  The template can be initialized within QNode as:

  ```python
  import pennylane as qp

  n_wires = 2
  dev = qml.device('default.qubit', wires=n_wires)

  coeffs = [1, -1]
  obs = [qml.PauliX(0) @ qml.PauliY(1), qml.PauliY(0) @ qml.PauliX(1)]
  hamiltonian = qml.Hamiltonian(coeffs, obs)
  frequencies = (2,4)

  @qml.qnode(dev)
  def circuit(time):
      qml.PauliX(0)
      qml.CommutingEvolution(hamiltonian, time, frequencies)
      return qml.expval(qml.PauliZ(0))
  ```

  Note that there is no internal validation that 1) the input `qml.Hamiltonian` is fully commuting
  and 2) the eigenvalue frequency spectrum is correct, since these checks become
  prohibitively expensive for large Hamiltonians.

* The `qml.Barrier()` operator has been added. With it we can separate blocks
  in compilation or use it as a visual tool.
  [(#1844)](https://github.com/PennyLaneAI/pennylane/pull/1844)

* Added the identity observable to be an operator. Now we can explicitly call the identity
  operation on our quantum circuits for both qubit and CV devices.
  [(#1829)](https://github.com/PennyLaneAI/pennylane/pull/1829)

* Added the `qml.QubitDensityMatrix` initialization gate for
  mixed state simulation.
  [(#1850)](https://github.com/PennyLaneAI/pennylane/pull/1850)

* A thermal relaxation channel is added to the Noisy channels. The channel description can be
  found on the supplementary information of [Quantum classifier with tailored quantum kernels](https://arxiv.org/abs/1909.02611).
  [(#1766)](https://github.com/PennyLaneAI/pennylane/pull/1766)

* Added a new `qml.PauliError` channel that allows the application of an
  arbitrary number of Pauli operators on an arbitrary number of wires.
  [(#1781)](https://github.com/PennyLaneAI/pennylane/pull/1781)

<h4>Manipulate QNodes to your ❤️s content with new transforms</h4>

* The `merge_amplitude_embedding` transformation has been created to
  automatically merge all gates of this type into one.
  [(#1933)](https://github.com/PennyLaneAI/pennylane/pull/1933)

  ```python
  from pennylane.transforms import merge_amplitude_embedding

  dev = qml.device("default.qubit", wires = 3)
  
  @qml.qnode(dev)
  @merge_amplitude_embedding
  def qfunc():
      qml.AmplitudeEmbedding([0,1,0,0], wires = [0,1])
      qml.AmplitudeEmbedding([0,1], wires = 2)
      return qml.expval(qml.PauliZ(wires = 0))
  ```

  ```pycon
  >>> print(qml.draw(qnode)())
   0: ──╭AmplitudeEmbedding(M0)──┤ ⟨Z⟩
   1: ──├AmplitudeEmbedding(M0)──┤
   2: ──╰AmplitudeEmbedding(M0)──┤
   M0 =
   [0.+0.j 0.+0.j 0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
  ```

* The `undo_swaps` transformation has been created to automatically remove all
  swaps of a circuit.
  [(#1960)](https://github.com/PennyLaneAI/pennylane/pull/1960)

  ```python
  dev = qml.device('default.qubit', wires=3)

  @qml.qnode(dev)
  @qml.transforms.undo_swaps
  def qfunc():
      qml.Hadamard(wires=0)
      qml.PauliX(wires=1)
      qml.SWAP(wires=[0,1])
      qml.SWAP(wires=[0,2])
      qml.PauliY(wires=0)
      return qml.expval(qml.PauliZ(0))
  ```
  
  ```pycon
  >>> print(qml.draw(qfunc)())
   0: ──Y──┤ ⟨Z⟩
   1: ──H──┤
   2: ──X──┤
  ```

<h3>Improvements</h3>

* Added functions for computing the values of atomic and molecular orbitals at a given position.
  [(#1867)](https://github.com/PennyLaneAI/pennylane/pull/1867)

  The functions `atomic_orbital` and `molecular_orbital` can be used, as shown in the
  following codeblock, to evaluate the orbitals. By generating values of the orbitals at different
  positions, one can plot the spatial shape of a desired orbital.

  ```python
  symbols  = ['H', 'H']
  geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
  mol = hf.Molecule(symbols, geometry)
  hf.generate_scf(mol)()

  ao = mol.atomic_orbital(0)
  mo = mol.molecular_orbital(1)
  ```

  ```pycon
  >>> print(ao(0.0, 0.0, 0.0))
  >>> print(mo(0.0, 0.0, 0.0))
  0.6282468778183719
  0.018251285973461928
  ```

* Added support for Python 3.10.
  [(#1964)](https://github.com/PennyLaneAI/pennylane/pull/1964)

* The execution of QNodes that have

  - multiple return types;
  - a return type other than Variance and Expectation

  now raises a descriptive error message when using the JAX interface.
  [(#2011)](https://github.com/PennyLaneAI/pennylane/pull/2011)

* The PennyLane `qchem` package is now lazily imported; it will only be imported
  the first time it is accessed.
  [(#1962)](https://github.com/PennyLaneAI/pennylane/pull/1962)

* `qml.math.scatter_element_add` now supports adding multiple values at
  multiple indices with a single function call, in all interfaces
  [(#1864)](https://github.com/PennyLaneAI/pennylane/pull/1864)

  For example, we may set five values of a three-dimensional tensor
  in the following way:

  ```pycon
  >>> X = tf.zeros((3, 2, 9), dtype=tf.float64)
  >>> indices = [(0, 0, 1, 2, 2), (0, 0, 0, 0, 1), (1, 3, 8, 6, 7)]
  >>> values = [1 * i for i in range(1,6)]
  >>> qml.math.scatter_element_add(X, indices, values)
  <tf.Tensor: shape=(3, 2, 9), dtype=float64, numpy=
  array([[[0., 1., 0., 2., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0., 0., 0., 0., 3.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0., 0., 4., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 5., 0.]]])>
  ```

* All instances of `str.format` have been replace with f-strings.
  [(#1970)](https://github.com/PennyLaneAI/pennylane/pull/1970)

* Tests do not loop over automatically imported and instantiated operations any more,
  which was opaque and created unnecessarily many tests.
  [(#1895)](https://github.com/PennyLaneAI/pennylane/pull/1895)

* A `decompose()` method has been added to the `Operator` class such that we can
  obtain (and queue) decompositions directly from instances of operations.
  [(#1873)](https://github.com/PennyLaneAI/pennylane/pull/1873)

  ```pycon
  >>> op = qml.PhaseShift(0.3, wires=0)
  >>> op.decompose()
  [RZ(0.3, wires=[0])]
  ```

* `qml.circuit_drawer.tape_mpl` produces a matplotlib figure and axes given a tape.
  [(#1787)](https://github.com/PennyLaneAI/pennylane/pull/1787)

* The `AngleEmbedding`, `BasicEntanglerLayers` and `MottonenStatePreparation`
  templates now support parameters with batch dimension when using the `@qml.batch_params` decorator.
  [(#1812)](https://github.com/PennyLaneAI/pennylane/pull/1812)
  [(#1883)](https://github.com/PennyLaneAI/pennylane/pull/1883)
  [(#1893)](https://github.com/PennyLaneAI/pennylane/pull/1893)

* `qml.draw` now supports a `max_length` argument to help prevent text overflows when printing circuits.
  [(#1892)](https://github.com/PennyLaneAI/pennylane/pull/1892)

* `Identity` operation is now part of both the `ops.qubit` and `ops.cv`
  modules.
  [(#1956)](https://github.com/PennyLaneAI/pennylane/pull/1956)

<h3>Breaking changes</h3>

* The QNode has been re-written to support batch execution across the board,
  custom gradients, better decomposition strategies, and higher-order derivatives.
  [(#1807)](https://github.com/PennyLaneAI/pennylane/pull/1807)
  [(#1969)](https://github.com/PennyLaneAI/pennylane/pull/1969)

  - Arbitrary :math:`n`-th order derivatives are supported on hardware using gradient transforms
    such as the parameter-shift rule. To specify that an :math:`n`-th order derivative of a QNode
    will be computed, the `max_diff` argument should be set. By default, this is set to 1
    (first-order derivatives only). Increasing this value allows for higher order derivatives to be
    extracted, at the cost of additional (classical) computational overhead during the backwards
    pass.

  - When decomposing the circuit, the default decomposition strategy `expansion_strategy="gradient"`
    will prioritize decompositions that result in the smallest number of parametrized operations
    required to satisfy the differentiation method. While this may lead to a slight increase in
    classical processing, it significantly reduces the number of circuit evaluations needed to
    compute gradients of complicated unitaries.

    To return to the old behaviour, `expansion_strategy="device"` can be specified.

  Note that the old QNode remains accessible at `@qml.qnode_old.qnode`, however this will
  be removed in the next release.

* Certain features deprecated in `v0.19.0` have been removed:
  [(#1981)](https://github.com/PennyLaneAI/pennylane/pull/1981)
  [(#1963)](https://github.com/PennyLaneAI/pennylane/pull/1963)

  - The `qml.template` decorator (use a [
    QuantumTape](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.tape.QuantumTape.html)
    as a context manager to record operations and its `operations` attribute to
    return them, see the linked page for examples);
  - The `default.tensor` and `default.tensor.tf` experimental devices;
  - The `qml.fourier.spectrum` function (use the `qml.fourier.circuit_spectrum`
    or `qml.fourier.qnode_spectrum` functions instead);
  - The `diag_approx` keyword argument of `qml.metric_tensor` and
    `qml.QNGOptimizer` (pass `approx='diag'` instead).

* The default behaviour of the `qml.metric_tensor` transform has been modified.
  By default, the full metric tensor is computed, leading to higher cost than the previous
  default of computing the block diagonal only. At the same time, the Hadamard tests for
  the full metric tensor require an additional wire on the device, so that

  ```pycon
  >>> qml.metric_tensor(some_qnode)(weights)
  ```

  will revert back to the block diagonal restriction and raise a warning if the
  used device does not have an additional wire.
  [(#1725)](https://github.com/PennyLaneAI/pennylane/pull/1725)

* The `circuit_drawer` module has been renamed `drawer`.
  [(#1949)](https://github.com/PennyLaneAI/pennylane/pull/1949)

* The `par_domain` attribute in the operator class has been removed.
  [(#1907)](https://github.com/PennyLaneAI/pennylane/pull/1907)

* The `mutable` keyword argument has been removed from the QNode,
  due to underlying bugs that result in incorrect results being
  returned from immutable QNodes. This functionality will return
  in an upcoming release.
  [(#1807)](https://github.com/PennyLaneAI/pennylane/pull/1807)

* The reversible QNode differentiation method has been removed; the adjoint
  differentiation method is preferred instead (`diff_method='adjoint'`).
  [(#1807)](https://github.com/PennyLaneAI/pennylane/pull/1807)

* `QuantumTape.trainable_params` now is a list instead of a set. This
  means that `tape.trainable_params` will return a list unlike before,
  but setting the `trainable_params` with a set works exactly as before.
  [(#1904)](https://github.com/PennyLaneAI/pennylane/pull/1904)

* The `num_params` attribute in the operator class is now dynamic. This makes it easier
  to define operator subclasses with a flexible number of parameters.
  [(#1898)](https://github.com/PennyLaneAI/pennylane/pull/1898)
  [(#1909)](https://github.com/PennyLaneAI/pennylane/pull/1909)

* The static method `decomposition()`, formerly in the `Operation` class, has
  been moved to the base `Operator` class.
  [(#1873)](https://github.com/PennyLaneAI/pennylane/pull/1873)

* `DiagonalOperation` is not a separate subclass any more.
  [(#1889)](https://github.com/PennyLaneAI/pennylane/pull/1889)

  Instead, devices can check for the diagonal
  property using attributes:

  ```python
  from pennylane.ops.qubit.attributes import diagonal_in_z_basis

  if op in diagonal_in_z_basis:
      # do something
  ```
  Custom operations can be added to this attribute at runtime via
  `diagonal_in_z_basis.add("MyCustomOp")`.

<h3>Bug fixes</h3>

* Fixes a bug with `qml.probs` when using `default.qubit.jax`.
  [(#1998)](https://github.com/PennyLaneAI/pennylane/pull/1998)

* Fixes a bug where output tensors of a QNode would always be put on the
  default GPU with `default.qubit.torch`.
  [(#1982)](https://github.com/PennyLaneAI/pennylane/pull/1982)

* Device test suite doesn't use empty circuits so that it can also
  test the IonQ plugin, and it checks if operations are supported in
  more places.
  [(#1979)](https://github.com/PennyLaneAI/pennylane/pull/1979)

* Fixes a bug where the metric tensor was computed incorrectly when using
  gates with `gate.inverse=True`.
  [(#1987)](https://github.com/PennyLaneAI/pennylane/pull/1987)

* Corrects the documentation of `qml.transforms.classical_jacobian`
  for the Autograd interface (and improves test coverage).
  [(#1978)](https://github.com/PennyLaneAI/pennylane/pull/1978)

* Fixes a bug where differentiating a QNode with `qml.state` using the JAX
  interface raised an error.
  [(#1906)](https://github.com/PennyLaneAI/pennylane/pull/1906)

* Fixes a bug with the adjoint of `qml.QFT`.
  [(#1955)](https://github.com/PennyLaneAI/pennylane/pull/1955)

* Fixes a bug where the `ApproxTimeEvolution` template was not correctly
  computing the operation wires from the input Hamiltonian. This did not
  affect computation with the `ApproxTimeEvolution` template, but did
  cause circuit drawing to fail.
  [(#1952)](https://github.com/PennyLaneAI/pennylane/pull/1952)

* Fixes a bug where the classical preprocessing Jacobian
  computed by `qml.transforms.classical_jacobian` with JAX
  returned a reduced submatrix of the Jacobian.
  [(#1948)](https://github.com/PennyLaneAI/pennylane/pull/1948)

* Fixes a bug where the operations are not accessed in the correct order
  in `qml.fourier.qnode_spectrum`, leading to wrong outputs.
  [(#1935)](https://github.com/PennyLaneAI/pennylane/pull/1935)

* Fixes several Pylint errors.
  [(#1951)](https://github.com/PennyLaneAI/pennylane/pull/1951)

* Fixes a bug where the device test suite wasn't testing certain operations.
  [(#1943)](https://github.com/PennyLaneAI/pennylane/pull/1943)

* Fixes a bug where batch transforms would mutate a QNodes execution options.
  [(#1934)](https://github.com/PennyLaneAI/pennylane/pull/1934)

* `qml.draw` now supports arbitrary templates with matrix parameters.
  [(#1917)](https://github.com/PennyLaneAI/pennylane/pull/1917)

* `QuantumTape.trainable_params` now is a list instead of a set, making
  it more stable in very rare edge cases.
  [(#1904)](https://github.com/PennyLaneAI/pennylane/pull/1904)

* `ExpvalCost` now returns corrects results shape when `optimize=True` with
  shots batch.
  [(#1897)](https://github.com/PennyLaneAI/pennylane/pull/1897)

* `qml.circuit_drawer.MPLDrawer` was slightly modified to work with
  matplotlib version 3.5.
  [(#1899)](https://github.com/PennyLaneAI/pennylane/pull/1899)

* `qml.CSWAP` and `qml.CRot` now define `control_wires`, and `qml.SWAP`
  returns the default empty wires object.
  [(#1830)](https://github.com/PennyLaneAI/pennylane/pull/1830)

* The `requires_grad` attribute of `qml.numpy.tensor` objects is now
  preserved when pickling/unpickling the object.
  [(#1856)](https://github.com/PennyLaneAI/pennylane/pull/1856)

* Device tests no longer throw warnings about the `requires_grad`
  attribute of variational parameters.
  [(#1913)](https://github.com/PennyLaneAI/pennylane/pull/1913)

* `AdamOptimizer` and `AdagradOptimizer` had small fixes to their
  optimization step updates.
  [(#1929)](https://github.com/PennyLaneAI/pennylane/pull/1929)

* Fixes a bug where differentiating a QNode with multiple array
  arguments via `qml.gradients.param_shift` throws an error.
  [(#1989)](https://github.com/PennyLaneAI/pennylane/pull/1989)

* `AmplitudeEmbedding` template no longer produces a `ComplexWarning`
  when the `features` parameter is batched and provided as a 2D array.
  [(#1990)](https://github.com/PennyLaneAI/pennylane/pull/1990)

* `qml.circuit_drawer.CircuitDrawer` no longer produces an error
  when attempting to draw tapes inside of circuits (e.g. from
  decomposition of an operation or manual placement).
  [(#1994)](https://github.com/PennyLaneAI/pennylane/pull/1994)

* Fixes a bug where using SciPy sparse matrices with the new QNode
  could lead to a warning being raised about prioritizing the TensorFlow
  and PyTorch interfaces.
  [(#2001)](https://github.com/PennyLaneAI/pennylane/pull/2001)

* Fixed a bug where the `QueueContext` was not empty when first importing PennyLane.
  [(#1957)](https://github.com/PennyLaneAI/pennylane/pull/1957)

* Fixed circuit drawing problem with `Interferometer` and `CVNeuralNet`.
  [(#1953)](https://github.com/PennyLaneAI/pennylane/pull/1953)

<h3>Documentation</h3>

* Added examples in documentation for some operations.
  [(#1902)](https://github.com/PennyLaneAI/pennylane/pull/1902)

* Improves the Developer's Guide Testing document.
  [(#1896)](https://github.com/PennyLaneAI/pennylane/pull/1896)

* Added documentation examples for `AngleEmbedding`, `BasisEmbedding`, `StronglyEntanglingLayers`,
  `SqueezingEmbedding`, `DisplacementEmbedding`, `MottonenStatePreparation` and `Interferometer`.
  [(#1910)](https://github.com/PennyLaneAI/pennylane/pull/1910)
  [(#1908)](https://github.com/PennyLaneAI/pennylane/pull/1908)
  [(#1912)](https://github.com/PennyLaneAI/pennylane/pull/1912)
  [(#1920)](https://github.com/PennyLaneAI/pennylane/pull/1920)
  [(#1936)](https://github.com/PennyLaneAI/pennylane/pull/1936)
  [(#1937)](https://github.com/PennyLaneAI/pennylane/pull/1937)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Catalina Albornoz, Guillermo Alonso-Linaje, Juan Miguel Arrazola, Ali Asadi, Utkarsh Azad, Samuel Banning, Benjamin Cordier, Alain Delgado,
Olivia Di Matteo, Anthony Hayes, David Ittah, Josh Izaac, Soran Jahangiri, Jalani Kanem, Ankit Khandelwal, Nathan Killoran, Shumpei
Kobayashi, Robert Lang, Christina Lee, Cedric Lin, Alejandro Montanez, Romain Moyard, Lee James O'Riordan, Chae-Yeun Park, Isidor Schoch,
Maria Schuld, Jay Soni, Antal Száva, Rodrigo Vargas, David Wierichs, Roeland Wiersema, Moritz Willmann.
