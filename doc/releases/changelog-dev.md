:orphan:

# Release 0.20.0-dev (development release)

<h3>New features since last release</h3>

* PennyLane now supports drawing a QNode with matplotlib!
  [(#1803)](https://github.com/PennyLaneAI/pennylane/pull/1803)

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

  <img src="https://pennylane.readthedocs.io/en/latest/_static/draw_mpl_qnode/main_example.png" width=70%/>

* It is now possible to use TensorFlow's [AutoGraph
  mode](https://www.tensorflow.org/guide/function) with QNodes on all devices and with arbitrary
  differentiation methods. Previously, AutoGraph mode only support `diff_method="backprop"`. This
  will result in significantly more performant model execution, at the cost of a more expensive
  initial compilation. [(#1866)](https://github.com/PennyLaneAI/pennylane/pull/1886)

  Use AutoGraph to convert your QNodes or cost functions into TensorFlow
  graphs by decorating them with `@tf.function`:

  ```python
  dev = qml.device("lightning.qubit", wires=2)

  @qml.beta.qnode(dev, diff_method="adjoint", interface="tf", max_diff=1)
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

* `qml.math.scatter_element_add` now supports adding multiple values at
  multiple indices with a single function call, in all interfaces
  [(#1864)](https://github.com/PennyLaneAI/pennylane/pull/1864)

  For example, we may set five values of a three-dimensional tensor
  in the following way:

  ```pycon
  >>> X = tf.zeros((3, 2, 9), dtype=tf.float64)
  >>> indices = [(0, 0, 1, 2, 2), (0, 0, 0, 0, 1), (1, 3, 8, 6, 7)]
  >>> values = [0.1 * i for i in range(5)]
  >>> qml.math.scatter_element_add(X, indices, values)
  <tf.Tensor: shape=(3, 2, 9), dtype=float64, numpy=
  array([[[0., 1., 0., 2., 0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0., 0., 0., 0., 3.],
          [0., 0., 0., 0., 0., 0., 0., 0., 0.]],

         [[0., 0., 0., 0., 0., 0., 4., 0., 0.],
          [0., 0., 0., 0., 0., 0., 0., 5., 0.]]])>
  ```

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
  [fourier.reconstruct docstring](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.fourier.reconstruct.html).

* A thermal relaxation channel is added to the Noisy channels. The channel description can be
  found on the supplementary information of [Quantum classifier with tailored quantum kernels](https://arxiv.org/abs/1909.02611).
  [(#1766)](https://github.com/PennyLaneAI/pennylane/pull/1766)

* Added the identity observable to be an operator. Now we can explicitly call the identity
  operation on our quantum circuits for both qubit and CV devices.
  [(#1829)](https://github.com/PennyLaneAI/pennylane/pull/1829)

* The qml.Barrier() operator has been added. With it we can separate blocks in compilation or use it as a visual tool.
  [(#1844)](https://github.com/PennyLaneAI/pennylane/pull/1844)
  
* Added density matrix initialization gate for mixed state simulation. [(#1686)](https://github.com/PennyLaneAI/pennylane/issues/1686)

<h3>Improvements</h3>


* The QNode has been re-written to support batch execution across the board,
  custom gradients, better decomposition strategies, and higher-order derivatives.
  [(#1807)](https://github.com/PennyLaneAI/pennylane/pull/1807)

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

* ``qml.circuit_drawer.tape_mpl`` produces a matplotlib figure and axes given a tape.
  [(#1787)](https://github.com/PennyLaneAI/pennylane/pull/1787)

* AngleEmbedding now supports `batch_params` decorator. [(#1812)](https://github.com/PennyLaneAI/pennylane/pull/1812)

* MottonenStatePreparation now supports `batch_params` decorator. [(#1893)](https://github.com/PennyLaneAI/pennylane/pull/1893)

* CircuitDrawer now supports a `max_length` argument to help prevent text overflows when printing circuits to the CLI. [#1841](https://github.com/PennyLaneAI/pennylane/pull/1841)

<h3>Breaking changes</h3>

* The `par_domain` attribute in the operator class has been removed. 
  [(#1907)](https://github.com/PennyLaneAI/pennylane/pull/1907)

- The `mutable` keyword argument has been removed from the QNode.
  [(#1807)](https://github.com/PennyLaneAI/pennylane/pull/1807)

- The reversible QNode differentiation method has been removed.
  [(#1807)](https://github.com/PennyLaneAI/pennylane/pull/1807)

* `QuantumTape.trainable_params` now is a list instead of a set. This
  means that `tape.trainable_params` will return a list unlike before,
  but setting the `trainable_params` with a set works exactly as before.
  [(#1904)](https://github.com/PennyLaneAI/pennylane/pull/1904)

* The `num_params` attribute in the operator class is now dynamic. This makes it easier
  to define operator subclasses with a flexible number of parameters. 
  [(#1898)](https://github.com/PennyLaneAI/pennylane/pull/1898)

* The static method `decomposition()`, formerly in the `Operation` class, has
  been moved to the base `Operator` class.
  [(#1873)](https://github.com/PennyLaneAI/pennylane/pull/1873)

* `DiagonalOperation` is not a separate subclass any more.
  [(#1889)](https://github.com/PennyLaneAI/pennylane/pull/1889)

  Instead, devices can check for the diagonal
  property using attributes:

  ``` python
  from pennylane.ops.qubit.attributes import diagonal_in_z_basis

  if op in diagonal_in_z_basis:
      # do something
  ```

<h3>Deprecations</h3>

<h3>Bug fixes</h3>

* `qml.CSWAP` and `qml.CRot` now define `control_wires`, and `qml.SWAP`

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

<h3>Documentation</h3>

* Added examples in documentation for some operations.
  [(#1902)](https://github.com/PennyLaneAI/pennylane/pull/1902)

* Improves the Developer's Guide Testing document.
  [(#1896)](https://github.com/PennyLaneAI/pennylane/pull/1896)

* Add documentation example for AngleEmbedding and BasisEmbedding.
  [(#1910)](https://github.com/PennyLaneAI/pennylane/pull/1910)
  [(#1908)](https://github.com/PennyLaneAI/pennylane/pull/1908)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso-Linaje, Benjamin Cordier, Olivia Di Matteo, David Ittah, Josh Izaac,
Jalani Kanem, Ankit Khandelwal, Shumpei Kobayashi, Christina Lee, Alejandro Montanez,
Romain Moyard, Maria Schuld, Jay Soni, David Wierichs
