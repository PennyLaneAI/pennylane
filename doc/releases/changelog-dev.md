:orphan:

# Release 0.25.0-dev (development release)

<h3>New features since last release</h3>

* `DefaultQubit` devices now natively support parameter broadcasting
  and `qml.gradients.param_shift` allows to make use of broadcasting.
  [(#2627)](https://github.com/PennyLaneAI/pennylane/pull/2627)
  
  Instead of utilizing the `broadcast_expand` transform, `DefaultQubit`
  devices now are able to directly execute broadcasted circuits, providing
  a faster way of executing the same circuit at varied parameter positions.

  Given a standard `QNode`

  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev)
  def circuit(x, y):
      qml.RX(x, wires=0)
      qml.RY(y, wires=0)
      return qml.expval(qml.PauliZ(0))
  ```

  we can call it with broadcasted parameters:

  ```pycon
  >>> x = np.array([0.4, 1.2, 0.6], requires_grad=True)
  >>> y = np.array([0.9, -0.7, 4.2], requires_grad=True)
  >>> circuit(x, y)
  tensor([ 0.5725407 ,  0.2771465 , -0.40462972], requires_grad=True)
  ```

  It's also possible to broadcast only some parameters:

  ```pycon
  >>> x = np.array([0.4, 1.2, 0.6], requires_grad=True)
  >>> y = np.array(0.23, requires_grad=True)
  >>> circuit(x, y)
  tensor([0.89680614, 0.35281557, 0.80360155], requires_grad=True)
  ```

  The gradient transform `qml.gradients.param_shift` now accepts the new Boolean keyword
  argument `broadcast`. If it is set to `True`, a single broadcasted
  tape is created per trainable operation, containing all shifted evaluations for that operation.
  For example, for the above circuit:

  ```pycon
  >>> x, y = np.array([0.4, 0.23], requires_grad=True)
  >>> circuit(x, y)
  >>> tapes, fn = qml.gradients.param_shift(circuit.qtape, broadcast=True)
  >>> len(tapes)
  2
  >>> (tapes[0].batch_size, tapes[1].batch_size)
  (2, 2)
  ```

  For `broadcast=False` (the default), multiple unbroadcasted tapes are created as before.

  ```pycon
  >>> tapes, fn = qml.gradients.param_shift(circuit.qtape, broadcast=False)
  >>> len(tapes)
  4
  >>> [t.batch_size for t in tapes]
  [None, None, None, None]
  ```

  An advantage of using `broadcast=True` is a speedup:

  ```pycon
  >>> number = 1000
  >>> broadcasted_call = "qml.gradients.param_shift(circuit, broadcast=True)(x, y)"
  >>> timeit.timeit(broadcasted_call, globals=globals(), number=number) / number
  0.004547867801011307
  >>> serial_call = "qml.gradients.param_shift(circuit, broadcast=False)(x, y)"
  >>> timeit.timeit(broadcasted_call, globals=globals(), number=number) / number
  0.006740601188008441
  ```

  This speedup grows with the number of shifts and qubits until all preprocessing and postprocessing
  overhead becomes negligible. While it will depend strongly on many details, a wide range of
  circuits will be differentiated significantly faster.

  Note that `QuantumTapes`/`QNodes` with multiple return values and shot vectors are not supported
  yet and that the differentiated operations are required to support broadcasting when using
  `broadcast=True`. One way of checking the latter is the `Attribute` `supports_broadcasting`:

  ```pycon
  >>> qml.RX in qml.ops.qubit.attributes.suports_broadcasting
  True
  ```

* Added the new optimizer, `qml.SPSAOptimizer` that implements the simultaneous
  perturbation stochastic approximation method based on
  [An Overview of the Simultaneous Perturbation Method for Efficient Optimization](https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF).
  [(#2661)](https://github.com/PennyLaneAI/pennylane/pull/2661)

  It is a suitable optimizer for cost functions whose evaluation may involve
  noise, as optimization with SPSA may significantly decrease the number of
  quantum executions for the entire optimization.

  ```pycon
  >>> dev = qml.device("default.qubit", wires=1)
  >>> def circuit(params):
  ...     qml.RX(params[0], wires=0)
  ...     qml.RY(params[1], wires=0)
  >>> coeffs = [1, 1]
  >>> obs = [qml.PauliX(0), qml.PauliZ(0)]
  >>> H = qml.Hamiltonian(coeffs, obs)
  >>> @qml.qnode(dev)
  ... def cost(params):
  ...     circuit(params)
  ...     return qml.expval(H)
  >>> params = np.random.normal(0, np.pi, (2), requires_grad=True)
  >>> print(params)
  [-5.92774911 -4.26420843]
  >>> print(cost(params))
  0.43866366253270167
  >>> max_iterations = 50
  >>> opt = qml.SPSAOptimizer(maxiter=max_iterations)
  >>> for _ in range(max_iterations):
  ...     params, energy = opt.step_and_cost(cost, params)
  >>> print(params)
  [-6.21193761 -2.99360548]
  >>> print(energy)
  -1.1258709813834058
  ```
  
* New PennyLane-inspired `sketch` and `sketch_dark` styles are now available for drawing circuit diagram graphics. 
  [(#2709)](https://github.com/PennyLaneAI/pennylane/pull/2709)

<h3>Improvements</h3>

* Adds a new function to compare operators. `qml.equal` can be used to compare equality of parametric operators taking into account their interfaces and trainability.
  [(#2651)](https://github.com/PennyLaneAI/pennylane/pull/2651)

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ankit Khandelwal, Ixchel Meza Chavez, David Wierichs, Moritz Willmann
