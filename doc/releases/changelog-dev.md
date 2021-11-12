:orphan:

# Release 0.20.0-dev (development release)

<h3>New features since last release</h3>

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

* A thermal relaxation channel is added to the Noisy channels. The channel description can be 
  found on the supplementary information of [Quantum classifier with tailored quantum kernels](https://arxiv.org/abs/1909.02611).
  [(#1766)](https://github.com/PennyLaneAI/pennylane/pull/1766)
  
* Added the identity observable to be an operator. Now we can explicitly call the identity 
  operation on our quantum circuits for both qubit and CV devices.
  [(#1829)](https://github.com/PennyLaneAI/pennylane/pull/1829) 

<h3>Improvements</h3>

* ``qml.circuit_drawer.draw_mpl`` produces a matplotlib figure and axes given a tape.
  [(#1787)](https://github.com/PennyLaneAI/pennylane/pull/1787)

* AngleEmbedding now supports `batch_params` decorator. [(#1812)](https://github.com/PennyLaneAI/pennylane/pull/1812)

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Bug fixes</h3>

* `qml.CSWAP` and `qml.CRot` now define `control_wires`, and `qml.SWAP` 
  returns the default empty wires object.
  [(#1830)](https://github.com/PennyLaneAI/pennylane/pull/1830)

* The `requires_grad` attribute of `qml.numpy.tensor` objects is now
  preserved when pickling/unpickling the object.
  [(#1856)](https://github.com/PennyLaneAI/pennylane/pull/1856)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order): 

Josh Izaac, Jalani Kanem, Christina Lee, Guillermo Alonso-Linaje, Alejandro Montanez, Jay Soni.
