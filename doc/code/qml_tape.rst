qml.tape
========

.. warning::

    The new PennyLane tape mode is **experimental**, and does not currently have feature-parity with
    the existing QNode. `Feedback and bug reports
    <https://github.com/PennyLaneAI/pennylane/issues>`__ are encouraged and will help improve the
    new tape mode.

    Tape mode can be enabled globally via the :func:`~.enable_tape` function, without changing your
    PennyLane code:

    >>> qml.enable_tape()


The PennyLane tape module provides a new QNode class, rewritten from the ground-up,
that uses a :class:`~.QuantumTape` to represent the internal variational quantum circuit.
Tape mode provides several advantanges over the standard PennyLane QNode.

* In tape mode, QNode arguments no longer become :class:`~.Variable` objects within the
  QNode; arguments can be easily inspected, printed, and manipulated.

  >>> dev = qml.device("default.qubit", wires=1)
  >>> @qml.qnode(dev)
  >>> def circuit(x):
  ...     print("Parameter value:", x)
  ...     qml.RX(x, wires=0)
  ...     return qml.expval(qml.PauliZ(0))
  >>> circuit(0.5)
  Parameter value: 0.5
  tensor(0.87758256, requires_grad=True)

* Tape mode allows for differentiable classical processing within the QNode.

  >>> dev = qml.device("default.qubit", wires=1)
  >>> @qml.qnode(dev, interface="tf")
  >>> def circuit(p):
  ...     qml.RX(tf.sin(p[0])**2 + p[1], wires=0)
  ...     return qml.expval(qml.PauliZ(0))
  >>> params = tf.Variable([0.5, 0.1], dtype=tf.float64)
  >>> with tf.GradientTape() as tape:
  ...     res = circuit(params)
  >>> grad = tape.gradient(res, params)
  >>> print(res)
  tf.Tensor(0.9460913127754935, shape=(), dtype=float64)
  >>> print(grad)
  tf.Tensor([-0.27255248 -0.32390003], shape=(2,), dtype=float64)

  As a result, quantum decompositions that require classical processing
  are fully supported and end-to-end differentiable in tape mode.

* There is no longer any restriction on the QNode signature; the QNode can be
  defined and called following the same rules as standard Python functions.

  For example, the following QNode uses positional, named, and variable
  keyword arguments:

  >>> x = torch.tensor(0.1, requires_grad=True)
  >>> y = torch.tensor([0.2, 0.3], requires_grad=True)
  >>> z = torch.tensor(0.4, requires_grad=True)
  >>> @qml.qnode(dev, interface="torch")
  ... def circuit(p1, p2=y, **kwargs):
  ...     qml.RX(p1, wires=0)
  ...     qml.RY(p2[0] * p2[1], wires=0)
  ...     qml.RX(kwargs["p3"], wires=0)
  ...     return qml.var(qml.PauliZ(0))

  When we call the QNode, we may pass the arguments by name
  even if defined positionally; any argument not provided will
  use the default value.

  >>> res = circuit(p1=x, p3=z)
  >>> print(res)
  tensor(0.2327, dtype=torch.float64, grad_fn=<SelectBackward>)
  >>> res.backward()
  >>> print(x.grad, y.grad, z.grad)
  tensor(0.8396) tensor([0.0289, 0.0193]) tensor(0.8387)

* Tape mode provides various optimizations, reducing pre- and post-processing overhead,
  and reduces the number of quantum evaluations in certain cases.



.. automodapi:: pennylane.tape
    :no-heading:
    :include-all-objects:
    :skip: enable_tape, disable_tape
