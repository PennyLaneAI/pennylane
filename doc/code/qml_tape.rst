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

    Once enabled, tape mode can be disabled via :func:`~.disable_tape`.


Tape-mode QNodes
----------------

The PennyLane tape module provides a new QNode class, rewritten from the ground-up,
that uses a :class:`~.QuantumTape` to represent the internal variational quantum circuit.
Tape mode provides several advantanges over the standard PennyLane QNode.

* **Support for in-QNode classical processing**: Tape mode allows for differentiable classical
  processing within the QNode.

  .. code-block:: python3

      qml.enable_tape()
      dev = qml.device("default.qubit", wires=1)

      @qml.qnode(dev, interface="tf")
      def circuit(p):
          qml.RX(tf.sin(p[0])**2 + p[1], wires=0)
          return qml.expval(qml.PauliZ(0))

  The classical processing functions used within the QNode must match
  the QNode interface. Here, we use TensorFlow:

  >>> params = tf.Variable([0.5, 0.1], dtype=tf.float64)
  >>> with tf.GradientTape() as tape:
  ...     res = circuit(params)
  >>> grad = tape.gradient(res, params)
  >>> print(res)
  tf.Tensor(0.9460913127754935, shape=(), dtype=float64)
  >>> print(grad)
  tf.Tensor([-0.27255248 -0.32390003], shape=(2,), dtype=float64)

  As a result of this change, quantum decompositions that require classical processing
  are fully supported and end-to-end differentiable in tape mode.

* **No more Variable wrapping**: In tape mode, QNode arguments no longer become :class:`~.Variable`
  objects within the QNode.

  .. code-block:: python3

      qml.enable_tape()
      dev = qml.device("default.qubit", wires=1)

      @qml.qnode(dev)
      def circuit(x):
          print("Parameter value:", x)
          qml.RX(x, wires=0)
          return qml.expval(qml.PauliZ(0))

  Internal QNode parameters can be easily inspected, printed, and manipulated:

  >>> circuit(0.5)
  Parameter value: 0.5
  tensor(0.87758256, requires_grad=True)

* **Return the quantum state**: In tape mode, QNodes bound to statevector simulators
  can return the quantum state using the :func:`~.state` function:

  .. code-block:: python3

      qml.enable_tape()
      dev = qml.device("default.qubit", wires=2)

      @qml.qnode(dev)
      def circuit():
          qml.Hadamard(wires=1)
          return qml.state()

  >>> circuit()
  array([0.70710678+0.j, 0.70710678+0.j, 0.        +0.j, 0.        +0.j])

  Calculating the derivative of :func:`~.state` is currently only supported when using the
  classical backpropagation differentiation method (``diff_method="backprop"``) with a
  compatible device.

* **Less restrictive QNode signatures**: There is no longer any restriction on the QNode signature; the QNode can be
  defined and called following the same rules as standard Python functions.

  For example, the following QNode uses positional, named, and variable
  keyword arguments:

  .. code-block:: python

      qml.enable_tape()

      x = torch.tensor(0.1, requires_grad=True)
      y = torch.tensor([0.2, 0.3], requires_grad=True)
      z = torch.tensor(0.4, requires_grad=True)

      @qml.qnode(dev, interface="torch")
      def circuit(p1, p2=y, **kwargs):
          qml.RX(p1, wires=0)
          qml.RY(p2[0] * p2[1], wires=0)
          qml.RX(kwargs["p3"], wires=0)
          return qml.var(qml.PauliZ(0))

  When we call the QNode, we may pass the arguments by name
  even if defined positionally; any argument not provided will
  use the default value.

  >>> res = circuit(p1=x, p3=z)
  >>> print(res)
  tensor(0.2327, dtype=torch.float64, grad_fn=<SelectBackward>)
  >>> res.backward()
  >>> print(x.grad, y.grad, z.grad)
  tensor(0.8396) tensor([0.0289, 0.0193]) tensor(0.8387)

* **Unifying all QNodes**: The tape-mode QNode merges all QNodes (including the :class:`~.JacobianQNode`
  and the :class:`~.PassthruQNode`) into a single unified QNode, with identicaly behaviour regardless
  of the differentiation type.

  In addition, it is now possible to inspect the internal variational quantum circuit structure
  of QNodes when using classical backpropagation (which is not support in the standard
  :class:`~.PassthruQNode`).

* **Optimizations**: Tape mode provides various performance optimizations, reducing pre- and post-processing
  overhead, and reduces the number of quantum evaluations in certain cases.

.. warning::

    In tape-mode, the QNode does not yet have feature-parity with the standard PennyLane
    QNode. Features currently not available in tape mode include:

    * Circuit drawing and visualization

    * Metric tensor computation

    * The ability to automatically extract the layer structure of variational circuits

    * Tape-mode QNodes cannot be used with the ``qml.qnn`` module yet


Quantum tapes
-------------

Under the hood, tape mode is able to provide these new features by significantly overhauling
the internal structure of the QNode. When tape mode is enabled, the QNode is no longer
responsible for recording quantum operations, executing devices, or computing gradients---these
tasks have been delegated to an internal object that is created by the QNode, the **quantum tape**.


In addition to being created internally by QNodes in tape mode, quantum tapes can also be created,
nested, expanded (via :meth:`~.QuantumTape.expand`), and executed manually. Tape subclasses also provide
additional gradient methods:

.. autosummary::

    ~pennylane.tape.QuantumTape
    ~pennylane.tape.QubitParamShiftTape
    ~pennylane.tape.CVParamShiftTape
    ~pennylane.tape.ReversibleTape

Finally, quantum tapes are fully compatible with autodifferentiating via NumPy/Autograd,
TensorFlow, and PyTorch:

.. autosummary::
    :toctree: api

    ~pennylane.tape.interfaces.tf.TFInterface
    ~pennylane.tape.interfaces.torch.TorchInterface
    ~pennylane.tape.interfaces.autograd.AutogradInterface

For more details and examples, please see the tape documentation.

.. automodapi:: pennylane.tape
    :no-main-docstr:
    :include-all-objects:
    :skip: enable_tape, disable_tape
