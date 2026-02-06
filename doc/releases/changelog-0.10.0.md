
# Release 0.10.0

<h3>New features since last release</h3>

<h4>New and improved simulators</h4>

* Added a new device, `default.qubit.tf`, a pure-state qubit simulator written using TensorFlow.
  As a result, it supports classical backpropagation as a means to compute the Jacobian. This can
  be faster than the parameter-shift rule for computing quantum gradients
  when the number of parameters to be optimized is large.

  `default.qubit.tf` is designed to be used with end-to-end classical backpropagation
  (`diff_method="backprop"`) with the TensorFlow interface. This is the default method
  of differentiation when creating a QNode with this device.

  Using this method, the created QNode is a 'white-box' that is
  tightly integrated with your TensorFlow computation, including
  [AutoGraph](https://www.tensorflow.org/guide/function) support:

  ```pycon
  >>> dev = qp.device("default.qubit.tf", wires=1)
  >>> @tf.function
  ... @qp.qnode(dev, interface="tf", diff_method="backprop")
  ... def circuit(x):
  ...     qp.RX(x[1], wires=0)
  ...     qp.Rot(x[0], x[1], x[2], wires=0)
  ...     return qp.expval(qp.PauliZ(0))
  >>> weights = tf.Variable([0.2, 0.5, 0.1])
  >>> with tf.GradientTape() as tape:
  ...     res = circuit(weights)
  >>> print(tape.gradient(res, weights))
  tf.Tensor([-2.2526717e-01 -1.0086454e+00  1.3877788e-17], shape=(3,), dtype=float32)
  ```

  See the `default.qubit.tf`
  [documentation](https://pennylane.ai/en/stable/code/api/pennylane.beta.plugins.DefaultQubitTF.html)
  for more details.

* The [default.tensor plugin](https://github.com/XanaduAI/pennylane/blob/master/pennylane/beta/plugins/default_tensor.py)
  has been significantly upgraded. It now allows two different
  tensor network representations to be used: `"exact"` and `"mps"`. The former uses a
  exact factorized representation of quantum states, while the latter uses a matrix product state
  representation.
  ([#572](https://github.com/XanaduAI/pennylane/pull/572))
  ([#599](https://github.com/XanaduAI/pennylane/pull/599))

<h4>New machine learning functionality and integrations</h4>

* PennyLane QNodes can now be converted into Torch layers, allowing for creation of quantum and
  hybrid models using the `torch.nn` API.
  [(#588)](https://github.com/XanaduAI/pennylane/pull/588)

  A PennyLane QNode can be converted into a `torch.nn` layer using the `qp.qnn.TorchLayer` class:

  ```pycon
  >>> @qp.qnode(dev)
  ... def qnode(inputs, weights_0, weight_1):
  ...    # define the circuit
  ...    # ...

  >>> weight_shapes = {"weights_0": 3, "weight_1": 1}
  >>> qlayer = qp.qnn.TorchLayer(qnode, weight_shapes)
  ```

  A hybrid model can then be easily constructed:

  ```pycon
  >>> model = torch.nn.Sequential(qlayer, torch.nn.Linear(2, 2))
  ```

* Added a new "reversible" differentiation method which can be used in simulators, but not hardware.

  The reversible approach is similar to backpropagation, but trades off extra computation for
  enhanced memory efficiency. Where backpropagation caches the state tensors at each step during
  a simulated evolution, the reversible method only caches the final pre-measurement state.

  Compared to the parameter-shift method, the reversible method can be faster or slower,
  depending on the density and location of parametrized gates in a circuit
  (circuits with higher density of parametrized gates near the end of the circuit will see a benefit).
  [(#670)](https://github.com/XanaduAI/pennylane/pull/670)

  ```pycon
  >>> dev = qp.device("default.qubit", wires=2)
  ... @qp.qnode(dev, diff_method="reversible")
  ... def circuit(x):
  ...     qp.RX(x, wires=0)
  ...     qp.RX(x, wires=0)
  ...     qp.CNOT(wires=[0,1])
  ...     return qp.expval(qp.PauliZ(0))
  >>> qp.grad(circuit)(0.5)
  (array(-0.47942554),)
  ```

<h4>New templates and cost functions</h4>

* Added the new templates `UCCSD`, `SingleExcitationUnitary`, and`DoubleExcitationUnitary`,
  which together implement the Unitary Coupled-Cluster Singles and Doubles (UCCSD) ansatz
  to perform VQE-based quantum chemistry simulations using PennyLane-QChem.
  [(#622)](https://github.com/XanaduAI/pennylane/pull/622)
  [(#638)](https://github.com/XanaduAI/pennylane/pull/638)
  [(#654)](https://github.com/XanaduAI/pennylane/pull/654)
  [(#659)](https://github.com/XanaduAI/pennylane/pull/659)
  [(#622)](https://github.com/XanaduAI/pennylane/pull/622)

* Added module `pennylane.qnn.cost` with class `SquaredErrorLoss`. The module contains classes
  to calculate losses and cost functions on circuits with trainable parameters.
  [(#642)](https://github.com/XanaduAI/pennylane/pull/642)

<h3>Improvements</h3>

* Improves the wire management by making the `Operator.wires` attribute a `wires` object.
  [(#666)](https://github.com/XanaduAI/pennylane/pull/666)

* A significant improvement with respect to how QNodes and interfaces mark quantum function
  arguments as differentiable when using Autograd, designed to improve performance and make
  QNodes more intuitive.
  [(#648)](https://github.com/XanaduAI/pennylane/pull/648)
  [(#650)](https://github.com/XanaduAI/pennylane/pull/650)

  In particular, the following changes have been made:

  - A new `ndarray` subclass `pennylane.numpy.tensor`, which extends NumPy arrays with
    the keyword argument and attribute `requires_grad`. Tensors which have `requires_grad=False`
    are treated as non-differentiable by the Autograd interface.

  - A new subpackage `pennylane.numpy`, which wraps `autograd.numpy` such that NumPy functions
    accept the `requires_grad` keyword argument, and allows Autograd to differentiate
    `pennylane.numpy.tensor` objects.

  - The `argnum` argument to `qp.grad` is now optional; if not provided, arguments explicitly
    marked as `requires_grad=False` are excluded for the list of differentiable arguments.
    The ability to pass `argnum` has been retained for backwards compatibility, and
    if present the old behaviour persists.

* The QNode Torch interface now inspects QNode positional arguments.
  If any argument does not have the attribute `requires_grad=True`, it
  is automatically excluded from quantum gradient computations.
  [(#652)](https://github.com/XanaduAI/pennylane/pull/652)
  [(#660)](https://github.com/XanaduAI/pennylane/pull/660)

* The QNode TF interface now inspects QNode positional arguments.
  If any argument is not being watched by a `tf.GradientTape()`,
  it is automatically excluded from quantum gradient computations.
  [(#655)](https://github.com/XanaduAI/pennylane/pull/655)
  [(#660)](https://github.com/XanaduAI/pennylane/pull/660)

* QNodes have two new public methods: `QNode.set_trainable_args()` and `QNode.get_trainable_args()`.
  These are designed to be called by interfaces, to specify to the QNode which of its
  input arguments are differentiable. Arguments which are non-differentiable will not be converted
  to PennyLane Variable objects within the QNode.
  [(#660)](https://github.com/XanaduAI/pennylane/pull/660)

* Added `decomposition` method to PauliX, PauliY, PauliZ, S, T, Hadamard, and PhaseShift gates, which
  decomposes each of these gates into rotation gates.
  [(#668)](https://github.com/XanaduAI/pennylane/pull/668)

* The `CircuitGraph` class now supports serializing contained circuit operations
  and measurement basis rotations to an OpenQASM2.0 script via the new
  `CircuitGraph.to_openqasm()` method.
  [(#623)](https://github.com/XanaduAI/pennylane/pull/623)

<h3>Breaking changes</h3>

* Removes support for Python 3.5.
  [(#639)](https://github.com/XanaduAI/pennylane/pull/639)

<h3>Documentation</h3>

* Various small typos were fixed.

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Thomas Bromley, Jack Ceroni, Alain Delgado Gran, Theodor Isacsson, Josh Izaac,
Nathan Killoran, Maria Schuld, Antal Száva, Nicola Vitucci.

