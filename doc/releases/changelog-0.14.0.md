
# Release 0.14.0

<h3>New features since last release</h3>

<h4>Perform quantum machine learning with JAX</h4>

* QNodes created with `default.qubit` now support a JAX interface, allowing JAX to be used
  to create, differentiate, and optimize hybrid quantum-classical models.
  [(#947)](https://github.com/PennyLaneAI/pennylane/pull/947)

  This is supported internally via a new `default.qubit.jax` device. This device runs end to end in
  JAX, meaning that it supports all of the awesome JAX transformations (`jax.vmap`, `jax.jit`,
  `jax.hessian`, etc).

  Here is an example of how to use the new JAX interface:

  ```python
  dev = qml.device("default.qubit", wires=1)
  @qml.qnode(dev, interface="jax", diff_method="backprop")
  def circuit(x):
      qml.RX(x[1], wires=0)
      qml.Rot(x[0], x[1], x[2], wires=0)
      return qml.expval(qml.PauliZ(0))

  weights = jnp.array([0.2, 0.5, 0.1])
  grad_fn = jax.grad(circuit)
  print(grad_fn(weights))
  ```

  Currently, only `diff_method="backprop"` is supported, with plans to support more in the future.

<h4>New, faster, quantum gradient methods</h4>

* A new differentiation method has been added for use with simulators. The `"adjoint"`
  method operates after a forward pass by iteratively applying inverse gates to scan backwards
  through the circuit.
  [(#1032)](https://github.com/PennyLaneAI/pennylane/pull/1032)

  This method is similar to the reversible method, but has a lower time
  overhead and a similar memory overhead. It follows the approach provided by
  [Jones and Gacon](https://arxiv.org/abs/2009.02823). This method is only compatible with certain
  statevector-based devices such as `default.qubit`.

  Example use:

  ```python
  import pennylane as qp

  wires = 1
  device = qml.device("default.qubit", wires=wires)

  @qml.qnode(device, diff_method="adjoint")
  def f(params):
      qml.RX(0.1, wires=0)
      qml.Rot(*params, wires=0)
      qml.RX(-0.3, wires=0)
      return qml.expval(qml.PauliZ(0))

  params = [0.1, 0.2, 0.3]
  qml.grad(f)(params)
  ```

* The default logic for choosing the 'best' differentiation method has been altered
  to improve performance.
  [(#1008)](https://github.com/PennyLaneAI/pennylane/pull/1008)

  - If the quantum device provides its own gradient, this is now the preferred
    differentiation method.

  - If the quantum device natively supports classical
    backpropagation, this is now preferred over the parameter-shift rule.

    This will lead to marked speed improvement during optimization when using
    `default.qubit`, with a sight penalty on the forward-pass evaluation.

  More details are available below in the 'Improvements' section for plugin developers.

* PennyLane now supports analytical quantum gradients for noisy channels, in addition to its
  existing support for unitary operations. The noisy channels `BitFlip`, `PhaseFlip`, and
  `DepolarizingChannel` all support analytic gradients out of the box.
  [(#968)](https://github.com/PennyLaneAI/pennylane/pull/968)

* A method has been added for calculating the Hessian of quantum circuits using the second-order
  parameter shift formula.
  [(#961)](https://github.com/PennyLaneAI/pennylane/pull/961)

  The following example shows the calculation of the Hessian:

  ```python
  n_wires = 5
  weights = [2.73943676, 0.16289932, 3.4536312, 2.73521126, 2.6412488]

  dev = qml.device("default.qubit", wires=n_wires)

  with qml.tape.QubitParamShiftTape() as tape:
      for i in range(n_wires):
          qml.RX(weights[i], wires=i)

      qml.CNOT(wires=[0, 1])
      qml.CNOT(wires=[2, 1])
      qml.CNOT(wires=[3, 1])
      qml.CNOT(wires=[4, 3])

      qml.expval(qml.PauliZ(1))

  print(tape.hessian(dev))
  ```

  The Hessian is not yet supported via classical machine learning interfaces, but will
  be added in a future release.

<h4>More operations and templates</h4>

* Two new error channels, `BitFlip` and `PhaseFlip` have been added.
  [(#954)](https://github.com/PennyLaneAI/pennylane/pull/954)

  They can be used in the same manner as existing error channels:

  ```python
  dev = qml.device("default.mixed", wires=2)

  @qml.qnode(dev)
  def circuit():
      qml.RX(0.3, wires=0)
      qml.RY(0.5, wires=1)
      qml.BitFlip(0.01, wires=0)
      qml.PhaseFlip(0.01, wires=1)
      return qml.expval(qml.PauliZ(0))
  ```

* Apply permutations to wires using the `Permute` subroutine.
  [(#952)](https://github.com/PennyLaneAI/pennylane/pull/952)

  ```python
  import pennylane as qp
  dev = qml.device('default.qubit', wires=5)

  @qml.qnode(dev)
  def apply_perm():
      # Send contents of wire 4 to wire 0, of wire 2 to wire 1, etc.
      qml.templates.Permute([4, 2, 0, 1, 3], wires=dev.wires)
      return qml.expval(qml.PauliZ(0))
  ```

<h4>QNode transformations</h4>

* The `qml.metric_tensor` function transforms a QNode to produce the Fubini-Study
  metric tensor with full autodifferentiation support---even on hardware.
  [(#1014)](https://github.com/PennyLaneAI/pennylane/pull/1014)

  Consider the following QNode:

  ```python
  dev = qml.device("default.qubit", wires=3)

  @qml.qnode(dev, interface="autograd")
  def circuit(weights):
      # layer 1
      qml.RX(weights[0, 0], wires=0)
      qml.RX(weights[0, 1], wires=1)

      qml.CNOT(wires=[0, 1])
      qml.CNOT(wires=[1, 2])

      # layer 2
      qml.RZ(weights[1, 0], wires=0)
      qml.RZ(weights[1, 1], wires=2)

      qml.CNOT(wires=[0, 1])
      qml.CNOT(wires=[1, 2])
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), qml.expval(qml.PauliY(2))
  ```

  We can use the `metric_tensor` function to generate a new function, that returns the
  metric tensor of this QNode:

  ```pycon
  >>> met_fn = qml.metric_tensor(circuit)
  >>> weights = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)
  >>> met_fn(weights)
  tensor([[0.25  , 0.    , 0.    , 0.    ],
          [0.    , 0.25  , 0.    , 0.    ],
          [0.    , 0.    , 0.0025, 0.0024],
          [0.    , 0.    , 0.0024, 0.0123]], requires_grad=True)
  ```

  The returned metric tensor is also fully differentiable, in all interfaces.
  For example, differentiating the `(3, 2)` element:

  ```pycon
  >>> grad_fn = qml.grad(lambda x: met_fn(x)[3, 2])
  >>> grad_fn(weights)
  array([[ 0.04867729, -0.00049502,  0.        ],
         [ 0.        ,  0.        ,  0.        ]])
  ```

  Differentiation is also supported using Torch, Jax, and TensorFlow.

* Adds the new function `qml.math.cov_matrix()`. This function accepts a list of commuting
  observables, and the probability distribution in the shared observable eigenbasis after the
  application of an ansatz. It uses these to construct the covariance matrix in a *framework
  independent* manner, such that the output covariance matrix is autodifferentiable.
  [(#1012)](https://github.com/PennyLaneAI/pennylane/pull/1012)

  For example, consider the following ansatz and observable list:

  ```python3
  obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliY(2)]
  ansatz = qml.templates.StronglyEntanglingLayers
  ```

  We can construct a QNode to output the probability distribution in the shared eigenbasis of the
  observables:

  ```python
  dev = qml.device("default.qubit", wires=3)

  @qml.qnode(dev, interface="autograd")
  def circuit(weights):
      ansatz(weights, wires=[0, 1, 2])
      # rotate into the basis of the observables
      for o in obs_list:
          o.diagonalizing_gates()
      return qml.probs(wires=[0, 1, 2])
  ```

  We can now compute the covariance matrix:

  ```pycon
  >>> weights = qml.init.strong_ent_layers_normal(n_layers=2, n_wires=3)
  >>> cov = qml.math.cov_matrix(circuit(weights), obs_list)
  >>> cov
  array([[0.98707611, 0.03665537],
         [0.03665537, 0.99998377]])
  ```

  Autodifferentiation is fully supported using all interfaces:

  ```pycon
  >>> cost_fn = lambda weights: qml.math.cov_matrix(circuit(weights), obs_list)[0, 1]
  >>> qml.grad(cost_fn)(weights)[0]
  array([[[ 4.94240914e-17, -2.33786398e-01, -1.54193959e-01],
          [-3.05414996e-17,  8.40072236e-04,  5.57884080e-04],
          [ 3.01859411e-17,  8.60411436e-03,  6.15745204e-04]],

         [[ 6.80309533e-04, -1.23162742e-03,  1.08729813e-03],
          [-1.53863193e-01, -1.38700657e-02, -1.36243323e-01],
          [-1.54665054e-01, -1.89018172e-02, -1.56415558e-01]]])
  ```

* A new  `qml.draw` function is available, allowing QNodes to be easily
  drawn without execution by providing example input.
  [(#962)](https://github.com/PennyLaneAI/pennylane/pull/962)

  ```python
  @qml.qnode(dev)
  def circuit(a, w):
      qml.Hadamard(0)
      qml.CRX(a, wires=[0, 1])
      qml.Rot(*w, wires=[1])
      qml.CRX(-a, wires=[0, 1])
      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  ```

  The QNode circuit structure may depend on the input arguments;
  this is taken into account by passing example QNode arguments
  to the `qml.draw()` drawing function:

  ```pycon
  >>> drawer = qml.draw(circuit)
  >>> result = drawer(a=2.3, w=[1.2, 3.2, 0.7])
  >>> print(result)
  0: ──H──╭C────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩
  1: ─────╰RX(2.3)──Rot(1.2, 3.2, 0.7)──╰RX(-2.3)──╰┤ ⟨Z ⊗ Z⟩
  ```

<h4>A faster, leaner, and more flexible core</h4>

* The new core of PennyLane, rewritten from the ground up and developed over the last few release
  cycles, has achieved feature parity and has been made the new default in PennyLane v0.14. The old
  core has been marked as deprecated, and will be removed in an upcoming release.
  [(#1046)](https://github.com/PennyLaneAI/pennylane/pull/1046)
  [(#1040)](https://github.com/PennyLaneAI/pennylane/pull/1040)
  [(#1034)](https://github.com/PennyLaneAI/pennylane/pull/1034)
  [(#1035)](https://github.com/PennyLaneAI/pennylane/pull/1035)
  [(#1027)](https://github.com/PennyLaneAI/pennylane/pull/1027)
  [(#1026)](https://github.com/PennyLaneAI/pennylane/pull/1026)
  [(#1021)](https://github.com/PennyLaneAI/pennylane/pull/1021)
  [(#1054)](https://github.com/PennyLaneAI/pennylane/pull/1054)
  [(#1049)](https://github.com/PennyLaneAI/pennylane/pull/1049)

  While high-level PennyLane code and tutorials remain unchanged, the new core
  provides several advantages and improvements:

  - **Faster and more optimized**: The new core provides various performance optimizations, reducing
    pre- and post-processing overhead, and reduces the number of quantum evaluations in certain
    cases.

  - **Support for in-QNode classical processing**: this allows for differentiable classical
    processing within the QNode.

    ```python
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface="tf")
    def circuit(p):
        qml.RX(tf.sin(p[0])**2 + p[1], wires=0)
        return qml.expval(qml.PauliZ(0))
    ```

    The classical processing functions used within the QNode must match
    the QNode interface. Here, we use TensorFlow:

    ```pycon
    >>> params = tf.Variable([0.5, 0.1], dtype=tf.float64)
    >>> with tf.GradientTape() as tape:
    ...     res = circuit(params)
    >>> grad = tape.gradient(res, params)
    >>> print(res)
    tf.Tensor(0.9460913127754935, shape=(), dtype=float64)
    >>> print(grad)
    tf.Tensor([-0.27255248 -0.32390003], shape=(2,), dtype=float64)
    ```

    As a result of this change, quantum decompositions that require classical processing
    are fully supported and end-to-end differentiable in tape mode.

  - **No more Variable wrapping**: QNode arguments no longer become `Variable`
    objects within the QNode.

    ```python
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(x):
        print("Parameter value:", x)
        qml.RX(x, wires=0)
        return qml.expval(qml.PauliZ(0))
    ```

    Internal QNode parameters can be easily inspected, printed, and manipulated:

    ```pycon
    >>> circuit(0.5)
    Parameter value: 0.5
    tensor(0.87758256, requires_grad=True)
    ```

  - **Less restrictive QNode signatures**: There is no longer any restriction on the QNode signature; the QNode can be
    defined and called following the same rules as standard Python functions.

    For example, the following QNode uses positional, named, and variable
    keyword arguments:

    ```python
    x = torch.tensor(0.1, requires_grad=True)
    y = torch.tensor([0.2, 0.3], requires_grad=True)
    z = torch.tensor(0.4, requires_grad=True)

    @qml.qnode(dev, interface="torch")
    def circuit(p1, p2=y, **kwargs):
        qml.RX(p1, wires=0)
        qml.RY(p2[0] * p2[1], wires=0)
        qml.RX(kwargs["p3"], wires=0)
        return qml.var(qml.PauliZ(0))
    ```

    When we call the QNode, we may pass the arguments by name
    even if defined positionally; any argument not provided will
    use the default value.

    ```pycon
    >>> res = circuit(p1=x, p3=z)
    >>> print(res)
    tensor(0.2327, dtype=torch.float64, grad_fn=<SelectBackward>)
    >>> res.backward()
    >>> print(x.grad, y.grad, z.grad)
    tensor(0.8396) tensor([0.0289, 0.0193]) tensor(0.8387)
    ```

    This extends to the `qnn` module, where `KerasLayer` and `TorchLayer` modules
    can be created from QNodes with unrestricted signatures.

  - **Smarter measurements:** QNodes can now measure wires more than once, as
    long as all observables are commuting:

    ```python
    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        return [
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
        ]
    ```

    Further, the `qml.ExpvalCost()` function allows for optimizing
    measurements to reduce the number of quantum evaluations required.

  With the new PennyLane core, there are a few small breaking changes, detailed
  below in the 'Breaking Changes' section.

<h3>Improvements</h3>

* The built-in PennyLane optimizers allow more flexible cost functions. The cost function passed to most optimizers
  may accept any combination of trainable arguments, non-trainable arguments, and keyword arguments.
  [(#959)](https://github.com/PennyLaneAI/pennylane/pull/959)
  [(#1053)](https://github.com/PennyLaneAI/pennylane/pull/1053)

  The full changes apply to:

  * `AdagradOptimizer`
  * `AdamOptimizer`
  * `GradientDescentOptimizer`
  * `MomentumOptimizer`
  * `NesterovMomentumOptimizer`
  * `RMSPropOptimizer`
  * `RotosolveOptimizer`

  The `requires_grad=False` property must mark any non-trainable constant argument.
  The `RotoselectOptimizer` allows passing only keyword arguments.

  Example use:

  ```python
  def cost(x, y, data, scale=1.0):
      return scale * (x[0]-data)**2 + scale * (y-data)**2

  x = np.array([1.], requires_grad=True)
  y = np.array([1.0])
  data = np.array([2.], requires_grad=False)

  opt = qml.GradientDescentOptimizer()

  # the optimizer step and step_and_cost methods can
  # now update multiple parameters at once
  x_new, y_new, data = opt.step(cost, x, y, data, scale=0.5)
  (x_new, y_new, data), value = opt.step_and_cost(cost, x, y, data, scale=0.5)

  # list and tuple unpacking is also supported
  params = (x, y, data)
  params = opt.step(cost, *params)
  ```

* The circuit drawer has been updated to support the inclusion of unused or inactive
  wires, by passing the `show_all_wires` argument.
  [(#1033)](https://github.com/PennyLaneAI/pennylane/pull/1033)

  ```python
  dev = qml.device('default.qubit', wires=[-1, "a", "q2", 0])

  @qml.qnode(dev)
  def circuit():
      qml.Hadamard(wires=-1)
      qml.CNOT(wires=[-1, "q2"])
      return qml.expval(qml.PauliX(wires="q2"))
  ```

  ```pycon
  >>> print(qml.draw(circuit, show_all_wires=True)())
  >>>
   -1: ──H──╭C──┤
    a: ─────│───┤
   q2: ─────╰X──┤ ⟨X⟩
    0: ─────────┤
  ```

* The logic for choosing the 'best' differentiation method has been altered
  to improve performance.
  [(#1008)](https://github.com/PennyLaneAI/pennylane/pull/1008)

  - If the device provides its own gradient, this is now the preferred
    differentiation method.

  - If a device provides additional interface-specific versions that natively support classical
    backpropagation, this is now preferred over the parameter-shift rule.

    Devices define additional interface-specific devices via their `capabilities()` dictionary. For
    example, `default.qubit` supports supplementary devices for TensorFlow, Autograd, and JAX:

    ```python
    {
      "passthru_devices": {
          "tf": "default.qubit.tf",
          "autograd": "default.qubit.autograd",
          "jax": "default.qubit.jax",
      },
    }
    ```

  As a result of this change, if the QNode `diff_method` is not explicitly provided,
  it is possible that the QNode will run on a *supplementary device* of the device that was
  specifically provided:

  ```python
  dev = qml.device("default.qubit", wires=2)
  qml.QNode(dev) # will default to backprop on default.qubit.autograd
  qml.QNode(dev, interface="tf") # will default to backprop on default.qubit.tf
  qml.QNode(dev, interface="jax") # will default to backprop on default.qubit.jax
  ```

* The `default.qubit` device has been updated so that internally it applies operations in a more
  functional style, i.e., by accepting an input state and returning an evolved state.
  [(#1025)](https://github.com/PennyLaneAI/pennylane/pull/1025)

* A new test series, `pennylane/devices/tests/test_compare_default_qubit.py`, has been added, allowing to test if
  a chosen device gives the same result as `default.qubit`.
  [(#897)](https://github.com/PennyLaneAI/pennylane/pull/897)

  Three tests are added:

  - `test_hermitian_expectation`,
  - `test_pauliz_expectation_analytic`, and
  - `test_random_circuit`.

* Adds the following agnostic tensor manipulation functions to the `qml.math` module: `abs`,
  `angle`, `arcsin`, `concatenate`, `dot`, `squeeze`, `sqrt`, `sum`, `take`, `where`. These functions are
  required to fully support end-to-end differentiable Mottonen and Amplitude embedding.
  [(#922)](https://github.com/PennyLaneAI/pennylane/pull/922)
  [(#1011)](https://github.com/PennyLaneAI/pennylane/pull/1011)

* The `qml.math` module now supports JAX.
  [(#985)](https://github.com/XanaduAI/software-docs/pull/274)

* Several improvements have been made to the `Wires` class to reduce overhead and simplify the logic
  of how wire labels are interpreted:
  [(#1019)](https://github.com/PennyLaneAI/pennylane/pull/1019)
  [(#1010)](https://github.com/PennyLaneAI/pennylane/pull/1010)
  [(#1005)](https://github.com/PennyLaneAI/pennylane/pull/1005)
  [(#983)](https://github.com/PennyLaneAI/pennylane/pull/983)
  [(#967)](https://github.com/PennyLaneAI/pennylane/pull/967)

  - If the input `wires` to a wires class instantiation `Wires(wires)` can be iterated over,
    its elements are interpreted as wire labels. Otherwise, `wires` is interpreted as a single wire label.
    The only exception to this are strings, which are always interpreted as a single
    wire label, so users can address wires with labels such as `"ancilla"`.

  - Any type can now be a wire label as long as it is hashable. The hash is used to establish
    the uniqueness of two labels.

  - Indexing wires objects now returns a label, instead of a new `Wires` object. For example:

    ```pycon
    >>> w = Wires([0, 1, 2])
    >>> w[1]
    >>> 1
    ```

  - The check for uniqueness of wires moved from `Wires` instantiation to
    the `qml.wires._process` function in order to reduce overhead from repeated
    creation of `Wires` instances.

  - Calls to the `Wires` class are substantially reduced, for example by avoiding to call
    Wires on Wires instances on `Operation` instantiation, and by using labels instead of
    `Wires` objects inside the default qubit device.

* Adds the `PauliRot` generator to the `qml.operation` module. This
  generator is required to construct the metric tensor.
  [(#963)](https://github.com/PennyLaneAI/pennylane/pull/963)

* The templates are modified to make use of the new `qml.math` module, for framework-agnostic
  tensor manipulation. This allows the template library to be differentiable
  in backpropagation mode (`diff_method="backprop"`).
  [(#873)](https://github.com/PennyLaneAI/pennylane/pull/873)

* The circuit drawer now allows for the wire order to be (optionally) modified:
  [(#992)](https://github.com/PennyLaneAI/pennylane/pull/992)

  ```pycon
  >>> dev = qml.device('default.qubit', wires=["a", -1, "q2"])
  >>> @qml.qnode(dev)
  ... def circuit():
  ...     qml.Hadamard(wires=-1)
  ...     qml.CNOT(wires=["a", "q2"])
  ...     qml.RX(0.2, wires="a")
  ...     return qml.expval(qml.PauliX(wires="q2"))
  ```

  Printing with default wire order of the device:

  ```pycon
  >>> print(circuit.draw())
    a: ─────╭C──RX(0.2)──┤
   -1: ──H──│────────────┤
   q2: ─────╰X───────────┤ ⟨X⟩
  ```

  Changing the wire order:

  ```pycon
  >>> print(circuit.draw(wire_order=["q2", "a", -1]))
   q2: ──╭X───────────┤ ⟨X⟩
    a: ──╰C──RX(0.2)──┤
   -1: ───H───────────┤
  ```

<h3>Breaking changes</h3>

* QNodes using the new PennyLane core will no longer accept ragged arrays as inputs.

* When using the new PennyLane core and the Autograd interface, non-differentiable data passed
  as a QNode argument or a gate must have the `requires_grad` property set to `False`:

  ```python
  @qml.qnode(dev)
  def circuit(weights, data):
      basis_state = np.array([1, 0, 1, 1], requires_grad=False)
      qml.BasisState(basis_state, wires=[0, 1, 2, 3])
      qml.templates.AmplitudeEmbedding(data, wires=[0, 1, 2, 3])
      qml.templates.BasicEntanglerLayers(weights, wires=[0, 1, 2, 3])
      return qml.probs(wires=0)

  data = np.array(data, requires_grad=False)
  weights = np.array(weights, requires_grad=True)
  circuit(weights, data)
  ```

<h3>Bug fixes</h3>

* Fixes an issue where if the constituent observables of a tensor product do not exist in the queue,
  an error is raised. With this fix, they are first queued before annotation occurs.
  [(#1038)](https://github.com/PennyLaneAI/pennylane/pull/1038)

* Fixes an issue with tape expansions where information about sampling
  (specifically the `is_sampled` tape attribute) was not preserved.
  [(#1027)](https://github.com/PennyLaneAI/pennylane/pull/1027)

* Tape expansion was not properly taking into devices that supported inverse operations,
  causing inverse operations to be unnecessarily decomposed. The QNode tape expansion logic, as well
  as the `Operation.expand()` method, has been modified to fix this.
  [(#956)](https://github.com/PennyLaneAI/pennylane/pull/956)

* Fixes an issue where the Autograd interface was not unwrapping non-differentiable
  PennyLane tensors, which can cause issues on some devices.
  [(#941)](https://github.com/PennyLaneAI/pennylane/pull/941)

* `qml.vqe.Hamiltonian` prints any observable with any number of strings.
  [(#987)](https://github.com/PennyLaneAI/pennylane/pull/987)

* Fixes a bug where parameter-shift differentiation would fail if the QNode
  contained a single probability output.
  [(#1007)](https://github.com/PennyLaneAI/pennylane/pull/1007)

* Fixes an issue when using trainable parameters that are lists/arrays with `tape.vjp`.
  [(#1042)](https://github.com/PennyLaneAI/pennylane/pull/1042)

* The `TensorN` observable is updated to support being copied without any parameters or wires passed.
  [(#1047)](https://github.com/PennyLaneAI/pennylane/pull/1047)

* Fixed deprecation warning when importing `Sequence` from `collections` instead of `collections.abc` in `vqe/vqe.py`.
  [(#1051)](https://github.com/PennyLaneAI/pennylane/pull/1051)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Thomas Bromley, Olivia Di Matteo, Theodor Isacsson, Josh Izaac, Christina Lee,
Alejandro Montanez, Steven Oud, Chase Roberts, Sankalp Sanand, Maria Schuld, Antal
Száva, David Wierichs, Jiahao Yao.
