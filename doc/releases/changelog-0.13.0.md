
# Release 0.13.0

<h3>New features since last release</h3>

<h4>Automatically optimize the number of measurements</h4>

* QNodes in tape mode now support returning observables on the same wire whenever the observables are
  qubit-wise commuting Pauli words. Qubit-wise commuting observables can be evaluated with a
  *single* device run as they are diagonal in the same basis, via a shared set of single-qubit rotations.
  [(#882)](https://github.com/PennyLaneAI/pennylane/pull/882)

  The following example shows a single QNode returning the expectation values of
  the qubit-wise commuting Pauli words `XX` and `XI`:

  ```python
  qp.enable_tape()

  @qp.qnode(dev)
  def f(x):
      qp.Hadamard(wires=0)
      qp.Hadamard(wires=1)
      qp.CRot(0.1, 0.2, 0.3, wires=[1, 0])
      qp.RZ(x, wires=1)
      return qp.expval(qp.PauliX(0) @ qp.PauliX(1)), qp.expval(qp.PauliX(0))
  ```

  ```pycon
  >>> f(0.4)
  tensor([0.89431013, 0.9510565 ], requires_grad=True)
  ```

* The `ExpvalCost` class (previously `VQECost`) now provides observable optimization using the
  `optimize` argument, resulting in potentially fewer device executions.
  [(#902)](https://github.com/PennyLaneAI/pennylane/pull/902)

  This is achieved by separating the observables composing the Hamiltonian into qubit-wise
  commuting groups and evaluating those groups on a single QNode using functionality from the
  `qp.grouping` module:

  ```python
  qp.enable_tape()
  commuting_obs = [qp.PauliX(0), qp.PauliX(0) @ qp.PauliZ(1)]
  H = qp.vqe.Hamiltonian([1, 1], commuting_obs)

  dev = qp.device("default.qubit", wires=2)
  ansatz = qp.templates.StronglyEntanglingLayers

  cost_opt = qp.ExpvalCost(ansatz, H, dev, optimize=True)
  cost_no_opt = qp.ExpvalCost(ansatz, H, dev, optimize=False)

  params = qp.init.strong_ent_layers_uniform(3, 2)
  ```

  Grouping these commuting observables leads to fewer device executions:

  ```pycon
  >>> cost_opt(params)
  >>> ex_opt = dev.num_executions
  >>> cost_no_opt(params)
  >>> ex_no_opt = dev.num_executions - ex_opt
  >>> print("Number of executions:", ex_no_opt)
  Number of executions: 2
  >>> print("Number of executions (optimized):", ex_opt)
  Number of executions (optimized): 1
  ```

<h4>New quantum gradient features</h4>

* Compute the analytic gradient of quantum circuits in parallel on supported devices.
  [(#840)](https://github.com/PennyLaneAI/pennylane/pull/840)

  This release introduces support for batch execution of circuits, via a new device API method
  `Device.batch_execute()`. Devices that implement this new API support submitting a batch of
  circuits for *parallel* evaluation simultaneously, which can significantly reduce the computation time.

  Furthermore, if using tape mode and a compatible device, gradient computations will
  automatically make use of the new batch API---providing a speedup during optimization.

* Gradient recipes are now much more powerful, allowing for operations to define their gradient
  via an arbitrary linear combination of circuit evaluations.
  [(#909)](https://github.com/PennyLaneAI/pennylane/pull/909)
  [(#915)](https://github.com/PennyLaneAI/pennylane/pull/915)

  With this change, gradient recipes can now be of the form
  :math:`\frac{\partial}{\partial\phi_k}f(\phi_k) = \sum_{i} c_i f(a_i \phi_k + s_i )`,
  and are no longer restricted to two-term shifts with identical (but opposite in sign) shift values.

  As a result, PennyLane now supports native analytic quantum gradients for the
  controlled rotation operations `CRX`, `CRY`, `CRZ`, and `CRot`. This allows for parameter-shift
  analytic gradients on hardware, without decomposition.

  Note that this is a breaking change for developers; please see the *Breaking Changes* section
  for more details.

* The `qnn.KerasLayer` class now supports differentiating the QNode through classical
  backpropagation in tape mode.
  [(#869)](https://github.com/PennyLaneAI/pennylane/pull/869)

  ```python
  qp.enable_tape()

  dev = qp.device("default.qubit.tf", wires=2)

  @qp.qnode(dev, interface="tf", diff_method="backprop")
  def f(inputs, weights):
      qp.templates.AngleEmbedding(inputs, wires=range(2))
      qp.templates.StronglyEntanglingLayers(weights, wires=range(2))
      return [qp.expval(qp.PauliZ(i)) for i in range(2)]

  weight_shapes = {"weights": (3, 2, 3)}

  qlayer = qp.qnn.KerasLayer(f, weight_shapes, output_dim=2)

  inputs = tf.constant(np.random.random((4, 2)), dtype=tf.float32)

  with tf.GradientTape() as tape:
      out = qlayer(inputs)

  tape.jacobian(out, qlayer.trainable_weights)
  ```

<h4>New operations, templates, and measurements</h4>

* Adds the `qp.density_matrix` QNode return with partial trace capabilities.
  [(#878)](https://github.com/PennyLaneAI/pennylane/pull/878)

  The density matrix over the provided wires is returned, with all other subsystems traced out.
  `qp.density_matrix` currently works for both the `default.qubit` and `default.mixed` devices.

  ```python
  qp.enable_tape()
  dev = qp.device("default.qubit", wires=2)

  def circuit(x):
      qp.PauliY(wires=0)
      qp.Hadamard(wires=1)
      return qp.density_matrix(wires=[1])  # wire 0 is traced out
  ```

* Adds the square-root X gate `SX`. [(#871)](https://github.com/PennyLaneAI/pennylane/pull/871)

  ```python
  dev = qp.device("default.qubit", wires=1)

  @qp.qnode(dev)
  def circuit():
      qp.SX(wires=[0])
      return qp.expval(qp.PauliZ(wires=[0]))
  ```

* Two new hardware-efficient particle-conserving templates have been implemented
  to perform VQE-based quantum chemistry simulations. The new templates apply
  several layers of the particle-conserving entanglers proposed in Figs. 2a and 2b
  of Barkoutsos *et al*., [arXiv:1805.04340](https://arxiv.org/abs/1805.04340)
  [(#875)](https://github.com/PennyLaneAI/pennylane/pull/875)
  [(#876)](https://github.com/PennyLaneAI/pennylane/pull/876)

<h4>Estimate and track resources</h4>

* The `QuantumTape` class now contains basic resource estimation functionality. The method
  `tape.get_resources()` returns a dictionary with a list of the constituent operations and the
  number of times they appear in the circuit. Similarly, `tape.get_depth()` computes the circuit depth.
  [(#862)](https://github.com/PennyLaneAI/pennylane/pull/862)

  ```pycon
  >>> with qp.tape.QuantumTape() as tape:
  ...    qp.Hadamard(wires=0)
  ...    qp.RZ(0.26, wires=1)
  ...    qp.CNOT(wires=[1, 0])
  ...    qp.Rot(1.8, -2.7, 0.2, wires=0)
  ...    qp.Hadamard(wires=1)
  ...    qp.CNOT(wires=[0, 1])
  ...    qp.expval(qp.PauliZ(0) @ qp.PauliZ(1))
  >>> tape.get_resources()
  {'Hadamard': 2, 'RZ': 1, 'CNOT': 2, 'Rot': 1}
  >>> tape.get_depth()
  4
  ```

* The number of device executions over a QNode's lifetime can now be returned using `num_executions`.
  [(#853)](https://github.com/PennyLaneAI/pennylane/pull/853)

  ```pycon
  >>> dev = qp.device("default.qubit", wires=2)
  >>> @qp.qnode(dev)
  ... def circuit(x, y):
  ...    qp.RX(x, wires=[0])
  ...    qp.RY(y, wires=[1])
  ...    qp.CNOT(wires=[0, 1])
  ...    return qp.expval(qp.PauliZ(0) @ qp.PauliX(1))
  >>> for _ in range(10):
  ...    circuit(0.432, 0.12)
  >>> print(dev.num_executions)
  10
  ```

<h3>Improvements</h3>

* Support for tape mode has improved across PennyLane. The following features now work in tape mode:

  - QNode collections [(#863)](https://github.com/PennyLaneAI/pennylane/pull/863)

  - `qnn.ExpvalCost` [(#863)](https://github.com/PennyLaneAI/pennylane/pull/863)
    [(#911)](https://github.com/PennyLaneAI/pennylane/pull/911)

  - `qp.qnn.KerasLayer` [(#869)](https://github.com/PennyLaneAI/pennylane/pull/869)

  - `qp.qnn.TorchLayer` [(#865)](https://github.com/PennyLaneAI/pennylane/pull/865)

  - The `qp.qaoa` module [(#905)](https://github.com/PennyLaneAI/pennylane/pull/905)

* A new function, `qp.refresh_devices()`, has been added, allowing PennyLane to
  rescan installed PennyLane plugins and refresh the device list. In addition, the `qp.device`
  loader will attempt to refresh devices if the required plugin device cannot be found.
  This will result in an improved experience if installing PennyLane and plugins within
  a running Python session (for example, on Google Colab), and avoid the need to
  restart the kernel/runtime.
  [(#907)](https://github.com/PennyLaneAI/pennylane/pull/907)

* When using `grad_fn = qp.grad(cost)` to compute the gradient of a cost function with the Autograd
  interface, the value of the intermediate forward pass is now available via the `grad_fn.forward`
  property
  [(#914)](https://github.com/PennyLaneAI/pennylane/pull/914):

  ```python
  def cost_fn(x, y):
      return 2 * np.sin(x[0]) * np.exp(-x[1]) + x[0] ** 3 + np.cos(y)

  params = np.array([0.1, 0.5], requires_grad=True)
  data = np.array(0.65, requires_grad=False)
  grad_fn = qp.grad(cost_fn)

  grad_fn(params, data)  # perform backprop and evaluate the gradient
  grad_fn.forward  # the cost function value
  ```

* Gradient-based optimizers now have a `step_and_cost` method that returns
  both the next step as well as the objective (cost) function output.
  [(#916)](https://github.com/PennyLaneAI/pennylane/pull/916)

  ```pycon
  >>> opt = qp.GradientDescentOptimizer()
  >>> params, cost = opt.step_and_cost(cost_fn, params)
  ```

* PennyLane provides a new experimental module `qp.proc` which provides framework-agnostic processing
  functions for array and tensor manipulations.
  [(#886)](https://github.com/PennyLaneAI/pennylane/pull/886)

  Given the input tensor-like object, the call is
  dispatched to the corresponding array manipulation framework, allowing for end-to-end
  differentiation to be preserved.

  ```pycon
  >>> x = torch.tensor([1., 2.])
  >>> qp.proc.ones_like(x)
  tensor([1, 1])
  >>> y = tf.Variable([[0], [5]])
  >>> qp.proc.ones_like(y, dtype=np.complex128)
  <tf.Tensor: shape=(2, 1), dtype=complex128, numpy=
  array([[1.+0.j],
         [1.+0.j]])>
  ```

  Note that these functions are experimental, and only a subset of common functionality is
  supported. Furthermore, the names and behaviour of these functions may differ from similar
  functions in common frameworks; please refer to the function docstrings for more details.

* The gradient methods in tape mode now fully separate the quantum and classical processing. Rather
  than returning the evaluated gradients directly, they now return a tuple containing the required
  quantum and classical processing steps.
  [(#840)](https://github.com/PennyLaneAI/pennylane/pull/840)

  ```python
  def gradient_method(idx, param, **options):
      # generate the quantum tapes that must be computed
      # to determine the quantum gradient
      tapes = quantum_gradient_tapes(self)

      def processing_fn(results):
          # perform classical processing on the evaluated tapes
          # returning the evaluated quantum gradient
          return classical_processing(results)

      return tapes, processing_fn
  ```

  The `JacobianTape.jacobian()` method has been similarly modified to accumulate all gradient
  quantum tapes and classical processing functions, evaluate all quantum tapes simultaneously,
  and then apply the post-processing functions to the evaluated tape results.

* The MultiRZ gate now has a defined generator, allowing it to be used in quantum natural gradient
  optimization.
  [(#912)](https://github.com/PennyLaneAI/pennylane/pull/912)

* The CRot gate now has a `decomposition` method, which breaks the gate down into rotations
  and CNOT gates. This allows `CRot` to be used on devices that do not natively support it.
  [(#908)](https://github.com/PennyLaneAI/pennylane/pull/908)

* The classical processing in the `MottonenStatePreparation` template has been largely
  rewritten to use dense matrices and tensor manipulations wherever possible.
  This is in preparation to support differentiation through the template in the future.
  [(#864)](https://github.com/PennyLaneAI/pennylane/pull/864)

* Device-based caching has replaced QNode caching. Caching is now accessed by passing a
  `cache` argument to the device.
  [(#851)](https://github.com/PennyLaneAI/pennylane/pull/851)

  The `cache` argument should be an integer specifying the size of the cache. For example, a
  cache of size 10 is created using:

  ```pycon
  >>> dev = qp.device("default.qubit", wires=2, cache=10)
  ```

* The `Operation`, `Tensor`, and `MeasurementProcess` classes now have the `__copy__` special method
  defined.
  [(#840)](https://github.com/PennyLaneAI/pennylane/pull/840)

  This allows us to ensure that, when a shallow copy is performed of an operation, the
  mutable list storing the operation parameters is *also* shallow copied. Both the old operation and
  the copied operation will continue to share the same parameter data,
  ```pycon
  >>> import copy
  >>> op = qp.RX(0.2, wires=0)
  >>> op2 = copy.copy(op)
  >>> op.data[0] is op2.data[0]
  True
  ```

  however the *list container* is not a reference:

  ```pycon
  >>> op.data is op2.data
  False
  ```

  This allows the parameters of the copied operation to be modified, without mutating
  the parameters of the original operation.

* The `QuantumTape.copy` method has been tweaked so that
  [(#840)](https://github.com/PennyLaneAI/pennylane/pull/840):

  - Optionally, the tape's operations are shallow copied in addition to the tape by passing the
    `copy_operations=True` boolean flag. This allows the copied tape's parameters to be mutated
    without affecting the original tape's parameters. (Note: the two tapes will share parameter data
    *until* one of the tapes has their parameter list modified.)

  - Copied tapes can be cast to another `QuantumTape` subclass by passing the `tape_cls` keyword
    argument.

<h3>Breaking changes</h3>

* Updated how parameter-shift gradient recipes are defined for operations, allowing for
  gradient recipes that are specified as an arbitrary number of terms.
  [(#909)](https://github.com/PennyLaneAI/pennylane/pull/909)

  Previously, `Operation.grad_recipe` was restricted to two-term parameter-shift formulas.
  With this change, the gradient recipe now contains elements of the form
  :math:`[c_i, a_i, s_i]`, resulting in a gradient recipe of
  :math:`\frac{\partial}{\partial\phi_k}f(\phi_k) = \sum_{i} c_i f(a_i \phi_k + s_i )`.

  As this is a breaking change, all custom operations with defined gradient recipes must be
  updated to continue working with PennyLane 0.13. Note though that if `grad_recipe = None`, the
  default gradient recipe remains unchanged, and corresponds to the two terms :math:`[c_0, a_0, s_0]=[1/2, 1, \pi/2]`
  and :math:`[c_1, a_1, s_1]=[-1/2, 1, -\pi/2]` for every parameter.

- The `VQECost` class has been renamed to `ExpvalCost` to reflect its general applicability
  beyond VQE. Use of `VQECost` is still possible but will result in a deprecation warning.
  [(#913)](https://github.com/PennyLaneAI/pennylane/pull/913)

<h3>Bug fixes</h3>

* The `default.qubit.tf` device is updated to handle TensorFlow objects (e.g.,
  `tf.Variable`) as gate parameters correctly when using the `MultiRZ` and
  `CRot` operations.
  [(#921)](https://github.com/PennyLaneAI/pennylane/pull/921)

* PennyLane tensor objects are now unwrapped in BaseQNode when passed as a
  keyword argument to the quantum function.
  [(#903)](https://github.com/PennyLaneAI/pennylane/pull/903)
  [(#893)](https://github.com/PennyLaneAI/pennylane/pull/893)

* The new tape mode now prevents multiple observables from being evaluated on the same wire
  if the observables are not qubit-wise commuting Pauli words.
  [(#882)](https://github.com/PennyLaneAI/pennylane/pull/882)

* Fixes a bug in `default.qubit` whereby inverses of common gates were not being applied
  via efficient gate-specific methods, instead falling back to matrix-vector multiplication.
  The following gates were affected: `PauliX`, `PauliY`, `PauliZ`, `Hadamard`, `SWAP`, `S`,
  `T`, `CNOT`, `CZ`.
  [(#872)](https://github.com/PennyLaneAI/pennylane/pull/872)

* The `PauliRot` operation now gracefully handles single-qubit Paulis, and all-identity Paulis
  [(#860)](https://github.com/PennyLaneAI/pennylane/pull/860).

* Fixes a bug whereby binary Python operators were not properly propagating the `requires_grad`
  attribute to the output tensor.
  [(#889)](https://github.com/PennyLaneAI/pennylane/pull/889)

* Fixes a bug which prevents `TorchLayer` from doing `backward` when CUDA is enabled.
  [(#899)](https://github.com/PennyLaneAI/pennylane/pull/899)

* Fixes a bug where multi-threaded execution of `QNodeCollection` sometimes fails
  because of simultaneous queuing. This is fixed by adding thread locking during queuing.
  [(#910)](https://github.com/PennyLaneAI/pennylane/pull/918)

* Fixes a bug in `QuantumTape.set_parameters()`. The previous implementation assumed
  that the `self.trainable_parms` set would always be iterated over in increasing integer
  order. However, this is not guaranteed behaviour, and can lead to the incorrect tape parameters
  being set if this is not the case.
  [(#923)](https://github.com/PennyLaneAI/pennylane/pull/923)

* Fixes broken error message if a QNode is instantiated with an unknown exception.
  [(#930)](https://github.com/PennyLaneAI/pennylane/pull/930)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Thomas Bromley, Christina Lee, Alain Delgado Gran, Olivia Di Matteo, Anthony
Hayes, Theodor Isacsson, Josh Izaac, Soran Jahangiri, Nathan Killoran, Shumpei Kobayashi, Romain
Moyard, Zeyue Niu, Maria Schuld, Antal Száva.
