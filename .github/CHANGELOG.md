# Release 0.14.0-dev (development release)

<h3>New features since last release</h3>

* A new `default.qubit.jax` device was added. This device runs end to end in JAX, meaning that it
  supports all of the awesome JAX transformations (`jax.vmap`, `jax.jit`, `jax.hessian`, etc).

  Here is an example of how to use the new device:

  ```python
  qml.enable_tape()

  dev = qml.device("default.qubit.jax", wires=1)
  @qml.qnode(dev, interface="jax", diff_method="backprop")
  def circuit(x):
      qml.RX(x[1], wires=0)
      qml.Rot(x[0], x[1], x[2], wires=0)
      return qml.expval(qml.PauliZ(0))

  weights = jnp.array([0.2, 0.5, 0.1])
  grad_fn = jax.grad(circuit)
  print(grad_fn(weights))
  ```

  Currently, only `diff_method="backprop"` is supported, with plans to add reverse mode support in
  the future.

* Two new error channels, `BitFlip` and `PhaseFlip` have been added.
  [#954](https://github.com/PennyLaneAI/pennylane/pull/954)

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

<h3>Improvements</h3>

* A new test series, pennylane/devices/tests/test_compare_default_qubit.py, has been added, allowing to test if
  a chosen device gives the same result as the default device. Three tests are added `test_hermitian_expectation`,
  `test_pauliz_expectation_analytic`, and `test_random_circuit`.
  [(#897)](https://github.com/PennyLaneAI/pennylane/pull/897)

* Adds the following agnostic tensor manipulation functions to the `qml.math` module: `abs`,
  `angle`, `arcsin`, `concatenate`, `dot`, `sqrt`, `sum`, `take`, `where`. These functions are
  required to fully support end-to-end differentiable Mottonen and Amplitude embedding.
  [(#922)](https://github.com/PennyLaneAI/pennylane/pull/922)

<h3>Breaking changes</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

* In tape mode, tape expansion was not properly taking into devices that supported inverse operations,
  causing inverse operations to be unnecessarily decomposed. The QNode tape expansion logic, as well
  as the `Operation.expand()` method, has been modified to fix this.
  [(#956)](https://github.com/PennyLaneAI/pennylane/pull/956)

* Fixes an issue where the Autograd interface was not unwrapping non-differentiable
  PennyLane tensors, which can cause issues on some devices.
  [(#941)](https://github.com/PennyLaneAI/pennylane/pull/941)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Olivia Di Matteo, Josh Izaac, Alejandro Montanez, Chase Roberts.

# Release 0.13.0 (current release)

<h3>New features since last release</h3>

<h4>Automatically optimize the number of measurements</h4>

* QNodes in tape mode now support returning observables on the same wire whenever the observables are
  qubit-wise commuting Pauli words. Qubit-wise commuting observables can be evaluated with a
  *single* device run as they are diagonal in the same basis, via a shared set of single-qubit rotations.
  [(#882)](https://github.com/PennyLaneAI/pennylane/pull/882)

  The following example shows a single QNode returning the expectation values of
  the qubit-wise commuting Pauli words `XX` and `XI`:

  ```python
  qml.enable_tape()

  @qml.qnode(dev)
  def f(x):
      qml.Hadamard(wires=0)
      qml.Hadamard(wires=1)
      qml.CRot(0.1, 0.2, 0.3, wires=[1, 0])
      qml.RZ(x, wires=1)
      return qml.expval(qml.PauliX(0) @ qml.PauliX(1)), qml.expval(qml.PauliX(0))
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
  `qml.grouping` module:

  ```python
  qml.enable_tape()
  commuting_obs = [qml.PauliX(0), qml.PauliX(0) @ qml.PauliZ(1)]
  H = qml.vqe.Hamiltonian([1, 1], commuting_obs)

  dev = qml.device("default.qubit", wires=2)
  ansatz = qml.templates.StronglyEntanglingLayers

  cost_opt = qml.ExpvalCost(ansatz, H, dev, optimize=True)
  cost_no_opt = qml.ExpvalCost(ansatz, H, dev, optimize=False)

  params = qml.init.strong_ent_layers_uniform(3, 2)
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
  qml.enable_tape()

  dev = qml.device("default.qubit.tf", wires=2)

  @qml.qnode(dev, interface="tf", diff_method="backprop")
  def f(inputs, weights):
      qml.templates.AngleEmbedding(inputs, wires=range(2))
      qml.templates.StronglyEntanglingLayers(weights, wires=range(2))
      return [qml.expval(qml.PauliZ(i)) for i in range(2)]

  weight_shapes = {"weights": (3, 2, 3)}

  qlayer = qml.qnn.KerasLayer(f, weight_shapes, output_dim=2)

  inputs = tf.constant(np.random.random((4, 2)), dtype=tf.float32)

  with tf.GradientTape() as tape:
      out = qlayer(inputs)

  tape.jacobian(out, qlayer.trainable_weights)
  ```

<h4>New operations, templates, and measurements</h4>

* Adds the `qml.density_matrix` QNode return with partial trace capabilities.
  [(#878)](https://github.com/PennyLaneAI/pennylane/pull/878)

  The density matrix over the provided wires is returned, with all other subsystems traced out.
  `qml.density_matrix` currently works for both the `default.qubit` and `default.mixed` devices.

  ```python
  qml.enable_tape()
  dev = qml.device("default.qubit", wires=2)

  def circuit(x):
      qml.PauliY(wires=0)
      qml.Hadamard(wires=1)
      return qml.density_matrix(wires=[1])  # wire 0 is traced out
  ```

* Adds the square-root X gate `SX`. [(#871)](https://github.com/PennyLaneAI/pennylane/pull/871)

  ```python
  dev = qml.device("default.qubit", wires=1)

  @qml.qnode(dev)
  def circuit():
      qml.SX(wires=[0])
      return qml.expval(qml.PauliZ(wires=[0]))
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
  >>> with qml.tape.QuantumTape() as tape:
  ...    qml.Hadamard(wires=0)
  ...    qml.RZ(0.26, wires=1)
  ...    qml.CNOT(wires=[1, 0])
  ...    qml.Rot(1.8, -2.7, 0.2, wires=0)
  ...    qml.Hadamard(wires=1)
  ...    qml.CNOT(wires=[0, 1])
  ...    qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  >>> tape.get_resources()
  {'Hadamard': 2, 'RZ': 1, 'CNOT': 2, 'Rot': 1}
  >>> tape.get_depth()
  4
  ```

* The number of device executions over a QNode's lifetime can now be returned using `num_executions`.
  [(#853)](https://github.com/PennyLaneAI/pennylane/pull/853)

  ```pycon
  >>> dev = qml.device("default.qubit", wires=2)
  >>> @qml.qnode(dev)
  ... def circuit(x, y):
  ...    qml.RX(x, wires=[0])
  ...    qml.RY(y, wires=[1])
  ...    qml.CNOT(wires=[0, 1])
  ...    return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))
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

  - `qml.qnn.KerasLayer` [(#869)](https://github.com/PennyLaneAI/pennylane/pull/869)

  - `qml.qnn.TorchLayer` [(#865)](https://github.com/PennyLaneAI/pennylane/pull/865)

  - The `qml.qaoa` module [(#905)](https://github.com/PennyLaneAI/pennylane/pull/905)

* A new function, `qml.refresh_devices()`, has been added, allowing PennyLane to
  rescan installed PennyLane plugins and refresh the device list. In addition, the `qml.device`
  loader will attempt to refresh devices if the required plugin device cannot be found.
  This will result in an improved experience if installing PennyLane and plugins within
  a running Python session (for example, on Google Colab), and avoid the need to
  restart the kernel/runtime.
  [(#907)](https://github.com/PennyLaneAI/pennylane/pull/907)

* When using `grad_fn = qml.grad(cost)` to compute the gradient of a cost function with the Autograd
  interface, the value of the intermediate forward pass is now available via the `grad_fn.forward`
  property
  [(#914)](https://github.com/PennyLaneAI/pennylane/pull/914):

  ```python
  def cost_fn(x, y):
      return 2 * np.sin(x[0]) * np.exp(-x[1]) + x[0] ** 3 + np.cos(y)

  params = np.array([0.1, 0.5], requires_grad=True)
  data = np.array(0.65, requires_grad=False)
  grad_fn = qml.grad(cost_fn)

  grad_fn(params, data)  # perform backprop and evaluate the gradient
  grad_fn.forward  # the cost function value
  ```

* Gradient-based optimizers now have a `step_and_cost` method that returns
  both the next step as well as the objective (cost) function output.
  [(#916)](https://github.com/PennyLaneAI/pennylane/pull/916)

  ```pycon
  >>> opt = qml.GradientDescentOptimizer()
  >>> params, cost = opt.step_and_cost(cost_fn, params)
  ```

* PennyLane provides a new experimental module `qml.proc` which provides framework-agnostic processing
  functions for array and tensor manipulations.
  [(#886)](https://github.com/PennyLaneAI/pennylane/pull/886)

  Given the input tensor-like object, the call is
  dispatched to the corresponding array manipulation framework, allowing for end-to-end
  differentiation to be preserved.

  ```pycon
  >>> x = torch.tensor([1., 2.])
  >>> qml.proc.ones_like(x)
  tensor([1, 1])
  >>> y = tf.Variable([[0], [5]])
  >>> qml.proc.ones_like(y, dtype=np.complex128)
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
  >>> dev = qml.device("default.qubit", wires=2, cache=10)
  ```

* The `Operation`, `Tensor`, and `MeasurementProcess` classes now have the `__copy__` special method
  defined.
  [(#840)](https://github.com/PennyLaneAI/pennylane/pull/840)

  This allows us to ensure that, when a shallow copy is performed of an operation, the
  mutable list storing the operation parameters is *also* shallow copied. Both the old operation and
  the copied operation will continue to share the same parameter data,
  ```pycon
  >>> import copy
  >>> op = qml.RX(0.2, wires=0)
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

# Release 0.12.0

<h3>New features since last release</h3>

<h4>New and improved simulators</h4>

* PennyLane now supports a new device, `default.mixed`, designed for
  simulating mixed-state quantum computations. This enables native
  support for implementing noisy channels in a circuit, which generally
  map pure states to mixed states.
  [(#794)](https://github.com/PennyLaneAI/pennylane/pull/794)
  [(#807)](https://github.com/PennyLaneAI/pennylane/pull/807)
  [(#819)](https://github.com/PennyLaneAI/pennylane/pull/819)

  The device can be initialized as
  ```pycon
  >>> dev = qml.device("default.mixed", wires=1)
  ```

  This allows the construction of QNodes that include non-unitary operations,
  such as noisy channels:

  ```pycon
  >>> @qml.qnode(dev)
  ... def circuit(params):
  ...     qml.RX(params[0], wires=0)
  ...     qml.RY(params[1], wires=0)
  ...     qml.AmplitudeDamping(0.5, wires=0)
  ...     return qml.expval(qml.PauliZ(0))
  >>> print(circuit([0.54, 0.12]))
  0.9257702929524184
  >>> print(circuit([0, np.pi]))
  0.0
  ```

<h4>New tools for optimizing measurements</h4>

* The new `grouping` module provides functionality for grouping simultaneously measurable Pauli word
  observables.
  [(#761)](https://github.com/PennyLaneAI/pennylane/pull/761)
  [(#850)](https://github.com/PennyLaneAI/pennylane/pull/850)
  [(#852)](https://github.com/PennyLaneAI/pennylane/pull/852)

  - The `optimize_measurements` function will take as input a list of Pauli word observables and
    their corresponding coefficients (if any), and will return the partitioned Pauli terms
    diagonalized in the measurement basis and the corresponding diagonalizing circuits.

    ```python
    from pennylane.grouping import optimize_measurements
    h, nr_qubits = qml.qchem.molecular_hamiltonian("h2", "h2.xyz")
    rotations, grouped_ops, grouped_coeffs = optimize_measurements(h.ops, h.coeffs, grouping="qwc")
    ```

    The diagonalizing circuits of `rotations` correspond to the diagonalized Pauli word groupings of
    `grouped_ops`.

  - Pauli word partitioning utilities are performed by the `PauliGroupingStrategy`
    class. An input list of Pauli words can be partitioned into mutually commuting,
    qubit-wise-commuting, or anticommuting groupings.

    For example, partitioning Pauli words into anticommutative groupings by the Recursive Largest
    First (RLF) graph colouring heuristic:

    ```python
    from pennylane import PauliX, PauliY, PauliZ, Identity
    from pennylane.grouping import group_observables
    pauli_words = [
        Identity('a') @ Identity('b'),
        Identity('a') @ PauliX('b'),
        Identity('a') @ PauliY('b'),
        PauliZ('a') @ PauliX('b'),
        PauliZ('a') @ PauliY('b'),
        PauliZ('a') @ PauliZ('b')
    ]
    groupings = group_observables(pauli_words, grouping_type='anticommuting', method='rlf')
    ```

  - Various utility functions are included for obtaining and manipulating Pauli
    words in the binary symplectic vector space representation.

    For instance, two Pauli words may be converted to their binary vector representation:

    ```pycon
    >>> from pennylane.grouping import pauli_to_binary
    >>> from pennylane.wires import Wires
    >>> wire_map = {Wires('a'): 0, Wires('b'): 1}
    >>> pauli_vec_1 = pauli_to_binary(qml.PauliX('a') @ qml.PauliY('b'))
    >>> pauli_vec_2 = pauli_to_binary(qml.PauliZ('a') @ qml.PauliZ('b'))
    >>> pauli_vec_1
    [1. 1. 0. 1.]
    >>> pauli_vec_2
    [0. 0. 1. 1.]
    ```

    Their product up to a phase may be computed by taking the sum of their binary vector
    representations, and returned in the operator representation.

    ```pycon
    >>> from pennylane.grouping import binary_to_pauli
    >>> binary_to_pauli((pauli_vec_1 + pauli_vec_2) % 2, wire_map)
    Tensor product ['PauliY', 'PauliX']: 0 params, wires ['a', 'b']
    ```

    For more details on the grouping module, see the
    [grouping module documentation](https://pennylane.readthedocs.io/en/stable/code/qml_grouping.html)


<h4>Returning the quantum state from simulators</h4>

* The quantum state of a QNode can now be returned using the `qml.state()` return function.
  [(#818)](https://github.com/XanaduAI/pennylane/pull/818)

  ```python
  import pennylane as qml

  dev = qml.device("default.qubit", wires=3)
  qml.enable_tape()

  @qml.qnode(dev)
  def qfunc(x, y):
      qml.RZ(x, wires=0)
      qml.CNOT(wires=[0, 1])
      qml.RY(y, wires=1)
      qml.CNOT(wires=[0, 2])
      return qml.state()

  >>> qfunc(0.56, 0.1)
  array([0.95985437-0.27601028j, 0.        +0.j        ,
         0.04803275-0.01381203j, 0.        +0.j        ,
         0.        +0.j        , 0.        +0.j        ,
         0.        +0.j        , 0.        +0.j        ])
  ```

  Differentiating the state is currently available when using the
  classical backpropagation differentiation method (`diff_method="backprop"`) with a compatible device,
  and when using the new tape mode.

<h4>New operations and channels</h4>

* PennyLane now includes standard channels such as the Amplitude-damping,
  Phase-damping, and Depolarizing channels, as well as the ability
  to make custom qubit channels.
  [(#760)](https://github.com/PennyLaneAI/pennylane/pull/760)
  [(#766)](https://github.com/PennyLaneAI/pennylane/pull/766)
  [(#778)](https://github.com/PennyLaneAI/pennylane/pull/778)

* The controlled-Y operation is now available via `qml.CY`. For devices that do
  not natively support the controlled-Y operation, it will be decomposed
  into `qml.RY`, `qml.CNOT`, and `qml.S` operations.
  [(#806)](https://github.com/PennyLaneAI/pennylane/pull/806)

<h4>Preview the next-generation PennyLane QNode</h4>

* The new PennyLane `tape` module provides a re-formulated QNode class, rewritten from the ground-up,
  that uses a new `QuantumTape` object to represent the QNode's quantum circuit. Tape mode
  provides several advantages over the standard PennyLane QNode.
  [(#785)](https://github.com/PennyLaneAI/pennylane/pull/785)
  [(#792)](https://github.com/PennyLaneAI/pennylane/pull/792)
  [(#796)](https://github.com/PennyLaneAI/pennylane/pull/796)
  [(#800)](https://github.com/PennyLaneAI/pennylane/pull/800)
  [(#803)](https://github.com/PennyLaneAI/pennylane/pull/803)
  [(#804)](https://github.com/PennyLaneAI/pennylane/pull/804)
  [(#805)](https://github.com/PennyLaneAI/pennylane/pull/805)
  [(#808)](https://github.com/PennyLaneAI/pennylane/pull/808)
  [(#810)](https://github.com/PennyLaneAI/pennylane/pull/810)
  [(#811)](https://github.com/PennyLaneAI/pennylane/pull/811)
  [(#815)](https://github.com/PennyLaneAI/pennylane/pull/815)
  [(#820)](https://github.com/PennyLaneAI/pennylane/pull/820)
  [(#823)](https://github.com/PennyLaneAI/pennylane/pull/823)
  [(#824)](https://github.com/PennyLaneAI/pennylane/pull/824)
  [(#829)](https://github.com/PennyLaneAI/pennylane/pull/829)

  - Support for in-QNode classical processing: Tape mode allows for differentiable classical
    processing within the QNode.

  - No more Variable wrapping: In tape mode, QNode arguments no longer become `Variable`
    objects within the QNode.

  - Less restrictive QNode signatures: There is no longer any restriction on the QNode signature;
    the QNode can be defined and called following the same rules as standard Python functions.

  - Unifying all QNodes: The tape-mode QNode merges all QNodes (including the
    `JacobianQNode` and the `PassthruQNode`) into a single unified QNode, with
    identical behaviour regardless of the differentiation type.

  - Optimizations: Tape mode provides various performance optimizations, reducing pre- and
    post-processing overhead, and reduces the number of quantum evaluations in certain cases.

  Note that tape mode is **experimental**, and does not currently have feature-parity with the
  existing QNode. [Feedback and bug reports](https://github.com/PennyLaneAI/pennylane/issues) are
  encouraged and will help improve the new tape mode.

  Tape mode can be enabled globally via the `qml.enable_tape` function, without changing your
  PennyLane code:

  ```python
  qml.enable_tape()
  dev = qml.device("default.qubit", wires=1)

  @qml.qnode(dev, interface="tf")
  def circuit(p):
      print("Parameter value:", p)
      qml.RX(tf.sin(p[0])**2 + p[1], wires=0)
      return qml.expval(qml.PauliZ(0))
  ```

  For more details, please see the [tape mode
  documentation](https://pennylane.readthedocs.io/en/stable/code/qml_tape.html).

<h3>Improvements</h3>

* QNode caching has been introduced, allowing the QNode to keep track of the results of previous
  device executions and reuse those results in subsequent calls.
  Note that QNode caching is only supported in the new and experimental tape-mode.
  [(#817)](https://github.com/PennyLaneAI/pennylane/pull/817)

  Caching is available by passing a `caching` argument to the QNode:

  ```python
  dev = qml.device("default.qubit", wires=2)
  qml.enable_tape()

  @qml.qnode(dev, caching=10)  # cache up to 10 evaluations
  def qfunc(x):
      qml.RX(x, wires=0)
      qml.RX(0.3, wires=1)
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.PauliZ(1))

  qfunc(0.1)  # first evaluation executes on the device
  qfunc(0.1)  # second evaluation accesses the cached result
  ```

* Sped up the application of certain gates in `default.qubit` by using array/tensor
  manipulation tricks. The following gates are affected: `PauliX`, `PauliY`, `PauliZ`,
  `Hadamard`, `SWAP`, `S`, `T`, `CNOT`, `CZ`.
  [(#772)](https://github.com/PennyLaneAI/pennylane/pull/772)

* The computation of marginal probabilities has been made more efficient for devices
  with a large number of wires, achieving in some cases a 5x speedup.
  [(#799)](https://github.com/PennyLaneAI/pennylane/pull/799)

* Adds arithmetic operations (addition, tensor product,
  subtraction, and scalar multiplication) between `Hamiltonian`,
  `Tensor`, and `Observable` objects, and inline arithmetic
  operations between Hamiltonians and other observables.
  [(#765)](https://github.com/PennyLaneAI/pennylane/pull/765)

  Hamiltonians can now easily be defined as sums of observables:

  ```pycon3
  >>> H = 3 * qml.PauliZ(0) - (qml.PauliX(0) @ qml.PauliX(1)) + qml.Hamiltonian([4], [qml.PauliZ(0)])
  >>> print(H)
  (7.0) [Z0] + (-1.0) [X0 X1]
  ```

* Adds `compare()` method to `Observable` and `Hamiltonian` classes, which allows
  for comparison between observable quantities.
  [(#765)](https://github.com/PennyLaneAI/pennylane/pull/765)

  ```pycon3
  >>> H = qml.Hamiltonian([1], [qml.PauliZ(0)])
  >>> obs = qml.PauliZ(0) @ qml.Identity(1)
  >>> print(H.compare(obs))
  True
  ```

  ```pycon3
  >>> H = qml.Hamiltonian([2], [qml.PauliZ(0)])
  >>> obs = qml.PauliZ(1) @ qml.Identity(0)
  >>> print(H.compare(obs))
  False
  ```

* Adds `simplify()` method to the `Hamiltonian` class.
  [(#765)](https://github.com/PennyLaneAI/pennylane/pull/765)

  ```pycon3
  >>> H = qml.Hamiltonian([1, 2], [qml.PauliZ(0), qml.PauliZ(0) @ qml.Identity(1)])
  >>> H.simplify()
  >>> print(H)
  (3.0) [Z0]
  ```

* Added a new bit-flip mixer to the `qml.qaoa` module.
  [(#774)](https://github.com/PennyLaneAI/pennylane/pull/774)

* Summation of two `Wires` objects is now supported and will return
  a `Wires` object containing the set of all wires defined by the
  terms in the summation.
  [(#812)](https://github.com/PennyLaneAI/pennylane/pull/812)

<h3>Breaking changes</h3>

* The PennyLane NumPy module now returns scalar (zero-dimensional) arrays where
  Python scalars were previously returned.
  [(#820)](https://github.com/PennyLaneAI/pennylane/pull/820)
  [(#833)](https://github.com/PennyLaneAI/pennylane/pull/833)

  For example, this affects array element indexing, and summation:

  ```pycon
  >>> x = np.array([1, 2, 3], requires_grad=False)
  >>> x[0]
  tensor(1, requires_grad=False)
  >>> np.sum(x)
  tensor(6, requires_grad=True)
  ```

  This may require small updates to user code. A convenience method, `np.tensor.unwrap()`,
  has been added to help ease the transition. This converts PennyLane NumPy tensors
  to standard NumPy arrays and Python scalars:

  ```pycon
  >>> x = np.array(1.543, requires_grad=False)
  >>> x.unwrap()
  1.543
  ```

  Note, however, that information regarding array differentiability will be
  lost.

* The device capabilities dictionary has been redesigned, for clarity and robustness. In particular,
  the capabilities dictionary is now inherited from the parent class, various keys have more
  expressive names, and all keys are now defined in the base device class. For more details, please
  [refer to the developer
  documentation](https://pennylane.readthedocs.io/en/stable/development/plugins.html#device-capabilities).
  [(#781)](https://github.com/PennyLaneAI/pennylane/pull/781/files)

<h3>Bug fixes</h3>

* Changed to use lists for storing variable values inside `BaseQNode`
  allowing complex matrices to be passed to `QubitUnitary`.
  [(#773)](https://github.com/PennyLaneAI/pennylane/pull/773)

* Fixed a bug within `default.qubit`, resulting in greater efficiency
  when applying a state vector to all wires on the device.
  [(#849)](https://github.com/PennyLaneAI/pennylane/pull/849)

<h3>Documentation</h3>

* Equations have been added to the `qml.sample` and `qml.probs` docstrings
  to clarify the mathematical foundation of the performed measurements.
  [(#843)](https://github.com/PennyLaneAI/pennylane/pull/843)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Aroosa Ijaz, Juan Miguel Arrazola, Thomas Bromley, Jack Ceroni, Alain Delgado Gran, Josh Izaac,
Soran Jahangiri, Nathan Killoran, Robert Lang, Cedric Lin, Olivia Di Matteo, Nicolás Quesada, Maria
Schuld, Antal Száva.

# Release 0.11.0

<h3>New features since last release</h3>

<h4>New and improved simulators</h4>

* Added a new device, `default.qubit.autograd`, a pure-state qubit simulator written using Autograd.
  This device supports classical backpropagation (`diff_method="backprop"`); this can
  be faster than the parameter-shift rule for computing quantum gradients
  when the number of parameters to be optimized is large.
  [(#721)](https://github.com/XanaduAI/pennylane/pull/721)

  ```pycon
  >>> dev = qml.device("default.qubit.autograd", wires=1)
  >>> @qml.qnode(dev, diff_method="backprop")
  ... def circuit(x):
  ...     qml.RX(x[1], wires=0)
  ...     qml.Rot(x[0], x[1], x[2], wires=0)
  ...     return qml.expval(qml.PauliZ(0))
  >>> weights = np.array([0.2, 0.5, 0.1])
  >>> grad_fn = qml.grad(circuit)
  >>> print(grad_fn(weights))
  array([-2.25267173e-01, -1.00864546e+00,  6.93889390e-18])
  ```

  See the [device documentation](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.devices.default_qubit_autograd.DefaultQubitAutograd.html) for more details.

* A new experimental C++ state-vector simulator device is now available, `lightning.qubit`. It
  uses the C++ Eigen library to perform fast linear algebra calculations for simulating quantum
  state-vector evolution.

  `lightning.qubit` is currently in beta; it can be installed via `pip`:

  ```console
  $ pip install pennylane-lightning
  ```

  Once installed, it can be used as a PennyLane device:

  ```pycon
  >>> dev = qml.device("lightning.qubit", wires=2)
  ```

  For more details, please see the [lightning qubit documentation](https://pennylane-lightning.readthedocs.io).

<h4>New algorithms and templates</h4>

* Added built-in QAOA functionality via the new `qml.qaoa` module.
  [(#712)](https://github.com/PennyLaneAI/pennylane/pull/712)
  [(#718)](https://github.com/PennyLaneAI/pennylane/pull/718)
  [(#741)](https://github.com/PennyLaneAI/pennylane/pull/741)
  [(#720)](https://github.com/PennyLaneAI/pennylane/pull/720)

  This includes the following features:

  * New `qml.qaoa.x_mixer` and `qml.qaoa.xy_mixer` functions for defining Pauli-X and XY
    mixer Hamiltonians.

  * MaxCut: The `qml.qaoa.maxcut` function allows easy construction of the cost Hamiltonian
    and recommended mixer Hamiltonian for solving the MaxCut problem for a supplied graph.

  * Layers: `qml.qaoa.cost_layer` and `qml.qaoa.mixer_layer` take cost and mixer
    Hamiltonians, respectively, and apply the corresponding QAOA cost and mixer layers
    to the quantum circuit

  For example, using PennyLane to construct and solve a MaxCut problem with QAOA:

  ```python
  wires = range(3)
  graph = Graph([(0, 1), (1, 2), (2, 0)])
  cost_h, mixer_h = qaoa.maxcut(graph)

  def qaoa_layer(gamma, alpha):
      qaoa.cost_layer(gamma, cost_h)
      qaoa.mixer_layer(alpha, mixer_h)

  def antatz(params, **kwargs):

      for w in wires:
          qml.Hadamard(wires=w)

      # repeat the QAOA layer two times
      qml.layer(qaoa_layer, 2, params[0], params[1])

  dev = qml.device('default.qubit', wires=len(wires))
  cost_function = qml.VQECost(ansatz, cost_h, dev)
  ```

* Added an `ApproxTimeEvolution` template to the PennyLane templates module, which
  can be used to implement Trotterized time-evolution under a Hamiltonian.
  [(#710)](https://github.com/XanaduAI/pennylane/pull/710)

  <img src="https://pennylane.readthedocs.io/en/latest/_static/templates/subroutines/approx_time_evolution.png" width=50%/>

* Added a `qml.layer` template-constructing function, which takes a unitary, and
  repeatedly applies it on a set of wires to a given depth.
  [(#723)](https://github.com/PennyLaneAI/pennylane/pull/723)

  ```python
  def subroutine():
      qml.Hadamard(wires=[0])
      qml.CNOT(wires=[0, 1])
      qml.PauliX(wires=[1])

  dev = qml.device('default.qubit', wires=3)

  @qml.qnode(dev)
  def circuit():
      qml.layer(subroutine, 3)
      return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]
  ```

  This creates the following circuit:
  ```pycon
  >>> circuit()
  >>> print(circuit.draw())
  0: ──H──╭C──X──H──╭C──X──H──╭C──X──┤ ⟨Z⟩
  1: ─────╰X────────╰X────────╰X─────┤ ⟨Z⟩
  ```

* Added the `qml.utils.decompose_hamiltonian` function. This function can be used to
  decompose a Hamiltonian into a linear combination of Pauli operators.
  [(#671)](https://github.com/XanaduAI/pennylane/pull/671)

  ```pycon
  >>> A = np.array(
  ... [[-2, -2+1j, -2, -2],
  ... [-2-1j,  0,  0, -1],
  ... [-2,  0, -2, -1],
  ... [-2, -1, -1,  0]])
  >>> coeffs, obs_list = decompose_hamiltonian(A)
  ```

<h4>New device features</h4>

* It is now possible to specify custom wire labels, such as `['anc1', 'anc2', 0, 1, 3]`, where the labels
  can be strings or numbers.
  [(#666)](https://github.com/XanaduAI/pennylane/pull/666)

  Custom wire labels are defined by passing a list to the `wires` argument when creating the device:

  ```pycon
  >>> dev = qml.device("default.qubit", wires=['anc1', 'anc2', 0, 1, 3])
  ```

  Quantum operations should then be invoked with these custom wire labels:

  ``` pycon
  >>> @qml.qnode(dev)
  >>> def circuit():
  ...    qml.Hadamard(wires='anc2')
  ...    qml.CNOT(wires=['anc1', 3])
  ...    ...
  ```

  The existing behaviour, in which the number of wires is specified on device initialization,
  continues to work as usual. This gives a default behaviour where wires are labelled
  by consecutive integers.

  ```pycon
  >>> dev = qml.device("default.qubit", wires=5)
  ```

* An integrated device test suite has been added, which can be used
  to run basic integration tests on core or external devices.
  [(#695)](https://github.com/PennyLaneAI/pennylane/pull/695)
  [(#724)](https://github.com/PennyLaneAI/pennylane/pull/724)
  [(#733)](https://github.com/PennyLaneAI/pennylane/pull/733)

  The test can be invoked against a particular device by calling the `pl-device-test`
  command line program:

  ```console
  $ pl-device-test --device=default.qubit --shots=1234 --analytic=False
  ```

  If the tests are run on external devices, the device and its dependencies must be
  installed locally. For more details, please see the
  [plugin test documentation](http://pennylane.readthedocs.io/en/latest/code/api/pennylane.devices.tests.html).

<h3>Improvements</h3>

* The functions implementing the quantum circuits building the Unitary Coupled-Cluster
  (UCCSD) VQE ansatz have been improved, with a more consistent naming convention and
  improved docstrings.
  [(#748)](https://github.com/PennyLaneAI/pennylane/pull/748)

  The changes include:

  - The terms *1particle-1hole (ph)* and *2particle-2hole (pphh)* excitations
    were replaced with the names *single* and *double* excitations, respectively.

  - The non-differentiable arguments in the `UCCSD` template were renamed accordingly:
    `ph` → `s_wires`, `pphh` → `d_wires`

  - The term *virtual*, previously used to refer the *unoccupied* orbitals, was discarded.

  - The Usage Details sections were updated and improved.

* Added support for TensorFlow 2.3 and PyTorch 1.6.
  [(#725)](https://github.com/PennyLaneAI/pennylane/pull/725)

* Returning probabilities is now supported from photonic QNodes.
  As with qubit QNodes, photonic QNodes returning probabilities are
  end-to-end differentiable.
  [(#699)](https://github.com/XanaduAI/pennylane/pull/699/)

  ```pycon
  >>> dev = qml.device("strawberryfields.fock", wires=2, cutoff_dim=5)
  >>> @qml.qnode(dev)
  ... def circuit(a):
  ...     qml.Displacement(a, 0, wires=0)
  ...     return qml.probs(wires=0)
  >>> print(circuit(0.5))
  [7.78800783e-01 1.94700196e-01 2.43375245e-02 2.02812704e-03 1.26757940e-04]
  ```

<h3>Breaking changes</h3>

* The `pennylane.plugins` and `pennylane.beta.plugins` folders have been renamed to
  `pennylane.devices` and `pennylane.beta.devices`, to reflect their content better.
  [(#726)](https://github.com/XanaduAI/pennylane/pull/726)

<h3>Bug fixes</h3>

* The PennyLane interface conversion functions can now convert QNodes with
  pre-existing interfaces.
  [(#707)](https://github.com/XanaduAI/pennylane/pull/707)

<h3>Documentation</h3>

* The interfaces section of the documentation has been renamed to 'Interfaces and training',
  and updated with the latest variable handling details.
  [(#753)](https://github.com/PennyLaneAI/pennylane/pull/753)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Thomas Bromley, Jack Ceroni, Alain Delgado Gran, Shadab Hussain, Theodor
Isacsson, Josh Izaac, Nathan Killoran, Maria Schuld, Antal Száva, Nicola Vitucci.

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
  >>> dev = qml.device("default.qubit.tf", wires=1)
  >>> @tf.function
  ... @qml.qnode(dev, interface="tf", diff_method="backprop")
  ... def circuit(x):
  ...     qml.RX(x[1], wires=0)
  ...     qml.Rot(x[0], x[1], x[2], wires=0)
  ...     return qml.expval(qml.PauliZ(0))
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

  A PennyLane QNode can be converted into a `torch.nn` layer using the `qml.qnn.TorchLayer` class:

  ```pycon
  >>> @qml.qnode(dev)
  ... def qnode(inputs, weights_0, weight_1):
  ...    # define the circuit
  ...    # ...

  >>> weight_shapes = {"weights_0": 3, "weight_1": 1}
  >>> qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
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
  >>> dev = qml.device("default.qubit", wires=2)
  ... @qml.qnode(dev, diff_method="reversible")
  ... def circuit(x):
  ...     qml.RX(x, wires=0)
  ...     qml.RX(x, wires=0)
  ...     qml.CNOT(wires=[0,1])
  ...     return qml.expval(qml.PauliZ(0))
  >>> qml.grad(circuit)(0.5)
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

  - The `argnum` argument to `qml.grad` is now optional; if not provided, arguments explicitly
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


# Release 0.9.0

<h3>New features since last release</h3>

<h4>New machine learning integrations</h4>

* PennyLane QNodes can now be converted into Keras layers, allowing for creation of quantum and
  hybrid models using the Keras API.
  [(#529)](https://github.com/XanaduAI/pennylane/pull/529)

  A PennyLane QNode can be converted into a Keras layer using the `KerasLayer` class:

  ```python
  from pennylane.qnn import KerasLayer

  @qml.qnode(dev)
  def circuit(inputs, weights_0, weight_1):
     # define the circuit
     # ...

  weight_shapes = {"weights_0": 3, "weight_1": 1}
  qlayer = qml.qnn.KerasLayer(circuit, weight_shapes, output_dim=2)
  ```

  A hybrid model can then be easily constructed:

  ```python
  model = tf.keras.models.Sequential([qlayer, tf.keras.layers.Dense(2)])
  ```

* Added a new type of QNode, `qml.qnodes.PassthruQNode`. For simulators which are coded in an
  external library which supports automatic differentiation, PennyLane will treat a PassthruQNode as
  a "white box", and rely on the external library to directly provide gradients via backpropagation.
  This can be more efficient than the using parameter-shift rule for a large number of parameters.
  [(#488)](https://github.com/XanaduAI/pennylane/pull/488)

  Currently this behaviour is supported by PennyLane's `default.tensor.tf` device backend,
  compatible with the `'tf'` interface using TensorFlow 2:

  ```python
  dev = qml.device('default.tensor.tf', wires=2)

  @qml.qnode(dev, diff_method="backprop")
  def circuit(params):
      qml.RX(params[0], wires=0)
      qml.RX(params[1], wires=1)
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.PauliZ(0))

  qnode = PassthruQNode(circuit, dev)
  params = tf.Variable([0.3, 0.1])

  with tf.GradientTape() as tape:
      tape.watch(params)
      res = qnode(params)

  grad = tape.gradient(res, params)
  ```

<h4>New optimizers</h4>

* Added the `qml.RotosolveOptimizer`, a gradient-free optimizer
  that minimizes the quantum function by updating each parameter,
  one-by-one, via a closed-form expression while keeping other parameters
  fixed.
  [(#636)](https://github.com/XanaduAI/pennylane/pull/636)
  [(#539)](https://github.com/XanaduAI/pennylane/pull/539)

* Added the `qml.RotoselectOptimizer`, which uses Rotosolve to
  minimizes a quantum function with respect to both the
  rotation operations applied and the rotation parameters.
  [(#636)](https://github.com/XanaduAI/pennylane/pull/636)
  [(#539)](https://github.com/XanaduAI/pennylane/pull/539)

  For example, given a quantum function `f` that accepts parameters `x`
  and a list of corresponding rotation operations `generators`,
  the Rotoselect optimizer will, at each step, update both the parameter
  values and the list of rotation gates to minimize the loss:

  ```pycon
  >>> opt = qml.optimize.RotoselectOptimizer()
  >>> x = [0.3, 0.7]
  >>> generators = [qml.RX, qml.RY]
  >>> for _ in range(100):
  ...     x, generators = opt.step(f, x, generators)
  ```


<h4>New operations</h4>

* Added the `PauliRot` gate, which performs an arbitrary
  Pauli rotation on multiple qubits, and the `MultiRZ` gate,
  which performs a rotation generated by a tensor product
  of Pauli Z operators.
  [(#559)](https://github.com/XanaduAI/pennylane/pull/559)

  ```python
  dev = qml.device('default.qubit', wires=4)

  @qml.qnode(dev)
  def circuit(angle):
      qml.PauliRot(angle, "IXYZ", wires=[0, 1, 2, 3])
      return [qml.expval(qml.PauliZ(wire)) for wire in [0, 1, 2, 3]]
  ```

  ```pycon
  >>> circuit(0.4)
  [1.         0.92106099 0.92106099 1.        ]
  >>> print(circuit.draw())
   0: ──╭RI(0.4)──┤ ⟨Z⟩
   1: ──├RX(0.4)──┤ ⟨Z⟩
   2: ──├RY(0.4)──┤ ⟨Z⟩
   3: ──╰RZ(0.4)──┤ ⟨Z⟩
  ```

  If the `PauliRot` gate is not supported on the target device, it will
  be decomposed into `Hadamard`, `RX` and `MultiRZ` gates. Note that
  identity gates in the Pauli word result in untouched wires:

  ```pycon
  >>> print(circuit.draw())
   0: ───────────────────────────────────┤ ⟨Z⟩
   1: ──H──────────╭RZ(0.4)──H───────────┤ ⟨Z⟩
   2: ──RX(1.571)──├RZ(0.4)──RX(-1.571)──┤ ⟨Z⟩
   3: ─────────────╰RZ(0.4)──────────────┤ ⟨Z⟩
  ```

  If the `MultiRZ` gate is not supported, it will be decomposed into
  `CNOT` and `RZ` gates:

  ```pycon
  >>> print(circuit.draw())
   0: ──────────────────────────────────────────────────┤ ⟨Z⟩
   1: ──H──────────────╭X──RZ(0.4)──╭X──────H───────────┤ ⟨Z⟩
   2: ──RX(1.571)──╭X──╰C───────────╰C──╭X──RX(-1.571)──┤ ⟨Z⟩
   3: ─────────────╰C───────────────────╰C──────────────┤ ⟨Z⟩
  ```

* PennyLane now provides `DiagonalQubitUnitary` for diagonal gates, that are e.g.,
  encountered in IQP circuits. These kinds of gates can be evaluated much faster on
  a simulator device.
  [(#567)](https://github.com/XanaduAI/pennylane/pull/567)

  The gate can be used, for example, to efficiently simulate oracles:

  ```python
  dev = qml.device('default.qubit', wires=3)

  # Function as a bitstring
  f = np.array([1, 0, 0, 1, 1, 0, 1, 0])

  @qml.qnode(dev)
  def circuit(weights1, weights2):
      qml.templates.StronglyEntanglingLayers(weights1, wires=[0, 1, 2])

      # Implements the function as a phase-kickback oracle
      qml.DiagonalQubitUnitary((-1)**f, wires=[0, 1, 2])

      qml.templates.StronglyEntanglingLayers(weights2, wires=[0, 1, 2])
      return [qml.expval(qml.PauliZ(w)) for w in range(3)]
  ```

* Added the `TensorN` CVObservable that can represent the tensor product of the
  `NumberOperator` on photonic backends.
  [(#608)](https://github.com/XanaduAI/pennylane/pull/608)

<h4>New templates</h4>

* Added the `ArbitraryUnitary` and `ArbitraryStatePreparation` templates, which use
  `PauliRot` gates to perform an arbitrary unitary and prepare an arbitrary basis
  state with the minimal number of parameters.
  [(#590)](https://github.com/XanaduAI/pennylane/pull/590)

  ```python
  dev = qml.device('default.qubit', wires=3)

  @qml.qnode(dev)
  def circuit(weights1, weights2):
        qml.templates.ArbitraryStatePreparation(weights1, wires=[0, 1, 2])
        qml.templates.ArbitraryUnitary(weights2, wires=[0, 1, 2])
        return qml.probs(wires=[0, 1, 2])
  ```

* Added the `IQPEmbedding` template, which encodes inputs into the diagonal gates of an
  IQP circuit.
  [(#605)](https://github.com/XanaduAI/pennylane/pull/605)

  <img src="https://pennylane.readthedocs.io/en/latest/_images/iqp.png"
  width=50%></img>

* Added the `SimplifiedTwoDesign` template, which implements the circuit
  design of [Cerezo et al. (2020)](<https://arxiv.org/abs/2001.00550>).
  [(#556)](https://github.com/XanaduAI/pennylane/pull/556)

  <img src="https://pennylane.readthedocs.io/en/latest/_images/simplified_two_design.png"
  width=50%></img>

* Added the `BasicEntanglerLayers` template, which is a simple layer architecture
  of rotations and CNOT nearest-neighbour entanglers.
  [(#555)](https://github.com/XanaduAI/pennylane/pull/555)

  <img src="https://pennylane.readthedocs.io/en/latest/_images/basic_entangler.png"
  width=50%></img>

* PennyLane now offers a broadcasting function to easily construct templates:
  `qml.broadcast()` takes single quantum operations or other templates and applies
  them to wires in a specific pattern.
  [(#515)](https://github.com/XanaduAI/pennylane/pull/515)
  [(#522)](https://github.com/XanaduAI/pennylane/pull/522)
  [(#526)](https://github.com/XanaduAI/pennylane/pull/526)
  [(#603)](https://github.com/XanaduAI/pennylane/pull/603)

  For example, we can use broadcast to repeat a custom template
  across multiple wires:

  ```python
  from pennylane.templates import template

  @template
  def mytemplate(pars, wires):
      qml.Hadamard(wires=wires)
      qml.RY(pars, wires=wires)

  dev = qml.device('default.qubit', wires=3)

  @qml.qnode(dev)
  def circuit(pars):
      qml.broadcast(mytemplate, pattern="single", wires=[0,1,2], parameters=pars)
      return qml.expval(qml.PauliZ(0))
  ```

  ```pycon
  >>> circuit([1, 1, 0.1])
  -0.841470984807896
  >>> print(circuit.draw())
   0: ──H──RY(1.0)──┤ ⟨Z⟩
   1: ──H──RY(1.0)──┤
   2: ──H──RY(0.1)──┤
  ```

  For other available patterns, see the
  [broadcast function documentation](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.broadcast.html).

<h3>Breaking changes</h3>

* The `QAOAEmbedding` now uses the new `MultiRZ` gate as a `ZZ` entangler,
  which changes the convention. While
  previously, the `ZZ` gate in the embedding was implemented as

  ```python
  CNOT(wires=[wires[0], wires[1]])
  RZ(2 * parameter, wires=wires[0])
  CNOT(wires=[wires[0], wires[1]])
  ```

  the `MultiRZ` corresponds to

  ```python
  CNOT(wires=[wires[1], wires[0]])
  RZ(parameter, wires=wires[0])
  CNOT(wires=[wires[1], wires[0]])
  ```

  which differs in the factor of `2`, and fixes a bug in the
  wires that the `CNOT` was applied to.
  [(#609)](https://github.com/XanaduAI/pennylane/pull/609)

* Probability methods are handled by `QubitDevice` and device method
  requirements are modified to simplify plugin development.
  [(#573)](https://github.com/XanaduAI/pennylane/pull/573)

* The internal variables `All` and `Any` to mark an `Operation` as acting on all or any
  wires have been renamed to `AllWires` and `AnyWires`.
  [(#614)](https://github.com/XanaduAI/pennylane/pull/614)

<h3>Improvements</h3>

* A new `Wires` class was introduced for the internal
  bookkeeping of wire indices.
  [(#615)](https://github.com/XanaduAI/pennylane/pull/615)

* Improvements to the speed/performance of the `default.qubit` device.
  [(#567)](https://github.com/XanaduAI/pennylane/pull/567)
  [(#559)](https://github.com/XanaduAI/pennylane/pull/559)

* Added the `"backprop"` and `"device"` differentiation methods to the `qnode`
  decorator.
  [(#552)](https://github.com/XanaduAI/pennylane/pull/552)

  - `"backprop"`: Use classical backpropagation. Default on simulator
    devices that are classically end-to-end differentiable.
    The returned QNode can only be used with the same machine learning
    framework (e.g., `default.tensor.tf` simulator with the `tensorflow` interface).

  - `"device"`: Queries the device directly for the gradient.

  Using the `"backprop"` differentiation method with the `default.tensor.tf`
  device, the created QNode is a 'white-box', and is tightly integrated with
  the overall TensorFlow computation:

  ```python
  >>> dev = qml.device("default.tensor.tf", wires=1)
  >>> @qml.qnode(dev, interface="tf", diff_method="backprop")
  >>> def circuit(x):
  ...     qml.RX(x[1], wires=0)
  ...     qml.Rot(x[0], x[1], x[2], wires=0)
  ...     return qml.expval(qml.PauliZ(0))
  >>> vars = tf.Variable([0.2, 0.5, 0.1])
  >>> with tf.GradientTape() as tape:
  ...     res = circuit(vars)
  >>> tape.gradient(res, vars)
  <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-2.2526717e-01, -1.0086454e+00,  1.3877788e-17], dtype=float32)>
  ```

* The circuit drawer now displays inverted operations, as well as wires
  where probabilities are returned from the device:
  [(#540)](https://github.com/XanaduAI/pennylane/pull/540)

  ```python
  >>> @qml.qnode(dev)
  ... def circuit(theta):
  ...     qml.RX(theta, wires=0)
  ...     qml.CNOT(wires=[0, 1])
  ...     qml.S(wires=1).inv()
  ...     return qml.probs(wires=[0, 1])
  >>> circuit(0.2)
  array([0.99003329, 0.        , 0.        , 0.00996671])
  >>> print(circuit.draw())
  0: ──RX(0.2)──╭C───────╭┤ Probs
  1: ───────────╰X──S⁻¹──╰┤ Probs
  ```

* You can now evaluate the metric tensor of a VQE Hamiltonian via the new
  `VQECost.metric_tensor` method. This allows `VQECost` objects to be directly
  optimized by the quantum natural gradient optimizer (`qml.QNGOptimizer`).
  [(#618)](https://github.com/XanaduAI/pennylane/pull/618)

* The input check functions in `pennylane.templates.utils` are now public
  and visible in the API documentation.
  [(#566)](https://github.com/XanaduAI/pennylane/pull/566)

* Added keyword arguments for step size and order to the `qnode` decorator, as well as
  the `QNode` and `JacobianQNode` classes. This enables the user to set the step size
  and order when using finite difference methods. These options are also exposed when
  creating QNode collections.
  [(#530)](https://github.com/XanaduAI/pennylane/pull/530)
  [(#585)](https://github.com/XanaduAI/pennylane/pull/585)
  [(#587)](https://github.com/XanaduAI/pennylane/pull/587)

* The decomposition for the `CRY` gate now uses the simpler form `RY @ CNOT @ RY @ CNOT`
  [(#547)](https://github.com/XanaduAI/pennylane/pull/547)

* The underlying queuing system was refactored, removing the `qml._current_context`
  property that held the currently active `QNode` or `OperationRecorder`. Now, all
  objects that expose a queue for operations inherit from `QueuingContext` and
  register their queue globally.
  [(#548)](https://github.com/XanaduAI/pennylane/pull/548)

* The PennyLane repository has a new benchmarking tool which supports the comparison of different git revisions.
  [(#568)](https://github.com/XanaduAI/pennylane/pull/568)
  [(#560)](https://github.com/XanaduAI/pennylane/pull/560)
  [(#516)](https://github.com/XanaduAI/pennylane/pull/516)

<h3>Documentation</h3>

* Updated the development section by creating a landing page with links to sub-pages
  containing specific guides.
  [(#596)](https://github.com/XanaduAI/pennylane/pull/596)

* Extended the developer's guide by a section explaining how to add new templates.
  [(#564)](https://github.com/XanaduAI/pennylane/pull/564)

<h3>Bug fixes</h3>

* `tf.GradientTape().jacobian()` can now be evaluated on QNodes using the TensorFlow interface.
  [(#626)](https://github.com/XanaduAI/pennylane/pull/626)

* `RandomLayers()` is now compatible with the qiskit devices.
  [(#597)](https://github.com/XanaduAI/pennylane/pull/597)

* `DefaultQubit.probability()` now returns the correct probability when called with
  `device.analytic=False`.
  [(#563)](https://github.com/XanaduAI/pennylane/pull/563)

* Fixed a bug in the `StronglyEntanglingLayers` template, allowing it to
  work correctly when applied to a single wire.
  [(544)](https://github.com/XanaduAI/pennylane/pull/544)

* Fixed a bug when inverting operations with decompositions; operations marked as inverted
  are now correctly inverted when the fallback decomposition is called.
  [(#543)](https://github.com/XanaduAI/pennylane/pull/543)

* The `QNode.print_applied()` method now correctly displays wires where
  `qml.prob()` is being returned.
  [#542](https://github.com/XanaduAI/pennylane/pull/542)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ville Bergholm, Lana Bozanic, Thomas Bromley, Theodor Isacsson, Josh Izaac, Nathan Killoran,
Maggie Li, Johannes Jakob Meyer, Maria Schuld, Sukin Sim, Antal Száva.

# Release 0.8.1

<h3>Improvements</h3>

* Beginning of support for Python 3.8, with the test suite
  now being run in a Python 3.8 environment.
  [(#501)](https://github.com/XanaduAI/pennylane/pull/501)

<h3>Documentation</h3>

* Present templates as a gallery of thumbnails showing the
  basic circuit architecture.
  [(#499)](https://github.com/XanaduAI/pennylane/pull/499)

<h3>Bug fixes</h3>

* Fixed a bug where multiplying a QNode parameter by 0 caused a divide
  by zero error when calculating the parameter shift formula.
  [(#512)](https://github.com/XanaduAI/pennylane/pull/512)

* Fixed a bug where the shape of differentiable QNode arguments
  was being cached on the first construction, leading to indexing
  errors if the QNode was re-evaluated if the argument changed shape.
  [(#505)](https://github.com/XanaduAI/pennylane/pull/505)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ville Bergholm, Josh Izaac, Johannes Jakob Meyer, Maria Schuld, Antal Száva.

# Release 0.8.0

<h3>New features since last release</h3>

* Added a quantum chemistry package, `pennylane.qchem`, which supports
  integration with OpenFermion, Psi4, PySCF, and OpenBabel.
  [(#453)](https://github.com/XanaduAI/pennylane/pull/453)

  Features include:

  - Generate the qubit Hamiltonians directly starting with the atomic structure of the molecule.
  - Calculate the mean-field (Hartree-Fock) electronic structure of molecules.
  - Allow to define an active space based on the number of active electrons and active orbitals.
  - Perform the fermionic-to-qubit transformation of the electronic Hamiltonian by
    using different functions implemented in OpenFermion.
  - Convert OpenFermion's QubitOperator to a Pennylane `Hamiltonian` class.
  - Perform a Variational Quantum Eigensolver (VQE) computation with this Hamiltonian in PennyLane.

  Check out the [quantum chemistry quickstart](https://pennylane.readthedocs.io/en/latest/introduction/chemistry.html), as well the quantum chemistry and VQE tutorials.

* PennyLane now has some functions and classes for creating and solving VQE
  problems. [(#467)](https://github.com/XanaduAI/pennylane/pull/467)

  - `qml.Hamiltonian`: a lightweight class for representing qubit Hamiltonians
  - `qml.VQECost`: a class for quickly constructing a differentiable cost function
    given a circuit ansatz, Hamiltonian, and one or more devices

    ```python
    >>> H = qml.vqe.Hamiltonian(coeffs, obs)
    >>> cost = qml.VQECost(ansatz, hamiltonian, dev, interface="torch")
    >>> params = torch.rand([4, 3])
    >>> cost(params)
    tensor(0.0245, dtype=torch.float64)
    ```

* Added a circuit drawing feature that provides a text-based representation
  of a QNode instance. It can be invoked via `qnode.draw()`. The user can specify
  to display variable names instead of variable values and choose either an ASCII
  or Unicode charset.
  [(#446)](https://github.com/XanaduAI/pennylane/pull/446)

  Consider the following circuit as an example:
  ```python3
  @qml.qnode(dev)
  def qfunc(a, w):
      qml.Hadamard(0)
      qml.CRX(a, wires=[0, 1])
      qml.Rot(w[0], w[1], w[2], wires=[1])
      qml.CRX(-a, wires=[0, 1])

      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
  ```

  We can draw the circuit after it has been executed:

  ```python
  >>> result = qfunc(2.3, [1.2, 3.2, 0.7])
  >>> print(qfunc.draw())
   0: ──H──╭C────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩
   1: ─────╰RX(2.3)──Rot(1.2, 3.2, 0.7)──╰RX(-2.3)──╰┤ ⟨Z ⊗ Z⟩
  >>> print(qfunc.draw(charset="ascii"))
   0: --H--+C----------------------------+C---------+| <Z @ Z>
   1: -----+RX(2.3)--Rot(1.2, 3.2, 0.7)--+RX(-2.3)--+| <Z @ Z>
  >>> print(qfunc.draw(show_variable_names=True))
   0: ──H──╭C─────────────────────────────╭C─────────╭┤ ⟨Z ⊗ Z⟩
   1: ─────╰RX(a)──Rot(w[0], w[1], w[2])──╰RX(-1*a)──╰┤ ⟨Z ⊗ Z⟩
  ```

* Added `QAOAEmbedding` and its parameter initialization
  as a new trainable template.
  [(#442)](https://github.com/XanaduAI/pennylane/pull/442)

  <img src="https://pennylane.readthedocs.io/en/latest/_images/qaoa_layers.png"
  width=70%></img>

* Added the `qml.probs()` measurement function, allowing QNodes
  to differentiate variational circuit probabilities
  on simulators and hardware.
  [(#432)](https://github.com/XanaduAI/pennylane/pull/432)

  ```python
  @qml.qnode(dev)
  def circuit(x):
      qml.Hadamard(wires=0)
      qml.RY(x, wires=0)
      qml.RX(x, wires=1)
      qml.CNOT(wires=[0, 1])
      return qml.probs(wires=[0])
  ```
  Executing this circuit gives the marginal probability of wire 1:
  ```python
  >>> circuit(0.2)
  [0.40066533 0.59933467]
  ```
  QNodes that return probabilities fully support autodifferentiation.

* Added the convenience load functions `qml.from_pyquil`, `qml.from_quil` and
  `qml.from_quil_file` that convert pyQuil objects and Quil code to PennyLane
  templates. This feature requires version 0.8 or above of the PennyLane-Forest
  plugin.
  [(#459)](https://github.com/XanaduAI/pennylane/pull/459)

* Added a `qml.inv` method that inverts templates and sequences of Operations.
  Added a `@qml.template` decorator that makes templates return the queued Operations.
  [(#462)](https://github.com/XanaduAI/pennylane/pull/462)

  For example, using this function to invert a template inside a QNode:

  ```python3
      @qml.template
      def ansatz(weights, wires):
          for idx, wire in enumerate(wires):
              qml.RX(weights[idx], wires=[wire])

          for idx in range(len(wires) - 1):
              qml.CNOT(wires=[wires[idx], wires[idx + 1]])

      dev = qml.device('default.qubit', wires=2)

      @qml.qnode(dev)
      def circuit(weights):
          qml.inv(ansatz(weights, wires=[0, 1]))
          return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    ```

* Added the `QNodeCollection` container class, that allows independent
  QNodes to be stored and evaluated simultaneously. Experimental support
  for asynchronous evaluation of contained QNodes is provided with the
  `parallel=True` keyword argument.
  [(#466)](https://github.com/XanaduAI/pennylane/pull/466)

* Added a high level `qml.map` function, that maps a quantum
  circuit template over a list of observables or devices, returning
  a `QNodeCollection`.
  [(#466)](https://github.com/XanaduAI/pennylane/pull/466)

  For example:

  ```python3
  >>> def my_template(params, wires, **kwargs):
  >>>    qml.RX(params[0], wires=wires[0])
  >>>    qml.RX(params[1], wires=wires[1])
  >>>    qml.CNOT(wires=wires)

  >>> obs_list = [qml.PauliX(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliX(1)]
  >>> dev = qml.device("default.qubit", wires=2)
  >>> qnodes = qml.map(my_template, obs_list, dev, measure="expval")
  >>> qnodes([0.54, 0.12])
  array([-0.06154835  0.99280864])
  ```

* Added high level `qml.sum`, `qml.dot`, `qml.apply` functions
  that act on QNode collections.
  [(#466)](https://github.com/XanaduAI/pennylane/pull/466)

  `qml.apply` allows vectorized functions to act over the entire QNode
  collection:
  ```python
  >>> qnodes = qml.map(my_template, obs_list, dev, measure="expval")
  >>> cost = qml.apply(np.sin, qnodes)
  >>> cost([0.54, 0.12])
  array([-0.0615095  0.83756375])
  ```

  `qml.sum` and `qml.dot` take the sum of a QNode collection, and a
  dot product of tensors/arrays/QNode collections, respectively.

<h3>Breaking changes</h3>

* Deprecated the old-style `QNode` such that only the new-style `QNode` and its syntax can be used,
  moved all related files from the `pennylane/beta` folder to `pennylane`.
  [(#440)](https://github.com/XanaduAI/pennylane/pull/440)

<h3>Improvements</h3>

* Added the `Tensor.prune()` method and the `Tensor.non_identity_obs` property for extracting
  non-identity instances from the observables making up a `Tensor` instance.
  [(#498)](https://github.com/XanaduAI/pennylane/pull/498)

* Renamed the `expt.tensornet` and `expt.tensornet.tf` devices to `default.tensor` and
  `default.tensor.tf`.
  [(#495)](https://github.com/XanaduAI/pennylane/pull/495)

* Added a serialization method to the `CircuitGraph` class that is used to create a unique
  hash for each quantum circuit graph.
  [(#470)](https://github.com/XanaduAI/pennylane/pull/470)

* Added the `Observable.eigvals` method to return the eigenvalues of observables.
  [(#449)](https://github.com/XanaduAI/pennylane/pull/449)

* Added the `Observable.diagonalizing_gates` method to return the gates
  that diagonalize an observable in the computational basis.
  [(#454)](https://github.com/XanaduAI/pennylane/pull/454)

* Added the `Operator.matrix` method to return the matrix representation
  of an operator in the computational basis.
  [(#454)](https://github.com/XanaduAI/pennylane/pull/454)

* Added a `QubitDevice` class which implements common functionalities of plugin devices such that
  plugin devices can rely on these implementations. The new `QubitDevice` also includes
  a new `execute` method, which allows for more convenient plugin design. In addition, `QubitDevice`
  also unifies the way samples are generated on qubit-based devices.
  [(#452)](https://github.com/XanaduAI/pennylane/pull/452)
  [(#473)](https://github.com/XanaduAI/pennylane/pull/473)

* Improved documentation of `AmplitudeEmbedding` and `BasisEmbedding` templates.
  [(#441)](https://github.com/XanaduAI/pennylane/pull/441)
  [(#439)](https://github.com/XanaduAI/pennylane/pull/439)

* Codeblocks in the documentation now have a 'copy' button for easily
  copying examples.
  [(#437)](https://github.com/XanaduAI/pennylane/pull/437)

<h3>Documentation</h3>

* Update the developers plugin guide to use QubitDevice.
  [(#483)](https://github.com/XanaduAI/pennylane/pull/483)

<h3>Bug fixes</h3>

* Fixed a bug in `CVQNode._pd_analytic`, where non-descendant observables were not
  Heisenberg-transformed before evaluating the partial derivatives when using the
  order-2 parameter-shift method, resulting in an erroneous Jacobian for some circuits.
  [(#433)](https://github.com/XanaduAI/pennylane/pull/433)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Ville Bergholm, Alain Delgado Gran, Olivia Di Matteo,
Theodor Isacsson, Josh Izaac, Soran Jahangiri, Nathan Killoran, Johannes Jakob Meyer,
Zeyue Niu, Maria Schuld, Antal Száva.

# Release 0.7.0

<h3>New features since last release</h3>

* Custom padding constant in `AmplitudeEmbedding` is supported (see 'Breaking changes'.)
  [(#419)](https://github.com/XanaduAI/pennylane/pull/419)

* `StronglyEntanglingLayer` and `RandomLayer` now work with a single wire.
  [(#409)](https://github.com/XanaduAI/pennylane/pull/409)
  [(#413)](https://github.com/XanaduAI/pennylane/pull/413)

* Added support for applying the inverse of an `Operation` within a circuit.
  [(#377)](https://github.com/XanaduAI/pennylane/pull/377)

* Added an `OperationRecorder()` context manager, that allows templates
  and quantum functions to be executed while recording events. The
  recorder can be used with and without QNodes as a debugging utility.
  [(#388)](https://github.com/XanaduAI/pennylane/pull/388)

* Operations can now specify a decomposition that is used when the desired operation
  is not supported on the target device.
  [(#396)](https://github.com/XanaduAI/pennylane/pull/396)

* The ability to load circuits from external frameworks as templates
  has been added via the new `qml.load()` function. This feature
  requires plugin support --- this initial release provides support
  for Qiskit circuits and QASM files when `pennylane-qiskit` is installed,
  via the functions `qml.from_qiskit` and `qml.from_qasm`.
  [(#418)](https://github.com/XanaduAI/pennylane/pull/418)

* An experimental tensor network device has been added
  [(#416)](https://github.com/XanaduAI/pennylane/pull/416)
  [(#395)](https://github.com/XanaduAI/pennylane/pull/395)
  [(#394)](https://github.com/XanaduAI/pennylane/pull/394)
  [(#380)](https://github.com/XanaduAI/pennylane/pull/380)

* An experimental tensor network device which uses TensorFlow for
  backpropagation has been added
  [(#427)](https://github.com/XanaduAI/pennylane/pull/427)

* Custom padding constant in `AmplitudeEmbedding` is supported (see 'Breaking changes'.)
  [(#419)](https://github.com/XanaduAI/pennylane/pull/419)

<h3>Breaking changes</h3>

* The `pad` parameter in `AmplitudeEmbedding()` is now either `None` (no automatic padding), or a
  number that is used as the padding constant.
  [(#419)](https://github.com/XanaduAI/pennylane/pull/419)

* Initialization functions now return a single array of weights per function. Utilities for multi-weight templates
  `Interferometer()` and `CVNeuralNetLayers()` are provided.
  [(#412)](https://github.com/XanaduAI/pennylane/pull/412)

* The single layer templates `RandomLayer()`, `CVNeuralNetLayer()` and `StronglyEntanglingLayer()`
  have been turned into private functions `_random_layer()`, `_cv_neural_net_layer()` and
  `_strongly_entangling_layer()`. Recommended use is now via the corresponding `Layers()` templates.
  [(#413)](https://github.com/XanaduAI/pennylane/pull/413)

<h3>Improvements</h3>

* Added extensive input checks in templates.
  [(#419)](https://github.com/XanaduAI/pennylane/pull/419)

* Templates integration tests are rewritten - now cover keyword/positional argument passing,
  interfaces and combinations of templates.
  [(#409)](https://github.com/XanaduAI/pennylane/pull/409)
  [(#419)](https://github.com/XanaduAI/pennylane/pull/419)

* State vector preparation operations in the `default.qubit` plugin can now be
  applied to subsets of wires, and are restricted to being the first operation
  in a circuit.
  [(#346)](https://github.com/XanaduAI/pennylane/pull/346)

* The `QNode` class is split into a hierarchy of simpler classes.
  [(#354)](https://github.com/XanaduAI/pennylane/pull/354)
  [(#398)](https://github.com/XanaduAI/pennylane/pull/398)
  [(#415)](https://github.com/XanaduAI/pennylane/pull/415)
  [(#417)](https://github.com/XanaduAI/pennylane/pull/417)
  [(#425)](https://github.com/XanaduAI/pennylane/pull/425)

* Added the gates U1, U2 and U3 parametrizing arbitrary unitaries on 1, 2 and 3
  qubits and the Toffoli gate to the set of qubit operations.
  [(#396)](https://github.com/XanaduAI/pennylane/pull/396)

* Changes have been made to accomodate the movement of the main function
  in `pytest._internal` to `pytest._internal.main` in pip 19.3.
  [(#404)](https://github.com/XanaduAI/pennylane/pull/404)

* Added the templates `BasisStatePreparation` and `MottonenStatePreparation` that use
  gates to prepare a basis state and an arbitrary state respectively.
  [(#336)](https://github.com/XanaduAI/pennylane/pull/336)

* Added decompositions for `BasisState` and `QubitStateVector` based on state
  preparation templates.
  [(#414)](https://github.com/XanaduAI/pennylane/pull/414)

* Replaces the pseudo-inverse in the quantum natural gradient optimizer
  (which can be numerically unstable) with `np.linalg.solve`.
  [(#428)](https://github.com/XanaduAI/pennylane/pull/428)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ville Bergholm, Josh Izaac, Nathan Killoran, Angus Lowe, Johannes Jakob Meyer,
Oluwatobi Ogunbayo, Maria Schuld, Antal Száva.

# Release 0.6.1

<h3>New features since last release</h3>

* Added a `print_applied` method to QNodes, allowing the operation
  and observable queue to be printed as last constructed.
  [(#378)](https://github.com/XanaduAI/pennylane/pull/378)

<h3>Improvements</h3>

* A new `Operator` base class is introduced, which is inherited by both the
  `Observable` class and the `Operation` class.
  [(#355)](https://github.com/XanaduAI/pennylane/pull/355)

* Removed deprecated `@abstractproperty` decorators
  in `_device.py`.
  [(#374)](https://github.com/XanaduAI/pennylane/pull/374)

* The `CircuitGraph` class is updated to deal with `Operation` instances directly.
  [(#344)](https://github.com/XanaduAI/pennylane/pull/344)

* Comprehensive gradient tests have been added for the interfaces.
  [(#381)](https://github.com/XanaduAI/pennylane/pull/381)

<h3>Documentation</h3>

* The new restructured documentation has been polished and updated.
  [(#387)](https://github.com/XanaduAI/pennylane/pull/387)
  [(#375)](https://github.com/XanaduAI/pennylane/pull/375)
  [(#372)](https://github.com/XanaduAI/pennylane/pull/372)
  [(#370)](https://github.com/XanaduAI/pennylane/pull/370)
  [(#369)](https://github.com/XanaduAI/pennylane/pull/369)
  [(#367)](https://github.com/XanaduAI/pennylane/pull/367)
  [(#364)](https://github.com/XanaduAI/pennylane/pull/364)

* Updated the development guides.
  [(#382)](https://github.com/XanaduAI/pennylane/pull/382)
  [(#379)](https://github.com/XanaduAI/pennylane/pull/379)

* Added all modules, classes, and functions to the API section
  in the documentation.
  [(#373)](https://github.com/XanaduAI/pennylane/pull/373)

<h3>Bug fixes</h3>

* Replaces the existing `np.linalg.norm` normalization with hand-coded
  normalization, allowing `AmplitudeEmbedding` to be used with differentiable
  parameters. AmplitudeEmbedding tests have been added and improved.
  [(#376)](https://github.com/XanaduAI/pennylane/pull/376)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ville Bergholm, Josh Izaac, Nathan Killoran, Maria Schuld, Antal Száva

# Release 0.6.0

<h3>New features since last release</h3>

* The devices `default.qubit` and `default.gaussian` have a new initialization parameter
  `analytic` that indicates if expectation values and variances should be calculated
  analytically and not be estimated from data.
  [(#317)](https://github.com/XanaduAI/pennylane/pull/317)

* Added C-SWAP gate to the set of qubit operations
  [(#330)](https://github.com/XanaduAI/pennylane/pull/330)

* The TensorFlow interface has been renamed from `"tfe"` to `"tf"`, and
  now supports TensorFlow 2.0.
  [(#337)](https://github.com/XanaduAI/pennylane/pull/337)

* Added the S and T gates to the set of qubit operations.
  [(#343)](https://github.com/XanaduAI/pennylane/pull/343)

* Tensor observables are now supported within the `expval`,
  `var`, and `sample` functions, by using the `@` operator.
  [(#267)](https://github.com/XanaduAI/pennylane/pull/267)


<h3>Breaking changes</h3>

* The argument `n` specifying the number of samples in the method `Device.sample` was removed.
  Instead, the method will always return `Device.shots` many samples.
  [(#317)](https://github.com/XanaduAI/pennylane/pull/317)

<h3>Improvements</h3>

* The number of shots / random samples used to estimate expectation values and variances, `Device.shots`,
  can now be changed after device creation.
  [(#317)](https://github.com/XanaduAI/pennylane/pull/317)

* Unified import shortcuts to be under qml in qnode.py
  and test_operation.py
  [(#329)](https://github.com/XanaduAI/pennylane/pull/329)

* The quantum natural gradient now uses `scipy.linalg.pinvh` which is more efficient for symmetric matrices
  than the previously used `scipy.linalg.pinv`.
  [(#331)](https://github.com/XanaduAI/pennylane/pull/331)

* The deprecated `qml.expval.Observable` syntax has been removed.
  [(#267)](https://github.com/XanaduAI/pennylane/pull/267)

* Remainder of the unittest-style tests were ported to pytest.
  [(#310)](https://github.com/XanaduAI/pennylane/pull/310)

* The `do_queue` argument for operations now only takes effect
  within QNodes. Outside of QNodes, operations can now be instantiated
  without needing to specify `do_queue`.
  [(#359)](https://github.com/XanaduAI/pennylane/pull/359)

<h3>Documentation</h3>

* The docs are rewritten and restructured to contain a code introduction section as well as an API section.
  [(#314)](https://github.com/XanaduAI/pennylane/pull/275)

* Added Ising model example to the tutorials
  [(#319)](https://github.com/XanaduAI/pennylane/pull/319)

* Added tutorial for QAOA on MaxCut problem
  [(#328)](https://github.com/XanaduAI/pennylane/pull/328)

* Added QGAN flow chart figure to its tutorial
  [(#333)](https://github.com/XanaduAI/pennylane/pull/333)

* Added missing figures for gallery thumbnails of state-preparation
  and QGAN tutorials
  [(#326)](https://github.com/XanaduAI/pennylane/pull/326)

* Fixed typos in the state preparation tutorial
  [(#321)](https://github.com/XanaduAI/pennylane/pull/321)

* Fixed bug in VQE tutorial 3D plots
  [(#327)](https://github.com/XanaduAI/pennylane/pull/327)

<h3>Bug fixes</h3>

* Fixed typo in measurement type error message in qnode.py
  [(#341)](https://github.com/XanaduAI/pennylane/pull/341)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Shahnawaz Ahmed, Ville Bergholm, Aroosa Ijaz, Josh Izaac, Nathan Killoran, Angus Lowe,
Johannes Jakob Meyer, Maria Schuld, Antal Száva, Roeland Wiersema.

# Release 0.5.0

<h3>New features since last release</h3>

* Adds a new optimizer, `qml.QNGOptimizer`, which optimizes QNodes using
  quantum natural gradient descent. See https://arxiv.org/abs/1909.02108
  for more details.
  [(#295)](https://github.com/XanaduAI/pennylane/pull/295)
  [(#311)](https://github.com/XanaduAI/pennylane/pull/311)

* Adds a new QNode method, `QNode.metric_tensor()`,
  which returns the block-diagonal approximation to the Fubini-Study
  metric tensor evaluated on the attached device.
  [(#295)](https://github.com/XanaduAI/pennylane/pull/295)

* Sampling support: QNodes can now return a specified number of samples
  from a given observable via the top-level `pennylane.sample()` function.
  To support this on plugin devices, there is a new `Device.sample` method.

  Calculating gradients of QNodes that involve sampling is not possible.
  [(#256)](https://github.com/XanaduAI/pennylane/pull/256)

* `default.qubit` has been updated to provide support for sampling.
  [(#256)](https://github.com/XanaduAI/pennylane/pull/256)

* Added controlled rotation gates to PennyLane operations and `default.qubit` plugin.
  [(#251)](https://github.com/XanaduAI/pennylane/pull/251)

<h3>Breaking changes</h3>

* The method `Device.supported` was removed, and replaced with the methods
  `Device.supports_observable` and `Device.supports_operation`.
  Both methods can be called with string arguments (`dev.supports_observable('PauliX')`) and
  class arguments (`dev.supports_observable(qml.PauliX)`).
  [(#276)](https://github.com/XanaduAI/pennylane/pull/276)

* The following CV observables were renamed to comply with the new Operation/Observable
  scheme: `MeanPhoton` to `NumberOperator`, `Homodyne` to `QuadOperator` and `NumberState` to `FockStateProjector`.
  [(#254)](https://github.com/XanaduAI/pennylane/pull/254)

<h3>Improvements</h3>

* The `AmplitudeEmbedding` function now provides options to normalize and
  pad features to ensure a valid state vector is prepared.
  [(#275)](https://github.com/XanaduAI/pennylane/pull/275)

* Operations can now optionally specify generators, either as existing PennyLane
  operations, or by providing a NumPy array.
  [(#295)](https://github.com/XanaduAI/pennylane/pull/295)
  [(#313)](https://github.com/XanaduAI/pennylane/pull/313)

* Adds a `Device.parameters` property, so that devices can view a dictionary mapping free
  parameters to operation parameters. This will allow plugin devices to take advantage
  of parametric compilation.
  [(#283)](https://github.com/XanaduAI/pennylane/pull/283)

* Introduces two enumerations: `Any` and `All`, representing any number of wires
  and all wires in the system respectively. They can be imported from
  `pennylane.operation`, and can be used when defining the `Operation.num_wires`
  class attribute of operations.
  [(#277)](https://github.com/XanaduAI/pennylane/pull/277)

  As part of this change:

  - `All` is equivalent to the integer 0, for backwards compatibility with the
    existing test suite

  - `Any` is equivalent to the integer -1 to allow numeric comparison
    operators to continue working

  - An additional validation is now added to the `Operation` class,
    which will alert the user that an operation with `num_wires = All`
    is being incorrectly.

* The one-qubit rotations in `pennylane.plugins.default_qubit` no longer depend on Scipy's `expm`. Instead
  they are calculated with Euler's formula.
  [(#292)](https://github.com/XanaduAI/pennylane/pull/292)

* Creates an `ObservableReturnTypes` enumeration class containing `Sample`,
  `Variance` and `Expectation`. These new values can be assigned to the `return_type`
  attribute of an `Observable`.
  [(#290)](https://github.com/XanaduAI/pennylane/pull/290)

* Changed the signature of the `RandomLayer` and `RandomLayers` templates to have a fixed seed by default.
  [(#258)](https://github.com/XanaduAI/pennylane/pull/258)

* `setup.py` has been cleaned up, removing the non-working shebang,
  and removing unused imports.
  [(#262)](https://github.com/XanaduAI/pennylane/pull/262)

<h3>Documentation</h3>

* A documentation refactor to simplify the tutorials and
  include Sphinx-Gallery.
  [(#291)](https://github.com/XanaduAI/pennylane/pull/291)

  - Examples and tutorials previously split across the `examples/`
    and `doc/tutorials/` directories, in a mixture of ReST and Jupyter notebooks,
    have been rewritten as Python scripts with ReST comments in a single location,
    the `examples/` folder.

  - Sphinx-Gallery is used to automatically build and run the tutorials.
    Rendered output is displayed in the Sphinx documentation.

  - Links are provided at the top of every tutorial page for downloading the
    tutorial as an executable python script, downloading the tutorial
    as a Jupyter notebook, or viewing the notebook on GitHub.

  - The tutorials table of contents have been moved to a single quick start page.

* Fixed a typo in `QubitStateVector`.
  [(#296)](https://github.com/XanaduAI/pennylane/pull/296)

* Fixed a typo in the `default_gaussian.gaussian_state` function.
  [(#293)](https://github.com/XanaduAI/pennylane/pull/293)

* Fixed a typo in the gradient recipe within the `RX`, `RY`, `RZ`
  operation docstrings.
  [(#248)](https://github.com/XanaduAI/pennylane/pull/248)

* Fixed a broken link in the tutorial documentation, as a
  result of the `qml.expval.Observable` deprecation.
  [(#246)](https://github.com/XanaduAI/pennylane/pull/246)

<h3>Bug fixes</h3>

* Fixed a bug where a `PolyXP` observable would fail if applied to subsets
  of wires on `default.gaussian`.
  [(#277)](https://github.com/XanaduAI/pennylane/pull/277)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Simon Cross, Aroosa Ijaz, Josh Izaac, Nathan Killoran, Johannes Jakob Meyer,
Rohit Midha, Nicolás Quesada, Maria Schuld, Antal Száva, Roeland Wiersema.

# Release 0.4.0

<h3>New features since last release</h3>

* `pennylane.expval()` is now a top-level *function*, and is no longer
  a package of classes. For now, the existing `pennylane.expval.Observable`
  interface continues to work, but will raise a deprecation warning.
  [(#232)](https://github.com/XanaduAI/pennylane/pull/232)

* Variance support: QNodes can now return the variance of observables,
  via the top-level `pennylane.var()` function. To support this on
  plugin devices, there is a new `Device.var` method.

  The following observables support analytic gradients of variances:

  - All qubit observables (requiring 3 circuit evaluations for involutory
    observables such as `Identity`, `X`, `Y`, `Z`; and 5 circuit evals for
    non-involutary observables, currently only `qml.Hermitian`)

  - First-order CV observables (requiring 5 circuit evaluations)

  Second-order CV observables support numerical variance gradients.

* `pennylane.about()` function added, providing details
  on current PennyLane version, installed plugins, Python,
  platform, and NumPy versions [(#186)](https://github.com/XanaduAI/pennylane/pull/186)

* Removed the logic that allowed `wires` to be passed as a positional
  argument in quantum operations. This allows us to raise more useful
  error messages for the user if incorrect syntax is used.
  [(#188)](https://github.com/XanaduAI/pennylane/pull/188)

* Adds support for multi-qubit expectation values of the `pennylane.Hermitian()`
  observable [(#192)](https://github.com/XanaduAI/pennylane/pull/192)

* Adds support for multi-qubit expectation values in `default.qubit`.
  [(#202)](https://github.com/XanaduAI/pennylane/pull/202)

* Organize templates into submodules [(#195)](https://github.com/XanaduAI/pennylane/pull/195).
  This included the following improvements:

  - Distinguish embedding templates from layer templates.

  - New random initialization functions supporting the templates available
    in the new submodule `pennylane.init`.

  - Added a random circuit template (`RandomLayers()`), in which rotations and 2-qubit gates are randomly
    distributed over the wires

  - Add various embedding strategies

<h3>Breaking changes</h3>

* The `Device` methods `expectations`, `pre_expval`, and `post_expval` have been
  renamed to `observables`, `pre_measure`, and `post_measure` respectively.
  [(#232)](https://github.com/XanaduAI/pennylane/pull/232)

<h3>Improvements</h3>

* `default.qubit` plugin now uses `np.tensordot` when applying quantum operations
  and evaluating expectations, resulting in significant speedup
  [(#239)](https://github.com/XanaduAI/pennylane/pull/239),
  [(#241)](https://github.com/XanaduAI/pennylane/pull/241)

* PennyLane now allows division of quantum operation parameters by a constant
  [(#179)](https://github.com/XanaduAI/pennylane/pull/179)

* Portions of the test suite are in the process of being ported to pytest.
  Note: this is still a work in progress.

  Ported tests include:

  - `test_ops.py`
  - `test_about.py`
  - `test_classical_gradients.py`
  - `test_observables.py`
  - `test_measure.py`
  - `test_init.py`
  - `test_templates*.py`
  - `test_ops.py`
  - `test_variable.py`
  - `test_qnode.py` (partial)

<h3>Bug fixes</h3>

* Fixed a bug in `Device.supported`, which would incorrectly
  mark an operation as supported if it shared a name with an
  observable [(#203)](https://github.com/XanaduAI/pennylane/pull/203)

* Fixed a bug in `Operation.wires`, by explicitly casting the
  type of each wire to an integer [(#206)](https://github.com/XanaduAI/pennylane/pull/206)

* Removed code in PennyLane which configured the logger,
  as this would clash with users' configurations
  [(#208)](https://github.com/XanaduAI/pennylane/pull/208)

* Fixed a bug in `default.qubit`, in which `QubitStateVector` operations
  were accidentally being cast to `np.float` instead of `np.complex`.
  [(#211)](https://github.com/XanaduAI/pennylane/pull/211)


<h3>Contributors</h3>

This release contains contributions from:

Shahnawaz Ahmed, riveSunder, Aroosa Ijaz, Josh Izaac, Nathan Killoran, Maria Schuld.

# Release 0.3.1

<h3>Bug fixes</h3>

* Fixed a bug where the interfaces submodule was not correctly being packaged via setup.py

# Release 0.3.0

<h3>New features since last release</h3>

* PennyLane now includes a new `interfaces` submodule, which enables QNode integration with additional machine learning libraries.
* Adds support for an experimental PyTorch interface for QNodes
* Adds support for an experimental TensorFlow eager execution interface for QNodes
* Adds a PyTorch+GPU+QPU tutorial to the documentation
* Documentation now includes links and tutorials including the new [PennyLane-Forest](https://github.com/rigetti/pennylane-forest) plugin.

<h3>Improvements</h3>

* Printing a QNode object, via `print(qnode)` or in an interactive terminal, now displays more useful information regarding the QNode,
  including the device it runs on, the number of wires, it's interface, and the quantum function it uses:

  ```python
  >>> print(qnode)
  <QNode: device='default.qubit', func=circuit, wires=2, interface=PyTorch>
  ```

<h3>Contributors</h3>

This release contains contributions from:

Josh Izaac and Nathan Killoran.


# Release 0.2.0

<h3>New features since last release</h3>

* Added the `Identity` expectation value for both CV and qubit models [(#135)](https://github.com/XanaduAI/pennylane/pull/135)
* Added the `templates.py` submodule, containing some commonly used QML models to be used as ansatz in QNodes [(#133)](https://github.com/XanaduAI/pennylane/pull/133)
* Added the `qml.Interferometer` CV operation [(#152)](https://github.com/XanaduAI/pennylane/pull/152)
* Wires are now supported as free QNode parameters [(#151)](https://github.com/XanaduAI/pennylane/pull/151)
* Added ability to update stepsizes of the optimizers [(#159)](https://github.com/XanaduAI/pennylane/pull/159)

<h3>Improvements</h3>

* Removed use of hardcoded values in the optimizers, made them parameters (see [#131](https://github.com/XanaduAI/pennylane/pull/131) and [#132](https://github.com/XanaduAI/pennylane/pull/132))
* Created the new `PlaceholderExpectation`, to be used when both CV and qubit expval modules contain expectations with the same name
* Provide a way for plugins to view the operation queue _before_ applying operations. This allows for on-the-fly modifications of
  the queue, allowing hardware-based plugins to support the full range of qubit expectation values. [(#143)](https://github.com/XanaduAI/pennylane/pull/143)
* QNode return values now support _any_ form of sequence, such as lists, sets, etc. [(#144)](https://github.com/XanaduAI/pennylane/pull/144)
* CV analytic gradient calculation is now more robust, allowing for operations which may not themselves be differentiated, but have a
  well defined `_heisenberg_rep` method, and so may succeed operations that are analytically differentiable [(#152)](https://github.com/XanaduAI/pennylane/pull/152)

<h3>Bug fixes</h3>

* Fixed a bug where the variational classifier example was not batching when learning parity (see [#128](https://github.com/XanaduAI/pennylane/pull/128) and [#129](https://github.com/XanaduAI/pennylane/pull/129))
* Fixed an inconsistency where some initial state operations were documented as accepting complex parameters - all operations
  now accept real values [(#146)](https://github.com/XanaduAI/pennylane/pull/146)

<h3>Contributors</h3>

This release contains contributions from:

Christian Gogolin, Josh Izaac, Nathan Killoran, and Maria Schuld.


# Release 0.1.0

Initial public release.

<h3>Contributors</h3>
This release contains contributions from:

Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, and Nathan Killoran.
