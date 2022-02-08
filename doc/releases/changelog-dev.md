:orphan:

# Release 0.21.0-dev (development release)

<h3>New features since last release</h3>

* Continued development of the circuit-cutting compiler:
  
  A method for converting a quantum tape to a directed multigraph that is amenable
  to graph partitioning algorithms for circuit cutting has been added.
  [(#2107)](https://github.com/PennyLaneAI/pennylane/pull/2107)
  
  A method to replace `WireCut` nodes in a directed multigraph with `MeasureNode` 
  and `PrepareNode` placeholders has been added.
  [(#2124)](https://github.com/PennyLaneAI/pennylane/pull/2124)
  
  A method has been added that takes a directed multigraph with `MeasureNode` and
  `PrepareNode` placeholders and fragments into subgraphs and a communication graph.
  [(#2153)](https://github.com/PennyLaneAI/pennylane/pull/2153)

<h3>Improvements</h3>

* The new function `qml.drawer.tape_text` produces a string drawing of a tape. This function
  differs in implementation and minor stylistic details from the old string circuit drawing
  infrastructure.
  [(#1885)](https://github.com/PennyLaneAI/pennylane/pull/1885)

* The `RotosolveOptimizer` now raises an error if no trainable arguments are
  detected, instead of silently skipping update steps for all arguments.
  [(#2109)](https://github.com/PennyLaneAI/pennylane/pull/2109)

* The function `qml.math.safe_squeeze` is introduced and `gradient_transform` allows
  for QNode argument axes of size `1`.
  [(#2080)](https://github.com/PennyLaneAI/pennylane/pull/2080)

  `qml.math.safe_squeeze` wraps `qml.math.squeeze`, with slight modifications:

  - When provided the `axis` keyword argument, axes that do not have size `1` will be
    ignored, instead of raising an error.

  - The keyword argument `exclude_axis` allows to explicitly exclude axes from the
    squeezing.

* The `adjoint` transform now raises and error whenever the object it is applied to
  is not callable.
  [(#2060)](https://github.com/PennyLaneAI/pennylane/pull/2060)

  An example is a list of operations to which one might apply `qml.adjoint`:

  ```python
  dev = qml.device("default.qubit", wires=2)
  @qml.qnode(dev)
  def circuit_wrong(params):
      # Note the difference:                  v                         v
      qml.adjoint(qml.templates.AngleEmbedding(params, wires=dev.wires))
      return qml.state()

  @qml.qnode(dev)
  def circuit_correct(params):
      # Note the difference:                  v                         v
      qml.adjoint(qml.templates.AngleEmbedding)(params, wires=dev.wires)
      return qml.state()

  params = list(range(1, 3))
  ```

  The produced state is

  ```pycon
  >>> circuit_wrong(params)
  [ 0.47415988+0.j          0.        -0.73846026j  0.        -0.25903472j
   -0.40342268+0.j        ]
  ```

  but if we apply the `adjoint` correctly, we get

  ```pycon
  >>> circuit_correct(params)
  [ 0.47415988+0.j          0.         0.73846026j  0.         0.25903472j
   -0.40342268+0.j        ]
  ```

* A precision argument has been added to the tape's ``to_openqasm`` function
  to control the precision of parameters.
  [(#2071)](https://github.com/PennyLaneAI/pennylane/pull/2071)

* Insert transform now supports adding operation after or before certain specific gates.
  [(#1980)](https://github.com/PennyLaneAI/pennylane/pull/1980)

* Interferometer is now a class with `shape` method.
  [(#1946)](https://github.com/PennyLaneAI/pennylane/pull/1946)

* The `CircuitGraph`, used to represent circuits via directed acyclic graphs, now
  uses RetworkX for its internal representation. This results in significant speedup
  for algorithms that rely on a directed acyclic graph representation.
  [(#1791)](https://github.com/PennyLaneAI/pennylane/pull/1791)

* The QAOA module now accepts both NetworkX and RetworkX graphs as function inputs.
  [(#1791)](https://github.com/PennyLaneAI/pennylane/pull/1791)

* The Barrier and Identity operations now support the `adjoint` method.
  [(#2062)](https://github.com/PennyLaneAI/pennylane/pull/2062)
  [(#2063)](https://github.com/PennyLaneAI/pennylane/pull/2063)

* `qml.BasisStatePreparation` now supports the `batch_params` decorator.
  [(#2091)](https://github.com/PennyLaneAI/pennylane/pull/2091)

* Added a new `multi_dispatch` decorator that helps ease the definition of new functions
  inside PennyLane. The decorator is used throughout the math module, demonstrating use cases.
  [(#2082)](https://github.com/PennyLaneAI/pennylane/pull/2084)

  [(#2096)](https://github.com/PennyLaneAI/pennylane/pull/2096)

  We can decorate a function, indicating the arguments that are
  tensors handled by the interface:

  ```pycon
  >>> @qml.math.multi_dispatch(argnum=[0, 1])
  ... def some_function(tensor1, tensor2, option, like):
  ...     # the interface string is stored in ``like``.
  ...     ...
  ```

  Previously, this was done using the private utility function `_multi_dispatch`.

  ```pycon
  >>> def some_function(tensor1, tensor2, option):
  ...     interface = qml.math._multi_dispatch([tensor1, tensor2])
  ...     ...
  ```

* The `IsingZZ` gate was added to the `diagonal_in_z_basis` attribute. For this
  an explicit `_eigvals` method was added.
  [(#2113)](https://github.com/PennyLaneAI/pennylane/pull/2113)

* The `IsingXX`, `IsingYY` and `IsingZZ` gates were added to
  the `composable_rotations` attribute.
  [(#2113)](https://github.com/PennyLaneAI/pennylane/pull/2113)

<h3>Breaking changes</h3>

* `qml.metric_tensor`, `qml.adjoint_metric_tensor` and `qml.transforms.classical_jacobian`
  now follow a different convention regarding their output shape when being used
  with the Autograd interface
  [(#2059)](https://github.com/PennyLaneAI/pennylane/pull/2059)

  See the previous entry for details. This breaking change immediately follows from
  the change in `qml.jacobian` whenever `hybrid=True` is used in the above methods.

* `qml.jacobian` now follows a different convention regarding its output shape.
  [(#2059)](https://github.com/PennyLaneAI/pennylane/pull/2059)

  Previously, `qml.jacobian` would attempt to stack the Jacobian for multiple
  QNode arguments, which succeeded whenever the arguments have the same shape.
  In this case, the stacked Jacobian would also be transposed, leading to the
  output shape `(*reverse_QNode_args_shape, *reverse_output_shape, num_QNode_args)`

  If no stacking and transposing occurs, the output shape instead is a `tuple`
  where each entry corresponds to one QNode argument and has the shape
  `(*output_shape, *QNode_arg_shape)`.

  This breaking change alters the behaviour in the first case and removes the attempt
  to stack and transpose, so that the output always has the shape of the second
  type.

  Note that the behaviour is unchanged --- that is, the Jacobian tuple is unpacked into
  a single Jacobian --- if `argnum=None` and there is only one QNode argument
  with respect to which the differentiation takes place, or if an integer
  is provided as `argnum`.

  A workaround that allowed `qml.jacobian` to differentiate multiple QNode arguments
  will no longer support higher-order derivatives. In such cases, combining multiple
  arguments into a single array is recommended.

* The behaviour of `RotosolveOptimizer` has been changed regarding
  its keyword arguments.
  [(#2081)](https://github.com/PennyLaneAI/pennylane/pull/2081)

  The keyword arguments `optimizer` and `optimizer_kwargs` for the
  `RotosolveOptimizer` have been renamed to `substep_optimizer`
  and `substep_kwargs`, respectively. Furthermore they have been
  moved from `step` and `step_and_cost` to the initialization `__init__`.

  The keyword argument `num_freqs` has been renamed to `nums_frequency`
  and is expected to take a different shape now:
  Previously, it was expected to be an `int` or a list of entries, with
  each entry in turn being either an `int` or a `list` of `int` entries.
  Now the expected structure is a nested dictionary, matching the
  formatting expected by
  [qml.fourier.reconstruct](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.fourier.reconstruct.html)
  This also matches the expected formatting of the new keyword arguments
  `spectra` and `shifts`.

  For more details, see the
  [RotosolveOptimizer documentation](https://pennylane.readthedocs.io/en/stable/code/api/pennylane.RotosolveOptimizer.html).

* QNode arguments will no longer be considered trainable by default when using
  the Autograd interface. In order to obtain derivatives with respect to a parameter,
  it should be instantiated via PennyLane's NumPy wrapper using the `requires_grad=True`
  attribute. The previous behaviour was deprecated in version v0.19.0 of PennyLane.
  [(#2116)](https://github.com/PennyLaneAI/pennylane/pull/2116)
  [(#2125)](https://github.com/PennyLaneAI/pennylane/pull/2125)
  [(#2139)](https://github.com/PennyLaneAI/pennylane/pull/2139)
  [(#2148)](https://github.com/PennyLaneAI/pennylane/pull/2148)

  ```python
  from pennylane import numpy as np

  @qml.qnode(qml.device("default.qubit", wires=2))
  def circuit(x):
    ...

  x = np.array([0.1, 0.2], requires_grad=True)
  qml.grad(circuit)(x)
  ```

  For the `qml.grad` and `qml.jacobian` functions, trainability can alternatively be
  indicated via the `argnum` keyword:

  ```python
  import numpy as np

  @qml.qnode(qml.device("default.qubit", wires=2))
  def circuit(hyperparam, param):
    ...

  x = np.array([0.1, 0.2])
  qml.grad(circuit, argnum=1)(0.5, x)
  ```

<h3>Bug fixes</h3>

* Fixes a bug where the `default.qubit.jax` device can't be used with `diff_method=None` and jitting.
  [(#2136)](https://github.com/PennyLaneAI/pennylane/pull/2136)

* Fixes a bug where the Torch interface was not properly unwrapping Torch tensors
  to NumPy arrays before executing gradient tapes on devices.
  [(#2117)](https://github.com/PennyLaneAI/pennylane/pull/2117)

* Fixes a bug for the TensorFlow interface where the dtype of input tensors was
  not cast.
  [(#2120)](https://github.com/PennyLaneAI/pennylane/pull/2120)

* Fixes a bug where batch transformed QNodes would fail to apply batch transforms
  provided by the underlying device.
  [(#2111)](https://github.com/PennyLaneAI/pennylane/pull/2111)

* An error is raised during QNode creation if backpropagation is requested on a device with
  finite-shots specified.
  [(#2114)](https://github.com/PennyLaneAI/pennylane/pull/2114)

* Pytest now ignores any `DeprecationWarning` raised within autograd's `numpy_wrapper` module.
  Other assorted minor test warnings are fixed.
  [(#2007)](https://github.com/PennyLaneAI/pennylane/pull/2007)

* Fixes a bug where the QNode was not correctly diagonalizing qubit-wise
  commuting observables.
  [(#2097)](https://github.com/PennyLaneAI/pennylane/pull/2097)

* Fixes a bug in `gradient_transform` where the hybrid differentiation
  of circuits with a single parametrized gate failed and QNode argument
  axes of size `1` where removed from the output gradient.
  [(#2080)](https://github.com/PennyLaneAI/pennylane/pull/2080)

* The available `diff_method` options for QNodes has been corrected in both the
  error messages and the documentation.
  [(#2078)](https://github.com/PennyLaneAI/pennylane/pull/2078)

* Fixes a bug in `DefaultQubit` where the second derivative of QNodes at
  positions corresponding to vanishing state vector amplitudes is wrong.
  [(#2057)](https://github.com/PennyLaneAI/pennylane/pull/2057)

* Fixes a bug where PennyLane didn't require v0.20.0 of PennyLane-Lightning,
  but raised an error with versions of Lightning earlier than v0.20.0 due to
  the new batch execution pipeline.
  [(#2033)](https://github.com/PennyLaneAI/pennylane/pull/2033)

* Fixes a bug in `classical_jacobian` when used with Torch, where the
  Jacobian of the preprocessing was also computed for non-trainable
  parameters.
  [(#2020)](https://github.com/PennyLaneAI/pennylane/pull/2020)

* Fixes a bug in queueing of the `two_qubit_decomposition` method that
  originally led to circuits with >3 two-qubit unitaries failing when passed
  through the `unitary_to_rot` optimization transform.
  [(#2015)](https://github.com/PennyLaneAI/pennylane/pull/2015)

* Fixes a bug which allows using `jax.jit` to be compatible with circuits
  which return `qml.probs` when the `default.qubit.jax` is provided with a custom shot
  vector.
  [(#2028)](https://github.com/PennyLaneAI/pennylane/pull/2028)

<h3>Documentation</h3>

* Improves the documentation of `RotosolveOptimizer` regarding the
  usage of the passed `substep_optimizer` and its keyword arguments.
  [(#2160)](https://github.com/PennyLaneAI/pennylane/pull/2160)

* Fixes an error in the signs of equations in the `DoubleExcitation` page.
  [(#2072)](https://github.com/PennyLaneAI/pennylane/pull/2072)

* Extended the interfaces description page to explicitly mention device
  compatibility.
  [(#2031)](https://github.com/PennyLaneAI/pennylane/pull/2031)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola, Ali Asadi, Utkarsh Azad, Esther Cruz, Christian Gogolin Christina Lee, Olivia Di Matteo, Diego Guala,
Anthony Hayes, Josh Izaac, Soran Jahangiri, Edward Jiang, Ankit Khandelwal, Korbinian Kottmann, Jay Soni, Antal Száva,
David Wierichs, Shaoming Zhang
