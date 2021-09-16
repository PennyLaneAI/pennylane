:orphan:

# Release 0.18.0 (Current release)

<h3>New features since last release</h3>

* Custom gradient transforms can now be created using the new
  `@qml.gradients.gradient_transform` decorator on a batch-tape transform.
  [(#1589)](https://github.com/PennyLaneAI/pennylane/pull/1589)

  Quantum gradient transforms are a specific case of `qml.batch_transform`.
  To create a quantum gradient transform, simply write a function that accepts a tape,
  and returns a batch of tapes to be independently executed on a quantum device, alongside
  a post-processing function that processes the tape results into the gradient.

  Furthermore, a smart default expansion function is provided, which automatically expands tape
  operations which are not differentiable prior to applying the quantum gradient.
  All gradient transforms in `qml.gradients` are now decorated with this decorator.

  Supported gradient transforms must be of the following form:

  ```python
  @qml.gradients.gradient_transform
  def my_custom_gradient(tape, argnum=None, **kwargs):
      ...
      return gradient_tapes, processing_fn
  ```

  Various built-in quantum gradient transforms are provided within the
  `qml.gradients` module, including `qml.gradients.param_shift`.
  Once defined, quantum gradient transforms can be applied directly
  to QNodes:

  ```pycon
  >>> @qml.qnode(dev)
  ... def circuit(x):
  ...     qml.RX(x, wires=0)
  ...     qml.CNOT(wires=[0, 1])
  ...     return qml.expval(qml.PauliZ(0))
  >>> circuit(0.3)
  tensor(0.95533649, requires_grad=True)
  >>> qml.gradients.param_shift(circuit)(0.5)
  array([[-0.47942554]])
  ```

  Quantum gradient transforms are fully differentiable, allowing higher order derivatives to be
  accessed:

  ```pycon
  >>> qml.grad(qml.gradients.param_shift(circuit))(0.5)
  tensor(-0.87758256, requires_grad=True)
  ```

* A new pytorch device, `qml.device('default.qubit.torch', wires=wires)`, supports
  backpropogation with the torch interface.
  [(#1225)](https://github.com/PennyLaneAI/pennylane/pull/1360)
  [(#1598)](https://github.com/PennyLaneAI/pennylane/pull/1598)


* The ability to define *batch* transforms has been added via the new
  `@qml.batch_transform` decorator.
  [(#1493)](https://github.com/PennyLaneAI/pennylane/pull/1493)

  A batch transform is a transform that takes a single tape or QNode as input,
  and executes multiple tapes or QNodes independently. The results may then be post-processed
  before being returned.

  For example, consider the following batch transform:

  ```python
  @qml.batch_transform
  def my_transform(tape, a, b):
      """Generates two tapes, one with all RX replaced with RY,
      and the other with all RX replaced with RZ."""
      tape1 = qml.tape.JacobianTape()
      tape2 = qml.tape.JacobianTape()

      # loop through all operations on the input tape
      for op in tape.operations + tape.measurements:
          if op.name == "RX":
              with tape1:
                  qml.RY(a * qml.math.abs(op.parameters[0]), wires=op.wires)
              with tape2:
                  qml.RZ(b * qml.math.abs(op.parameters[0]), wires=op.wires)
          else:
              for t in [tape1, tape2]:
                  with t:
                      qml.apply(op)

      def processing_fn(results):
          return qml.math.sum(qml.math.stack(results))

      return [tape1, tape2], processing_fn
  ```

  We can transform a QNode directly using decorator syntax:

  ```pycon
  >>> @my_transform(0.65, 2.5)
  ... @qml.qnode(dev)
  ... def circuit(x):
  ...     qml.Hadamard(wires=0)
  ...     qml.RX(x, wires=0)
  ...     return qml.expval(qml.PauliX(0))
  >>> print(circuit(-0.5))
  1.2629730888100839
  ```

  Batch tape transforms are fully differentiable:

  ```pycon
  >>> gradient = qml.grad(circuit)(-0.5)
  >>> print(gradient)
  2.5800122591960153
  ```

  Batch transforms can also be applied to existing QNodes,

  ```pycon
  >>> new_qnode = my_transform(existing_qnode, *transform_weights)
  >>> new_qnode(weights)
  ```

  or to tapes (in which case, the processed tapes and classical post-processing
  functions are returned):

  ```pycon
  >>> tapes, fn = my_transform(tape, 0.65, 2.5)
  >>> from pennylane.interfaces.batch import execute
  >>> dev = qml.device("default.qubit", wires=1)
  >>> res = execute(tapes, dev, interface="autograd", gradient_fn=qml.gradients.param_shift)
  1.2629730888100839
  ```

* Added a new `SISWAP` operation and a `SQISW` alias with support to the `default_qubit` device.
  [#1563](https://github.com/PennyLaneAI/pennylane/pull/1563)

* The `RotosolveOptimizer` now can tackle general parametrized circuits, and is no longer
  restricted to single-qubit Pauli rotations.
  [(#1489)](https://github.com/PennyLaneAI/pennylane/pull/1489)

  This includes:

  - layers of gates controlled by the same parameter,
  - controlled variants of parametrized gates, and
  - Hamiltonian time evolution.

  Note that the eigenvalue spectrum of the gate generator needs to be known to
  use `RotosolveOptimizer` for a general gate, and it
  is required to produce equidistant frequencies.
  For details see [Vidal and Theis, 2018](https://arxiv.org/abs/1812.06323)
  and [Wierichs, Izaac, Wang, Lin 2021](https://arxiv.org/abs/2107.12390).

  Consider a circuit with a mixture of Pauli rotation gates, controlled Pauli rotations, and
  single-parameter layers of Pauli rotations:
  ```python
  dev = qml.device('default.qubit', wires=3, shots=None)

  @qml.qnode(dev)
  def cost_function(rot_param, layer_par, crot_param):
      for i, par in enumerate(rot_param):
          qml.RX(par, wires=i)
      for w in dev.wires:
          qml.RX(layer_par, wires=w)
      for i, par in enumerate(crot_param):
          qml.CRY(par, wires=[i, (i+1) % 3])

      return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))
  ```
  This cost function has one frequency for each of the first `RX` rotation angles,
  three frequencies for the layer of `RX` gates that depend on `layer_par`, and two
  frequencies for each of the `CRY` gate parameters. Rotosolve can then be used to minimize
  the `cost_function`:

  ```python
  # Initial parameters
  init_param = [
      np.array([0.3, 0.2, 0.67], requires_grad=True),
      np.array(1.1, requires_grad=True),
      np.array([-0.2, 0.1, -2.5], requires_grad=True),
  ]
  # Numbers of frequencies per parameter
  num_freqs = [[1, 1, 1], 3, [2, 2, 2]]

  opt = qml.RotosolveOptimizer()
  param = init_param.copy()
  ```

  In addition, the optimization technique for the Rotosolve substeps can be chosen via the
  `optimizer` and `optimizer_kwargs` keyword arguments and the minimized cost of the
  intermediate univariate reconstructions can be read out via `full_output`, including the
  cost _after_ the full Rotosolve step:

  ```python
  for step in range(3):
      param, cost, sub_cost = opt.step_and_cost(
          cost_function,
          *param,
          num_freqs=num_freqs,
          full_output=True,
          optimizer="brute",
      )
      print(f"Cost before step: {cost}")
      print(f"Minimization substeps: {np.round(sub_cost, 6)}")
  ```
  ``` pycon
  Cost before step: 0.042008210392535605
  Minimization substeps: [-0.230905 -0.863336 -0.980072 -0.980072 -1.       -1.       -1.      ]
  Cost before step: -0.999999999068121
  Minimization substeps: [-1. -1. -1. -1. -1. -1. -1.]
  Cost before step: -1.0
  Minimization substeps: [-1. -1. -1. -1. -1. -1. -1.]
  ```

  For usage details please consider the docstring.

* The `frobenius_inner_product` function has been moved to the `qml.math`
  module, and is now differentiable using all autodiff frameworks.
  [(#1388)](https://github.com/PennyLaneAI/pennylane/pull/1388)

* Vector-Jacobian product transforms have been added to the `qml.gradients` package.
  [(#1494)](https://github.com/PennyLaneAI/pennylane/pull/1494)

  The new transforms include:

  - `qml.gradients.vjp`
  - `qml.gradients.batch_vjp`

* The Hamiltonian can now store grouping information, which can be accessed by a device to
  speed up computations of the expectation value of a Hamiltonian.
  [(#1515)](https://github.com/PennyLaneAI/pennylane/pull/1515)

  ```python
  obs = [qml.PauliX(0), qml.PauliX(1), qml.PauliZ(0)]
  coeffs = np.array([1., 2., 3.])
  H = qml.Hamiltonian(coeffs, obs, grouping_type='qwc')
  ```

  Initialization with a ``grouping_type`` other than ``None`` stores the indices
  required to make groups of commuting observables and their coefficients.

  ``` pycon
  >>> H.grouping_indices
  [[0, 1], [2]]
  ```

* Hamiltonians are now trainable with respect to their coefficients.
  [(#1483)](https://github.com/PennyLaneAI/pennylane/pull/1483)

  ``` python
  from pennylane import numpy as np

  dev = qml.device("default.qubit", wires=2)
  @qml.qnode(dev)
  def circuit(coeffs, param):
      qml.RX(param, wires=0)
      qml.RY(param, wires=0)
      return qml.expval(
          qml.Hamiltonian(coeffs, [qml.PauliX(0), qml.PauliZ(0)], simplify=True)
      )

  coeffs = np.array([-0.05, 0.17])
  param = np.array(1.7)
  grad_fn = qml.grad(circuit)
  ```
  ``` pycon
  >>> grad_fn(coeffs, param)
  (array([-0.12777055,  0.0166009 ]), array(0.0917819))
  ```

* Support for differentiable execution of batches of circuits has been
  added, via the beta `pennylane.interfaces.batch` module.
  [(#1501)](https://github.com/PennyLaneAI/pennylane/pull/1501)
  [(#1508)](https://github.com/PennyLaneAI/pennylane/pull/1508)
  [(#1542)](https://github.com/PennyLaneAI/pennylane/pull/1542)
  [(#1549)](https://github.com/PennyLaneAI/pennylane/pull/1549)
  [(#1608)](https://github.com/PennyLaneAI/pennylane/pull/1608)
  [(#1618)](https://github.com/PennyLaneAI/pennylane/pull/1618)
  [(#1637)](https://github.com/PennyLaneAI/pennylane/pull/1637)

  For example:

  ```python
  from pennylane.interfaces.batch import execute

  def cost_fn(x):
      with qml.tape.JacobianTape() as tape1:
          qml.RX(x[0], wires=[0])
          qml.RY(x[1], wires=[1])
          qml.CNOT(wires=[0, 1])
          qml.var(qml.PauliZ(0) @ qml.PauliX(1))

      with qml.tape.JacobianTape() as tape2:
          qml.RX(x[0], wires=0)
          qml.RY(x[0], wires=1)
          qml.CNOT(wires=[0, 1])
          qml.probs(wires=1)

      result = execute(
          [tape1, tape2], dev,
          gradient_fn=qml.gradients.param_shift,
          interface="autograd"
      )
      return result[0] + result[1][0, 0]

  res = qml.grad(cost_fn)(params)
  ```

<h3>Improvements</h3>

* The slowest tests, more than 1.5 seconds, now have the pytest mark `slow`, and can be
  selected or deselected during local execution of tests.
  [(#1633)](https://github.com/PennyLaneAI/pennylane/pull/1633)

* Hamiltonians are now natively supported on the `default.qubit` device if `shots=None`.
  This makes VQE workflows a lot faster in some cases.
  [(#1551)](https://github.com/PennyLaneAI/pennylane/pull/1551)
  [(#1596)](https://github.com/PennyLaneAI/pennylane/pull/1596)

* A gradient recipe for Hamiltonian coefficients has been added. This makes it possible
  to compute parameter-shift gradients of these coefficients on devices that natively
  support Hamiltonians.
  [(#1551)](https://github.com/PennyLaneAI/pennylane/pull/1551)

* The device test suite has been expanded to cover more qubit operations and observables.
  [(#1510)](https://github.com/PennyLaneAI/pennylane/pull/1510)

* The `MultiControlledX` class now inherits from `Operation` instead of `ControlledQubitUnitary` which makes the `MultiControlledX` gate a non-parameterized gate.
  [(#1557)](https://github.com/PennyLaneAI/pennylane/pull/1557)

* The `utils.sparse_hamiltonian` function can now deal with non-integer
  wire labels, and it throws an error for the edge case of observables that are
  created from multi-qubit operations.
  [(#1550)](https://github.com/PennyLaneAI/pennylane/pull/1550)

* Added the matrix attribute to `qml.templates.subroutines.GroverOperator`
  [(#1553)](https://github.com/PennyLaneAI/pennylane/pull/1553)

* The `tape.to_openqasm()` method now has a `measure_all` argument that specifies whether the
  serialized OpenQASM script includes computational basis measurements on all of the qubits or
  just those specified by the tape.
  [(#1559)](https://github.com/PennyLaneAI/pennylane/pull/1559)

* An error is raised when no arguments are passed to a `qml.operation.Observable` to inform the user about specifying wires.
  [(#1547)](https://github.com/PennyLaneAI/pennylane/pull/1547)

* The Hamiltonian class was moved to the `ops/qubit` folder from the `vqe` module, since it is now an observable.
  [(#1534)](https://github.com/PennyLaneAI/pennylane/pull/1534)

* The `group_observables` transform is now differentiable.
  [(#1483)](https://github.com/PennyLaneAI/pennylane/pull/1483)

  For example:

  ``` python
  import jax
  from jax import numpy as jnp

  coeffs = jnp.array([1., 2., 3.])
  obs = [PauliX(wires=0), PauliX(wires=1), PauliZ(wires=1)]

  def group(coeffs, select=None):
    _, grouped_coeffs = qml.grouping.group_observables(obs, coeffs)
    # in this example, grouped_coeffs is a list of two jax tensors
    # [DeviceArray([1., 2.], dtype=float32), DeviceArray([3.], dtype=float32)]
    return grouped_coeffs[select]

  jac_fn = jax.jacobian(group)
  ```
  ```pycon
  >>> jac_fn(coeffs, select=0)
  [[1. 0. 0.]
  [0. 1. 0.]]

  >>> jac_fn(coeffs, select=1)
  [[0., 0., 1.]]
  ```

* The tape does not verify any more that all Observables have owners in the annotated queue.
  [(#1505)](https://github.com/PennyLaneAI/pennylane/pull/1505)

  This allows manipulation of Observables inside a tape context. An example is
  `expval(Tensor(qml.PauliX(0), qml.Identity(1)).prune())` which makes the expval
  an owner of the pruned tensor and its constituent observables, but leaves the
  original tensor in the queue without an owner.

* Create a separate requirements file for the CI issue , to have a separate requirements.txt (pinned)
and requirements-ci.txt (unpinned). This latter would be used by the CI.
  [(#1535)](https://github.com/PennyLaneAI/pennylane/pull/1535)

* The QFT operation is moved to template
  [(#1548)](https://github.com/PennyLaneAI/pennylane/pull/1548)

* The `qml.ResetError` is now supported for `default.mixed` device.
  [(#1541)](https://github.com/PennyLaneAI/pennylane/pull/1541)

* `QNode.diff_method` will now reflect which method was selected from `diff_method="best"`.
  [(#1568)](https://github.com/PennyLaneAI/pennylane/pull/1568)

* QNodes now support `diff_method=None`. This works the same as `interface=None`. Such QNodes accept
  floats, ints, lists and numpy arrays and return numpy output but can not be differentiated.
  [(#1585)](https://github.com/PennyLaneAI/pennylane/pull/1585)

* QNodes now include validation to warn users if a supplied keyword argument is not one of the
  recognized arguments. [(#1496)](https://github.com/PennyLaneAI/pennylane/pull/1591)

<h3>Breaking changes</h3>

* Specifying `shots=None` with `qml.sample` was previously deprecated.
  From this release onwards, setting `shots=None` when sampling will
  raise an error also for `default.qubit.jax`.
  [(#1629)](https://github.com/PennyLaneAI/pennylane/pull/1629)

* An error is raised during QNode creation when a user requests backpropagation on
  a device with finite-shots.
  [(#1588)](https://github.com/PennyLaneAI/pennylane/pull/1588)

* The class `qml.Interferometer` is deprecated and will be renamed `qml.InterferometerUnitary`
  after one release cycle.
  [(#1546)](https://github.com/PennyLaneAI/pennylane/pull/1546)

*  All optimizers except for Rotosolve and Rotoselect now have a public attribute `stepsize`.
  Temporary backward compatibility has been added to support the use of `_stepsize` for one
  release cycle. `update_stepsize` method is deprecated.
  [(#1625)](https://github.com/PennyLaneAI/pennylane/pull/1625)


<h3>Bug fixes</h3>

* `MottonenStatepreparation` can now be run with a single wire label not in a list.
  [(#1620)](https://github.com/PennyLaneAI/pennylane/pull/1620)

* Fixed the circuit representation of CY gates to align with CNOT and CZ gates when calling the circuit drawer.
  [(#1504)](https://github.com/PennyLaneAI/pennylane/issues/1504)

* Dask and CVXPY dependent tests are skipped if those packages are not installed.
[(#1617)](https://github.com/PennyLaneAI/pennylane/pull/1617)

* The `qml.layer` template now works with tensorflow variables.
[(#1615)](https://github.com/PennyLaneAI/pennylane/pull/1615)

* Remove `QFT` from possible operations in `default.qubit` and `default.mixed`.
  [(#1600)](https://github.com/PennyLaneAI/pennylane/pull/1600)

* Fix bug when computing expectations of Hamiltonians using TensorFlow.
  [(#1586)](https://github.com/PennyLaneAI/pennylane/pull/1586)

* Fix bug when computing the specs of a circuit with a Hamiltonian observable.
  [(#1533)](https://github.com/PennyLaneAI/pennylane/pull/1533)

<h3>Documentation</h3>

* The `qml.Identity` operation is placed under the sections Qubit observables and CV observables.
  [(#1576)](https://github.com/PennyLaneAI/pennylane/pull/1576)

* Updated the documentation of `qml.grouping`, `qml.kernels` and `qml.qaoa` modules to present
  the list of functions first followed by the technical details of the module.
  [(#1581)](https://github.com/PennyLaneAI/pennylane/pull/1581)

* Recategorized Qubit operations into new and existing categories so that code for each
  operation is easier to locate.
  [(#1566)](https://github.com/PennyLaneAI/pennylane/pull/1583)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Vishnu Ajith, Akash Narayanan B, Thomas Bromley, Olivia Di Matteo, Sahaj Dhamija, Tanya Garg, Josh Izaac,
Prateek Jain, Ankit Khandelwal, Christina Lee, Ian McLean, Johannes Jakob Meyer, Romain Moyard, Esteban Payares,
Pratul Saini, Maria Schuld, Arshpreet Singh, Ingrid Strandberg, Slimane Thabet, Antal Száva, David Wierichs,
Vincent Wong.
