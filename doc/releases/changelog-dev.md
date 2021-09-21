:orphan:

# Release 0.19.0-dev (development release)

<h3>New features since last release</h3>

* Arbitrary two-qubit unitaries can now be decomposed into elementary gates. This
  functionality has been incorporated into the `qml.transforms.unitary_to_rot` transform, and is
  available separately as `qml.transforms.two_qubit_decomposition`.
  [(#1552)](https://github.com/PennyLaneAI/pennylane/pull/1552)

  As an example, consider the following randomly-generated matrix and circuit that uses it:

  ```python
  U = np.array([
      [-0.03053706-0.03662692j,  0.01313778+0.38162226j, 0.4101526 -0.81893687j, -0.03864617+0.10743148j],
      [-0.17171136-0.24851809j,  0.06046239+0.1929145j, -0.04813084-0.01748555j, -0.29544883-0.88202604j],
      [ 0.39634931-0.78959795j, -0.25521689-0.17045233j, -0.1391033 -0.09670952j, -0.25043606+0.18393466j],
      [ 0.29599198-0.19573188j,  0.55605806+0.64025769j, 0.06140516+0.35499559j,  0.02674726+0.1563311j ]
  ])

  dev = qml.device('default.qubit', wires=2)

  @qml.qnode(dev)
  @qml.transforms.unitary_to_rot
  def circuit(x, y):
      qml.RX(x, wires=0)
      qml.QubitUnitary(U, wires=[0, 1])
      qml.RY(y, wires=0)
      return qml.expval(qml.PauliZ(wires=0))
  ```

  If we run the circuit, we can see the new decomposition:

  ```pycon
  >>> circuit(0.3, 0.4)
  tensor(-0.70520073, requires_grad=True)
  >>> print(qml.draw(circuit)(0.3, 0.4))
  0: ──RX(0.3)─────────────────Rot(-3.5, 0.242, 0.86)──╭X──RZ(0.176)───╭C─────────────╭X──Rot(5.56, 0.321, -2.09)───RY(0.4)──┤ ⟨Z⟩
  1: ──Rot(-1.64, 2.69, 1.58)──────────────────────────╰C──RY(-0.883)──╰X──RY(-1.47)──╰C──Rot(-1.46, 0.337, 0.587)───────────┤
  ```

* The transform for the Jacobian of the classical preprocessing within a QNode,
  `qml.transforms.classical_jacobian`, now takes a keyword argument `argnum` to specify
  the QNode argument indices with respect to which the Jacobian is computed.
  [(#1645)](https://github.com/PennyLaneAI/pennylane/pull/1645)

  An example for the usage of ``argnum`` is

  ```python
  @qml.qnode(dev)
  def circuit(x, y, z):
      qml.RX(qml.math.sin(x), wires=0)
      qml.CNOT(wires=[0, 1])
      qml.RY(y ** 2, wires=1)
      qml.RZ(1 / z, wires=1)
      return qml.expval(qml.PauliZ(0))

  jac_fn = qml.transforms.classical_jacobian(circuit, argnum=[1, 2])
  ```

  The Jacobian can then be computed at specified parameters.

  ```pycon
  >>> x, y, z = np.array([0.1, -2.5, 0.71])
  >>> jac_fn(x, y, z)
  (array([-0., -5., -0.]), array([-0.        , -0.        , -1.98373339]))
  ```

  The returned arrays are the derivatives of the three parametrized gates in the circuit
  with respect to `y` and `z` respectively.

  There also are explicit tests for `classical_jacobian` now, which previously was tested
  implicitly via its use in the `metric_tensor` transform.

  For more usage details, please see the
  [classical Jacobian docstring](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.transforms.classical_jacobian.html).

* A new, experimental QNode has been added, that adds support for batch execution of circuits,
  custom quantum gradient support, and arbitrary order derivatives. This QNode is available via
  `qml.beta.QNode`, and `@qml.beta.qnode`.
  [(#1642)](https://github.com/PennyLaneAI/pennylane/pull/1642)
  [(#1646)](https://github.com/PennyLaneAI/pennylane/pull/1646)

  It differs from the standard QNode in several ways:

  - Custom gradient transforms can be specified as the differentiation method:

    ```python
    @qml.gradients.gradient_transform
    def my_gradient_transform(tape):
        ...
        return tapes, processing_fn

    @qml.beta.qnode(dev, diff_method=my_gradient_transform)
    def circuit():
    ```

  - Arbitrary :math:`n`-th order derivatives are supported on hardware using
    gradient transforms such as the parameter-shift rule. To specify that an :math:`n`-th
    order derivative of a QNode will be computed, the `max_diff` argument should be set.
    By default, this is set to 1 (first-order derivatives only).

  - Internally, if multiple circuits are generated for execution simultaneously, they
    will be packaged into a single job for execution on the device. This can lead to
    significant performance improvement when executing the QNode on remote
    quantum hardware.

  In an upcoming release, this QNode will replace the existing one. If you come across any bugs
  while using this QNode, please let us know via a [bug
  report](https://github.com/PennyLaneAI/pennylane/issues/new?assignees=&labels=bug+%3Abug%3A&template=bug_report.yml&title=%5BBUG%5D)
  on our GitHub bug tracker.

  Currently, this beta QNode does not support the following features:

  - Circuit decompositions
  - Non-mutability via the `mutable` keyword argument
  - Viewing specifications with `qml.specs`
  - The `reversible` QNode differentiation method
  - The ability to specify a `dtype` when using PyTorch and TensorFlow.

  It is also not tested with the `qml.qnn` module.

<h3>Improvements</h3>

* The tests for qubit operations are split into multiple files.
  [(#1661)](https://github.com/PennyLaneAI/pennylane/pull/1661)

* The `qml.metric_tensor` transform has been improved with regards to
  both function and performance.
  [(#1638)](https://github.com/PennyLaneAI/pennylane/pull/1638)

  - If the underlying device supports batch execution of circuits, the quantum circuits required to
    compute the metric tensor elements will be automatically submitted as a batched job. This can
    lead to significant performance improvements for devices with a non-trivial job submission
    overhead.

  - Previously, the transform would only return the metric tensor with respect to gate arguments,
    and ignore any classical processing inside the QNode, even very trivial classical processing
    such as parameter permutation. The metric tensor now takes into account classical processing,
    and returns the metric tensor with respect to QNode arguments, not simply gate arguments:

    ```pycon
    >>> @qml.qnode(dev)
    ... def circuit(x):
    ...     qml.Hadamard(wires=1)
    ...     qml.RX(x[0], wires=0)
    ...     qml.CNOT(wires=[0, 1])
    ...     qml.RY(x[1] ** 2, wires=1)
    ...     qml.RY(x[1], wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> x = np.array([0.1, 0.2], requires_grad=True)
    >>> qml.metric_tensor(circuit)(x)
    array([[0.25      , 0.        ],
           [0.        , 0.28750832]])
    ```

    To revert to the previous behaviour of returning the metric tensor with respect to gate
    arguments, `qml.metric_tensor(qnode, hybrid=False)` can be passed.

    ```pycon
    >>> qml.metric_tensor(circuit, hybrid=False)(x)
    array([[0.25      , 0.        , 0.        ],
           [0.        , 0.25      , 0.        ],
           [0.        , 0.        , 0.24750832]])
    ```

* ``qml.circuit_drawer.CircuitDrawer`` can accept a string for the ``charset`` keyword, instead of a ``CharSet`` object.
  [(#1640)](https://github.com/PennyLaneAI/pennylane/pull/1640)

<h3>Breaking changes</h3>

- The `QNode.metric_tensor` method has been deprecated, and will be removed in an upcoming release.
  Please use the `qml.metric_tensor` transform instead.
  [(#1638)](https://github.com/PennyLaneAI/pennylane/pull/1638)

- The utility function `qml.math.requires_grad` now returns `True` when using Autograd
  if and only if the `requires_grad=True` attribute is set on the NumPy array. Previously,
  this function would return `True` for *all* NumPy arrays and Python floats, unless
  `requires_grad=False` was explicitly set.
  [(#1638)](https://github.com/PennyLaneAI/pennylane/pull/1638)

<h3>Bug fixes</h3>

* The device suite tests can now execute successfully if no shots configuration variable is given.
  [(#1641)](https://github.com/PennyLaneAI/pennylane/pull/1641)

* Fixes a bug where the `qml.gradients.param_shift` transform would raise an error while attempting
  to compute the variance of a QNode with ragged output.
  [(#1646)](https://github.com/PennyLaneAI/pennylane/pull/1646)

<h3>Documentation</h3>

* Adds a link to https://pennylane.ai/qml/demonstrations.html in the navbar.
  [(#1624)](https://github.com/PennyLaneAI/pennylane/pull/1624)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Olivia Di Matteo, Andrew Gardhouse, Josh Izaac, Christina Lee, David Wierichs.
