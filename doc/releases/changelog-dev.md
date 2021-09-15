:orphan:

# Release 0.19.0-dev (development release)

<h3>New features since last release</h3>

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

<h3>Improvements</h3>

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

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Josh Izaac, Christina Lee, David Wierichs.