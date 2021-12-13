:orphan:

# Release 0.21.0-dev (development release)

<h3>New features since last release</h3>

* Added the adjoint method for the metric tensor.
  [(#1992)](https://github.com/PennyLaneAI/pennylane/pull/1992)

  This method, detailed in [Jones 2020](https://arxiv.org/abs/2011.02991),
  computes the metric tensor using four copies of the state vector and
  a number of operations that scales quadratically in the number of trainable
  parameters. As it makes use of state cloning, it is inherently classical
  and to be used on state vector simulators only.
  It is particular useful for larger circuits for which backpropagation requires
  inconvenient or even unfeasible amounts of storage, but is slower.
  Furthermore, the adjoint method is only available for analytic computation, not
  for measurements simulation with `shots!=None`.

  ```python
  dev = qml.device("default.qubit", wires=3)
  
  @qml.qnode(dev)
  def circuit(x, y):
      qml.Rot(*x[0], wires=0)
      qml.Rot(*x[1], wires=1)
      qml.Rot(*x[2], wires=2)
      qml.CNOT(wires=[0, 1])
      qml.CNOT(wires=[1, 2])
      qml.CNOT(wires=[2, 0])
      qml.RY(y[0], wires=0)
      qml.RY(y[1], wires=1)
      qml.RY(y[0], wires=2)

  x = np.array([[0.2, 0.4, -0.1], [-2.1, 0.5, -0.2], [0.1, 0.7, -0.6]], requires_grad=False)
  y = np.array([1.3, 0.2], requires_grad=True)
  ```

  ```pycon
  >>> qml.adjoint_metric_tensor(circuit)(x, y)
  tensor([[ 0.25495723, -0.07086695],
          [-0.07086695,  0.24945606]], requires_grad=True)
  ```

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Bug fixes</h3>

* Fixes a bug in queueing of the `two_qubit_decomposition` method that
  originally led to circuits with >3 two-qubit unitaries failing when passed
  through the `unitary_to_rot` optimization transform.
  [(#2015)](https://github.com/PennyLaneAI/pennylane/pull/2015)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Olivia Di Matteo, David Wierichs