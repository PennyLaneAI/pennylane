:orphan:

# Release 0.28.0-dev (development release)

<h3>New features since last release</h3>

<h4>Entanglement Entropy</h4>

* Support for entanglement entropy computation is added. The `qml.math.vn_entanglement_entropy` function computes the von Neumann entanglement entropy from a state vector or a density matrix:

  ```pycon
  >>> x = np.array([0, -1, 1, 0]) / np.sqrt(2)
  >>> qml.math.vn_entanglement_entropy(x, indices0=[0], indices1=[1])
  0.6931471805599453
  >>> y = np.array([[1, 1, -1, -1], [1, 1, -1, -1], [-1, -1, 1, 1], [-1, -1, 1, 1]]) * 0.25
  >>> qml.math.vn_entanglement_entropy(y, indices0=[0], indices1=[1])
  0
  ```
  The `qml.vn_entanglement_entropy` measurement process can be returned from a QNode:
  ```python3
  dev = qml.device("default.qubit", wires=2)
  @qml.qnode(dev)
  def circuit(params):
    qml.RY(params, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.vn_entanglement_entropy(wires0=[0], wires1=[1])
  ```
  ```pycon
  >>> circuit(np.pi / 2)
  tensor(0.69314718, requires_grad=True)
  ```
  The `qml.qinfo.vn_entanglement_entropy` can be used to transform a QNode returning
  a state to a function that returns the mutual information:
  ```python3
  dev = qml.device("default.qubit", wires=2)
  @qml.qnode(dev)
  def circuit(x):
    qml.RY(x, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()
  ```

  ```pycon
  >>> entanglement_entropy_circuit = qinfo.vn_entanglement_entropy(circuit, wires0=[0], wires1=[1])
  >>> entanglement_entropy_circuit(np.pi / 2)
  0.69314718
  >>> x = np.array(np.pi / 4, requires_grad=True)
  >>> qml.grad(entanglement_entropy_circuit)(x)
  0.62322524
  ```

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

<h3>Contributors</h3>

Astral Cai

This release contains contributions from (in alphabetical order):
