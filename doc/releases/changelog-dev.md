:orphan:

# Release 0.21.0-dev (development release)

<h3>New features since last release</h3>

* A tensor network template has been added. Quantum circuits with the shape of a matrix product state tensor network can now be easily implemented. Motivation and theory can be found in [arXiv:1803.11537](https://arxiv.org/abs/1803.11537). [(#1871)](https://github.com/PennyLaneAI/pennylane/pull/1871)

  An example circuit that uses the `MPS` template is:
  ```python
  def block(weights, wires):
    qml.CNOT(wires=[wires[0],wires[1]])
    qml.RY(weights[0],wires=wires[0])
    qml.RY(weights[1],wires=wires[1])

  weights = np.array([[1,2],[3,4],[5,6]])
  dev = qml.device('default.qubit',wires=4)
  @qml.qnode(dev)
  def circuit(weights):
    qml.MPS(wires=range(4),loc=2,block=block,n_params_block=2,weights=weights)
    return qml.expval(qml.PauliZ(wires=3))
  ```
  Running this circuit gives:
  ```pycon
  >>> circuit(weights)
  tensor(0.26117758, requires_grad=True)
  >>> print(qml.draw(circuit)(weights))
  0: ──╭C──RY(1)────────────────────────┤     
  1: ──╰X──RY(2)──╭C──RY(3)─────────────┤     
  2: ─────────────╰X──RY(4)──╭C──RY(5)──┤     
  3: ────────────────────────╰X──RY(6)──┤ ⟨Z⟩
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

Olivia Di Matteo
