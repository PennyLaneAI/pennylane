:orphan:

# Release 0.21.0-dev (development release)

<h3>New features since last release</h3>

* A tensor network template has been added. Quantum circuits with the shape of a matrix product state tensor network can now be easily implemented. Motivation and theory can be found in [arXiv:1803.11537](https://arxiv.org/abs/1803.11537). [(#1871)](https://github.com/PennyLaneAI/pennylane/pull/1871)

  An example circuit that uses the `MPS` template is:
  ```python
  import pennylane as qml
  import numpy as np

  def block(weights, wires):
      qml.CNOT(wires=[wires[0],wires[1]])
      qml.RY(weights[0], wires=wires[0])
      qml.RY(weights[1], wires=wires[1])

  n_wires = 4
  n_block_wires = 2
  n_params_block = 2
  template_weights = [[0.1,-0.3],[0.4,0.2],[-0.15,0.5]]

  dev= qml.device('default.qubit',wires=range(n_wires))
  @qml.qnode(dev)
  def circuit(weights):
      qml.MPS(range(n_wires),n_block_wires,block, n_params_block, weights)
      return qml.expval(qml.PauliZ(wires=n_wires-1))
  ```

  The resulting circuit is:
  ```pycon
  >>> print(qml.draw(circuit,expansion_strategy='device')(template_weights))
  0: ──╭C──RY(0.1)───────────────────────────────┤
  1: ──╰X──RY(-0.3)──╭C──RY(0.4)─────────────────┤
  2: ────────────────╰X──RY(0.2)──╭C──RY(-0.15)──┤
  3: ─────────────────────────────╰X──RY(0.5)────┤ ⟨Z⟩
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

Olivia Di Matteo, Juan Miguel Arrazola, Esther Cruz, Diego Guala, Shaoming Zhang
