:orphan:

# Release 0.21.0-dev (development release)

<h3>New features since last release</h3>

* A tensor network template has been added. Quantum circuits with the shape of a matrix product state tensor network can now be easily implemented. Motivation and theory can be found in [arXiv:1803.11537](https://arxiv.org/abs/1803.11537). [(#1871)](https://github.com/PennyLaneAI/pennylane/pull/1871)

  An example circuit that uses the `MPS` template is:
  ```python
  import pennylane as qml
  import numpy as np

  def block(block_weights, block_wires):
      qml.CNOT(wires=[block_wires[0],block_wires[1]])
      qml.Rot(block_weights[0],block_weights[1],block_weights[2],wires=block_wires[0])
      qml.Rot(block_weights[3],block_weights[4],block_weights[5],wires=block_wires[1])

  n_wires = 4
  loc = 2
  n_params_block = 6
  template_weights = [[1,2,3,4,5,6],[3,4,5,6,7,8],[4,5,6,7,8,9]]

  dev= qml.device('default.qubit',wires=n_wires)
  @qml.qnode(dev)
  def circuit(weights):
      qml.MPS(wires = range(n_wires),loc=loc,block=block, n_params_block=n_params_block, weights=weights)
      return qml.expval(qml.PauliZ(wires=n_wires-1))

  
  ```
  The resulting circuit is:
  ```pycon
  >>> print(qml.draw(circuit,expansion_strategy='device')(template_weights))
  0: ──╭C──Rot(1, 2, 3)──────────────────────────────────────┤
  1: ──╰X──Rot(4, 5, 6)──╭C──Rot(3, 4, 5)────────────────────┤
  2: ────────────────────╰X──Rot(6, 7, 8)──╭C──Rot(4, 5, 6)──┤
  3: ──────────────────────────────────────╰X──Rot(7, 8, 9)──┤ ⟨Z⟩
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

Olivia Di Matteo, Juan Miguel Arrazola, Esther Cruz,  Diego Guala, Shaoming Zhang
