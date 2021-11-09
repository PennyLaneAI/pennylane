:orphan:

# Release 0.20.0-dev (development release)

<h3>New features since last release</h3>

* The `metric_tensor` transform can now be used to compute the full
  tensor, beyond the block diagonal approximation. 
  [(#1725)](https://github.com/PennyLaneAI/pennylane/pull/1725)

  This is performed using Hadamard tests, and requires an additional wire 
  on the device to execute the circuits produced by the transform, 
  as compared to the number of wires required by the original circuit.
  The transform defaults to computing the full tensor, which can
  be controlled by the `approx` keyword argument.
  See the 
  [qml.metric_tensor docstring](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.transforms.metric_tensor.html).
  for more information and usage details.

  As an example, consider the QNode

  ```python
  dev = qml.device("default.qubit", wires=3)

  @qml.qnode(dev)
  def circuit(weights):
      qml.RX(weights[0], wires=0)
      qml.RY(weights[1], wires=0)
      qml.CNOT(wires=[0, 1])
      qml.RZ(weights[2], wires=1)
      return qml.expval(qml.PauliZ(0)@qml.PauliZ(1))

  weights = np.array([0.2, 1.2, -0.9], requires_grad=True)
  ```

  Then we can compute the (block) diagonal metric tensor as before, now using the
  ``approx="block-diag"`` keyword:

  ```pycon
  >>> qml.metric_tensor(circuit, approx="block-diag")(weights)
  [[0.25       0.         0.        ]
   [0.         0.24013262 0.        ]
   [0.         0.         0.21846983]]
  ```

  Instead, we now can also compute the full metric tensor:

  ```pycon
  >>> qml.metric_tensor(circuit)(weights)
  [[ 0.25        0.         -0.23300977]
   [ 0.          0.24013262  0.01763859]
   [-0.23300977  0.01763859  0.21846983]]
  ```

* A thermal relaxation channel is added to the Noisy channels. The channel description can be 
  found on the supplementary information of [Quantum classifier with tailored quantum kernels](https://arxiv.org/abs/1909.02611).
  [(#1766)](https://github.com/PennyLaneAI/pennylane/pull/1766)

<h3>Improvements</h3>

* AngleEmbedding now supports `batch_params` decorator. [(#1812)](https://github.com/PennyLaneAI/pennylane/pull/1812)

<h3>Breaking changes</h3>

* The default behaviour of the `qml.metric_tensor` transform has been modified:
  By default, the full metric tensor is computed, leading to higher cost than the previous
  default of computing the block diagonal only. At the same time, the Hadamard tests for
  the full metric tensor require an additional wire on the device, so that 

  ```pycon
  >>> qml.metric_tensor(some_qnode)(weights)
  ```

  will raise an error if the used device does not have an additional wire.
  [(#1725)](https://github.com/PennyLaneAI/pennylane/pull/1725)

<h3>Deprecations</h3>

<h3>Bug fixes</h3>

* `qml.CSWAP` and `qml.CRot` now define `control_wires`, and `qml.SWAP` 
  returns the default empty wires object.
  [(#1830)](https://github.com/PennyLaneAI/pennylane/pull/1830)

* The `requires_grad` attribute of `qml.numpy.tensor` objects is now
  preserved when pickling/unpickling the object.
  [(#1856)](https://github.com/PennyLaneAI/pennylane/pull/1856)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order): 

Jalani Kanem, Christina Lee, Guillermo Alonso-Linaje, Alejandro Montanez, David Wierichs

