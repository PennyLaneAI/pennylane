:orphan:

# Release 0.20.0-dev (development release)

<h3>New features since last release</h3>
* A thermal relaxation channel is added to the Noisy channels. The channel description can be 
  found on the supplementary information of [Quantum classifier with tailored quantum kernels](https://arxiv.org/abs/1909.02611).
  [(#1766)](https://github.com/PennyLaneAI/pennylane/pull/1766)
  
* Added the identity observable to be an operator. Now we can explicitly call the identity 
  operation on our quantum circuits for both qubit and CV devices.
  [(#1829)](https://github.com/PennyLaneAI/pennylane/pull/1829) 

<h3>Improvements</h3>

* AngleEmbedding now supports `batch_params` decorator. [(#1812)](https://github.com/PennyLaneAI/pennylane/pull/1812)

<h3>Breaking changes</h3>

* The `decomposition` method of operations has changed from a static
  class method to an operation-dependent method.
  [(#1873)](https://github.com/PennyLaneAI/pennylane/pull/1873)

  Instead of the original syntax

  ```python
  >>> qml.CRX.decomposition(0.3, wires=[0, 1])
  ```

  the decomposition must be called on an instantiated version of the operation:

  ```python
  >>> qml.CRX(0.3, wires=[0, 1]).decomposition()
  ```

  This has consequences when decompositions are called from within a
  QNode, as the instantiation of the operation itself will be queued in addition
  to the decomposition. This can be solved by stopping the recording
  while instantiating an operator, and then calling its decomposition:

  ```python
  @qml.qnode(dev)
  def my_qnode(x):
      with qml.tape.stop_recording():
          op = qml.CRX(x, wires=[0, 1])
      op.decomposition()
  ```

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

Guillermo Alonso-Linaje, Olivia Di Matteo, Jalani Kanem, Christina Lee, Alejandro Montanez, Jay Soni
