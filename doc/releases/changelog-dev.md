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

* The `decomposition` method of `Operation` has been updated. Instead of the
  single static method, one can now also retrieve decompositions of instantiated
  operations directly.
  [(#1873)](https://github.com/PennyLaneAI/pennylane/pull/1873)

  To obtain a decomposition using the static method, we now use

  ```pycon
  >>> qml.PhaseShift._decomposition(0.3, wires=[0])
  [RZ(0.3, wires=[0])]
  ```

  Previously, the static method was named `decomposition`. Following this
  change, `decomposition` without the `_` is a regular method in the class that
  we can call from instantiated operations:

  ```pycon
  >>> op = qml.PhaseShift(0.3, wires=0)
  >>> op.decomposition()
  [RZ(0.3, wires=[0])]
  ```

  New `Operation`s should therefore define the `_decomposition` static method. This
  change upgrades the `decomposition` functionality to work just like the existing
  `_matrix` methods, as both are equally valid representations of `Operation`s.

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
