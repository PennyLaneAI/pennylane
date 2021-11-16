:orphan:

# Release 0.20.0-dev (development release)

<h3>New features since last release</h3>

* Custom decompositions can now be applied to operations at the device level.
  [(#1900)](https://github.com/PennyLaneAI/pennylane/pull/1900)

  For example, suppose we would like to implement the following QNode:

  ```python
  def circuit(weights):
    qml.BasicEntanglerLayers(weights, wires=[0, 1, 2])
    return qml.expval(qml.PauliZ(0))

  original_dev = qml.device("default.qubit", wires=3)
  original_qnode = qml.beta.QNode(circuit, original_dev)
  ```

  ```pycon
  >>> weights = np.array([[0.4, 0.5, 0.6]])
  >>> print(qml.draw(original_qnode, expansion_strategy="device")(weights))
   0: ──RX(0.4)──╭C──────╭X──┤ ⟨Z⟩
   1: ──RX(0.5)──╰X──╭C──│───┤
   2: ──RX(0.6)──────╰X──╰C──┤
  ```

  Now, let's swap out the decomposition of the `CNOT` gate into `CZ`
  and `Hadamard`, and furthermore the decomposition of `Hadamard` into
  `RZ` and `RY` rather than the decomposition already available in PennyLane.
  We define the two decompositions like so, and pass them to a device:

  ```python
  def custom_cnot(wires):
      return [
          qml.Hadamard(wires=wires[1]),
          qml.CZ(wires=[wires[0], wires[1]]),
          qml.Hadamard(wires=wires[1])
      ]

  def custom_hadamard(wires):
      return [
          qml.RZ(np.pi, wires=wires),
	  qml.RY(np.pi / 2, wires=wires)
      ]

  # Can pass the operation itself, or a string
  custom_decomps = {qml.CNOT : custom_cnot, "Hadamard" : custom_hadamard}

  decomp_dev = qml.device("default.qubit", wires=3, custom_decomps=custom_decomps)
  decomp_qnode = qml.beta.QNode(circuit, decomp_dev)
  ```

  Now when we draw or run a QNode on this device, the gates will be expanded
  according to our specifications:

  ```pycon
  >>> print(qml.draw(decomp_qnode, expansion_strategy="device")(weights))
   0: ──RX(0.4)──────────────────────╭C──RZ(3.14)──RY(1.57)──────────────────────────╭Z──RZ(3.14)──RY(1.57)──┤ ⟨Z⟩
   1: ──RX(0.5)──RZ(3.14)──RY(1.57)──╰Z──RZ(3.14)──RY(1.57)──╭C──────────────────────│───────────────────────┤
   2: ──RX(0.6)──RZ(3.14)──RY(1.57)──────────────────────────╰Z──RZ(3.14)──RY(1.57)──╰C──────────────────────┤
  ```

* A thermal relaxation channel is added to the Noisy channels. The channel description can be 
  found on the supplementary information of [Quantum classifier with tailored quantum kernels](https://arxiv.org/abs/1909.02611).
  [(#1766)](https://github.com/PennyLaneAI/pennylane/pull/1766)
  
* Added the identity observable to be an operator. Now we can explicitly call the identity 
  operation on our quantum circuits for both qubit and CV devices.
  [(#1829)](https://github.com/PennyLaneAI/pennylane/pull/1829) 

<h3>Improvements</h3>

* A `decompose()` method has been added to the `Operator` class such that we can
  obtain (and queue) decompositions directly from instances of operations.
  [(#1873)](https://github.com/PennyLaneAI/pennylane/pull/1873)

  ```pycon
  >>> op = qml.PhaseShift(0.3, wires=0)
  >>> op.decompose()
  [RZ(0.3, wires=[0])]
  ```
  
* ``qml.circuit_drawer.draw_mpl`` produces a matplotlib figure and axes given a tape.
  [(#1787)](https://github.com/PennyLaneAI/pennylane/pull/1787)

* AngleEmbedding now supports `batch_params` decorator. [(#1812)](https://github.com/PennyLaneAI/pennylane/pull/1812)

<h3>Breaking changes</h3>

* The static method `decomposition()`, formerly in the `Operation` class, has
  been moved to the base `Operator` class.
  [(#1873)](https://github.com/PennyLaneAI/pennylane/pull/1873)
  
* `DiagonalOperation` is not a separate subclass any more. 
  [(#1889)](https://github.com/PennyLaneAI/pennylane/pull/1889) 

  Instead, devices can check for the diagonal 
  property using attributes:

  ``` python
  from pennylane.ops.qubit.attributes import diagonal_in_z_basis

  if op in diagonal_in_z_basis:
      # do something
  ``` 

<h3>Deprecations</h3>

<h3>Bug fixes</h3>

* `qml.circuit_drawer.MPLDrawer` was slightly modified to work with
  matplotlib version 3.5.
  [(#1899)](https://github.com/PennyLaneAI/pennylane/pull/1899)

* `qml.CSWAP` and `qml.CRot` now define `control_wires`, and `qml.SWAP` 
  returns the default empty wires object.
  [(#1830)](https://github.com/PennyLaneAI/pennylane/pull/1830)

* The `requires_grad` attribute of `qml.numpy.tensor` objects is now
  preserved when pickling/unpickling the object.
  [(#1856)](https://github.com/PennyLaneAI/pennylane/pull/1856)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order): 

Guillermo Alonso-Linaje, Olivia Di Matteo, Jalani Kanem, Christina Lee, Alejandro Montanez, Maria Schuld, Jay Soni
