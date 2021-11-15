:orphan:

# Release 0.20.0-dev (development release)

<h3>New features since last release</h3>

* It is now possible to apply custom decompositions to operations using the
  `custom_decomposition` context manager.
  [(#1872)](https://github.com/PennyLaneAI/pennylane/pull/1872)

  For example, suppose we are running on an ion trap machine, and would
  like to implement a CNOT gate. This can be done using the `IsingXX` gate,
  but no decomposition is implemented in PennyLane for a CNOT. We can define
  this decomposition manually:

  ```python
  def ion_trap_cnot(wires):
      return [
          qml.RY(np.pi/2, wires=wires[0]),
      	  qml.IsingXX(np.pi/2, wires=wires),
          qml.RX(-np.pi/2, wires=wires[0]),
          qml.RY(-np.pi/2, wires=wires[0]),
          qml.RY(-np.pi/2, wires=wires[1])
      ]
   ```

   We can now execute this on a device using the context manager. We begin
   by defining a QNode:

   ```python
   dev = qml.device('default.qubit', wires=2)
 
   @qml.beta.qnode(dev)
   def run_cnot():
       qml.CNOT(wires=[0, 1])
       return qml.expval(qml.PauliX(wires=1))
   ```

   Now we can draw or execute the QNode within the context manager, and the
   device will perform the desired decomposition:

   ```pycon
   >>> with custom_decomposition({"CNOT" : ion_trap_cnot}, dev):
   ...    print(qml.draw(run_cnot, expansion_strategy="device")())
    0: ──RY(1.57)──╭IsingXX(1.57)──RX(-1.57)──RY(-1.57)──┤     
    1: ────────────╰IsingXX(1.57)──RY(-1.57)─────────────┤ ⟨X⟩ 
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
