:orphan:

# Release 0.21.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Bug fixes</h3>

* Fixes a bug in `classical_jacobian` when used with Torch, where the
  Jacobian of the preprocessing was also computed for non-trainable
  parameters.
  [(#2020)](https://github.com/PennyLaneAI/pennylane/pull/2020)

* Fixes a bug in queueing of the `two_qubit_decomposition` method that
  originally led to circuits with >3 two-qubit unitaries failing when passed
  through the `unitary_to_rot` optimization transform.
  [(#2015)](https://github.com/PennyLaneAI/pennylane/pull/2015)

<h3>Documentation</h3>

<h3>Operator class refactor</h3>

The Operator class has undergone a major refactor with the following changes:

* The `diagonalizing_gates()` representation has been moved to the highest-level 
  `Operator` class and is therefore available to all subclasses. A condition 
  `qml.operation.defines_diagonalizing_gates` has been added, which can be used 
  in tape contexts without queueing.
  [(#1985)](https://github.com/PennyLaneAI/pennylane/pull/1985)

* A `hyperparameters` attribute was added to the operator class.
  [(#2017)](https://github.com/PennyLaneAI/pennylane/pull/2017)
  
* The representation of an operator as a matrix has been overhauled. 
  
  The `matrix()` method now accepts a 
  `wire_order` argument and calculates the correct numerical representation 
  with respect to that ordering. 
    
  ```pycon
  >>> op = qml.RX(0.5, wires="b")
  >>> op.matrix()
  [[0.96891242+0.j         0.        -0.24740396j]
   [0.        -0.24740396j 0.96891242+0.j        ]]
  >>> op.matrix(wire_order=["a", "b"])
  [[0.9689+0.j  0.-0.2474j 0.+0.j         0.+0.j]
   [0.-0.2474j  0.9689+0.j 0.+0.j         0.+0.j]
   [0.+0.j          0.+0.j 0.9689+0.j 0.-0.2474j]
   [0.+0.j          0.+0.j 0.-0.2474j 0.9689+0.j]]
  ```
    
  The "canonical matrix", which is independent of wires,
  is now defined in the static method `compute_matrix()` instead of `_matrix`.
  By default, this method is assumed to take all parameters and non-trainable 
  hyperparameters that define the operation. 
    
  ```pycon
  >>> qml.RX.compute_matrix(0.5)
  [[0.96891242+0.j         0.        -0.24740396j]
   [0.        -0.24740396j 0.96891242+0.j        ]]
  ```
       
  If no canonical matrix is specified for a gate, `compute_matrix()` 
  returns `None`, whereas previously `NotImplementedErrors` 
  were raised.
  
  The new `matrix()` method is now used in the 
  `pennylane.transforms.get_qubit_unitary()` transform.
  [(#1996)](https://github.com/PennyLaneAI/pennylane/pull/1996)

* The eigenvalues representation of operators has been overhauled.
  XXX

* The `string_for_inverse` attribute is removed.
  [(#2021)](https://github.com/PennyLaneAI/pennylane/pull/2021)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):


Olivia Di Matteo, Christina Lee, Maria Schuld, David Wierichs