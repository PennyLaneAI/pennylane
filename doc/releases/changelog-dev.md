:orphan:

# Release 0.30.0-dev (development release)

<h3>New features since last release</h3>

* The diagonal of the Hessian of a `QuantumTape` can now be computed on simulators
  via `adjoint_hessian_diagonal`. Restrictions to the tape structure apply.
  [(#3083)](https://github.com/PennyLaneAI/pennylane/pull/3083)

  Similar to `adjoint_jacobian`, it is now possible to apply `adjoint_hessian_diagonal`
  to a `QuantumTape` to obtain the second-order derivatives of a tape.

  **Example**
  Consider the tape

  ```pycon
  >>> with qml.tape.QuantumTape() as tape:
  ...     qml.PauliRot(0.4, "XYXY", wires=[0, 1, 2, 3])
  ...     qml.RZ(-0.9, 1)
  ...     qml.RZ(-0.3, 2)
  ...     qml.expval(qml.PauliY(0) @ qml.PauliX(1))
  ...     qml.expval(qml.PauliY(2) @ qml.PauliX(3))
  ```
  and a simulator device like `"default.qubit"`. Then we can compute the second-order
  derivatives of the tape via
  
  ```pycon
  >>> dev = qml.device("default.qubit", wires=4)
  >>> hess_diag = dev.adjoint_hessian_diagonal(tape)
  [[ 0.30504187  0.          0.30504187]
   [ 0.         -0.95533649  0.        ]]
  ```
  Here the first axis corresponds to the two returned expectation values and the second
  axis is the axis for the parameters of the tape.

  **Usage conditions**

  For this method, the following conditions need to be satisfied:

    1. The used device must be a statevector simulator

    2. All differentiated operations must have a single parameter, and be generated
       by some generator `G` that squares to the identity operation. This includes
       `RX`, `RY`, `RZ`, `PauliRot`, `IsingXX`, `IsingYY`, `IsingZZ`, but for example
       does not include `CRX`, `IsingXY` or `CPauliRot`.

    3. The return values of the tape are measured via `qml.expval`.

    4. The differentiated parameters are not part of a measured `Hermitian` or `Hamiltonian`.

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

David Wierichs
