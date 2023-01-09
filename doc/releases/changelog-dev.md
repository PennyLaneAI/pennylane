:orphan:

# Release 0.29.0-dev (development release)

<h3>New features since last release</h3>

* `qml.purity` is added as a measurement process for purity
  [(#3551)](https://github.com/PennyLaneAI/pennylane/pull/3551)

* Added a new template that implements a canonical 2-complete linear (2-CCL) swap network
  described in [arXiv:1905.05118](https://arxiv.org/abs/1905.05118).
  [(#3447)](https://github.com/PennyLaneAI/pennylane/pull/3447)

  ```python3
  dev = qml.device('default.qubit', wires=5)
  weights = np.random.random(size=TwoLocalSwapNetwork.shape(len(dev.wires)))
  acquaintances = lambda index, wires, param: (qml.CRY(param, wires=index)
                                   if np.abs(wires[0]-wires[1]) else qml.CRZ(param, wires=index))
  @qml.qnode(dev)
  def swap_network_circuit():
     qml.templates.TwoLocalSwapNetwork(dev.wires, acquaintances, weights, fermionic=False)
     return qml.state()
  ```

  ```pycon
  >>> print(weights)
  tensor([0.20308242, 0.91906199, 0.67988804, 0.81290256, 0.08708985,
          0.81860084, 0.34448344, 0.05655892, 0.61781612, 0.51829044], requires_grad=True)
  >>> qml.draw(swap_network_circuit, expansion_strategy = 'device')()
  0: ─╭●────────╭SWAP─────────────────╭●────────╭SWAP─────────────────╭●────────╭SWAP─┤  State
  1: ─╰RY(0.20)─╰SWAP─╭●────────╭SWAP─╰RY(0.09)─╰SWAP─╭●────────╭SWAP─╰RY(0.62)─╰SWAP─┤  State
  2: ─╭●────────╭SWAP─╰RY(0.68)─╰SWAP─╭●────────╭SWAP─╰RY(0.34)─╰SWAP─╭●────────╭SWAP─┤  State
  3: ─╰RY(0.92)─╰SWAP─╭●────────╭SWAP─╰RY(0.82)─╰SWAP─╭●────────╭SWAP─╰RY(0.52)─╰SWAP─┤  State
  4: ─────────────────╰RY(0.81)─╰SWAP─────────────────╰RY(0.06)─╰SWAP─────────────────┤  State
  ```

* The JAX-JIT interface now supports higher-order gradient computation with the new return types system.
  [(#3498)](https://github.com/PennyLaneAI/pennylane/pull/3498)

  ```python
  import pennylane as qml
  import jax
  from jax import numpy as jnp
  
  jax.config.update("jax_enable_x64", True)
  
  qml.enable_return()
  
  dev = qml.device("lightning.qubit", wires=2)
  
  @jax.jit
  @qml.qnode(dev, interface="jax-jit", diff_method="parameter-shift", max_diff=2)
  def circuit(a, b):
      qml.RY(a, wires=0)
      qml.RX(b, wires=1)
      return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))
  
  a, b = jnp.array(1.0), jnp.array(2.0)
  ```

  ```pycon
  >>> jax.hessian(circuit, argnums=[0, 1])(a, b)
  (((DeviceArray(-0.54030231, dtype=float64, weak_type=True),
     DeviceArray(1.76002563e-17, dtype=float64, weak_type=True)),
    (DeviceArray(1.76002563e-17, dtype=float64, weak_type=True),
     DeviceArray(1.11578284e-34, dtype=float64, weak_type=True))),
   ((DeviceArray(2.77555756e-17, dtype=float64, weak_type=True),
     DeviceArray(-4.54411427e-17, dtype=float64, weak_type=True)),
    (DeviceArray(-1.76855671e-17, dtype=float64, weak_type=True),
     DeviceArray(0.41614684, dtype=float64, weak_type=True))))
  ```

* The qchem workflow is modified to support both Autograd and JAX frameworks.
  [(#3458)](https://github.com/PennyLaneAI/pennylane/pull/3458)
  [(#3462)](https://github.com/PennyLaneAI/pennylane/pull/3462)
  [(#3495)](https://github.com/PennyLaneAI/pennylane/pull/3495)

  The JAX interface is automatically used when the differentiable parameters are JAX objects. Here
  is an example for computing the Hartree-Fock energy gradients with respect to the atomic
  coordinates.

  ```python
  import pennylane as qml
  from pennylane import numpy as np
  import jax
  
  symbols = ["H", "H"]
  geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

  mol = qml.qchem.Molecule(symbols, geometry)

  args = [jax.numpy.array(mol.coordinates)]
  ```

  ```pycon
  >>> jax.grad(qml.qchem.hf_energy(mol))(*args)
  >>> DeviceArray([[0.0, 0.0, 0.3650435], [0.0, 0.0, -0.3650435]], dtype=float32)
  ```
  
* The function `load_basisset` is added to extract qchem basis set data from the Basis Set Exchange
  library.
  [(#3363)](https://github.com/PennyLaneAI/pennylane/pull/3363)

* Added `qml.ops.dot` function to compute the dot product between a vector and a list of operators.

  ```pycon
  >>> coeffs = np.array([1.1, 2.2])
  >>> ops = [qml.PauliX(0), qml.PauliY(0)]
  >>> qml.ops.dot(coeffs, ops)
  (1.1*(PauliX(wires=[0]))) + (2.2*(PauliY(wires=[0])))
  >>> qml.ops.dot(coeffs, ops, pauli=True)
  1.1 * X(0)
  + 2.2 * Y(0)
  ```

  [(#3586)](https://github.com/PennyLaneAI/pennylane/pull/3586)

<h3>Improvements</h3>

* Extended the `qml.equal` function to compare `Prod` and `Sum` operators.
  [(#3516)](https://github.com/PennyLaneAI/pennylane/pull/3516)

* The `qml.generator` function now checks if the generator is hermitian, rather than whether it is a subclass of
  `Observable`, allowing it to return valid generators from `SymbolicOp` and `CompositeOp` classes.
 [(#3485)](https://github.com/PennyLaneAI/pennylane/pull/3485)

* Added support for two-qubit unitary decomposition with JAX-JIT.
  [(#3569)](https://github.com/PennyLaneAI/pennylane/pull/3569)

* Limit the `numpy` version to `<1.24`.
  [(#3563)](https://github.com/PennyLaneAI/pennylane/pull/3563)

* Validation has been added on the `gradient_kwargs` when initializing a QNode, and if unexpected kwargs are passed,
  a `UserWarning` is raised. A list of the current expected gradient function kwargs has been added as
  `qml.gradients.SUPPORTED_GRADIENT_KWARGS`.
  [(#3526)](https://github.com/PennyLaneAI/pennylane/pull/3526)

* Improve the `PauliSentence.operation()` method to avoid instantiating an `SProd` operator when
  the coefficient is equal to 1.
  [(#3595)](https://github.com/PennyLaneAI/pennylane/pull/3595)

* Write Hamiltonians to file in a condensed format when using the data module.
  [(#3592)](https://github.com/PennyLaneAI/pennylane/pull/3592)

<h3>Breaking changes</h3>

* The tape constructed by a QNode is no longer queued to surrounding contexts.
  [(#3509)](https://github.com/PennyLaneAI/pennylane/pull/3509)

* Nested operators like `Tensor`, `Hamiltonian` and `Adjoint` now remove their owned operators
  from the queue instead of updating their metadata to have an `"owner"`.
  [(#3282)](https://github.com/PennyLaneAI/pennylane/pull/3282)

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

* Pins networkx version <3.0 till a bug with tensorflow-jit, networkx, and qcut is resolved.
  [(#3609)](https://github.com/PennyLaneAI/pennylane/pull/3609)

* Fixed the wires for the Y decomposition in the ZX calculus transform.
  [(#3598)](https://github.com/PennyLaneAI/pennylane/pull/3598)
* 
* `qml.pauli.PauliWord` is now pickle-able.
  [(#3588)](https://github.com/PennyLaneAI/pennylane/pull/3588)

* Child classes of `QuantumScript` now return their own type when using `SomeChildClass.from_queue`.
  [(#3501)](https://github.com/PennyLaneAI/pennylane/pull/3501)

* Fixed typo in calculation error message and comment in operation.py
  [(#3536)](https://github.com/PennyLaneAI/pennylane/pull/3536)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola
Ikko Ashimine
Utkarsh Azad
Astral Cai
Lillian M. A. Frederiksen
Soran Jahangiri
Christina Lee
Albert Mitjans Coma
Romain Moyard
Matthew Silverman
Antal Száva
