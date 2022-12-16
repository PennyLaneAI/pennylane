:orphan:

# Release 0.29.0-dev (development release)

<h3>New features since last release</h3>

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

<h3>Improvements</h3>

* The `qml.generator` function now checks if the generator is hermitian, rather than whether it is a subclass of 
  `Observable`, allowing it to return valid generators from `SymbolicOp` and `CompositeOp` classes.
 [(#3485)](https://github.com/PennyLaneAI/pennylane/pull/3485)

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

* Child classes of `QuantumScript` now return their own type when using `SomeChildClass.from_queue`.
  [(#3501)](https://github.com/PennyLaneAI/pennylane/pull/3501)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):
 
Lillian M. A. Frederiksen
Matthew Silverman
Antal Száva
