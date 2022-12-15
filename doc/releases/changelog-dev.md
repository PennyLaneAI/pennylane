:orphan:

 # Release 0.29.0-dev (development release)

 <h3>New features since last release</h3>

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

 <h3>Improvements</h3>
* The `qml.generator` function now checks if the generator is hermitian, rather than whether it is a subclass of 
  `Observable`, allowing it to return valid generators from `SymbolicOp` and `CompositeOp` classes.
 [(#3485)](https://github.com/PennyLaneAI/pennylane/pull/3485)

 <h3>Breaking changes</h3>

 <h3>Deprecations</h3>

 <h3>Documentation</h3>

 <h3>Bug fixes</h3>

 <h3>Contributors</h3>
 This release contains contributions from (in alphabetical order):

Utkarsh Azad
Lillian M. A. Frederiksen
Antal Száva
