:orphan:

# Release 0.29.0-dev (development release)

<h3>New features since last release</h3>

* The gradient transform `qml.gradients.spsa_grad` is now registered as a
  differentiation method for `QNode`s.
  [#3440](https://github.com/PennyLaneAI/pennylane/pull/3440)

  The SPSA gradient transform can now also be used implicitly by marking a `QNode`
  as differentiable with SPSA. It can be selected via

  ```pycon
  >>> dev = qml.device("default.qubit", wires=2)
  >>> @qml.qnode(dev, interface="jax", diff_method="spsa", h=0.05, num_directions=20)
  ... def circuit(x):
  ...     qml.RX(x, 0)
  ...     qml.RX(x, 1)
  ...     return qml.expval(qml.PauliZ(0))
  >>> jax.jacobian(circuit)(jax.numpy.array(0.5)) 
  DeviceArray(-0.4792258, dtype=float32, weak_type=True)
  ```

  The argument `num_directions` determines how many directions of simultaneous
  perturbation are used and therefore the number of circuit evaluations, up
  to a prefactor. See the
  [SPSA gradient transform documentation](https://docs.pennylane.ai/en/stable/code/api/pennylane.gradients.spsa_grad.html) for details.
  Note: The full SPSA optimization method is already available as `SPSAOptimizer`.

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
  
* The function `max_entropy` is added to compute the maximum entropy $H=\log(rank(\rho))$ of a quantum state.
  [(#3594)](https://github.com/PennyLaneAI/pennylane/pull/3594)

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

* Support `qml.math.size` with torch tensors.
  [(#3606)](https://github.com/PennyLaneAI/pennylane/pull/3606)

* Added `ParametrizedEvolution`, which computes the time evolution of a `ParametrizedHamiltonian`.
  [(#3617)](https://github.com/PennyLaneAI/pennylane/pull/3617)

* Added `qml.evolve`, which accepts an operator or a `ParametrizedHamiltonian` and returns another
  operator that computes its evolution.
  [(#3617)](https://github.com/PennyLaneAI/pennylane/pull/3617)

* Support `qml.math.matmul` with a torch tensor and an autograd tensor.
  [(#3613)](https://github.com/PennyLaneAI/pennylane/pull/3613)

* Added `qml.qchem.givens_decomposition` method that decompose a unitary into a sequence
  of Givens rotation gates with phase shifts and a diagonal phase matrix.
  [(#3573)](https://github.com/PennyLaneAI/pennylane/pull/3573)

  ```python
  unitary = np.array([[ 0.73678+0.27511j, -0.5095 +0.10704j, -0.06847+0.32515j]
                      [-0.21271+0.34938j, -0.38853+0.36497j,  0.61467-0.41317j]
                      [ 0.41356-0.20765j, -0.00651-0.66689j,  0.32839-0.48293j]])

  phase_mat, ordered_rotations = givens_decomposition(matrix)
  ```

  ```pycon
  >>> phase_mat
  [-0.20606284+0.97853876j -0.82993403+0.55786154j  0.56230707-0.82692851j]
  >>> ordered_rotations
  [(tensor([[-0.65088844-0.63936314j, -0.40933972-0.j],
            [-0.29202076-0.28684994j,  0.91238204-0.j]], requires_grad=True), (0, 1)),
    (tensor([[ 0.47970417-0.33309047j, -0.8117479 -0.j],
            [ 0.66676972-0.46298251j,  0.584008  -0.j]], requires_grad=True), (1, 2)),
    (tensor([[ 0.36147511+0.73779414j, -0.57008381-0.j],
            [ 0.25082094+0.5119418j ,  0.82158655-0.j]], requires_grad=True), (0, 1))]
  ```

* Added a new template `qml.BasisRotation` that performs basis transformation defined by a set of
  fermionic ladder operators.
  [(#3573)](https://github.com/PennyLaneAI/pennylane/pull/3573)

  ```python
  import pennylane as qml
  from pennylane import numpy as np

  V = np.array([[ 0.53672126+0.j        , -0.1126064 -2.41479668j],
                [-0.1126064 +2.41479668j,  1.48694623+0.j        ]])
  eigen_vals, eigen_vecs = np.linalg.eigh(V)
  umat = eigen_vecs.T
  wires = range(len(umat))
  def circuit():
      qml.adjoint(qml.BasisRotation(wires=wires, unitary_matrix=umat))
      for idx, eigenval in enumerate(eigen_vals):
          qml.RZ(eigenval, wires=[idx])
      qml.BasisRotation(wires=wires, unitary_matrix=umat)
  ```

  ```pycon
  >>> circ_unitary = qml.matrix(circuit)()
  >>> np.round(circ_unitary/circ_unitary[0][0], 3)
  tensor([[ 1.   +0.j   ,  0.   +0.j   ,  0.   +0.j   ,  0.   +0.j   ],
          [ 0.   +0.j   , -0.516-0.596j, -0.302-0.536j,  0.   +0.j   ],
          [ 0.   +0.j   ,  0.35 +0.506j, -0.311-0.724j,  0.   +0.j   ],
          [ 0.   +0.j   ,  0.   +0.j   ,  0.   +0.j   , -0.438+0.899j]])
  ```
  
* Added `ParametrizedHamiltonian`, a callable that holds information representing a linear combination of operators 
  with parametrized coefficents. The `ParametrizedHamiltonian` can be passed parameters to create the `Operator` for 
  the specified parameters.
  [(#3617)](https://github.com/PennyLaneAI/pennylane/pull/3617)
  
  ```pycon
  f1 = lambda p, t: p * np.sin(t) * (t - 1)
  f2 = lambda p, t: p[0] * np.cos(p[1]* t ** 2)

  XX = qml.PauliX(1) @ qml.PauliX(1)
  YY = qml.PauliY(0) @ qml.PauliY(0)
  ZZ = qml.PauliZ(0) @ qml.PauliZ(1)

  H =  2 * XX + f1 * YY + f2 * ZZ
  ```
  ```pycon
  >>> H
  ParametrizedHamiltonian: terms=3
  >>> params = [1.2, [2.3, 3.4]]
  >>> H(params, t=0.5)
  (2*(PauliX(wires=[1]) @ PauliX(wires=[1]))) + ((-0.2876553535461426*(PauliY(wires=[0]) @ PauliY(wires=[0]))) + (1.5179612636566162*(PauliZ(wires=[0]) @ PauliZ(wires=[1]))))
  ```
  The same `ParametrizedHamiltonian` can also be constructed via a list of coefficients and operators:

  ```pycon
  >>> coeffs = [2, f1, f2]
  >>> ops = [XX, YY, ZZ]
  >>> H =  qml.ops.dot(coeffs, ops)
  ```
  
<h3>Improvements</h3>

* Most channels in are now fully differentiable in all interfaces.
  [(#3612)](https://github.com/PennyLaneAI/pennylane/pull/3612)

* Extended the `qml.equal` function to compare `Prod` and `Sum` operators.
  [(#3516)](https://github.com/PennyLaneAI/pennylane/pull/3516)

* Reorganize `ControlledQubitUnitary` to inherit from `ControlledOp`. The class methods
  `decomposition`, `expand`, and `sparse_matrix` are now defined rather than raising an error.
  [(#3450)](https://github.com/PennyLaneAI/pennylane/pull/3450)

* Parameter broadcasting support is added for the `Controlled` class if the base operator supports
  broadcasting.
  [(#3450)](https://github.com/PennyLaneAI/pennylane/pull/3450)

* The `qml.generator` function now checks if the generator is hermitian, rather than whether it is a subclass of
  `Observable`, allowing it to return valid generators from `SymbolicOp` and `CompositeOp` classes.
 [(#3485)](https://github.com/PennyLaneAI/pennylane/pull/3485)

* Added support for two-qubit unitary decomposition with JAX-JIT.
  [(#3569)](https://github.com/PennyLaneAI/pennylane/pull/3569)

* Limit the `numpy` version to `<1.24`.
  [(#3563)](https://github.com/PennyLaneAI/pennylane/pull/3563)

* Removes qutrit operations use of in-place inversion in preparation for the
  removal of in-place inversion.
  [(#3566)](https://github.com/PennyLaneAI/pennylane/pull/3566)

* Validation has been added on the `gradient_kwargs` when initializing a QNode, and if unexpected kwargs are passed,
  a `UserWarning` is raised. A list of the current expected gradient function kwargs has been added as
  `qml.gradients.SUPPORTED_GRADIENT_KWARGS`.
  [(#3526)](https://github.com/PennyLaneAI/pennylane/pull/3526)

* Improve the `PauliSentence.operation()` method to avoid instantiating an `SProd` operator when
  the coefficient is equal to 1.
  [(#3595)](https://github.com/PennyLaneAI/pennylane/pull/3595)

* Write Hamiltonians to file in a condensed format when using the data module.
  [(#3592)](https://github.com/PennyLaneAI/pennylane/pull/3592)

* Improve lazy-loading in `Dataset.read()` so it is more universally supported. Also added the `assign_to`
  keyword argument to specify that the contents of the file being read should be directly assigned to an attribute.
  [(#3605)](https://github.com/PennyLaneAI/pennylane/pull/3605)

* Implemented the XYX single-qubit unitary decomposition. 
  [(#3628)](https://github.com/PennyLaneAI/pennylane/pull/3628) 

* Allow `Sum` and `Prod` to have broadcasted operands.
  [(#3611)](https://github.com/PennyLaneAI/pennylane/pull/3611)

* Make `qml.ops.dot` jax-jittable.
  [(#3636)](https://github.com/PennyLaneAI/pennylane/pull/3636)

* All dunder methods now return `NotImplemented`, allowing the right dunder method (e.g. `__radd__`)
  of the other class to be called.
  [(#3631)](https://github.com/PennyLaneAI/pennylane/pull/3631)

* The GellMann operators now include their index in the displayed representation.
  [(#3641)](https://github.com/PennyLaneAI/pennylane/pull/3641)

* Introduce the `ExecutionConfig` data class.
  [(#3649)](https://github.com/PennyLaneAI/pennylane/pull/3649)

<h3>Breaking changes</h3>

* `Operator.inv()` and the `Operator.inverse` setter are removed. Please use `qml.adjoint` or `qml.pow` instead.
  [(#3618)](https://github.com/PennyLaneAI/pennylane/pull/3618)
  
  Instead of 
  
  >>> qml.PauliX(0).inv()
  
  use
  
  >>> qml.adjoint(qml.PauliX(0))

* The target wires of the unitary for `ControlledQubitUnitary` are no longer available via `op.hyperparameters["u_wires"]`.
  Instead, they can be accesses via `op.base.wires` or `op.target_wires`.
  [(#3450)](https://github.com/PennyLaneAI/pennylane/pull/3450)

* The tape constructed by a QNode is no longer queued to surrounding contexts.
  [(#3509)](https://github.com/PennyLaneAI/pennylane/pull/3509)

* Nested operators like `Tensor`, `Hamiltonian` and `Adjoint` now remove their owned operators
  from the queue instead of updating their metadata to have an `"owner"`.
  [(#3282)](https://github.com/PennyLaneAI/pennylane/pull/3282)

* `qchem.scf`, `RandomLayers.compute_decomposition`, and `Wires.select_random` all use
  local random number generators now instead of global random number generators. This may lead to slighlty
  different random numbers, and an independence of the results from the global random number generation state.
  Please provide a seed to each individual function instead if you want controllable results.
  [(#3624)](https://github.com/PennyLaneAI/pennylane/pull/3624)

<h3>Deprecations</h3>

<h3>Documentation</h3>

* Updated the code example in `qml.SparseHamiltonian` with the correct wire range.
  [(#3643)](https://github.com/PennyLaneAI/pennylane/pull/3643)
  
* Added hyperlink text for an URL in the `qml.qchem.mol_data` docstring.
  [(#3644)](https://github.com/PennyLaneAI/pennylane/pull/3644)

<h3>Bug fixes</h3>

* Fixed a bug in `qml.transforms.metric_tensor` where prefactors of operation generators were taken
  into account multiple times, leading to wrong outputs for non-standard operations.
  [(#3579)](https://github.com/PennyLaneAI/pennylane/pull/3579)

* Uses a local random number generator where possible to avoid mutating the global random state.
  [(#3624)](https://github.com/PennyLaneAI/pennylane/pull/3624)

* Handles breaking networkx version change by selectively skipping a qcut tensorflow-jit test.
  [(#3609)](https://github.com/PennyLaneAI/pennylane/pull/3609)
  [(#3619)](https://github.com/PennyLaneAI/pennylane/pull/3619)

* Fixed the wires for the Y decomposition in the ZX calculus transform.
  [(#3598)](https://github.com/PennyLaneAI/pennylane/pull/3598)

* `qml.pauli.PauliWord` is now pickle-able.
  [(#3588)](https://github.com/PennyLaneAI/pennylane/pull/3588)

* Child classes of `QuantumScript` now return their own type when using `SomeChildClass.from_queue`.
  [(#3501)](https://github.com/PennyLaneAI/pennylane/pull/3501)

* Fixed typo in calculation error message and comment in operation.py
  [(#3536)](https://github.com/PennyLaneAI/pennylane/pull/3536)

* `Dataset.write()` now ensures that any lazy-loaded values are loaded before they are written to a file.
  [(#3605)](https://github.com/PennyLaneAI/pennylane/pull/3605)

* Set `Tensor._batch_size` to None during initialization, copying and `map_wires`.
  [(#3642)](https://github.com/PennyLaneAI/pennylane/pull/3642)
  [(#3661)](https://github.com/PennyLaneAI/pennylane/pull/3661)

* Set `Tensor.has_matrix` to `True`.
  [(#3647)](https://github.com/PennyLaneAI/pennylane/pull/3647)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Juan Miguel Arrazola
Ikko Ashimine
Utkarsh Azad
Cristian Boghiu
Astral Cai
Lillian M. A. Frederiksen
Soran Jahangiri
Christina Lee
Albert Mitjans Coma
Romain Moyard
Borja Requena
Matthew Silverman
Antal Száva
David Wierichs
