:orphan:

# Release 0.29.0-dev (development release)

<h3>New features since last release</h3>

<h4>Add new features here</h4>

<h4>Feel the pulse 🔊</h4>

* Parameterized Hamiltonians can now be created with the addition of `ParametrizedHamiltonian`.
  [(#3617)](https://github.com/PennyLaneAI/pennylane/pull/3617)

  A `ParametrizedHamiltonian` holds information representing a linear combination of operators 
  with parametrized coefficents. The `ParametrizedHamiltonian` can be passed parameters to create the operator for 
  the specified parameters.
  
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

* A `ParametrizedHamiltonian` can be time-evolved by using `ParametrizedEvolution`.
  [(#3617)](https://github.com/PennyLaneAI/pennylane/pull/3617)

* A new function called `qml.evolve` has been added that returns the evolution of an operator or a `ParametrizedHamiltonian`.
  [(#3617)](https://github.com/PennyLaneAI/pennylane/pull/3617)

* A new function called `qml.ops.dot` has been added to compute the dot product between a vector and a list of operators.
  [(#3586)](https://github.com/PennyLaneAI/pennylane/pull/3586)

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

<h4>Always differentiable 📈</h4>

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

* The JAX-JIT interface now supports higher-order gradient computation with the new return types system.
  [(#3498)](https://github.com/PennyLaneAI/pennylane/pull/3498)

  Here is an example of using JAX-JIT to compute the Hessian of a circuit:

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

* The `qchem` workflow has been modified to support both Autograd and JAX frameworks.
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

<h4>Tools for quantum chemistry and other applications 🛠️</h4>

* A new method called `qml.qchem.givens_decomposition` has been added, which decomposes a unitary into a sequence
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

* A new template called `qml.BasisRotation` has been added, which performs a basis transformation defined by a set of
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

* A new function called `load_basisset` has been added to extract `qchem` basis set data from the Basis Set Exchange
  library.
  [(#3363)](https://github.com/PennyLaneAI/pennylane/pull/3363)

* A new function called `max_entropy` has been added to compute the maximum entropy of a quantum state.
  [(#3594)](https://github.com/PennyLaneAI/pennylane/pull/3594)

* A new template called `TwoLocalSwapNetwork` has been added that implements a canonical 2-complete linear (2-CCL) swap network
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

<h3>Improvements</h3>

* `qml.purity` is added as a measurement process for purity
  [(#3551)](https://github.com/PennyLaneAI/pennylane/pull/3551)

* `qml.math.matmul` now supports PyTorch and Autograd tensors.
  [(#3613)](https://github.com/PennyLaneAI/pennylane/pull/3613)

* `qml.math.size` now supports PyTorch tensors.
  [(#3606)](https://github.com/PennyLaneAI/pennylane/pull/3606)

* `qml.purity` can now be used as a measurement process.
  [(#3551)](https://github.com/PennyLaneAI/pennylane/pull/3551)

* Most quantum channels are now fully differentiable on all interfaces.
  [(#3612)](https://github.com/PennyLaneAI/pennylane/pull/3612)

* The `qml.equal` function has been extended to compare `Prod` and `Sum` operators.
  [(#3516)](https://github.com/PennyLaneAI/pennylane/pull/3516)

* `qml.ControlledQubitUnitary` now inherits from `ControlledOp`, which defines `decomposition`, `expand`, and `sparse_matrix` rather than raising an error.
  [(#3450)](https://github.com/PennyLaneAI/pennylane/pull/3450)

* Parameter broadcasting support has been added for the `Controlled` class if the base operator supports
  broadcasting.
  [(#3450)](https://github.com/PennyLaneAI/pennylane/pull/3450)

* The `qml.generator` function now checks if the generator is hermitian, rather than whether it is a subclass of
  `Observable`. This allows it to return valid generators from `SymbolicOp` and `CompositeOp` classes.
 [(#3485)](https://github.com/PennyLaneAI/pennylane/pull/3485)

* Support for two-qubit unitary decomposition with JAX-JIT has been added.
  [(#3569)](https://github.com/PennyLaneAI/pennylane/pull/3569)

* The `numpy` version has been constrained to `<1.24`.
  [(#3563)](https://github.com/PennyLaneAI/pennylane/pull/3563)

* In-place inversion has been removed for qutrit operations in preparation for the
  removal of in-place inversion.
  [(#3566)](https://github.com/PennyLaneAI/pennylane/pull/3566)

* Validation has been added on gradient keyword arguments when initializing a QNode — if unexpected keyword arguments are passed,
  a `UserWarning` is raised. A list of the current expected gradient function keyword arguments can be accessed via
  `qml.gradients.SUPPORTED_GRADIENT_KWARGS`.
  [(#3526)](https://github.com/PennyLaneAI/pennylane/pull/3526)

* The `PauliSentence.operation()` method has been improved to avoid instantiating an `SProd` operator when
  the coefficient is equal to 1.
  [(#3595)](https://github.com/PennyLaneAI/pennylane/pull/3595)

* Writing Hamiltonians to a file using the `data` module has been improved by employing a condensed writing format.
  [(#3592)](https://github.com/PennyLaneAI/pennylane/pull/3592)

* Lazy-loading in the `Dataset.read()` method is more universally supported.
  [(#3605)](https://github.com/PennyLaneAI/pennylane/pull/3605)

* Implemented the XYX single-qubit unitary decomposition. 
  [(#3628)](https://github.com/PennyLaneAI/pennylane/pull/3628) 

* `Sum` and `Prod` operations now have broadcasted operands.
  [(#3611)](https://github.com/PennyLaneAI/pennylane/pull/3611)

* `qml.ops.dot` is now compatible with JAX-JIT.
  [(#3636)](https://github.com/PennyLaneAI/pennylane/pull/3636)

* All dunder methods now return `NotImplemented`, allowing the right dunder method (e.g. `__radd__`)
  of the other class to be called.
  [(#3631)](https://github.com/PennyLaneAI/pennylane/pull/3631)

* The `qml.GellMann` operators now include their index when displayed.
  [(#3641)](https://github.com/PennyLaneAI/pennylane/pull/3641)

* The `ExecutionConfig` data class has been added.
  [(#3649)](https://github.com/PennyLaneAI/pennylane/pull/3649)

* All `Operator`'s input parameters that are lists are cast into vanilla numpy arrays.
  [(#3659)](https://github.com/PennyLaneAI/pennylane/pull/3659)

* The `StatePrep` class has been added as an interface that state-prep operators must implement.
  [(#3654)](https://github.com/PennyLaneAI/pennylane/pull/3654)

<h3>Breaking changes</h3>

* The tape method `get_operation` can also now return the operation index in the tape, and it can be
  activated by setting the `return_op_index` to `True`: `get_operation(idx, return_op_index=True)`. It will become
  the default in version 0.30.
  [(#3667)](https://github.com/PennyLaneAI/pennylane/pull/3667)

* `Operator.inv()` and the `Operator.inverse` setter have been removed. Please use `qml.adjoint` or `qml.pow` instead.
  [(#3618)](https://github.com/PennyLaneAI/pennylane/pull/3618)
  
  For example, instead of 
  
  ```pycon
  >>> qml.PauliX(0).inv()
  ```
  
  use

  ```pycon
  >>> qml.adjoint(qml.PauliX(0))
  ```

* The target wires of `qml.ControlledQubitUnitary` are no longer available via `op.hyperparameters["u_wires"]`.
  Instead, they can be accesses via `op.base.wires` or `op.target_wires`.
  [(#3450)](https://github.com/PennyLaneAI/pennylane/pull/3450)

* The tape constructed by a QNode is no longer queued to surrounding contexts.
  [(#3509)](https://github.com/PennyLaneAI/pennylane/pull/3509)

* Nested operators like `Tensor`, `Hamiltonian`, and `Adjoint` now remove their owned operators
  from the queue instead of updating their metadata to have an `"owner"`.
  [(#3282)](https://github.com/PennyLaneAI/pennylane/pull/3282)

* `qchem.scf`, `RandomLayers.compute_decomposition`, and `Wires.select_random` now use
  local random number generators instead of global random number generators. This may lead to slighlty
  different random numbers and an independence of the results from the global random number generation state.
  Please provide a seed to each individual function instead if you want controllable results.
  [(#3624)](https://github.com/PennyLaneAI/pennylane/pull/3624)

<h3>Deprecations</h3>

<h3>Documentation</h3>

* Organizes the module for documentation for ``operation``.
  [(#3664)](https://github.com/PennyLaneAI/pennylane/pull/3664)

* Updated the code example in `qml.SparseHamiltonian` with the correct wire range.
  [(#3643)](https://github.com/PennyLaneAI/pennylane/pull/3643)
  
* A hyperlink has been added in the text for a URL in the `qml.qchem.mol_data` docstring.
  [(#3644)](https://github.com/PennyLaneAI/pennylane/pull/3644)

<h3>Bug fixes</h3>

* Fixed a bug in `qml.transforms.metric_tensor` where prefactors of operation generators were taken
  into account multiple times, leading to wrong outputs for non-standard operations.
  [(#3579)](https://github.com/PennyLaneAI/pennylane/pull/3579)

* Local random number generators are now used where possible to avoid mutating the global random state.
  [(#3624)](https://github.com/PennyLaneAI/pennylane/pull/3624)

* Handles breaking the `networkx` version change by selectively skipping a `qcut` TensorFlow-JIT test.
  [(#3609)](https://github.com/PennyLaneAI/pennylane/pull/3609)
  [(#3619)](https://github.com/PennyLaneAI/pennylane/pull/3619)

* Fixed the wires for the `Y` decomposition in the ZX calculus transform.
  [(#3598)](https://github.com/PennyLaneAI/pennylane/pull/3598)

* `qml.pauli.PauliWord` is now pickle-able.
  [(#3588)](https://github.com/PennyLaneAI/pennylane/pull/3588)

* Child classes of `QuantumScript` now return their own type when using `SomeChildClass.from_queue`.
  [(#3501)](https://github.com/PennyLaneAI/pennylane/pull/3501)

* A typo has been fixed in the calculation and error messages in `operation.py`
  [(#3536)](https://github.com/PennyLaneAI/pennylane/pull/3536)

* `Dataset.write()` now ensures that any lazy-loaded values are loaded before they are written to a file.
  [(#3605)](https://github.com/PennyLaneAI/pennylane/pull/3605)

* `Tensor._batch_size` is now set to `None` during initialization, copying and `map_wires`.
  [(#3642)](https://github.com/PennyLaneAI/pennylane/pull/3642)
  [(#3661)](https://github.com/PennyLaneAI/pennylane/pull/3661)

* `Tensor.has_matrix` is now set to `True`.
  [(#3647)](https://github.com/PennyLaneAI/pennylane/pull/3647)

* Fixed typo in the example of IsingZZ gate decomposition
  [(#3676)](https://github.com/PennyLaneAI/pennylane/pull/3676)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso-Linaje
Juan Miguel Arrazola
Ikko Ashimine
Utkarsh Azad
Cristian Boghiu
Astral Cai
Isaac De Vlugt
Lillian M. A. Frederiksen
Soran Jahangiri
Christina Lee
Albert Mitjans Coma
Romain Moyard
Borja Requena
Matthew Silverman
Antal Száva
David Wierichs
