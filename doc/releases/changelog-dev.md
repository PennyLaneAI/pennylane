:orphan:

# Release 0.31.0-dev (development release)

<h3>New features since last release</h3>

<h4>Fermionic operators 🔬</h4>

<h4>Workflow-level resource estimation 🧮</h4>

* PennyLane's [Tracker](https://docs.pennylane.ai/en/stable/code/api/pennylane.Tracker.html) now
  monitors the resource requirements of circuits being executed by the device.
  [(#4045)](https://github.com/PennyLaneAI/pennylane/pull/4045)
  [(#4110)](https://github.com/PennyLaneAI/pennylane/pull/4110)

  Suppose we have a workflow that involves executing circuits with different qubit numbers. This
  can be achieved by executing the workflow with the `Tracker` context:

  ```python
  dev = qml.device("default.qubit", wires=4)

  @qml.qnode(dev)
  def circuit(n_wires):
      for i in range(n_wires):
          qml.Hadamard(i)
      return qml.probs(range(n_wires))

  with qml.Tracker(dev) as tracker:
      for i in range(1, 5):
          circuit(i)
  ```

  It is then possible to inspect the resource requirements of individual circuits:

  ```pycon
  >>> resources = tracker.history["resources"]
  >>> resources[0]
  wires: 1
  gates: 1
  depth: 1
  shots: Shots(total=None)
  gate_types:
  {'Hadamard': 1}
  gate_sizes:
  {1: 1}
  >>> [r.num_wires for r in resources]
  [1, 2, 3, 4]
  ```
  
  Moreover, it is possible to predict the resource requirements without evaluating circuits
  using the `null.qubit` device, which follows the standard execution pipeline but returns numeric
  zeros. Consider the following workflow that takes the gradient of a
  `50`-qubit circuit:

  ```python
  from pennylane import numpy as np

  n_wires = 50
  dev = qml.device("null.qubit", wires=n_wires)

  weight_shape = qml.StronglyEntanglingLayers.shape(2, n_wires)
  weights = np.random.random(weight_shape, requires_grad=True)

  @qml.qnode(dev, diff_method="parameter-shift")
  def circuit(weights):
      qml.StronglyEntanglingLayers(weights, wires=range(n_wires))
      return qml.expval(qml.PauliZ(0))

  with qml.Tracker(dev) as tracker:
      qml.grad(circuit)(weights)
  ```

  The tracker can be inspected to extract resource requirements without requiring a 50-qubit circuit
  run:

  ```pycon
  >>> tracker.totals
  {'executions': 451, 'batches': 2, 'batch_len': 451}
  >>> tracker.history["resources"][0]
  wires: 50
  gates: 200
  depth: 77
  shots: Shots(total=None)
  gate_types:
  {'Rot': 100, 'CNOT': 100}
  gate_sizes:
  {1: 100, 2: 100}
  ```

* `qml.specs` is compatible with custom operations that have `depth` bigger than 1.
  [(#4033)](https://github.com/PennyLaneAI/pennylane/pull/4033)

<h4>Community contributions from UnitaryHack 🤝</h4>

* Updated repr for ParametrizedHamiltonian.
  [(##4176)](https://github.com/PennyLaneAI/pennylane/pull/4176)

<h4>Broadcasting and other tweaks to Torch and Keras layers 🦾</h4>

* The `qml.qnn.KerasLayer` and `qml.qnn.TorchLayer` classes now natively support parameter broadcasting.
  [(#4131)](https://github.com/PennyLaneAI/pennylane/pull/4131)

<h3>Improvements 🛠</h3>

<h4>Extended support for differentiating pulses</h4>

* `pulse.ParametrizedEvolution` now uses _batched_ compressed sparse row (`BCSR`) format. This allows computing Jacobians of the unitary directly even when `dense=False`.
  ```python
  def U(params):
      H = jnp.polyval * qml.PauliZ(0) # time dependent Hamiltonian
      Um = qml.evolve(H)(params, t=10., dense=False)
      return qml.matrix(Um)
  params = jnp.array([[0.5]], dtype=complex)
  jac = jax.jacobian(U, holomorphic=True)(params)
  ```
  [(#4126)](https://github.com/PennyLaneAI/pennylane/pull/4126)

* The stochastic parameter-shift gradient transform for pulses, `stoch_pulse_grad`, now
  supports arbitrary Hermitian generating terms in pulse Hamiltonians.
  [(4132)](https://github.com/PennyLaneAI/pennylane/pull/4132)

<h4>The qchem module</h4>

* The `qchem.molecular_hamiltonian` function is upgraded to support custom wires for constructing
  differentiable Hamiltonians. The zero imaginary component of the Hamiltonian coefficients are
  removed.
  [(#4050)](https://github.com/PennyLaneAI/pennylane/pull/4050)
  [(#4094)](https://github.com/PennyLaneAI/pennylane/pull/4094)

* An error is now raised by `qchem.molecular_hamiltonian` when the `dhf` method is used for an 
  open-shell system. This duplicates a similar error in `qchem.Molecule` but makes it easier to
  inform the users that the `pyscf` backend can be used for open-shell calculations.
  [(#4058)](https://github.com/PennyLaneAI/pennylane/pull/4058)

* Accelerate Jordan-Wigner transforms caching Pauli gate objects.
  [(#4046)](https://github.com/PennyLaneAI/pennylane/pull/4046)

<h4>A more flexible projector</h4>

<h4>Do more with qutrits</h4>

* Updated `pennylane/qnode.py` to support parameter-shift differentiation on qutrit devices.
  [(#2845)](https://github.com/PennyLaneAI/pennylane/pull/2845)

* Added the `TRX` qutrit rotation operator, which allows applying a Pauli X rotation on a
  given subspace.
  [(#2845)](https://github.com/PennyLaneAI/pennylane/pull/2845)

* Added the `TRY` qutrit rotation operator, which allows applying a Y rotation on a
  given subspace.
  [(#2846)](https://github.com/PennyLaneAI/pennylane/pull/2846)

<h4>Next-generation device API</h4>

* The new device interface in integrated with `qml.execute` for autograd, backpropagation, and no differentiation.
  [(#3903)](https://github.com/PennyLaneAI/pennylane/pull/3903)

* Support for adjoint differentiation has been added to the `DefaultQubit2` device.
  [(#4037)](https://github.com/PennyLaneAI/pennylane/pull/4037)

* Added a function `measure_with_samples` that returns a sample-based measurement result given a state
  [(#4083)](https://github.com/PennyLaneAI/pennylane/pull/4083)
  [(#4093)](https://github.com/PennyLaneAI/pennylane/pull/4093)

* `DefaultQubit2.preprocess` now returns a new `ExecutionConfig` object with decisions for `gradient_method`,
  `use_device_gradient`, and `grad_on_execution`.
  [(#4102)](https://github.com/PennyLaneAI/pennylane/pull/4102)

* Support for sample-based measurements has been added to the `DefaultQubit2` device.
  [(#4105)](https://github.com/PennyLaneAI/pennylane/pull/4105)
  [(#4114)](https://github.com/PennyLaneAI/pennylane/pull/4114)
  [(#4133)](https://github.com/PennyLaneAI/pennylane/pull/4133)
  [(#4172)](https://github.com/PennyLaneAI/pennylane/pull/4172)

* Added a keyword argument `seed` to the `DefaultQubit2` device.
  [(#4120)](https://github.com/PennyLaneAI/pennylane/pull/4120)

* The new device interface in integrated with `qml.execute` for Jax.
  [(#4137)](https://github.com/PennyLaneAI/pennylane/pull/4137)

* The experimental device `devices.experimental.DefaultQubit2` now supports `qml.Snapshot`.
  [(#4193)](https://github.com/PennyLaneAI/pennylane/pull/4193)

<h4>Handling shots</h4>

* Added a `shots` property to `QuantumScript`. This will allow shots to be tied to executions instead of devices more
  concretely.
  [(#4067)](https://github.com/PennyLaneAI/pennylane/pull/4067)
  [(#4103)](https://github.com/PennyLaneAI/pennylane/pull/4103)
  [(#4106)](https://github.com/PennyLaneAI/pennylane/pull/4106)
  [(#4112)](https://github.com/PennyLaneAI/pennylane/pull/4112)

* Added `__repr__` and `__str__` methods to the `Shots` class.
  [(#4081)](https://github.com/PennyLaneAI/pennylane/pull/4081)

* Added `__eq__` and `__hash__` methods to the `Shots` class.
  [(#4082)](https://github.com/PennyLaneAI/pennylane/pull/4082)

* `qml.devices.ExecutionConfig` no longer has a `shots` property, as it is now on the `QuantumScript`.  It now has a `use_device_gradient` property. `ExecutionConfig.grad_on_execution = None` indicates a request for `"best"`, instead of a string.
  [(#4102)](https://github.com/PennyLaneAI/pennylane/pull/4102)

* Integrated `QuantumScript.shots` with `QNode` so that shots are placed on the `QuantumScript`
  during `QNode` construction.
  [(#4110)](https://github.com/PennyLaneAI/pennylane/pull/4110)

* Updated the `gradients` module to use the new `Shots` object internally.
  [(#4152)](https://github.com/PennyLaneAI/pennylane/pull/4152)

<h4>Operators</h4>

* `qml.prod` now accepts a single qfunc input for creating new `Prod` operators.
  [(#4011)](https://github.com/PennyLaneAI/pennylane/pull/4011)

* `DiagonalQubitUnitary` now decomposes into `RZ`, `IsingZZ` and `MultiRZ` gates
  instead of a `QubitUnitary` operation with a dense matrix.
  [(#4035)](https://github.com/PennyLaneAI/pennylane/pull/4035)

* Wrap all objects being queued in an `AnnotatedQueue` so that `AnnotatedQueue` is not dependent on
  the hash of any operators/measurement processes.
  [(#4087)](https://github.com/PennyLaneAI/pennylane/pull/4087)

* Added a `dense` keyword to `ParametrizedEvolution` that allows forcing dense or sparse matrices.
  [(#4079)](https://github.com/PennyLaneAI/pennylane/pull/4079)
  [(#4095)](https://github.com/PennyLaneAI/pennylane/pull/4095)

* Added a new function `qml.ops.functions.bind_new_parameters` that creates a copy of an operator with new parameters
  without mutating the original operator.
  [(#4113)](https://github.com/PennyLaneAI/pennylane/pull/4113)

* `qml.CY` has been moved from `qml.ops.qubit.non_parametric_ops` to `qml.ops.op_math.controlled_ops`
  and now inherits from `qml.ops.op_math.ControlledOp`.
  [(#4116)](https://github.com/PennyLaneAI/pennylane/pull/4116/)

* `CZ` now inherits from the `ControlledOp` class. It now supports exponentiation to arbitrary powers with `pow`, which is no longer limited to integers. It also supports `sparse_matrix` and `decomposition` representations.
  [(#4117)](https://github.com/PennyLaneAI/pennylane/pull/4117)

* The construction of the pauli representation for the `Sum` class is now faster.
  [(#4142)](https://github.com/PennyLaneAI/pennylane/pull/4142)

* `qml.drawer.drawable_layers.drawable_layers` and `qml.CircuitGraph` have been updated to not rely on `Operator`
  equality or hash to work correctly.
  [(#4143)](https://github.com/PennyLaneAI/pennylane/pull/4143)

<h4>Other improvements</h4>

* All drawing methods changed their default value for the keyword argument `show_matrices`
  to `True`. This allows quick insights into broadcasted tapes for example.
  [(#3920)](https://github.com/PennyLaneAI/pennylane/pull/3920)

* Adds the Type variables `pennylane.typing.Result` and `pennylane.typing.ResultBatch` for type hinting the result of
  an execution.
  [(#4018)](https://github.com/PennyLaneAI/pennylane/pull/4108)
  
* The Jax-JIT interface now uses symbolic zeros to determine trainable parameters.
  [(4075)](https://github.com/PennyLaneAI/pennylane/pull/4075)

* A function `pauli.pauli_word_prefactor()` is added to extract the prefactor for a given Pauli word.
  [(#4164)](https://github.com/PennyLaneAI/pennylane/pull/4164)

* Added `qml.math.reduce_dm` and `qml.math.reduce_statevector` to produce reduced density matrices.
  Both functions have broadcasting support.
  [(#4173)](https://github.com/PennyLaneAI/pennylane/pull/4173)

<h3>Breaking changes 💔</h3>

* All drawing methods changed their default value for the keyword argument `show_matrices` to `True`.
  [(#3920)](https://github.com/PennyLaneAI/pennylane/pull/3920)

* `DiagonalQubitUnitary` does not decompose into `QubitUnitary` any longer, but into `RZ`, `IsingZZ`
  and `MultiRZ` gates.
  [(#4035)](https://github.com/PennyLaneAI/pennylane/pull/4035)

* Jax trainable parameters are now `Tracer` instead of `JVPTracer`, it is not always the right definition for the JIT 
  interface, but we update them in the custom JVP using symbolic zeros.
  [(4075)](https://github.com/PennyLaneAI/pennylane/pull/4075)

* The experimental Device interface `qml.devices.experimental.Device` now requires that the `preprocess` method
  also returns an `ExecutionConfig` object. This allows the device to choose what `"best"` means for various
  hyperparameters like `gradient_method` and `grad_on_execution`.
  [(#4007)](https://github.com/PennyLaneAI/pennylane/pull/4007)
  [(#4102)](https://github.com/PennyLaneAI/pennylane/pull/4102)

* Gradient transforms with Jax do not support `argnum` anymore,  `argnums` needs to be used.
  [(#4076)](https://github.com/PennyLaneAI/pennylane/pull/4076)

* `pennylane.collections`, `pennylane.op_sum`, and `pennylane.utils.sparse_hamiltonian` are removed.

<h3>Deprecations 👋</h3>

* `LieAlgebraOptimizer` is renamed. Please use `RiemannianGradientOptimizer` instead.
  [(#4153)(https://github.com/PennyLaneAI/pennylane/pull/4153)]

* `Operation.base_name` is deprecated. Please use `Operation.name` or `type(op).__name__` instead.

* `QuantumScript`'s `name` keyword argument and property are deprecated.
  This also affects `QuantumTape` and `OperationRecorder`.
  [(#4141)](https://github.com/PennyLaneAI/pennylane/pull/4141)

* `qml.grouping` module is removed. The functionality has been reorganized in the `qml.pauli` module.

* `qml.math.reduced_dm` has been deprecated. Please use `qml.math.reduce_dm` or `qml.math.reduce_statevector` instead.
  [(#4173)](https://github.com/PennyLaneAI/pennylane/pull/4173)

* `do_queue` keyword argument in `qml.operation.Operator` is deprecated. Instead of
  setting `do_queue=False`, use the `qml.QueuingManager.stop_recording()` context.
  [(#4148)](https://github.com/PennyLaneAI/pennylane/pull/4148)

<h3>Documentation 📝</h3>

* The docstring for `qml.grad` now states that it should be used with the Autograd interface only.
  [(#4202)](https://github.com/PennyLaneAI/pennylane/pull/4202)

* The description of `mult` in the `qchem.Molecule` docstring now correctly states the value
  of `mult` that is supported.
  [(#4058)](https://github.com/PennyLaneAI/pennylane/pull/4058)

<h3>Bug fixes 🐛</h3>

* Fixes a bug where `stoch_pulse_grad` would ignore prefactors of rescaled Pauli words in the
  generating terms of a pulse Hamiltonian.
  [(4156)](https://github.com/PennyLaneAI/pennylane/pull/4156)
  
* Fixes a bug where the wire ordering of the `wires` argument to `qml.density_matrix`
  was not taken into account.
  [(#4072)](https://github.com/PennyLaneAI/pennylane/pull/4072)

* Removes a patch in `interfaces/autograd.py` that checks for the `strawberryfields.gbs` device.  That device
  is pinned to PennyLane <= v0.29.0, so that patch is no longer necessary.

* `qml.pauli.are_identical_pauli_words` now treats all identities as equal. Identity terms on Hamiltonians with non-standard
  wire orders are no longer eliminated.
  [(#4161)](https://github.com/PennyLaneAI/pennylane/pull/4161)

* `qml.pauli_sentence()` is now compatible with empty Hamiltonians `qml.Hamiltonian([], [])`.
  [(#4171)](https://github.com/PennyLaneAI/pennylane/pull/4171)

* Fixes a bug with Jax where executing multiple tapes with `gradient_fn="device"` would fail.
  [(#4190)](https://github.com/PennyLaneAI/pennylane/pull/4190)

* A more meaningful error message is raised when broadcasting with adjoint differentation on `DefaultQubit`.
  [(#4203)](https://github.com/PennyLaneAI/pennylane/pull/4203)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Venkatakrishnan AnushKrishna,
Isaac De Vlugt,
Soran Jahangiri,
Edward Jiang,
Korbinian Kottmann,
Christina Lee,
Vincent Michaud-Rioux,
Romain Moyard,
Mudit Pandey,
Borja Requena,
Matthew Silverman,
Jay Soni,
David Wierichs,
Frederik Wilde.
