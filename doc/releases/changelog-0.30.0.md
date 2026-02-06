
# Release 0.30.0

<h3>New features since last release</h3>

<h4>Pulse programming on hardware ⚛️🔬</h4>

* Support for loading time-dependent Hamiltonians that are compatible with quantum hardware has been
  added, making it possible to load a Hamiltonian that describes an ensemble of Rydberg atoms or a
  collection of transmon qubits.
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)
  [(#3911)](https://github.com/PennyLaneAI/pennylane/pull/3911)
  [(#3930)](https://github.com/PennyLaneAI/pennylane/pull/3930)
  [(#3936)](https://github.com/PennyLaneAI/pennylane/pull/3936)
  [(#3966)](https://github.com/PennyLaneAI/pennylane/pull/3966)
  [(#3987)](https://github.com/PennyLaneAI/pennylane/pull/3987)
  [(#4021)](https://github.com/PennyLaneAI/pennylane/pull/4021)
  [(#4040)](https://github.com/PennyLaneAI/pennylane/pull/4040)

  [Rydberg atoms](https://en.wikipedia.org/wiki/Rydberg_atom) are the foundational 
  unit for neutral atom quantum computing. 
  A Rydberg-system Hamiltonian can be constructed from a
  [drive term](https://docs.pennylane.ai/en/stable/code/api/pennylane.pulse.rydberg_drive.html)
  — `qml.pulse.rydberg_drive` — and an
  [interaction term](https://docs.pennylane.ai/en/stable/code/api/pennylane.pulse.rydberg_interaction.html)
  — `qml.pulse.rydberg_interaction`:

  ```python
  from jax import numpy as jnp
  
  atom_coordinates = [[0, 0], [0, 4], [4, 0], [4, 4]]
  wires = [0, 1, 2, 3]
    
  amplitude = lambda p, t: p * jnp.sin(jnp.pi * t)
  phase = jnp.pi / 2
  detuning = 3 * jnp.pi / 4
  
  H_d = qml.pulse.rydberg_drive(amplitude, phase, detuning, wires)
  H_i = qml.pulse.rydberg_interaction(atom_coordinates, wires)
  H = H_d + H_i
  ```
  
  The time-dependent Hamiltonian `H` can be used in a PennyLane pulse-level differentiable circuit:

  ```python
  dev = qml.device("default.qubit.jax", wires=wires)

  @qml.qnode(dev, interface="jax")
  def circuit(params):
      qml.evolve(H)(params, t=[0, 10])
      return qml.expval(qml.PauliZ(0))
  ```
  
  ```pycon
  >>> params = jnp.array([2.4])
  >>> circuit(params)
  Array(0.6316659, dtype=float32)
  >>> import jax
  >>> jax.grad(circuit)(params)
  Array([1.3116529], dtype=float32)
  ```
  
  The [qml.pulse](https://docs.pennylane.ai/en/stable/code/qml_pulse.html) page contains
  additional details. Check out our
  [release blog post](https://pennylane.ai/blog/2023/05/pennylane-v030-released/) for a
  demonstration of how to perform the execution on actual hardware!

* A pulse-level circuit can now be differentiated using a
  [stochastic parameter-shift](https://arxiv.org/abs/2210.15812) method. 
  [(#3780)](https://github.com/PennyLaneAI/pennylane/pull/3780)
  [(#3900)](https://github.com/PennyLaneAI/pennylane/pull/3900)
  [(#4000)](https://github.com/PennyLaneAI/pennylane/pull/4000)
  [(#4004)](https://github.com/PennyLaneAI/pennylane/pull/4004)

  The new 
  [qml.gradient.stoch_pulse_grad](https://docs.pennylane.ai/en/stable/code/api/pennylane.gradients.stoch_pulse_grad.html) 
  differentiation method unlocks stochastic-parameter-shift differentiation for pulse-level circuits.
  The current version of this new method is restricted to Hamiltonians composed of parametrized
  [Pauli words](https://docs.pennylane.ai/en/stable/code/api/pennylane.pauli.PauliWord.html), but
  future updates to extend to parametrized
  [Pauli sentences](https://docs.pennylane.ai/en/stable/code/api/pennylane.pauli.PauliSentence.html)
  can allow this method to be compatible with hardware-based systems such as an ensemble of Rydberg
  atoms.

  This method can be activated by setting `diff_method` to
  [qml.gradient.stoch_pulse_grad](https://docs.pennylane.ai/en/stable/code/api/pennylane.gradients.stoch_pulse_grad.html):

  ```pycon
  >>> dev = qml.device("default.qubit.jax", wires=2)
  >>> sin = lambda p, t: jax.numpy.sin(p * t)
  >>> ZZ = qml.PauliZ(0) @ qml.PauliZ(1)
  >>> H = 0.5 * qml.PauliX(0) + qml.pulse.constant * ZZ + sin * qml.PauliX(1)
  >>> @qml.qnode(dev, interface="jax", diff_method=qml.gradients.stoch_pulse_grad)
  >>> def ansatz(params):
  ...     qml.evolve(H)(params, (0.2, 1.))
  ...     return qml.expval(qml.PauliY(1))
  >>> params = [jax.numpy.array(0.4), jax.numpy.array(1.3)]
  >>> jax.grad(ansatz)(params)
  [Array(0.16921353, dtype=float32, weak_type=True),
   Array(-0.2537478, dtype=float32, weak_type=True)]
  ```

<h4>Quantum singular value transformation 🐛➡️🦋</h4>

* PennyLane now supports the
  [quantum singular value transformation](https://arxiv.org/abs/1806.01838) (QSVT), which describes
  how a quantum circuit can be constructed to apply a polynomial transformation to the singular
  values of an input matrix.
  [(#3756)](https://github.com/PennyLaneAI/pennylane/pull/3756)
  [(#3757)](https://github.com/PennyLaneAI/pennylane/pull/3757)
  [(#3758)](https://github.com/PennyLaneAI/pennylane/pull/3758)
  [(#3905)](https://github.com/PennyLaneAI/pennylane/pull/3905)
  [(#3909)](https://github.com/PennyLaneAI/pennylane/pull/3909)
  [(#3926)](https://github.com/PennyLaneAI/pennylane/pull/3926)
  [(#4023)](https://github.com/PennyLaneAI/pennylane/pull/4023)

  Consider a matrix `A` along with a vector `angles` that describes the target polynomial
  transformation. The `qml.qsvt`
  function creates a corresponding circuit:

  ```python
  dev = qml.device("default.qubit", wires=2)

  A = np.array([[0.1, 0.2], [0.3, 0.4]])
  angles = np.array([0.1, 0.2, 0.3])

  @qml.qnode(dev)
  def example_circuit(A):
      qml.qsvt(A, angles, wires=[0, 1])
      return qml.expval(qml.PauliZ(wires=0))
  ```
  
  This circuit is composed of `qml.BlockEncode` and `qml.PCPhase` operations.

  ```pycon
  >>> example_circuit(A)
  tensor(0.97777078, requires_grad=True)
  >>> print(example_circuit.qtape.expand(depth=1).draw(decimals=2)) 
  0: ─╭∏_ϕ(0.30)─╭BlockEncode(M0)─╭∏_ϕ(0.20)─╭BlockEncode(M0)†─╭∏_ϕ(0.10)─┤  <Z>
  1: ─╰∏_ϕ(0.30)─╰BlockEncode(M0)─╰∏_ϕ(0.20)─╰BlockEncode(M0)†─╰∏_ϕ(0.10)─┤
  ```

  The [qml.qsvt](https://docs.pennylane.ai/en/stable/code/api/pennylane.qsvt.html) function
  creates a circuit that is targeted at simulators due to the use of matrix-based operations.
  For advanced users, you can use the 
  [operation-based](https://docs.pennylane.ai/en/stable/code/api/pennylane.QSVT.html)
  `qml.QSVT` template to perform
  the transformation with a custom choice of unitary and projector operations, which may be
  hardware compatible if a decomposition is provided.

  The QSVT is a complex but powerful transformation capable of
  [generalizing important algorithms](https://arxiv.org/abs/2105.02859)
  like amplitude amplification. Stay tuned for a demo in the coming few weeks to learn more!

<h4>Intuitive QNode returns ↩️</h4>

* An updated QNode return system has been introduced. PennyLane QNodes now return exactly what you 
  tell them to! 🎉
  [(#3957)](https://github.com/PennyLaneAI/pennylane/pull/3957)
  [(#3969)](https://github.com/PennyLaneAI/pennylane/pull/3969)
  [(#3946)](https://github.com/PennyLaneAI/pennylane/pull/3946)
  [(#3913)](https://github.com/PennyLaneAI/pennylane/pull/3913)
  [(#3914)](https://github.com/PennyLaneAI/pennylane/pull/3914)
  [(#3934)](https://github.com/PennyLaneAI/pennylane/pull/3934)

  This was an experimental feature introduced in version 0.25 of PennyLane that was enabled
  via `qml.enable_return()`. Now, it's the default return system. Let's see how it works.

  Consider the following circuit:

  ```python
  import pennylane as qp
  
  dev = qml.device("default.qubit", wires=1)
  
  @qml.qnode(dev)
  def circuit(x):
      qml.RX(x, wires=0)
      return qml.expval(qml.PauliZ(0)), qml.probs(0)
  ```

  In version 0.29 and earlier of PennyLane, `circuit()` would return a single length-3 array:

  ```pycon
  >>> circuit(0.5)
  tensor([0.87758256, 0.93879128, 0.06120872], requires_grad=True)
  ```

  In versions 0.30 and above, `circuit()` returns a length-2 tuple containing the expectation
  value and probabilities separately:

  ```pycon
  >>> circuit(0.5)
  (tensor(0.87758256, requires_grad=True),
   tensor([0.93879128, 0.06120872], requires_grad=True))
  ```

  You can find
  [more details about this change](https://docs.pennylane.ai/en/stable/introduction/returns.html),
  along with help and troubleshooting tips to solve any issues.
  If you still have questions, comments, or concerns, we encourage you to post on the
  PennyLane [discussion forum](https://discuss.pennylane.ai).

<h4>A bunch of performance tweaks 🏃💨</h4>

* Single-qubit operations that have multi-qubit control can now be decomposed more efficiently
  using fewer CNOT gates.
  [(#3851)](https://github.com/PennyLaneAI/pennylane/pull/3851)

  Three decompositions from [arXiv:2302.06377](https://arxiv.org/abs/2302.06377) are provided and
  compare favourably to the already-available `qml.ops.ctrl_decomp_zyz`:

  ```python
  wires = [0, 1, 2, 3, 4, 5]
  control_wires = wires[1:]

  @qml.qnode(qml.device('default.qubit', wires=6))
  def circuit():
      with qml.QueuingManager.stop_recording():
          # the decomposition does not un-queue the target
          target = qml.RX(np.pi/2, wires=0)
      qml.ops.ctrl_decomp_bisect(target, (1, 2, 3, 4, 5))
      return qml.state()

  print(qml.draw(circuit, expansion_strategy="device")())
  ```
  
  ```
  0: ──H─╭X──U(M0)─╭X──U(M0)†─╭X──U(M0)─╭X──U(M0)†──H─┤  State
  1: ────├●────────│──────────├●────────│─────────────┤  State
  2: ────├●────────│──────────├●────────│─────────────┤  State
  3: ────╰●────────│──────────╰●────────│─────────────┤  State
  4: ──────────────├●───────────────────├●────────────┤  State
  5: ──────────────╰●───────────────────╰●────────────┤  State
  ```

* A new decomposition to `qml.SingleExcitation` has been added that halves the number of
  CNOTs required.
  [(3976)](https://github.com/PennyLaneAI/pennylane/pull/3976)

  ```pycon
  >>> qml.SingleExcitation.compute_decomposition(1.23, wires=(0,1))
  [Adjoint(T(wires=[0])), Hadamard(wires=[0]), S(wires=[0]), 
   Adjoint(T(wires=[1])), Adjoint(S(wires=[1])), Hadamard(wires=[1]),
   CNOT(wires=[1, 0]), RZ(-0.615, wires=[0]), RY(0.615, wires=[1]),
   CNOT(wires=[1, 0]), Adjoint(S(wires=[0])), Hadamard(wires=[0]),
   T(wires=[0]), Hadamard(wires=[1]), S(wires=[1]), T(wires=[1])]
  ```

* The adjoint differentiation method can now be more efficient, avoiding the decomposition of operations
  that can be differentiated directly. Any operation that defines a ``generator()`` can
  be differentiated with the adjoint method.
  [(#3874)](https://github.com/PennyLaneAI/pennylane/pull/3874)

  For example, in version 0.29 the ``qml.CRY`` operation would be decomposed when calculating the
  adjoint-method gradient. Executing the code below shows that this decomposition no longer takes
  place in version 0.30 and ``qml.CRY`` is differentiated directly:

  ```python
  import jax
  from jax import numpy as jnp

  def compute_decomposition(self, phi, wires):
      print("A decomposition has been performed!")
      decomp_ops = [
          qml.RY(phi / 2, wires=wires[1]),
          qml.CNOT(wires=wires),
          qml.RY(-phi / 2, wires=wires[1]),
          qml.CNOT(wires=wires),
      ]
      return decomp_ops

  qml.CRY.compute_decomposition = compute_decomposition

  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev, diff_method="adjoint")
  def circuit(phi):
      qml.Hadamard(wires=0)
      qml.CRY(phi, wires=[0, 1])
      return qml.expval(qml.PauliZ(1))

  phi = jnp.array(0.5)
  jax.grad(circuit)(phi)
  ```

* Derivatives are computed more efficiently when using `jax.jit` with gradient transforms; the
  trainable parameters are now set correctly instead of every parameter having to be set as
  trainable.
  [(#3697)](https://github.com/PennyLaneAI/pennylane/pull/3697)

  In the circuit below, only the derivative with respect to parameter `b` is now calculated:

  ```python
  dev = qml.device("default.qubit", wires=2)

  @qml.qnode(dev, interface="jax-jit")
  def circuit(a, b):
      qml.RX(a, wires=0)
      qml.RY(b, wires=0)
      qml.CNOT(wires=[0, 1])
      return qml.expval(qml.PauliZ(0))

  a = jnp.array(0.4)
  b = jnp.array(0.5)

  jac = jax.jacobian(circuit, argnums=[1])
  jac_jit = jax.jit(jac)

  jac_jit(a, b)
  assert len(circuit.tape.trainable_params) == 1
  ```

<h3>Improvements 🛠</h3>

<h4>Next-generation device API</h4>

In this release and future releases, we will be making changes to our device API with the goal in mind to make 
developing plugins much easier for developers and unlock new device capabilities. Users shouldn't yet feel any of 
these changes when using PennyLane, but here is what has changed this release:

* Several functions in `devices/qubit` have been added or improved:
  - `sample_state`: returns a series of samples based on a given state vector and a number of shots.
    [(#3720)](https://github.com/PennyLaneAI/pennylane/pull/3720)
  - `simulate`: supports measuring expectation values of large observables such as `qml.Hamiltonian`, `qml.SparseHamiltonian`, and `qml.Sum`.
    [(#3759)](https://github.com/PennyLaneAI/pennylane/pull/3759) 
  - `apply_operation`: supports broadcasting.
    [(#3852)](https://github.com/PennyLaneAI/pennylane/pull/3852)
  - `adjoint_jacobian`: supports adjoint differentiation in the new qubit state-vector device.
    [(#3790)](https://github.com/PennyLaneAI/pennylane/pull/3790)
  
* `qml.devices.qubit.preprocess` now allows circuits with non-commuting observables.
  [(#3857)](https://github.com/PennyLaneAI/pennylane/pull/3857)

* `qml.devices.qubit.measure` now computes the expectation values of `Hamiltonian` and `Sum`
  in a backpropagation-compatible way.
  [(#3862)](https://github.com/PennyLaneAI/pennylane/pull/3862/)

<h4>Pulse programming</h4>

* Here are the functions, classes, and more that were added or improved to facilitate simulating ensembles of Rydberg atoms:
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)
  [(#3911)](https://github.com/PennyLaneAI/pennylane/pull/3911)
  [(#3930)](https://github.com/PennyLaneAI/pennylane/pull/3930)
  [(#3936)](https://github.com/PennyLaneAI/pennylane/pull/3936)
  [(#3966)](https://github.com/PennyLaneAI/pennylane/pull/3966)
  [(#3987)](https://github.com/PennyLaneAI/pennylane/pull/3987)
  [(#3889)](https://github.com/PennyLaneAI/pennylane/pull/3889)
  [(#4021)](https://github.com/PennyLaneAI/pennylane/pull/4021)
  - `HardwareHamiltonian`: an internal class that contains additional information about pulses and settings.
  - `rydberg_interaction`: a user-facing function that returns a `HardwareHamiltonian` containing
    the Hamiltonian of the interaction of all the Rydberg atoms.
  - `transmon_interaction`: a user-facing function for constructing
    the Hamiltonian that describes the circuit QED interaction Hamiltonian of superconducting transmon systems.
  - `drive`: a user-facing function function that returns a `ParametrizedHamiltonian` (`HardwareHamiltonian`) containing
    the Hamiltonian of the interaction between a driving electro-magnetic field and a group of qubits.
  - `rydberg_drive`: a user-facing function that returns a `ParametrizedHamiltonian` (`HardwareHamiltonian`) containing
    the Hamiltonian of the interaction between a driving laser field and a group of Rydberg atoms.
  - `max_distance`: a keyword argument added to `qml.pulse.rydberg_interaction` to allow for the removal of negligible contributions from atoms beyond `max_distance` from each other.

* `ParametrizedEvolution` now takes two new Boolean keyword arguments: `return_intermediate` and
  `complementary`. They allow computing intermediate time evolution matrices.
  [(#3900)](https://github.com/PennyLaneAI/pennylane/pull/3900)
  
  Activating `return_intermediate` will return intermediate time evolution steps, for example
  for the matrix of the Operation, or of a quantum circuit when used in a QNode.
  Activating `complementary` will make these intermediate steps be the _remaining_
  time evolution complementary to the output for `complementary=False`.
  See the [docstring](https://docs.pennylane.ai/en/stable/code/api/pennylane.pulse.ParametrizedEvolution.html)
  for details.

* Hardware-compatible pulse sequence gradients with `qml.gradient.stoch_pulse_grad` can now be calculated faster using
  the new keyword argument `use_broadcasting`. Executing a `ParametrizedEvolution` that returns
  intermediate evolutions has increased performance using the state vector ODE solver, as well.
  [(#4000)](https://github.com/PennyLaneAI/pennylane/pull/4000)
  [(#4004)](https://github.com/PennyLaneAI/pennylane/pull/4004)

<h4>Intuitive QNode returns</h4>

* The QNode keyword argument `mode` has been replaced by the boolean `grad_on_execution`.
  [(#3969)](https://github.com/PennyLaneAI/pennylane/pull/3969)

* The `"default.gaussian"` device and parameter-shift CV both support the new return system, but only for single measurements.
  [(#3946)](https://github.com/PennyLaneAI/pennylane/pull/3946)

* Keras and Torch NN modules are now compatible with the new return type system.
  [(#3913)](https://github.com/PennyLaneAI/pennylane/pull/3913)
  [(#3914)](https://github.com/PennyLaneAI/pennylane/pull/3914)

* `DefaultQutrit` now supports the new return system.
  [(#3934)](https://github.com/PennyLaneAI/pennylane/pull/3934)

<h4>Performance improvements</h4>

* The efficiency of `tapering()`, `tapering_hf()` and `clifford()` have been improved.
  [(3942)](https://github.com/PennyLaneAI/pennylane/pull/3942)

* The peak memory requirements of `tapering()` and `tapering_hf()` have been improved when used for larger observables.
  [(3977)](https://github.com/PennyLaneAI/pennylane/pull/3977)

* Pauli arithmetic has been updated to convert to a Hamiltonian more efficiently.
  [(#3939)](https://github.com/PennyLaneAI/pennylane/pull/3939)

* `Operator` has a new Boolean attribute `has_generator`. It returns whether or not the `Operator`
  has a `generator` defined. `has_generator` is used in `qml.operation.has_gen`, which improves its performance
  and extends differentiation support. 
  [(#3875)](https://github.com/PennyLaneAI/pennylane/pull/3875)

* The performance of `CompositeOp` has been significantly improved now that it overrides
  determining whether it is being used with a batch of parameters (see `Operator._check_batching`).
  `Hamiltonian` also now overrides this, but it does nothing since it does not support batching.
  [(#3915)](https://github.com/PennyLaneAI/pennylane/pull/3915)

* The performance of a `Sum` operator has been significantly improved now that `is_hermitian` checks 
  that all coefficients are real if the operator has a pre-computed Pauli representation.
  [(#3915)](https://github.com/PennyLaneAI/pennylane/pull/3915)

* The `coefficients` function and the `visualize` submodule of the `qml.fourier` module
  now allow assigning different degrees for different parameters of the input function.
  [(#3005)](https://github.com/PennyLaneAI/pennylane/pull/3005)

  Previously, the arguments `degree` and `filter_threshold` to `qml.fourier.coefficients` were
  expected to be integers. Now, they can be a sequences of integers with one integer per function
  parameter (i.e. `len(degree)==n_inputs`), resulting in a returned array with shape
  `(2*degrees[0]+1,..., 2*degrees[-1]+1)`.
  The functions in `qml.fourier.visualize` accordingly accept such arrays of coefficients.

<h4>Other improvements</h4>

* A `Shots` class has been added to the `measurements` module to hold shot-related data.
  [(#3682)](https://github.com/PennyLaneAI/pennylane/pull/3682)

* The custom JVP rules in PennyLane also now support non-scalar and mixed-shape tape parameters as
  well as multi-dimensional tape return types, like broadcasted `qml.probs`, for example.
  [(#3766)](https://github.com/PennyLaneAI/pennylane/pull/3766)

* The `qchem.jordan_wigner` function has been extended to support more fermionic operator orders.
  [(#3754)](https://github.com/PennyLaneAI/pennylane/pull/3754)
  [(#3751)](https://github.com/PennyLaneAI/pennylane/pull/3751)

* The `AdaptiveOptimizer` has been updated to use non-default user-defined QNode arguments.
  [(#3765)](https://github.com/PennyLaneAI/pennylane/pull/3765)

* Operators now use `TensorLike` types dunder methods.
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)

* `qml.QubitStateVector.state_vector` now supports broadcasting.
  [(#3852)](https://github.com/PennyLaneAI/pennylane/pull/3852)

* `qml.SparseHamiltonian` can now be applied to any wires in a circuit rather than being restricted to all wires
  in the circuit.
  [(#3888)](https://github.com/PennyLaneAI/pennylane/pull/3888)

* Operators can now be divided by scalars with `/` with the addition of the `Operation.__truediv__` dunder 
  method.
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)

* Printing an instance of `MutualInfoMP` now displays the distribution of the wires between the two subsystems.
  [(#3898)](https://github.com/PennyLaneAI/pennylane/pull/3898)

* `Operator.num_wires` has been changed from an abstract value to `AnyWires`.
  [(#3919)](https://github.com/PennyLaneAI/pennylane/pull/3919)

* `qml.transforms.sum_expand` is not run in `Device.batch_transform` if the device supports `Sum` observables.
  [(#3915)](https://github.com/PennyLaneAI/pennylane/pull/3915)

* The type of `n_electrons` in `qml.qchem.Molecule` has been set to `int`.
  [(#3885)](https://github.com/PennyLaneAI/pennylane/pull/3885)

* Explicit errors have been added to `QutritDevice` if `classical_shadow` or `shadow_expval` is measured.
  [(#3934)](https://github.com/PennyLaneAI/pennylane/pull/3934)

* `QubitDevice` now defines the private `_get_diagonalizing_gates(circuit)` method and uses it when executing circuits.
  This allows devices that inherit from `QubitDevice` to override and customize their definition of diagonalizing gates.
  [(#3938)](https://github.com/PennyLaneAI/pennylane/pull/3938)

* `retworkx` has been renamed to `rustworkx` to accommodate the change in the package name.
  [(#3975)](https://github.com/PennyLaneAI/pennylane/pull/3975)

* `Exp`, `Sum`, `Prod`, and `SProd` operator data is now a flat list instead of nested.
  [(#3958)](https://github.com/PennyLaneAI/pennylane/pull/3958)
  [(#3983)](https://github.com/PennyLaneAI/pennylane/pull/3983)

* `qml.transforms.convert_to_numpy_parameters` has been added to convert a circuit with interface-specific parameters to one
  with only numpy parameters. This transform is designed to replace `qml.tape.Unwrap`.
  [(#3899)](https://github.com/PennyLaneAI/pennylane/pull/3899)

* `qml.operation.WiresEnum.AllWires` is now -2 instead of 0 to avoid the
  ambiguity between `op.num_wires = 0` and `op.num_wires = AllWires`.
  [(#3978)](https://github.com/PennyLaneAI/pennylane/pull/3978)

* Execution code has been updated to use the new `qml.transforms.convert_to_numpy_parameters` instead of `qml.tape.Unwrap`.
  [(#3989)](https://github.com/PennyLaneAI/pennylane/pull/3989)

* A sub-routine of `expand_tape` has been converted into `qml.tape.tape.rotations_and_diagonal_measurements`,
  a helper function that computes rotations and diagonal measurements for a tape with measurements
  with overlapping wires.
  [(#3912)](https://github.com/PennyLaneAI/pennylane/pull/3912)

* Various operators and templates have been updated to ensure that their decompositions only return lists of operators.
  [(#3243)](https://github.com/PennyLaneAI/pennylane/pull/3243)

* The `qml.operation.enable_new_opmath` toggle has been introduced to cause dunder methods to return arithmetic
  operators instead of a `Hamiltonian` or `Tensor`.
  [(#4008)](https://github.com/PennyLaneAI/pennylane/pull/4008)

  ```pycon
  >>> type(qml.PauliX(0) @ qml.PauliZ(1))
  <class 'pennylane.operation.Tensor'>
  >>> qml.operation.enable_new_opmath()
  >>> type(qml.PauliX(0) @ qml.PauliZ(1))
  <class 'pennylane.ops.op_math.prod.Prod'>
  >>> qml.operation.disable_new_opmath()
  >>> type(qml.PauliX(0) @ qml.PauliZ(1))
  <class 'pennylane.operation.Tensor'>
  ```

* A new data class called `Resources` has been added to store resources like the number of gates and
  circuit depth throughout a quantum circuit.
  [(#3981)](https://github.com/PennyLaneAI/pennylane/pull/3981/)

* A new function called `_count_resources()` has been added to count the resources required when executing a 
  `QuantumTape` for a given number of shots.
  [(#3996)](https://github.com/PennyLaneAI/pennylane/pull/3996)

* `QuantumScript.specs` has been modified to make use of the new `Resources` class. This also modifies the 
  output of `qml.specs()`. 
  [(#4015)](https://github.com/PennyLaneAI/pennylane/pull/4015)

* A new class called `ResourcesOperation` has been added to allow users to define operations with custom resource information.
  [(#4026)](https://github.com/PennyLaneAI/pennylane/pull/4026)

  For example, users can define a custom operation by inheriting from this new class:

  ```pycon
  >>> class CustomOp(qml.resource.ResourcesOperation):
  ...     def resources(self):
  ...         return qml.resource.Resources(num_wires=1, num_gates=2,
  ...                                       gate_types={"PauliX": 2})
  ... 
  >>> CustomOp(wires=1)
  CustomOp(wires=[1])
  ```
  
  Then, we can track and display the resources of the workflow using `qml.specs()`:

  ```pycon
  >>> dev = qml.device("default.qubit", wires=[0,1])
  >>> @qml.qnode(dev)
  ... def circ():
  ...     qml.PauliZ(wires=0)
  ...     CustomOp(wires=1)
  ...     return qml.state()
  ... 
  >>> print(qml.specs(circ)()['resources'])
  wires: 2
  gates: 3
  depth: 1
  shots: 0
  gate_types:
  {'PauliZ': 1, 'PauliX': 2}
  ```

* `MeasurementProcess.shape` now accepts a `Shots` object as one of its arguments to reduce exposure to unnecessary
  execution details.
  [(#4012)](https://github.com/PennyLaneAI/pennylane/pull/4012)

<h3>Breaking changes 💔</h3>

* The `seed_recipes` argument has been removed from `qml.classical_shadow` and `qml.shadow_expval`.
  [(#4020)](https://github.com/PennyLaneAI/pennylane/pull/4020)

* The tape method `get_operation` has an updated signature.
  [(#3998)](https://github.com/PennyLaneAI/pennylane/pull/3998)

* Both JIT interfaces are no longer compatible with JAX `>0.4.3` (we raise an error for those versions).
  [(#3877)](https://github.com/PennyLaneAI/pennylane/pull/3877)

* An operation that implements a custom `generator` method, but does not always return a valid generator, also has
  to implement a `has_generator` property that reflects in which scenarios a generator will be returned.
  [(#3875)](https://github.com/PennyLaneAI/pennylane/pull/3875)
 
* Trainable parameters for the Jax interface are the parameters that are `JVPTracer`, defined by setting
  `argnums`. Previously, all JAX tracers, including those used for JIT compilation, were interpreted to be trainable.
  [(#3697)](https://github.com/PennyLaneAI/pennylane/pull/3697)

* The keyword argument `argnums` is now used for gradient transforms using Jax instead of `argnum`.
  `argnum` is automatically converted to `argnums` when using Jax and will no longer be supported in v0.31 of PennyLane.
  [(#3697)](https://github.com/PennyLaneAI/pennylane/pull/3697)
  [(#3847)](https://github.com/PennyLaneAI/pennylane/pull/3847)

* `qml.OrbitalRotation` and, consequently, `qml.GateFabric` are now more consistent with the interleaved Jordan-Wigner ordering.
  Previously, they were consistent with the sequential Jordan-Wigner ordering.
  [(#3861)](https://github.com/PennyLaneAI/pennylane/pull/3861)

* Some `MeasurementProcess` classes can now only be instantiated with arguments that they will actually use.
  For example, you can no longer create `StateMP(qml.PauliX(0))` or `PurityMP(eigvals=(-1,1), wires=Wires(0))`.
  [(#3898)](https://github.com/PennyLaneAI/pennylane/pull/3898)

* `Exp`, `Sum`, `Prod`, and `SProd` operator data is now a flat list, instead of nested.
  [(#3958)](https://github.com/PennyLaneAI/pennylane/pull/3958)
  [(#3983)](https://github.com/PennyLaneAI/pennylane/pull/3983)

* `qml.tape.tape.expand_tape` and, consequentially, `QuantumScript.expand` no longer update the input tape
  with rotations and diagonal measurements. Note that the newly expanded tape that is returned will still have
  the rotations and diagonal measurements.
  [(#3912)](https://github.com/PennyLaneAI/pennylane/pull/3912)

* `qml.Evolution` now initializes the coefficient with a factor of `-1j` instead of `1j`.
  [(#4024)](https://github.com/PennyLaneAI/pennylane/pull/4024)

<h3>Deprecations 👋</h3>

Nothing for this release!

<h3>Documentation 📝</h3>

* The documentation of `QubitUnitary` and `DiagonalQubitUnitary` was clarified regarding the
  parameters of the operations.
  [(#4031)](https://github.com/PennyLaneAI/pennylane/pull/4031)

* A typo has been corrected in the documentation for the introduction to `inspecting_circuits` and `chemistry`.
  [(#3844)](https://github.com/PennyLaneAI/pennylane/pull/3844)

* `Usage Details` and `Theory` sections have been separated in the documentation for `qml.qchem.taper_operation`.
  [(3977)](https://github.com/PennyLaneAI/pennylane/pull/3977)

<h3>Bug fixes 🐛</h3>

* `ctrl_decomp_bisect` and `ctrl_decomp_zyz` are no longer used by default when decomposing
  controlled operations due to the presence of a global phase difference in the zyz decomposition of some target operators.
  [(#4065)](https://github.com/PennyLaneAI/pennylane/pull/4065)

* Fixed a bug where `qml.math.dot` returned a numpy array instead of an autograd array, breaking autograd derivatives
  in certain circumstances.
  [(#4019)](https://github.com/PennyLaneAI/pennylane/pull/4019)

* Operators now cast a `tuple` to an `np.ndarray` as well as `list`. 
  [(#4022)](https://github.com/PennyLaneAI/pennylane/pull/4022)

* Fixed a bug where `qml.ctrl` with parametric gates was incompatible with PyTorch tensors on GPUs.
  [(#4002)](https://github.com/PennyLaneAI/pennylane/pull/4002)

* Fixed a bug where the broadcast expand results were stacked along the wrong axis for the new return system.
  [(#3984)](https://github.com/PennyLaneAI/pennylane/pull/3984)

* A more informative error message is raised in `qml.jacobian` to explain potential
  problems with the new return types specification.
  [(#3997)](https://github.com/PennyLaneAI/pennylane/pull/3997)

* Fixed a bug where calling `Evolution.generator` with `coeff` being a complex ArrayBox raised an error.
  [(#3796)](https://github.com/PennyLaneAI/pennylane/pull/3796)
  
* `MeasurementProcess.hash` now uses the hash property of the observable. The property now depends on all
  properties that affect the behaviour of the object, such as `VnEntropyMP.log_base` or the distribution of wires between
  the two subsystems in `MutualInfoMP`.
  [(#3898)](https://github.com/PennyLaneAI/pennylane/pull/3898)

* The enum `measurements.Purity` has been added so that `PurityMP.return_type` is defined. `str` and `repr` for `PurityMP` are also now defined.
  [(#3898)](https://github.com/PennyLaneAI/pennylane/pull/3898)

* `Sum.hash` and `Prod.hash` have been changed slightly
  to work with non-numeric wire labels. `sum_expand` should now return correct results and not treat some products as the same
  operation.
  [(#3898)](https://github.com/PennyLaneAI/pennylane/pull/3898)
  
* Fixed bug where the coefficients where not ordered correctly when summing a `ParametrizedHamiltonian`
  with other operators.
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)
  [(#3902)](https://github.com/PennyLaneAI/pennylane/pull/3902)

* The metric tensor transform is now fully compatible with Jax and therefore users can provide multiple parameters.
  [(#3847)](https://github.com/PennyLaneAI/pennylane/pull/3847)

* `qml.math.ndim` and `qml.math.shape` are now registered for built-ins and autograd to accomodate Autoray 0.6.1.
  [#3864](https://github.com/PennyLaneAI/pennylane/pull/3865)

* Ensured that `qml.data.load` returns datasets in a stable and expected order.
  [(#3856)](https://github.com/PennyLaneAI/pennylane/pull/3856)

* The `qml.equal` function now handles comparisons of `ParametrizedEvolution` operators.
  [(#3870)](https://github.com/PennyLaneAI/pennylane/pull/3870)

* `qml.devices.qubit.apply_operation` catches the `tf.errors.UnimplementedError` that occurs when `PauliZ` or `CNOT` gates
  are applied to a large (>8 wires) tensorflow state. When that occurs, the logic falls back to the tensordot logic instead.
  [(#3884)](https://github.com/PennyLaneAI/pennylane/pull/3884/)

* Fixed parameter broadcasting support with `qml.counts` in most cases and introduced explicit errors otherwise.
  [(#3876)](https://github.com/PennyLaneAI/pennylane/pull/3876)

* An error is now raised if a QNode with Jax-jit in use returns `counts` while having trainable parameters
  [(#3892)](https://github.com/PennyLaneAI/pennylane/pull/3892)

* A correction has been added to the reference values in `test_dipole_of` to account for small changes
  (~`2e-8`) in the computed dipole moment values resulting from the new [PySCF 2.2.0](https://github.com/pyscf/pyscf/releases/tag/v2.2.0) release.
  [(#3908)](https://github.com/PennyLaneAI/pennylane/pull/3908)

* `SampleMP.shape` is now correct when sampling only occurs on a subset of the device wires.
  [(#3921)](https://github.com/PennyLaneAI/pennylane/pull/3921)

* An issue has been fixed in `qchem.Molecule` to allow basis sets other than the hard-coded ones to be
  used in the `Molecule` class.
  [(#3955)](https://github.com/PennyLaneAI/pennylane/pull/3955)

* Fixed bug where all devices that inherit from `DefaultQubit` claimed to support `ParametrizedEvolution`.
  Now, only `DefaultQubitJax` supports the operator, as expected.
  [(#3964)](https://github.com/PennyLaneAI/pennylane/pull/3964)

* Ensured that parallel `AnnotatedQueues` do not queue each other's contents.
  [(#3924)](https://github.com/PennyLaneAI/pennylane/pull/3924)

* Added a `map_wires` method to `PauliWord` and `PauliSentence`, and ensured that operators call
  it in their respective `map_wires` methods if they have a Pauli rep.
  [(#3985)](https://github.com/PennyLaneAI/pennylane/pull/3985)

* Fixed a bug when a `Tensor` is multiplied by a `Hamiltonian` or vice versa.
  [(#4036)](https://github.com/PennyLaneAI/pennylane/pull/4036)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Komi Amiko,
Utkarsh Azad,
Thomas Bromley,
Isaac De Vlugt,
Olivia Di Matteo,
Lillian M. A. Frederiksen,
Diego Guala,
Soran Jahangiri,
Korbinian Kottmann,
Christina Lee,
Vincent Michaud-Rioux,
Albert Mitjans Coma,
Romain Moyard,
Lee J. O'Riordan,
Mudit Pandey,
Matthew Silverman,
Jay Soni,
David Wierichs.
