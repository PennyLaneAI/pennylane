:orphan:

# Release 0.30.0-dev (development release)

<h3>New features since last release</h3>

<h4>Pulse programming on hardware ⚛️🔬</h4>

* Support for loading time-dependent Hamiltonians that are compatible with quantum hardware has been
  added. It is now possible to load a Hamiltonian that describes an ensemble of Rydberg atoms or a
  collection of transmon qubits.
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)
  [(#3911)](https://github.com/PennyLaneAI/pennylane/pull/3911)
  [(#3930)](https://github.com/PennyLaneAI/pennylane/pull/3930)
  [(#3936)](https://github.com/PennyLaneAI/pennylane/pull/3936)
  [(#3966)](https://github.com/PennyLaneAI/pennylane/pull/3966)
  [(#3987)](https://github.com/PennyLaneAI/pennylane/pull/3987)

  A Rydberg system Hamiltonian can be constructed from a
  [drive term](https://docs.pennylane.ai/en/stable/code/api/pennylane.pulse.rydberg_drive.html)
  and an
  [interaction term](https://docs.pennylane.ai/en/stable/code/api/pennylane.pulse.rydberg_interaction.html):

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
  
  The time-dependent Hamiltonian `H` can be used in a PennyLane pulse-level circuit:

  ```python
  dev = qml.device("default.qubit.jax", wires=wires)

  @qml.qnode(dev, interface="jax")
  def circuit(params):
      qml.evolve(H)(params, t=[0, 10])
      return qml.expval(qml.PauliZ(0))
  ```
  
  This circuit can be executed and differentiated:

  ```pycon
  >>> params = jnp.array([2.4])
  >>> circuit(params)
  Array(0.94307977, dtype=float32)
  >>> import jax
  >>> jax.grad(circuit)(params)
  Array([0.5940717], dtype=float32)
  ```
  
  The [qml.pulse](https://docs.pennylane.ai/en/stable/code/qml_pulse.html) page contains additional
  details. Check out our
  [release blog post](https://pennylane.ai/blog/2023/04/pennylane-v030-released/) for a
  demonstration of how to perform the execution on actual hardware!  

<h4>Quantum singular value transform 🐛➡️🦋</h4>

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

  Consider a matrix `A` along with a vector `angles` that describes the target polynomial
  transformation. The [qml.qsvt](https://docs.pennylane.ai/en/stable/code/api/pennylane.qsvt.html)
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
  
  This circuit is composed of
  [qml.BlockEncode](https://docs.pennylane.ai/en/stable/code/api/pennylane.BlockEncode.html) and
  [qml.PCPhase](https://docs.pennylane.ai/en/stable/code/api/pennylane.PCPhase.html) operations.

  ```pycon
  >>> qml.draw(example_circuit, expansion_strategy="device")(A)  # TODO
  0: ─╭∏_ϕ─╭BlockEncode(M0)─╭∏_ϕ─╭BlockEncode(M0)†─╭∏_ϕ─┤  
  1: ─╰∏_ϕ─╰BlockEncode(M0)─╰∏_ϕ─╰BlockEncode(M0)†─╰∏_ϕ─┤
  ```

  The [qml.qsvt](https://docs.pennylane.ai/en/stable/code/api/pennylane.qsvt.html) function creates
  a circuit that is targeted at simulators due to the use of matrix-based operations.
  Advanced users are able to use the
  [qml.QSVT](https://docs.pennylane.ai/en/stable/code/api/pennylane.QSVT.html) template to perform
  the transformation with a custom choice of unitary and projector operations, which may be
  hardware compatible if a decomposition is provided.

  The QSVT is a complex but powerful transformation capable of
  [generalizing important algorithms](https://arxiv.org/abs/2105.02859)
  like amplitude amplification. Stay tuned for a demo in the coming few weeks to learn more!

<h4>Intuitive QNode returns</h4>

* The new return system is now activated and public-facing. The QNode keyword argument `mode` is replaced by the boolean  
  `grad_on_execution`.
  [(#3957)](https://github.com/PennyLaneAI/pennylane/pull/3957)
  [(#3969)](https://github.com/PennyLaneAI/pennylane/pull/3969)

<h3>Improvements 🛠</h3>

<h4>Next-generation device API</h4>

* The `sample_state` function is added to `devices/qubit` that returns a series of samples based on a given
  state vector and a number of shots.
  [(#3720)](https://github.com/PennyLaneAI/pennylane/pull/3720)

* The `simulate` function added to `devices/qubit` now supports measuring expectation values of large observables such as
  `qml.Hamiltonian`, `qml.SparseHamiltonian`, `qml.Sum`.
  [(#3759)](https://github.com/PennyLaneAI/pennylane/pull/3759)

* The `apply_operation` function added to `devices/qubit` now supports broadcasting.
  [(#3852)](https://github.com/PennyLaneAI/pennylane/pull/3852)

* `pennylane.devices.qubit.preprocess` now allows circuits with non-commuting observables.
  [(#3857)](https://github.com/PennyLaneAI/pennylane/pull/3857)

* Adjoint differentiation support for the new qubit state-vector device has been added via
  `adjoint_jacobian` in `devices/qubit`.
  [(#3790)](https://github.com/PennyLaneAI/pennylane/pull/3790)

* `qml.devices.qubit.measure` now computes the expectation values of `Hamiltonian` and `Sum`
  in a backpropagation-compatible way.
  [(#3862)](https://github.com/PennyLaneAI/pennylane/pull/3862/)

<h4>Performance improvements</h4>

* Added a new decomposition to `qml.SingleExcitation` that halves the number of
  CNOTs required.
  [(3976)](https://github.com/PennyLaneAI/pennylane/pull/3976)

* Improved efficiency of `tapering()`, `tapering_hf()` and `clifford()`.
  [(3942)](https://github.com/PennyLaneAI/pennylane/pull/3942)

* Updated Pauli arithmetic to more efficiently convert to a Hamiltonian.
  [(#3939)](https://github.com/PennyLaneAI/pennylane/pull/3939)

* The adjoint differentiation method now supports more operations, and does no longer decompose
  some operations that may be differentiated directly. In addition, all new operations with a
  generator are now supported by the method.
  [(#3874)](https://github.com/PennyLaneAI/pennylane/pull/3874)

* When using `jax.jit` with gradient transforms, the trainable parameters are set correctly (instead of every parameter having
  to be set as trainable), and therefore the derivatives are computed more efficiently.
  [(#3697)](https://github.com/PennyLaneAI/pennylane/pull/3697)

* `CompositeOp` now overrides `Operator._check_batching`, providing a significant performance improvement.
  `Hamiltonian` also overrides this method and does nothing, because it does not support batching.
  [(#3915)](https://github.com/PennyLaneAI/pennylane/pull/3915)

* If a `Sum` operator has a pre-computed Pauli representation, `is_hermitian` now checks that all coefficients
  are real, providing a significant performance improvement.
  [(#3915)](https://github.com/PennyLaneAI/pennylane/pull/3915)

* Three new decomposition algorithms have been added for n-controlled operations with a single-qubit target
  and are selected automatically when they produce a better result, i.e., fewer CNOT gates.
  They can be accessed via `ops.op_math.ctrl_decomp_bisect`.
  [(#3851)](https://github.com/PennyLaneAI/pennylane/pull/3851)

<h4>Pulse programming on hardware</h4>

* Added the needed functions and classes to simulate an ensemble of Rydberg atoms:
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)
  [(#3911)](https://github.com/PennyLaneAI/pennylane/pull/3911)
  [(#3930)](https://github.com/PennyLaneAI/pennylane/pull/3930)
  [(#3936)](https://github.com/PennyLaneAI/pennylane/pull/3936)
  [(#3966)](https://github.com/PennyLaneAI/pennylane/pull/3966)
  [(#3987)](https://github.com/PennyLaneAI/pennylane/pull/3987)
  [(#3889)](https://github.com/PennyLaneAI/pennylane/pull/3889)
  * A new internal `HardwareHamiltonian` class is added, which contains additional information about pulses and settings.
  * A new user-facing `rydberg_interaction` function is added, which returns a `HardwareHamiltonian` containing
    the Hamiltonian of the interaction of all the Rydberg atoms.
  * A new user-facing `transmon_interaction` function is added, constructing
    the Hamiltonian that describes the circuit QED interaction Hamiltonian of superconducting transmon systems.
  * A new user-facing `drive` function is added, which returns a `ParametrizedHamiltonian` (`HardwareHamiltonian`) containing
    the Hamiltonian of the interaction between a driving electro-magnetic field and a group of qubits.
  * A new user-facing `rydberg_drive` function is added, which returns a `ParametrizedHamiltonian` (`HardwareHamiltonian`) containing
    the Hamiltonian of the interaction between a driving laser field and a group of Rydberg atoms.
  * A new keyword argument called `max_distance` has been added to `qml.pulse.rydberg_interaction` to allow for the removal of negligible contributions from atoms beyond `max_distance` from each other.

<h4>Intuitive QNode returns</h4>

* The default Gaussian device and parameter shift CV support the new return system, but only for single measurements.
  [(3946)](https://github.com/PennyLaneAI/pennylane/pull/3946)

* Keras and Torch NN modules are now compatible with the new return type system.
  [(#3913)](https://github.com/PennyLaneAI/pennylane/pull/3913)
  [(#3914)](https://github.com/PennyLaneAI/pennylane/pull/3914)

* `DefaultQutrit` supports the new return system.
  [(#3934)](https://github.com/PennyLaneAI/pennylane/pull/3934)

  [(3946)](https://github.com/PennyLaneAI/pennylane/pull/3946)

<h4>Other improvements</h4>

* Added a `Shots` class to the `measurements` module to hold shot-related data.
  [(#3682)](https://github.com/PennyLaneAI/pennylane/pull/3682)

* The `coefficients` function and the `visualize` submodule of the `qml.fourier` module
  now allow assigning different degrees for different parameters of the input function.
  [(#3005)](https://github.com/PennyLaneAI/pennylane/pull/3005)

  The arguments `degree` and `filter_threshold` to `qml.fourier.coefficients` previously were
  expected to be integers, and now can be a sequences of integers with one integer per function
  parameter (i.e. `len(degree)==n_inputs`), resulting in a returned array with shape
  `(2*degrees[0]+1,..., 2*degrees[-1]+1)`.
  The functions in `qml.fourier.visualize` accordingly accept such arrays of coefficients.

* `Operator` now has a `has_generator` attribute that returns whether or not the operator
  has a generator defined. It is used in `qml.operation.has_gen`, improving its performance.
  [(#3875)](https://github.com/PennyLaneAI/pennylane/pull/3875)

* The custom JVP rules in PennyLane now also support non-scalar and mixed-shape tape parameters as
  well as multi-dimensional tape return types, like broadcasted `qml.probs`, for example.
  [(#3766)](https://github.com/PennyLaneAI/pennylane/pull/3766)

* The `qchem.jordan_wigner` function is extended to support more fermionic operator orders.
  [(#3754)](https://github.com/PennyLaneAI/pennylane/pull/3754)
  [(#3751)](https://github.com/PennyLaneAI/pennylane/pull/3751)

* `AdaptiveOptimizer` is updated to use non-default user-defined qnode arguments.
  [(#3765)](https://github.com/PennyLaneAI/pennylane/pull/3765)

* Use `TensorLike` type in `Operator` dunder methods.
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)

* `qml.QubitStateVector.state_vector` now supports broadcasting.
  [(#3852)](https://github.com/PennyLaneAI/pennylane/pull/3852)

* `qml.SparseHamiltonian` can now be applied to any wires in a circuit rather than being restricted to all wires
  in the circuit.
  [(#3888)](https://github.com/PennyLaneAI/pennylane/pull/3888)

* Added `Operation.__truediv__` dunder method to be able to divide operators.
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)

* `repr` for `MutualInfoMP` now displays the distribution of the wires between the two subsystems.
  [(#3898)](https://github.com/PennyLaneAI/pennylane/pull/3898)

* Changed `Operator.num_wires` from an abstract value to `AnyWires`.
  [(#3919)](https://github.com/PennyLaneAI/pennylane/pull/3919)

* `qml.transforms.sum_expand` is not run in `Device.batch_transform` if the device supports Sum observables.
  [(#3915)](https://github.com/PennyLaneAI/pennylane/pull/3915)

* The type of `n_electrons` in `qml.qchem.Molecule` has been set to `int`.
  [(#3885)](https://github.com/PennyLaneAI/pennylane/pull/3885)

* Added explicit errors to `QutritDevice` if `classical_shadow` or `shadow_expval` are measured.
  [(#3934)](https://github.com/PennyLaneAI/pennylane/pull/3934)

* `QubitDevice` now defines the private `_get_diagonalizing_gates(circuit)` method and uses it when executing circuits.
  This allows devices that inherit from `QubitDevice` to override and customize their definition of diagonalizing gates.
  [(#3938)](https://github.com/PennyLaneAI/pennylane/pull/3938)

* `retworkx` has been renamed to `rustworkx` to accommodate the change in the package name.
  [(#3975)](https://github.com/PennyLaneAI/pennylane/pull/3975)

* `Exp`, `Sum`, `Prod`, and `SProd` operator data is now a flat list, instead of nested.
  [(#3958)](https://github.com/PennyLaneAI/pennylane/pull/3958)
  [(#3983)](https://github.com/PennyLaneAI/pennylane/pull/3983)

* `qml.transforms.convert_to_numpy_parameters` is added to convert a circuit with interface-specific parameters to one
  with only numpy parameters. This transform is designed to replace `qml.tape.Unwrap`.
  [(#3899)](https://github.com/PennyLaneAI/pennylane/pull/3899)

* `qml.operation.WiresEnum.AllWires` is now -2 instead of 0 to avoid the
  ambiguity between `op.num_wires = 0` and `op.num_wires = AllWires`.
  [(#3978)](https://github.com/PennyLaneAI/pennylane/pull/3978)

<h3>Breaking changes 💔</h3>

* Both JIT interfaces are not compatible with JAX `>0.4.3`, we raise an error for those versions.
  [(#3877)](https://github.com/PennyLaneAI/pennylane/pull/3877)

* An operation that implements a custom `generator` method, but does not always return a valid generator, also has
  to implement a `has_generator` property that reflects in which scenarios a generator will be returned.
  [(#3875)](https://github.com/PennyLaneAI/pennylane/pull/3875)
 
* Trainable parameters for the Jax interface are the parameters that are `JVPTracer`, defined by setting
  `argnums`. Previously, all JAX tracers, including those used for JIT compilation, were interpreted to be trainable.
  [(#3697)](https://github.com/PennyLaneAI/pennylane/pull/3697)

* The keyword argument `argnums` is now used for gradient transform using Jax, instead of `argnum`.
  `argnum` is automatically converted to `argnums` when using JAX, and will no longer be supported in v0.31.
  [(#3697)](https://github.com/PennyLaneAI/pennylane/pull/3697)
  [(#3847)](https://github.com/PennyLaneAI/pennylane/pull/3847)

* Made `qml.OrbitalRotation` and consequently `qml.GateFabric` consistent with the interleaved Jordan-Wigner ordering.
  Previously, they were consistent with the sequential Jordan-Wigner ordering.
  [(#3861)](https://github.com/PennyLaneAI/pennylane/pull/3861)

* Some `MeasurementProcess` classes can now only be instantiated with arguments that they will actually use.
  For example, you can no longer create `StateMP(qml.PauliX(0))` or `PurityMP(eigvals=(-1,1), wires=Wires(0))`.
  [(#3898)](https://github.com/PennyLaneAI/pennylane/pull/3898)

* `Exp`, `Sum`, `Prod`, and `SProd` operator data is now a flat list, instead of nested.
  [(#3958)](https://github.com/PennyLaneAI/pennylane/pull/3958)
  [(#3983)](https://github.com/PennyLaneAI/pennylane/pull/3983)

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

* A typo was corrected in the documentation for introduction to `inspecting_circuits` and `chemistry`.
  [(#3844)](https://github.com/PennyLaneAI/pennylane/pull/3844)

<h3>Bug fixes 🐛</h3>

* Fixes a bug where the broadcast expand results where stacked along the wrong axis for the new return system.
  [(#3984)](https://github.com/PennyLaneAI/pennylane/pull/3984)

* Fixed a bug where calling `Evolution.generator` with `coeff` being a complex ArrayBox raised an error.
  [(#3796)](https://github.com/PennyLaneAI/pennylane/pull/3796)
  
* `MeasurementProcess.hash` now uses the hash property of the observable. The property now depends on all
  properties that affect the behaviour of the object, such as `VnEntropyMP.log_base` or the distribution of wires between
  the two subsystems in `MutualInfoMP`.
  [(#3898)](https://github.com/PennyLaneAI/pennylane/pull/3898)

* The enum `measurements.Purity` is added so that `PurityMP.return_type` is defined. `str` and `repr` for `PurityMP` are now defined.
  [(#3898)](https://github.com/PennyLaneAI/pennylane/pull/3898)

* `Sum.hash` and `Prod.hash` are slightly changed
  to work with non-numeric wire labels.  `sum_expand` should now return correct results and not treat some products as the same
  operation.
  [(#3898)](https://github.com/PennyLaneAI/pennylane/pull/3898)
  
* Fixed bug where the coefficients where not ordered correctly when summing a `ParametrizedHamiltonian`
  with other operators.
  [(#3749)](https://github.com/PennyLaneAI/pennylane/pull/3749)
  [(#3902)](https://github.com/PennyLaneAI/pennylane/pull/3902)

* The metric tensor transform is fully compatible with Jax and therefore users can provide multiple parameters.
  [(#3847)](https://github.com/PennyLaneAI/pennylane/pull/3847)

* Registers `math.ndim` and `math.shape` for built-ins and autograd to accomodate Autoray 0.6.1.
  [#3864](https://github.com/PennyLaneAI/pennylane/pull/3865)

* Ensure that `qml.data.load` returns datasets in a stable and expected order.
  [(#3856)](https://github.com/PennyLaneAI/pennylane/pull/3856)

* The `qml.equal` function now handles comparisons of `ParametrizedEvolution` operators.
  [(#3870)](https://github.com/PennyLaneAI/pennylane/pull/3870)

* Made `qml.OrbitalRotation` and consequently `qml.GateFabric` consistent with the interleaved Jordan-Wigner ordering.
  [(#3861)](https://github.com/PennyLaneAI/pennylane/pull/3861)

* `qml.devices.qubit.apply_operation` catches the `tf.errors.UnimplementedError` that occurs when `PauliZ` or `CNOT` gates
  are applied to a large (>8 wires) tensorflow state. When that occurs, the logic falls back to the tensordot logic instead.
  [(#3884)](https://github.com/PennyLaneAI/pennylane/pull/3884/)

* Fixed parameter broadcasting support with `qml.counts` in most cases, and introduced explicit errors otherwise.
  [(#3876)](https://github.com/PennyLaneAI/pennylane/pull/3876)

* An error is now raised if a `QNode` with Jax-jit in use returns `counts` while having trainable parameters
  [(#3892)](https://github.com/PennyLaneAI/pennylane/pull/3892)

* A correction is added to the reference values in `test_dipole_of` to account for small changes
  (~2e-8) in the computed dipole moment values, resulting from the new [PySCF 2.2.0](https://github.com/pyscf/pyscf/releases/tag/v2.2.0) release.
  [(#3908)](https://github.com/PennyLaneAI/pennylane/pull/3908)

* `SampleMP.shape` is now correct when sampling only occurs on a subset of the device wires.
  [(#3921)](https://github.com/PennyLaneAI/pennylane/pull/3921)

* An issue is fixed in `qchem.Molecule` to allow basis sets other than the hard-coded ones to be
  used in the `Molecule` class.
  [(#3955)](https://github.com/PennyLaneAI/pennylane/pull/3955)

* Fixed bug where all devices that inherit from DefaultQubit claimed to support `ParametrizedEvolution`.
  Now only `DefaultQubitJax` supports the operator, as expected.
  [(#3964)](https://github.com/PennyLaneAI/pennylane/pull/3964)

* Ensure that parallel `AnnotatedQueues` do not queue each other's contents.
  [(#3924)](https://github.com/PennyLaneAI/pennylane/pull/3924)

* Added a `map_wires` method to `PauliWord` and `PauliSentence`, and ensured that operators call
  it in their respective `map_wires` methods if they have a Pauli rep.
  [(#3985)](https://github.com/PennyLaneAI/pennylane/pull/3985)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Komi Amiko,
Utkarsh Azad,
Olivia Di Matteo,
Lillian M. A. Frederiksen,
Soran Jahangiri,
Christina Lee,
Vincent Michaud-Rioux,
Albert Mitjans,
Romain Moyard,
Mudit Pandey,
Matthew Silverman,
Jay Soni,
David Wierichs.
