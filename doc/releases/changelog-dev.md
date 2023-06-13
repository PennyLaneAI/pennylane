:orphan:

# Release 0.31.0-dev (development release)

<h3>New features since last release</h3>

<h4>Fermionic operators 🔬</h4>

* A new class called `Fermiword` has been added to represent a fermionic operator (e.g., $\hat{c}_1 c_0 \hat{c}_2 c_3$).
  [(#4191)](https://github.com/PennyLaneAI/pennylane/pull/4191)

* A new class called `FermiSentence` has been added to represent a linear combination of fermionic operators.
  [(#4195)](https://github.com/PennyLaneAI/pennylane/pull/4195)

<h4>Workflow-level resource estimation 🧮</h4>

* PennyLane's [Tracker](https://docs.pennylane.ai/en/stable/code/api/pennylane.Tracker.html) now
  monitors the resource requirements of circuits being executed by the device.
  [(#4045)](https://github.com/PennyLaneAI/pennylane/pull/4045)
  [(#4110)](https://github.com/PennyLaneAI/pennylane/pull/4110)

  Suppose we have a workflow that involves executing circuits with different qubit numbers. We
  can obtain the resource requirements as a function of the number of qubits by executing the 
  workflow with the `Tracker` context:

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
  zeros. Consider the following workflow that takes the gradient of a `50`-qubit circuit:

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

* Custom operations can now be defined that solely include resource requirements — an explicit
  decomposition or matrix representation is not needed.
  [(#4033)](https://github.com/PennyLaneAI/pennylane/pull/4033)

  PennyLane is now able to estimate the total resource requirements of circuits that include one
  or more of these operations, allowing you to estimate requirements for high-level algorithms 
  composed of abstract subroutines. 

  These operations can be defined by inheriting from
  [ResourcesOperation](https://docs.pennylane.ai/en/stable/code/api/pennylane.resource.ResourcesOperation.html)
  and overriding the `resources()` method to return an appropriate
  [Resources](https://docs.pennylane.ai/en/stable/code/api/pennylane.resource.Resources.html)
  object:

  ```python
  class CustomOp(qml.resource.ResourcesOperation):
      def resources(self):
          n = len(self.wires)
          r = qml.resource.Resources(
              num_wires=n,
              num_gates=n ** 2,
              depth=5,
          )
          return r
  ```

  ```pycon
  >>> wires = [0, 1, 2]
  >>> c = CustomOp(wires)
  >>> c.resources()
  wires: 3
  gates: 9
  depth: 5
  shots: Shots(total=None)
  gate_types:
  {}
  gate_sizes:
  {}
  ```
  
  A quantum circuit that contains `CustomOp` can be created and inspected using
  [qml.specs](https://docs.pennylane.ai/en/stable/code/api/pennylane.specs.html):

  ```python
  dev = qml.device("default.qubit", wires=wires)

  @qml.qnode(dev)
  def circ():
      qml.PauliZ(wires=0)
      CustomOp(wires)
      return qml.state()
  ```
  
  ```pycon
  >>> specs = qml.specs(circ)()
  >>> specs["resources"].depth
  6
  ``` 

<h4>Community contributions from UnitaryHack 🤝</h4>

* [ParametrizedHamiltonian](https://docs.pennylane.ai/en/stable/code/api/pennylane.pulse.ParametrizedHamiltonian.html)
  now has an improved string representation.
  [(#4176)](https://github.com/PennyLaneAI/pennylane/pull/4176)

  ```pycon
  >>> def f1(p, t): return p[0] * jnp.sin(p[1] * t)
  >>> def f2(p, t): return p * t
  >>> coeffs = [2., f1, f2]
  >>> observables =  [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]
  >>> qml.dot(coeffs, observables)
    (2.0*(PauliX(wires=[0])))
  + (f1(params_0, t)*(PauliY(wires=[0])))
  + (f2(params_1, t)*(PauliZ(wires=[0])))
  ```

<h4>Trace distance is now available in qml.qinfo 💥</h4>

* The quantum information module now supports [trace distance](https://en.wikipedia.org/wiki/Trace_distance).
  [(#4181)](https://github.com/PennyLaneAI/pennylane/pull/4181)

  Two cases are enabled for calculating the trace distance:
  
  - A QNode transform via `qinfo.trace_distance`:

    ```python
    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(param):
        qml.RY(param, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()
    ```

    ```pycon
    >>> trace_distance_circuit = qml.qinfo.trace_distance(circuit, circuit, wires0=[0], wires1=[0])
    >>> x, y = np.array(0.4), np.array(0.6)
    >>> trace_distance_circuit((x,), (y,))
    0.047862689546603415
    ```

  - Flexible post-processing via `math.trace_distance`:

    ```pycon
    >>> rho = np.array([[0.3, 0], [0, 0.7]])
    >>> sigma = np.array([[0.5, 0], [0, 0.5]])
    >>> qml.math.trace_distance(rho, sigma)
    0.19999999999999998
    ```

* It is now possible to use basis-state preparations in Qutrit circuits.
  [(#4185)](https://github.com/PennyLaneAI/pennylane/pull/4185)

  ```python
  wires = range(2)
  dev = qml.device("default.qutrit", wires=wires)

  @qml.qnode(dev)
  def qutrit_circuit():
      qml.QutritBasisState([1, 1], wires=wires)
      qml.TAdd(wires=wires)
      return qml.probs(wires=1)
  ```
  
  ```pycon
  >>> qutrit_circuit()
  array([0., 0., 1.])
  ```

* Added the `one_qubit_decomposition` function to provide a unified interface for decompositions
  of a single-qubit unitary matrix into sequences of X, Y, and Z rotations. 
  [(#4210)](https://github.com/PennyLaneAI/pennylane/pull/4210)

  ```pycon
  >>> from pennylane.transforms import one_qubit_decomposition
  >>> U = np.array([[-0.28829348-0.78829734j,  0.30364367+0.45085995j],
  ...               [ 0.53396245-0.10177564j,  0.76279558-0.35024096j]])
  >>> one_qubit_decomposition(U, 0, "ZYZ")
  [RZ(array(-0.2420953), wires=[0]),
   RY(array(1.14938178), wires=[0]),
   RZ(array(1.73305815), wires=[0])]
  >>> one_qubit_decomposition(U, 0, "XYX", return_global_phase=True)
  [RX(tensor(-1.72101925, requires_grad=True), wires=[0]),
   RY(tensor(1.39749741, requires_grad=True), wires=[0]),
   RX(tensor(0.45246584, requires_grad=True), wires=[0]),
   (0.38469215914523336-0.9230449299422961j)*(Identity(wires=[0]))]
  ```

* PennyLane Docker builds have been updated to include the latest plugins and interface versions.
  [(#4178)](https://github.com/PennyLaneAI/pennylane/pull/4178)

<h4>Extended support for differentiating pulses</h4>

* `pulse.ParametrizedEvolution` now uses _batched_ compressed sparse row (`BCSR`) format. 
  [(#4126)](https://github.com/PennyLaneAI/pennylane/pull/4126)

  This allows for computing Jacobians of the unitary directly even when `dense=False`.
  
  ```python
  def U(params):
      H = jnp.polyval * qml.PauliZ(0) # time dependent Hamiltonian
      Um = qml.evolve(H)(params, t=10., dense=False)
      return qml.matrix(Um)
  params = jnp.array([[0.5]], dtype=complex)
  jac = jax.jacobian(U, holomorphic=True)(params)
  ```

* The stochastic parameter-shift gradient transform for pulses, `stoch_pulse_grad`, now
  supports arbitrary Hermitian generating terms in pulse Hamiltonians.
  [(4132)](https://github.com/PennyLaneAI/pennylane/pull/4132)

<h4>Broadcasting and other tweaks to Torch and Keras layers 🦾</h4>

* The `TorchLayer` and `KerasLayer` integrations with `torch.nn` and `Keras` have been upgraded.
  Consider the `TorchLayer`:

  ```python
  n_qubits = 2
  dev = qml.device("default.qubit", wires=n_qubits)

  @qml.qnode(dev)
  def qnode(inputs, weights):
      qml.AngleEmbedding(inputs, wires=range(n_qubits))
      qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
      return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

  n_layers = 6
  weight_shapes = {"weights": (n_layers, n_qubits)}
  qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
  ```
  
  The following features are now available:

  - Native support for parameter broadcasting.
    [(#4131)](https://github.com/PennyLaneAI/pennylane/pull/4131)

    ```pycon
    >>> batch_size = 10
    >>> inputs = torch.rand((batch_size, n_qubits))
    >>> qlayer(inputs)
    >>> assert dev.num_executions == 1
    ```

  - Ability to draw a `TorchLayer` and `KerasLayer` using `qml.draw()` and
    `qml.draw_mpl()`.
    [(#4197)](https://github.com/PennyLaneAI/pennylane/pull/4197)

    ```pycon
    >>> print(qml.draw(qlayer, show_matrices=False)(inputs))
    0: ─╭AngleEmbedding(M0)─╭BasicEntanglerLayers(M1)─┤  <Z>
    1: ─╰AngleEmbedding(M0)─╰BasicEntanglerLayers(M1)─┤  <Z>
    ```

  - Support for `KerasLayer` model saving and clearer instructions on `TorchLayer` model saving.
    [(#4149)](https://github.com/PennyLaneAI/pennylane/pull/4149)
    [(#4158)](https://github.com/PennyLaneAI/pennylane/pull/4158)

    ```pycon
    >>> torch.save(qlayer.state_dict(), "weights.pt")  # Saving
    >>> qlayer.load_state_dict(torch.load("weights.pt"))  # Loading
    >>> qlayer.eval()
    ```
    
    Hybrid models containing `KerasLayer` or `TorchLayer` objects can also be saved and loaded.

<h3>Improvements 🛠</h3>

<h4>A more flexible projector</h4>

<h4>Do more with qutrits</h4>

* Qutrit devices now support parameter-shift differentiation.
  [(#2845)](https://github.com/PennyLaneAI/pennylane/pull/2845)

* Three qutrit rotation operators have been added that are analogous to `RX`, `RY`, and `RZ`:

  - `TRX`: an X rotation
  - `TRY`: a Y rotation
  - `TRZ`: a Z rotation

  [(#2845)](https://github.com/PennyLaneAI/pennylane/pull/2845)
  [(#2846)](https://github.com/PennyLaneAI/pennylane/pull/2846)
  [(#2847)](https://github.com/PennyLaneAI/pennylane/pull/2847)

<h4>The qchem module</h4>

* Non-cubic lattice support for all electron resource estimation has been added.
  [(3956)](https://github.com/PennyLaneAI/pennylane/pull/3956)

* The `qchem.molecular_hamiltonian` function has been upgraded to support custom wires for constructing
  differentiable Hamiltonians. The zero imaginary component of the Hamiltonian coefficients have been
  removed.
  [(#4050)](https://github.com/PennyLaneAI/pennylane/pull/4050)
  [(#4094)](https://github.com/PennyLaneAI/pennylane/pull/4094)

* An error is now raised by `qchem.molecular_hamiltonian` when the `dhf` method is used for an 
  open-shell system. This duplicates a similar error in `qchem.Molecule` but makes it clear
  that the `pyscf` backend can be used for open-shell calculations.
  [(#4058)](https://github.com/PennyLaneAI/pennylane/pull/4058)

* Jordan-Wigner transforms that cache Pauli gate objects have been accelerated.
  [(#4046)](https://github.com/PennyLaneAI/pennylane/pull/4046)

<h4>Next-generation device API</h4>

* The new device interface has been integrated with `qml.execute` for autograd, backpropagation, and no differentiation.
  [(#3903)](https://github.com/PennyLaneAI/pennylane/pull/3903)

* Support for adjoint differentiation has been added to the `DefaultQubit2` device.
  [(#4037)](https://github.com/PennyLaneAI/pennylane/pull/4037)

* A new function called `measure_with_samples` that returns a sample-based measurement result given a state has been added.
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

* The `DefaultQubit2` device now has a `seed` keyword argument.
  [(#4120)](https://github.com/PennyLaneAI/pennylane/pull/4120)

* The new device interface for Jax has been integrated with `qml.execute`.
  [(#4137)](https://github.com/PennyLaneAI/pennylane/pull/4137)

* The experimental device `devices.experimental.DefaultQubit2` now supports `qml.Snapshot`.
  [(#4193)](https://github.com/PennyLaneAI/pennylane/pull/4193)

<h4>Handling shots</h4>

* `QuantumScript` now has a `shots` property, allowing shots to be tied to executions instead of devices.
  [(#4067)](https://github.com/PennyLaneAI/pennylane/pull/4067)
  [(#4103)](https://github.com/PennyLaneAI/pennylane/pull/4103)
  [(#4106)](https://github.com/PennyLaneAI/pennylane/pull/4106)
  [(#4112)](https://github.com/PennyLaneAI/pennylane/pull/4112)

* Several Python built-in functions are now properly defined for instances of the `Shots` class:
  - `print`: printing `Shots` instances is now human-readable
  - `str`: converting `Shots` instances to human-readable strings
  - `==`: equating two different `Shots` instances
  - `hash`: obtaining the hash values of `Shots` instances
  [(#4081)](https://github.com/PennyLaneAI/pennylane/pull/4081)
  [(#4082)](https://github.com/PennyLaneAI/pennylane/pull/4082)

* `qml.devices.ExecutionConfig` no longer has a `shots` property, as it is now on the `QuantumScript`. It now has a `use_device_gradient` property. `ExecutionConfig.grad_on_execution = None` indicates a request for `"best"` instead of a string.
  [(#4102)](https://github.com/PennyLaneAI/pennylane/pull/4102)

* `QuantumScript.shots` has been integrated with QNodes so that shots are placed on the `QuantumScript`
  during `QNode` construction.
  [(#4110)](https://github.com/PennyLaneAI/pennylane/pull/4110)

* The `gradients` module has been updated to use the new `Shots` object internally
  [(#4152)](https://github.com/PennyLaneAI/pennylane/pull/4152)

<h4>Operators</h4>

* `qml.prod` now accepts a single quantum function input for creating new `Prod` operators.
  [(#4011)](https://github.com/PennyLaneAI/pennylane/pull/4011)

* `DiagonalQubitUnitary` now decomposes into `RZ`, `IsingZZ` and `MultiRZ` gates
  instead of a `QubitUnitary` operation with a dense matrix.
  [(#4035)](https://github.com/PennyLaneAI/pennylane/pull/4035)

* All objects being queued in an `AnnotatedQueue` are now wrapped so that `AnnotatedQueue` is not 
  dependent on the has of any operators or measurement processes.
  [(#4087)](https://github.com/PennyLaneAI/pennylane/pull/4087)

* A `dense` keyword to `ParametrizedEvolution` that allows forcing dense or sparse matrices has been added.
  [(#4079)](https://github.com/PennyLaneAI/pennylane/pull/4079)
  [(#4095)](https://github.com/PennyLaneAI/pennylane/pull/4095)

* A new function called `qml.ops.functions.bind_new_parameters` that creates a copy of an operator with new parameters
  without mutating the original operator has been added.
  [(#4113)](https://github.com/PennyLaneAI/pennylane/pull/4113)

* `qml.CY` has been moved from `qml.ops.qubit.non_parametric_ops` to `qml.ops.op_math.controlled_ops`
  and now inherits from `qml.ops.op_math.ControlledOp`.
  [(#4116)](https://github.com/PennyLaneAI/pennylane/pull/4116/)

* `CZ` now inherits from the `ControlledOp` class and supports exponentiation to arbitrary powers with `pow`, which is no longer limited to integers. It also supports `sparse_matrix` and `decomposition` representations.
  [(#4117)](https://github.com/PennyLaneAI/pennylane/pull/4117)

* The construction of the Pauli representation for the `Sum` class is now faster.
  [(#4142)](https://github.com/PennyLaneAI/pennylane/pull/4142)

* `qml.drawer.drawable_layers.drawable_layers` and `qml.CircuitGraph` have been updated to not rely on `Operator`
  equality or hash to work correctly.
  [(#4143)](https://github.com/PennyLaneAI/pennylane/pull/4143)

<h4>Other improvements</h4>

* Added broadcasting support for `qml.qinfo.reduced_dm`, `qml.qinfo.purity`, `qml.qinfo.vn_entropy`,
  `qml.qinfo.mutual_info`, `qml.qinfo.fidelity`, `qml.qinfo.relative_entropy`, and `qml.qinfo.trace_distance`.
  [(#4234)](https://github.com/PennyLaneAI/pennylane/pull/4234)

* Added broadcasting support for `qml.math.purity`, `qml.math.vn_entropy`, `qml.math.mutual_info`, `qml.math.fidelity`,
  `qml.math.relative_entropy`, `qml.math.max_entropy`, and `qml.math.sqrt_matrix`.
  [(#4186)](https://github.com/PennyLaneAI/pennylane/pull/4186)

* `pulse.ParametrizedEvolution` now raises an error if the number of input parameters does not match the number
  of parametrized coefficients in the `ParametrizedHamiltonian` that generates it. An exception is made for
  `HardwareHamiltonian`s which are not checked.
  [(#4216)](https://github.com/PennyLaneAI/pennylane/pull/4216)

* One qubit unitaries can now be decomposed into a `ZXZ` gate sequence (apart from the pre-existing `XYX` and `ZYZ`).
  [(#4210)](https://github.com/PennyLaneAI/pennylane/pull/4210)

* The default value for the `show_matrices` keyword argument in all drawing methods is now `True`. 
  This allows for quick insights into broadcasted tapes, for example.
  [(#3920)](https://github.com/PennyLaneAI/pennylane/pull/3920)

* Type variables for `qml.typing.Result` and `qml.typing.ResultBatch` have been added for type hinting the result of an execution.
  [(#4018)](https://github.com/PennyLaneAI/pennylane/pull/4108)
  
* The Jax-JIT interface now uses symbolic zeros to determine trainable parameters.
  [(4075)](https://github.com/PennyLaneAI/pennylane/pull/4075)

* A new function called `pauli.pauli_word_prefactor()` that extracts the prefactor for a given Pauli word has been added.
  [(#4164)](https://github.com/PennyLaneAI/pennylane/pull/4164)

* Reduced density matrix functionality has been added via `qml.math.reduce_dm` and `qml.math.reduce_statevector`.
  Both functions have broadcasting support.
  [(#4173)](https://github.com/PennyLaneAI/pennylane/pull/4173)

<h3>Breaking changes 💔</h3>

* The default value for the `show_matrices` keyword argument in all drawing methods is now `True`. 
  This allows for quick insights into broadcasted tapes, for example.
  [(#3920)](https://github.com/PennyLaneAI/pennylane/pull/3920)

* `DiagonalQubitUnitary` now decomposes into `RZ`, `IsingZZ`, and `MultiRZ` gates rather than a `QubitUnitary`.
  [(#4035)](https://github.com/PennyLaneAI/pennylane/pull/4035)

* Jax trainable parameters are now `Tracer` instead of `JVPTracer`, it is not always the right definition for the JIT 
  interface, but we update them in the custom JVP using symbolic zeros.
  [(4075)](https://github.com/PennyLaneAI/pennylane/pull/4075)

* The experimental Device interface `qml.devices.experimental.Device` now requires that the `preprocess` method
  also returns an `ExecutionConfig` object. This allows the device to choose what `"best"` means for various
  hyperparameters like `gradient_method` and `grad_on_execution`.
  [(#4007)](https://github.com/PennyLaneAI/pennylane/pull/4007)
  [(#4102)](https://github.com/PennyLaneAI/pennylane/pull/4102)

* Gradient transforms with Jax no longer support `argnum`. Use `argnums` instead.
  [(#4076)](https://github.com/PennyLaneAI/pennylane/pull/4076)

* `qml.collections`, `qml.op_sum`, and `qml.utils.sparse_hamiltonian` have been removed.

<h3>Deprecations 👋</h3>

* `LieAlgebraOptimizer` has been renamed to `RiemannianGradientOptimizer`.
  [(#4153)(https://github.com/PennyLaneAI/pennylane/pull/4153)]

* `Operation.base_name` has been deprecated. Please use `Operation.name` or `type(op).__name__` instead.

* `QuantumScript`'s `name` keyword argument and property have been deprecated.
  This also affects `QuantumTape` and `OperationRecorder`.
  [(#4141)](https://github.com/PennyLaneAI/pennylane/pull/4141)

* The `qml.grouping` module has been removed. Its functionality has been reorganized in the `qml.pauli` module.

* `qml.math.reduced_dm` has been deprecated. Please use `qml.math.reduce_dm` or `qml.math.reduce_statevector` instead.
  [(#4173)](https://github.com/PennyLaneAI/pennylane/pull/4173)

* `qml.math.purity`, `qml.math.vn_entropy`, `qml.math.mutual_info`, `qml.math.fidelity`,
  `qml.math.relative_entropy`, and `qml.math.max_entropy` no longer support state vectors as
  input. Please call `qml.math.dm_from_state_vector` on the input before passing to any of these functions.
  [(#4186)](https://github.com/PennyLaneAI/pennylane/pull/4186)

* The `do_queue` keyword argument in `qml.operation.Operator` has been deprecated. Instead of
  setting `do_queue=False`, use the `qml.QueuingManager.stop_recording()` context.
  [(#4148)](https://github.com/PennyLaneAI/pennylane/pull/4148)

* `zyz_decomposition` and `xyx_decomposition` are now deprecated in favour of `one_qubit_decomposition`.
  [(#4230)](https://github.com/PennyLaneAI/pennylane/pull/4230)

<h3>Documentation 📝</h3>

* The docstring for `qml.ops.op_math.Pow.__new__` is now complete and it has been updated along with
  `qml.ops.op_math.Adjoint.__new__`.
  [(#4231)](https://github.com/PennyLaneAI/pennylane/pull/4231)

* The docstring for `qml.grad` now states that it should be used with the Autograd interface only.
  [(#4202)](https://github.com/PennyLaneAI/pennylane/pull/4202)

* The description of `mult` in the `qchem.Molecule` docstring now correctly states the value
  of `mult` that is supported.
  [(#4058)](https://github.com/PennyLaneAI/pennylane/pull/4058)

<h3>Bug fixes 🐛</h3>

* Fixed adjoint jacobian results with `grad_on_execution=False` in the JAX-JIT interface.
  [(4217)](https://github.com/PennyLaneAI/pennylane/pull/4217)

* Fixed a bug where `stoch_pulse_grad` would ignore prefactors of rescaled Pauli words in the
  generating terms of a pulse Hamiltonian.
  [(4156)](https://github.com/PennyLaneAI/pennylane/pull/4156)
  
* Fixed a bug where the wire ordering of the `wires` argument to `qml.density_matrix`
  was not taken into account.
  [(#4072)](https://github.com/PennyLaneAI/pennylane/pull/4072)

* A patch in `interfaces/autograd.py` that checks for the `strawberryfields.gbs` device has been removed. 
  That device is pinned to PennyLane <= v0.29.0, so that patch is no longer necessary.

* `qml.pauli.are_identical_pauli_words` now treats all identities as equal. Identity terms on Hamiltonians with non-standard
  wire orders are no longer eliminated.
  [(#4161)](https://github.com/PennyLaneAI/pennylane/pull/4161)

* `qml.pauli_sentence()` is now compatible with empty Hamiltonians `qml.Hamiltonian([], [])`.
  [(#4171)](https://github.com/PennyLaneAI/pennylane/pull/4171)

* Fixed a bug with Jax where executing multiple tapes with `gradient_fn="device"` would fail.
  [(#4190)](https://github.com/PennyLaneAI/pennylane/pull/4190)

* A more meaningful error message is raised when broadcasting with adjoint differentiation on `DefaultQubit`.
  [(#4203)](https://github.com/PennyLaneAI/pennylane/pull/4203)

* Fixed a bug where `op = qml.qsvt()` was incorrect up to a global phase when using `convention="Wx""` and `qml.matrix(op)`.
  [(#4214)](https://github.com/PennyLaneAI/pennylane/pull/4214)

* Fixed buggy calculation of angle in `xyx_decomposition` causing it to give an incorrect decomposition.
  An if conditional was intended to prevent divide by zero errors but the division was by the sine of the argument so any multiple of $\pi$ should trigger the conditional, but it was only checking if the argument was 0. Example: `qml.Rot(2.3, 2.3, 2.3)`
  [(#4210)](https://github.com/PennyLaneAI/pennylane/pull/4210)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Venkatakrishnan AnushKrishna,
Thomas Bromley,
Isaac De Vlugt,
Emiliano Godinez Ramirez
Nikhil Harle
Soran Jahangiri,
Edward Jiang,
Korbinian Kottmann,
Christina Lee,
Vincent Michaud-Rioux,
Romain Moyard,
Tristan Nemoz,
Mudit Pandey,
Manul Patel,
Borja Requena,
Modjtaba Shokrian-Zini,
Mainak Roy,
Matthew Silverman,
Jay Soni,
David Wierichs,
Frederik Wilde.
