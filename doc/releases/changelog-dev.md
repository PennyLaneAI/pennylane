:orphan:

# Release 0.31.0-dev (development release)

<h3>New features since last release</h3>

<h4>Seamlessly create and combine fermionic operators 🔬</h4>

* Fermionic operators and arithmetic are now available. 
  [(#4191)](https://github.com/PennyLaneAI/pennylane/pull/4191)
  [(#4195)](https://github.com/PennyLaneAI/pennylane/pull/4195)
  [(#4200)](https://github.com/PennyLaneAI/pennylane/pull/4200)
  [(#4201)](https://github.com/PennyLaneAI/pennylane/pull/4201)
  [(#4209)](https://github.com/PennyLaneAI/pennylane/pull/4209)
  [(#4229)](https://github.com/PennyLaneAI/pennylane/pull/4229)
  [(#4253)](https://github.com/PennyLaneAI/pennylane/pull/4253)
  [(#4255)](https://github.com/PennyLaneAI/pennylane/pull/4255)

  There are a few ways to create fermionic operators with this new feature:

  - via `qml.string_to_fermi_word`: create a fermionic operator that represents multiple creation and annihilation operators being 
    multiplied by each other, which we call a Fermi word.

    ```pycon
    >>> qml.string_to_fermi_word('0+ 1- 0+ 1-')
    <FermiWord = '0+ 1- 0+ 1-'>
    >>> string_to_fermi_word('0^ 1 0^ 1')
    <FermiWord = '0+ 1- 0+ 1-'>
    ```

    Fermi words can be linearly combined to create a fermionic operator that we call a Fermi sentence:

    ```pycon 
    >>> word1 = qml.string_to_fermi_word('0+ 0- 3+ 3-')
    >>> word2 = string_to_fermi_word('3+ 3-')
    >>> sentence = 1.2 * word1 - 0.345 * word2
    >>> sentence
    1.2 * a⁺(0) a(0) a⁺(3) a(3)
    - 0.345 * a⁺(3) a(3)
    ```

  - via `qml.FermiC` and `qml.FermiA`: the fermionic creation and annihilation operators, respectively. These operators are 
    defined by passing the index of the orbital that the fermionic operator acts on. For instance, 
    the operators $a^{\dagger}_0$ and $a_3$ are respectively constructed as

    ```pycon
    >>> qml.FermiC(0)
    >>> qml.FermiA(3)
    ```

    These operators can be composed with (`@`), linearly combined with (`+` and `-`) and multiplied by (`*`) other Fermi operators
    to create Fermi words, sentences, and Hamiltonians. Here is an example of creating a Fermi sentence:
    
    ```
    >>> word = qml.FermiC(0) * qml.FermiA(0) * qml.FermiC(3) * qml.FermiA(3)
    >>> sentence = 1.2 * word - 0.345 * qml.FermiC(3) * qml.FermiA(3)
    >>> sentence
    1.2 * a⁺(0) a(0) a⁺(3) a(3)
    - 0.345 * a⁺(3) a(3)
    ```

  Fermi words and sentences can also be created via `qml.fermi.FermiWord` and `qml.fermi.FermiSentence`, respectively.
    
  Additionally, any fermionic operator, be it a single fermionic creation/annihilation operator, a Fermi word, or a Fermi sentence,
  can be mapped to the qubit basis by using `qml.jordan_wigner`:

  ```pycon
  >>> qml.jordan_wigner(h)
  ((-1.75+0j)*(Identity(wires=[0]))) + ((0.6+0j)*(PauliZ(wires=[0]))) + ((1.15+0j)*(PauliZ(wires=[3])))
  ```

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

  The resource requirements of individual circuits can then be inspected as follows:

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

* The quantum information module now supports [trace distance](https://en.wikipedia.org/wiki/Trace_distance).
  [(#4181)](https://github.com/PennyLaneAI/pennylane/pull/4181)

  Two cases are enabled for calculating the trace distance:
  
  - A QNode transform via `qml.qinfo.trace_distance`:

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

  - Flexible post-processing via `qml.math.trace_distance`:

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
  of a single-qubit unitary matrix into sequences of X, Y, and Z rotations. All
  decompositions simplify the rotations angles to be between `0` and `4` pi.
  [(#4210)](https://github.com/PennyLaneAI/pennylane/pull/4210)
  [(#4246)](https://github.com/PennyLaneAI/pennylane/pull/4246)

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

* The `has_unitary_generator` attribute in `qml.ops.qubit.attributes` no longer contains operators
  with non-unitary generators.
  [(#4183)](https://github.com/PennyLaneAI/pennylane/pull/4183)

* PennyLane Docker builds have been updated to include the latest plugins and interface versions.
  [(#4178)](https://github.com/PennyLaneAI/pennylane/pull/4178)

<h4>Extended support for differentiating pulses</h4>

* The stochastic parameter-shift gradient method can now be used with hardware-compatible
  Hamiltonians.
  [(#4132)](https://github.com/PennyLaneAI/pennylane/pull/4132)

  This new feature generalizes the stochastic parameter-shift gradient transform for pulses 
  (`stoch_pulse_grad`) to support Hermitian generating terms beyond just Pauli words in pulse Hamiltonians, 
  which makes it hardware-compatible.

* A new differentiation method called `qml.gradients.pulse_generator` is available, which combines classical processing 
  with the parameter-shift rule for multivariate gates to differentiate pulse programs. Access it in your pulse
  programs by setting `diff_method=qml.gradients.pulse_generator`. 
  [(#4160)](https://github.com/PennyLaneAI/pennylane/pull/4132) 

* `qml.pulse.ParametrizedEvolution` now uses _batched_ compressed sparse row (`BCSR`) format. 
  This allows for computing Jacobians of the unitary directly even when `dense=False`.
  [(#4126)](https://github.com/PennyLaneAI/pennylane/pull/4126)

  ```python
  def U(params):
      H = jnp.polyval * qml.PauliZ(0) # time dependent Hamiltonian
      Um = qml.evolve(H)(params, t=10., dense=False)
      return qml.matrix(Um)
  params = jnp.array([[0.5]], dtype=complex)
  jac = jax.jacobian(U, holomorphic=True)(params)
  ```

<h4>Broadcasting and other tweaks to Torch and Keras layers 🦾</h4>

* The `TorchLayer` and `KerasLayer` integrations with `torch.nn` and `Keras` have been upgraded.
  Consider the following `TorchLayer`:

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
    >>> dev.num_executions == 1
    True
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

* `qml.Projector` now accepts a state vector representation, which enables the creation of projectors
  in any basis.
  [(#4192)](https://github.com/PennyLaneAI/pennylane/pull/4192)

  ```python
  dev = qml.device("default.qubit", wires=2)
  @qml.qnode(dev)
  def circuit(state):
      return qml.expval(qml.Projector(state, wires=[0, 1]))
  zero_state = [0, 0]
  plusplus_state = np.array([1, 1, 1, 1]) / 2
  ```

  ```pycon
  >>> circuit(zero_state)
  1.
  >>> 
  >>> circuit(plusplus_state)
  0.25
  ```

<h4>Do more with qutrits</h4>

* It is now possible to prepare qutrit basis states with `qml.QutritBasisState`.
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

* Three qutrit rotation operators have been added that are analogous to `RX`, `RY`, and `RZ`:

  - `TRX`: an X rotation
  - `TRY`: a Y rotation
  - `TRZ`: a Z rotation

  [(#2845)](https://github.com/PennyLaneAI/pennylane/pull/2845)
  [(#2846)](https://github.com/PennyLaneAI/pennylane/pull/2846)
  [(#2847)](https://github.com/PennyLaneAI/pennylane/pull/2847)
  
* Qutrit devices now support parameter-shift differentiation.
  [(#2845)](https://github.com/PennyLaneAI/pennylane/pull/2845)

<h4>The qchem module</h4>

* `qchem.molecular_hamiltonian()` will now return an arithmetic operator if `enable_new_opmath()` is active.
  [(#4159)](https://github.com/PennyLaneAI/pennylane/pull/4159)

* `qchem.qubit_observable()` will now return an arithmetic operator if `enable_new_opmath()` is active. 
  [(#4138)](https://github.com/PennyLaneAI/pennylane/pull/4138)

* `qchem.import_operator()` will now return an arithmetic operator if `enable_new_opmath()` is active.
  [(#4204)](https://github.com/PennyLaneAI/pennylane/pull/4204)

* Non-cubic lattice support for all electron resource estimation has been added.
  [(3956)](https://github.com/PennyLaneAI/pennylane/pull/3956)

* The `qchem.molecular_hamiltonian` function has been upgraded to support custom wires for constructing
  differentiable Hamiltonians. The zero imaginary component of the Hamiltonian coefficients have been
  removed.
  [(#4050)](https://github.com/PennyLaneAI/pennylane/pull/4050)
  [(#4094)](https://github.com/PennyLaneAI/pennylane/pull/4094)

* Jordan-Wigner transforms that cache Pauli gate objects have been accelerated.
  [(#4046)](https://github.com/PennyLaneAI/pennylane/pull/4046)

* `qchem.qubit_observable()` will now return an arithmetic operator if `enable_new_opmath()` is active. 
  [(#4138)](https://github.com/PennyLaneAI/pennylane/pull/4138)

* `qchem.molecular_hamiltonian()` will now return an arithmetic operator if `enable_new_opmath()` is active.
  [(#4159)](https://github.com/PennyLaneAI/pennylane/pull/4159)

* An error is now raised by `qchem.molecular_hamiltonian` when the `dhf` method is used for an 
  open-shell system. This duplicates a similar error in `qchem.Molecule` but makes it clear
  that the `pyscf` backend can be used for open-shell calculations.
  [(#4058)](https://github.com/PennyLaneAI/pennylane/pull/4058)

* `qchem.qubit_observable()` will now return an arithmetic operator if `enable_new_opmath()` is active. 
  [(#4138)](https://github.com/PennyLaneAI/pennylane/pull/4138)

* `qml.dipole_moment()` will now return an arithmetic operator if `enable_new_opmath()` is active.
  [(#4189)](https://github.com/PennyLaneAI/pennylane/pull/4189)

<h4>Next-generation device API</h4>

* The new device interface has been integrated with `qml.execute` for autograd, backpropagation, and no differentiation.
  [(#3903)](https://github.com/PennyLaneAI/pennylane/pull/3903)

* Support for adjoint differentiation has been added to the `DefaultQubit2` device.
  [(#4037)](https://github.com/PennyLaneAI/pennylane/pull/4037)

* A new function called `measure_with_samples` that returns a sample-based measurement result given a state has been added.
  [(#4083)](https://github.com/PennyLaneAI/pennylane/pull/4083)
  [(#4093)](https://github.com/PennyLaneAI/pennylane/pull/4093)
  [(#4254)](https://github.com/PennyLaneAI/pennylane/pull/4254)

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

* The experimental device `DefaultQubit2` now supports `qml.Snapshot`.
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

<h4>Circuit drawing</h4>

<h4>Other improvements</h4>

* The new device interface in integrated with `qml.execute` for Torch.
  [(#4257)](https://github.com/PennyLaneAI/pennylane/pull/4257)

* Added a transform dispatcher.
  [(#4109)](https://github.com/PennyLaneAI/pennylane/pull/4109)
  
* Added a transform program.
  [(#4187)](https://github.com/PennyLaneAI/pennylane/pull/4187)

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

* The default value for the `show_matrices` keyword argument in all drawing methods is now `True`. 
  This allows for quick insights into broadcasted tapes, for example.
  [(#3920)](https://github.com/PennyLaneAI/pennylane/pull/3920)

* Type variables for `qml.typing.Result` and `qml.typing.ResultBatch` have been added for type hinting the result of an execution.
  [(#4108)](https://github.com/PennyLaneAI/pennylane/pull/4108)
  
* The Jax-JIT interface now uses symbolic zeros to determine trainable parameters.
  [(4075)](https://github.com/PennyLaneAI/pennylane/pull/4075)

* A new function called `pauli.pauli_word_prefactor()` that extracts the prefactor for a given Pauli word has been added.
  [(#4164)](https://github.com/PennyLaneAI/pennylane/pull/4164)

* Reduced density matrix functionality has been added via `qml.math.reduce_dm` and `qml.math.reduce_statevector`.
  Both functions have broadcasting support.
  [(#4173)](https://github.com/PennyLaneAI/pennylane/pull/4173)

* Added `qml.math.reduce_dm` and `qml.math.reduce_statevector` to produce reduced density matrices.
  Both functions have broadcasting support.
  [(#4173)](https://github.com/PennyLaneAI/pennylane/pull/4173)

* Fix unclear documentation and indicate variable-length argument lists of functions and methods in
  the respective docstrings.
  [(#4242)](https://github.com/PennyLaneAI/pennylane/pull/4242)

* `qml.drawer.drawable_layers.drawable_layers` and `qml.CircuitGraph` have been updated to not rely on `Operator`
  equality or hash to work correctly.
  [(#4143)](https://github.com/PennyLaneAI/pennylane/pull/4143)

* Added the ability to draw mid-circuit measurements connected by classical control signals
  to conditional operations.
  [(#4228)](https://github.com/PennyLaneAI/pennylane/pull/4228)

* The new device interface is integrated with `qml.execute` for Tensorflow.
  [(#4169)](https://github.com/PennyLaneAI/pennylane/pull/4169)

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

* The `pennylane.transforms.qcut` module now uses `(op, id(op))` as nodes in directed multigraphs that are used within
  the circuit cutting workflow instead of `op`. This change removes the dependency of the module on the hash of operators.
  [(#4227)](https://github.com/PennyLaneAI/pennylane/pull/4227)

* `Operator.data` now returns a `tuple` instead of a `list`.
  [(#4222)](https://github.com/PennyLaneAI/pennylane/pull/4222)

<h3>Deprecations 👋</h3>

* The pulse differentiation methods, `pulse_generator` and `stoch_pulse_grad`, now raise an error when they
  are applied to a QNode directly. Instead, use differentiation via a JAX entry point (`jax.grad`, `jax.jacobian`, ...).
  [(#4241)](https://github.com/PennyLaneAI/pennylane/pull/4241)

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

* The documentation is updated to construct `QuantumTape` upon initialization instead of with queuing.
  [(#4243)](https://github.com/PennyLaneAI/pennylane/pull/4243)

* The docstring for `qml.ops.op_math.Pow.__new__` is now complete and it has been updated along with
  `qml.ops.op_math.Adjoint.__new__`.
  [(#4231)](https://github.com/PennyLaneAI/pennylane/pull/4231)

* The docstring for `qml.grad` now states that it should be used with the Autograd interface only.
  [(#4202)](https://github.com/PennyLaneAI/pennylane/pull/4202)

* The description of `mult` in the `qchem.Molecule` docstring now correctly states the value
  of `mult` that is supported.
  [(#4058)](https://github.com/PennyLaneAI/pennylane/pull/4058)

<h3>Bug fixes 🐛</h3>

* Fixes adjoint jacobian results with `grad_on_execution=False` in the JAX-JIT interface.
  [(4217)](https://github.com/PennyLaneAI/pennylane/pull/4217)

* Fixes the matrix of `SProd` when the coefficient is tensorflow and the target matrix is not `complex128`.
  [(#4249)](https://github.com/PennyLaneAI/pennylane/pull/4249)
 
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
  
* The `has_unitary_generator` attribute in `qml.ops.qubit.attributes` no longer contains operators with non-unitary generators.
  [(#4183)](https://github.com/PennyLaneAI/pennylane/pull/4183)

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
Lillian M. A. Frederiksen,
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
Edward Thomas,
David Wierichs,
Frederik Wilde.
