:orphan:

# Release 0.39.0 (current release)

<h3>New features since last release</h3>

<h4>Creating spin Hamiltonians on lattices 💞</h4>

* Functionality for creating custom Hamiltonians on arbitrary lattices has been added.
  [(#6226)](https://github.com/PennyLaneAI/pennylane/pull/6226)
  [(6237)](https://github.com/PennyLaneAI/pennylane/pull/6237)

  Hamiltonians beyond the available boiler-plate ones in the `qml.spin` module can be created with the 
  addition of three new functions:

  * `qml.spin.Lattice`: a new object for instantiating customized lattices via primitive translation vectors and unit cell parameters,
  * `qml.spin.generate_lattice`: a utility function for creating standard `Lattice` objects, including `'chain'`, `'square'`, `'rectangle'`, `'triangle'`, `'honeycomb'`, `'kagome'`, `'lieb'`, `'cubic'`, `'bcc'`, `'fcc'`, and `'diamond'`,
  * `qml.spin.spin_hamiltonian`: generates a spin `Hamiltonian` object given a `Lattice` object with custom edges/nodes.

  An example is shown below for a :math:`3 \times 3` triangular lattice with open boundary conditions.

  ```python
  lattice = qml.spin.Lattice(
      n_cells=[3, 3],
      vectors=[[1, 0], [np.cos(np.pi/3), np.sin(np.pi/3)]],
      positions=[[0, 0]],
      boundary_condition=False
  )
  ```

  We can validate this `lattice` against `qml.spin.generate_lattice('triangle', ...)` by checking the 
  `lattice_points` (the :math:`(x, y)` coordinates of all sites in the lattice):

  ```pycon
  >>> lp = lattice.lattice_points
  >>> triangular_lattice = qml.spin.generate_lattice('triangle', n_cells=[3, 3])
  >>> np.allclose(lp, triangular_lattice.lattice_points)
  True
  ```

  The `edges` of the `Lattice` object are nearest-neighbour by default, where we can add edges by using
  its `add_edge` method. With nearest-neighbour interactions, we can construct a transverse-field Ising 
  model on the lattice, for example:

  ```python
  hamiltonian = 0.0
  J = 1.0
  h = 0.5

  for edge in lattice.edges:
      i, j = edge[0], edge[1]
      hamiltonian -= J * qml.Z(i) @ qml.Z(j)

  for node in range(lattice.n_sites):
      hamiltonian -= h * qml.X(node)
  ```

  This is equivalent to using `qml.spin.transverse_ising` in the following way:

  ```pycon
  >>> hamiltonian == qml.spin.transverse_ising('triangle', n_cells=[3, 3], coupling=J, h=h)
  True
  ```

  Optionally, a `Lattice` object can have interactions and fields endowed to it by specifying values 
  for its `custom_edges` and `custom_nodes` keyword arguments. The Hamiltonian can then be extracted
  with the `qml.spin.spin_hamiltonian` function. An example is shown below for the same transverse-field 
  Ising model Hamiltonian on a :math:`3 \times 3` triangular lattice. Note that the `custom_edges` and 
  `custom_nodes` keyword arguments only need to be defined for one unit cell repetition.

  ```python
  edges = [
      (0, 1), (0, 3), (1, 3)
  ]

  lattice = qml.spin.Lattice(
      n_cells=[3, 3],
      vectors=[[1, 0], [np.cos(np.pi/3), np.sin(np.pi/3)]],
      positions=[[0, 0]],
      boundary_condition=False,
      custom_edges=[[edge, ("ZZ", -J)] for edge in edges],
      custom_nodes=[[i, ("X", -h)] for i in range(3*3)],
  )
  ```

  ```pycon
  >>> hamiltonian == qml.spin.spin_hamiltonian(lattice=lattice)
  True
  ```

* More industry-standard spin Hamiltonians have been added in the `qml.spin` module.
  [(#6174)](https://github.com/PennyLaneAI/pennylane/pull/6174)
  [(#6201)](https://github.com/PennyLaneAI/pennylane/pull/6201/)

  Three new industry-standard spin Hamiltonians are now available with PennyLane v0.39:

  * `qml.spin.emery`: the [Emery model](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.58.2794)
  * `qml.spin.haldane`: the [Haldane model](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.61.2015)
  * `qml.spin.kitaev`: the [Kitaev model](https://arxiv.org/abs/cond-mat/0506438)

  These additions accompany `qml.spin.heisenberg`, `qml.spin.transverse_ising`, and `qml.spin.fermi_hubbard`, 
  which were introduced in v0.38. 

  Each Hamiltonian can be instantiated by specifying a `lattice`, the number of [unit cells](https://en.wikipedia.org/wiki/Unit_cell), 
  `n_cells`, and the Hamiltonian parameters as keyword arguments. The returned object can be used in 
  a circuit like any other `Hamiltonian` object in PennyLane. Here is an example with the Haldane 
  model on a `"square"` lattice:

  ```python
  n_cells = [2, 2]
  h1 = 0.5
  h2 = 1.0
  phi = 0.1

  spin_ham = qml.spin.haldane("square", n_cells, hopping=h1, hopping_next=h2, phi=phi)
  n_qubits = spin_ham.num_wires

  dev = qml.device("lightning.qubit", wires=n_qubits)

  @qml.qnode(dev)
  def circuit(params):
      for i in range(params.shape[0] - 1):
          qml.CNOT(wires=[i, i + 1])
          qml.RY(params[i], wires=i)

      return qml.expval(spin_ham)
  ```

  ```pycon
  >>> params = pnp.array([1.967]*n_qubits)
  >>> circuit(params)
  array(0.61344534)
  ```

<h4>Calculating Polynomials 🔢</h4>

* Polynomial functions can now be easily encoded into quantum circuits with `qml.OutPoly`.
  [(#6320)](https://github.com/PennyLaneAI/pennylane/pull/6320)

  A new template called `qml.OutPoly` is available, which provides the ability to encode a polynomial
  function in a quantum circuit. Given a polynomial function :math:`f(x_1, x_2, \cdots, x_N)`, `qml.OutPoly` 
  requires:

  * `f`: a standard Python function that represents :math:`f(x_1, x_2, \cdots, x_N)`,
  * `input_registers` (:math:`\vert x_1 \rangle`, :math:`\vert x_2 \rangle`, ..., :math:`\vert x_N \rangle`): a list/tuple containing `Wires` objects that correspond to the embedded numeric values of :math:`x_1, x_2, \cdots, x_N`,
  * `output_wires`: the `Wires` for which the numeric value of :math:`f(x_1, x_2, \cdots, x_N)` is stored.

  Here is an example of using `qml.OutPoly` to calculate :math:`f(x_1, x_2) = 3x_1^2 - x_1x_2` for :math:`f(1, 2) = 1`.

  ```python
  wires = qml.registers({"x1": 1, "x2": 2, "output": 2})

  def f(x1, x2):
      return 3 * x1 ** 2 - x1 * x2

  @qml.qnode(qml.device("default.qubit", shots = 1))
  def circuit():
      # load values of x1 and x2
      qml.BasisEmbedding(1, wires=wires["x1"])
      qml.BasisEmbedding(2, wires=wires["x2"])

      # apply the polynomial
      qml.OutPoly(
          f,
          input_registers = [wires["x1"], wires["x2"]],
          output_wires = wires["output"])

      return qml.sample(wires=wires["output"])
  ```

  ```pycon
  >>> circuit()
  [0 1]
  ```

  The result, `[0 1]`, is the binary representation of :math:`1`. By default, the result is calculated
  modulo :math:`2^\text{len(output_wires)}` but can be overridden with the `mod` keyword argument.

<h4>Readout Noise 📠</h4>

* Readout errors can now be included in `qml.NoiseModel` and `qml.add_noise` with the new `qml.noise.meas_eq`
  function.
  [(#6321)](https://github.com/PennyLaneAI/pennylane/pull/6321/)

  Measurement/readout errors can be specified in a similar fashion to regular gate noise in PennyLane: 
  a newly added Boolean function called `qml.noise.meas_eq` that accepts a measurement function 
  (e.g., `qml.expval`, `qml.sample`, or any other function that can be returned from a QNode) that, 
  when present in the QNode, inserts a noisy operation via `qml.noise.partial_wires` or a custom noise 
  function. Readout noise in PennyLane also follows the insertion convention, where the specified noise 
  is inserted *before* the measurement.

  Here is an example of adding `qml.PhaseFlip` noise to any `qml.expval` measurement:

  ```python
  c0 = qml.noise.meas_eq(qml.expval)
  n0 = qml.noise.partial_wires(qml.PhaseFlip, 0.2)
  ```

  To include this in a `qml.NoiseModel`, use its `meas_map` keyword argument:

  ```python
  # gate-based noise
  c1 = qml.noise.wires_in([0, 2]) 
  n1 = qml.noise.partial_wires(qml.RY, -0.42)

  noise_model = qml.NoiseModel({c1: n1}, meas_map={c0: n0})
  ```

  ```pycon
  >>> noise_model
  NoiseModel({
    WiresIn([0, 2]): RY(phi=-0.42)
  },
  meas_map = {
      MeasEq(expval): PhaseFlip(p=0.2)
  })
  ```

  `qml.noise.meas_eq` can also be combined with other Boolean functions in `qml.noise` via bitwise operators
  for more versatility.

  To add this `noise_model` to a circuit, use the `qml.add_noise` transform as per usual. For example, 

  ```python
  @qml.qnode(qml.device("default.mixed", wires=3))
  def circuit():
      qml.RX(0.1967, wires=0)
      for i in range(3):
          qml.Hadamard(i)

      return qml.expval(qml.X(0) @ qml.X(1))
  ```

  ```pycon
  >>> noisy_circuit = qml.add_noise(circuit, noise_model)
  >>> print(qml.draw(noisy_circuit)())
  0: ──RX(0.20)──RY(-0.42)────────H──RY(-0.42)──PhaseFlip(0.20)─┤ ╭<X@X>
  1: ──H─────────PhaseFlip(0.20)────────────────────────────────┤ ╰<X@X>
  2: ──H─────────RY(-0.42)──────────────────────────────────────┤    
  >>> print(circuit(), noisy_circuit())
  0.9807168489852615 0.35305806563469433
  ```

<h4>User-friendly decompositions 📠</h4>

* `qml.transforms.decompose` is added for stepping through decompositions to a target gate set. 
  [(#6334)](https://github.com/PennyLaneAI/pennylane/pull/6334)

<h3>Improvements 🛠</h3>

<h4>QJIT/JIT Compatibility and improvements</h4>

* The `SampleMP.process_samples` method has been updated to support using JAX tracers for samples, allowing 
  compatiblity with QJIT workflows.
  [(#6211)](https://github.com/PennyLaneAI/pennylane/pull/6211)

* All PL templates are now unit tested to ensure JIT compatibility.
  [(#6309)](https://github.com/PennyLaneAI/pennylane/pull/6309)

* The `qml.FABLE` template now returns the correct value when JIT is enabled.
  [(#6263)](https://github.com/PennyLaneAI/pennylane/pull/6263)

* `qml.QutritBasisStatePreparation` is now JIT compatible.
  [(#6308)](https://github.com/PennyLaneAI/pennylane/pull/6308)

* `qml.AmplitudeAmplification` is now compatible with QJIT.
  [(#6306)](https://github.com/PennyLaneAI/pennylane/pull/6306)

* The quantum arithmetic templates are now compatible with QJIT.
  [(#6307)](https://github.com/PennyLaneAI/pennylane/pull/6307)
  
* The `qml.Qubitization` template is now compatible with QJIT.
  [(#6305)](https://github.com/PennyLaneAI/pennylane/pull/6305)

<h4>Capturing and representing hybrid programs</h4>

* `qml.wires.Wires` now accepts JAX arrays as input. Furthermore, a `FutureWarning` is no longer raised 
  in `JAX 0.4.30+` when providing JAX tracers as input to `qml.wires.Wires`.
  [(#6312)](https://github.com/PennyLaneAI/pennylane/pull/6312)

* Differentiation of hybrid programs via `qml.grad` and `qml.jacobian` can now be captured with PLxPR. 
  When evaluating a captured `qml.grad` (`qml.jacobian`) instruction, it will dispatch to `jax.grad` 
  (`jax.jacobian`), which differs from the Autograd implementation without capture. Pytree inputs and 
  outputs are supported.
  [(#6120)](https://github.com/PennyLaneAI/pennylane/pull/6120)
  [(#6127)](https://github.com/PennyLaneAI/pennylane/pull/6127)
  [(#6134)](https://github.com/PennyLaneAI/pennylane/pull/6134)

* Unit testing for capturing nested control flow has been improved.
  [(#6111)](https://github.com/PennyLaneAI/pennylane/pull/6111)

* Some custom primitives for the capture project can now be imported via `from pennylane.capture.primitives import *`.
  [(#6129)](https://github.com/PennyLaneAI/pennylane/pull/6129)

* All higher order primitives now use `jax.core.Jaxpr` as metadata instead of sometimes using `jax.core.ClosedJaxpr` 
  or `jax.core.Jaxpr`.
  [(#6319)](https://github.com/PennyLaneAI/pennylane/pull/6319)

* A new function called `qml.capture.make_plxpr` has been added to take a function and create a `Callable` 
  that, when called, will return a PLxPR representation of the input function.
  [(#6326)](https://github.com/PennyLaneAI/pennylane/pull/6326)

<h4>Improvements to fermionic operators</h4>

* `qml.fermi.FermiWord` and `qml.fermi.FermiSentence` are now compatible with JAX arrays.
  [(#6324)](https://github.com/PennyLaneAI/pennylane/pull/6324)

* The `qml.fermi.FermiWord` class now has a method to apply anti-commutator relations.
  [(#6196)](https://github.com/PennyLaneAI/pennylane/pull/6196)

* The `qml.fermi.FermiWord` and `qml.fermi.FermiSentence` classes now have methods to compute adjoints.
  [(#6166)](https://github.com/PennyLaneAI/pennylane/pull/6166)

* The `to_mat` methods for `qml.fermi.FermiWord` and `qml.fermi.FermiSentence` now optionally return
  a sparse matrix.
  [(#6173)](https://github.com/PennyLaneAI/pennylane/pull/6173)

* When printed, `qml.fermi.FermiWord` and `qml.fermi.FermiSentence` now return a unique representation 
  of the object.
  [(#6167)](https://github.com/PennyLaneAI/pennylane/pull/6167)

* `qml.qchem.excitations` now optionally returns fermionic operators.
  [(#6171)](https://github.com/PennyLaneAI/pennylane/pull/6171)

<h4>A new optimizer</h4>

* A new class `MomentumQNGOptimizer` is added. It inherits the basic `QNGOptimizer` class and requires one additional hyperparameter (the momentum coefficient) :math:`0 \leq \rho < 1`, the default value being :math:`\rho=0.9`. For :math:`\rho=0` Momentum-QNG reduces to the basic QNG.
  [(#6240)](https://github.com/PennyLaneAI/pennylane/pull/6240)

<h4>Other Improvements</h4>

* `process_density_matrix` was implemented in 5 `StateMeasurement` subclasses: `ExpVal`, `Var`, `Purity`, 
  `MutualInformation`, and `VnEntropy`. This facilitates future support for mixed-state devices and 
  expanded density matrix operations. Also, a fix was made in the `ProbabilityMP` class to use `qml.math.sqrt` 
  instead of `np.sqrt`.
  [(#6330)](https://github.com/PennyLaneAI/pennylane/pull/6330)

* The decomposition for `qml.Qubitization` has been improved to use `qml.PrepSelPrep`.
  [(#6182)](https://github.com/PennyLaneAI/pennylane/pull/6182)

* A `ReferenceQubit` device (shortname: `"reference.qubit"`) has been introduced for testing purposes 
  and referencing for future plugin development.
  [(#6181)](https://github.com/PennyLaneAI/pennylane/pull/6181)

* `qml.transforms.mitigate_with_zne` now gives a clearer error message when being applied on circuits, not devices, 
  that include channel noise.
  [(#6346)](https://github.com/PennyLaneAI/pennylane/pull/6346)

* A new function called `get_best_diff_method` has been added to `qml.workflow`.
  [(#6399)](https://github.com/PennyLaneAI/pennylane/pull/6399)

* A new method called `construct_tape` has been added to `qml.workflow` for users to construct single 
  tapes from a `QNode`.
  [(#6419)](https://github.com/PennyLaneAI/pennylane/pull/6419)

* Datasets are now downloaded via Dataset API.
  [(#6126)](https://github.com/PennyLaneAI/pennylane/pull/6126)

* `qml.Hadamard` now has a more friendly alias: `qml.H`.
  [(#6450)](https://github.com/PennyLaneAI/pennylane/pull/6450)

* A Boolean keyword argument called `show_wire_labels` has been added to `draw` and `draw_mpl`, which 
  hides wire labels when set to `False` (the default is `True`).
  [(#6410)](https://github.com/PennyLaneAI/pennylane/pull/6410)

* A new function called `sample_probs` has been added to the `qml.devices.qubit` and `qml.devices.qutrit_mixed` 
  modules. This function takes probability distributions as input and returns sampled outcomes, simplifying 
  the sampling process by separating it from other operations in the measurement chain and improving modularity.
  [(#6354)](https://github.com/PennyLaneAI/pennylane/pull/6354)

* `qml.labs` has been added to the PennyLane documentation.
  [(#6397)](https://github.com/PennyLaneAI/pennylane/pull/6397)
  [(#6369)](https://github.com/PennyLaneAI/pennylane/pull/6369)

* A `has_sparse_matrix` property has been added to `Operator` to indicate whether a sparse matrix is 
  defined.
  [(#6278)](https://github.com/PennyLaneAI/pennylane/pull/6278)
  [(#6310)](https://github.com/PennyLaneAI/pennylane/pull/6310)

* `qml.matrix` now works with empty objects (e.g., empty tapes, QNodes and quantum functions that do
  not call operations, and single operators with empty decompositions).
  [(#6347)](https://github.com/PennyLaneAI/pennylane/pull/6347)
  
* PennyLane is now compatible with NumPy 2.0.
  [(#6061)](https://github.com/PennyLaneAI/pennylane/pull/6061)
  [(#6258)](https://github.com/PennyLaneAI/pennylane/pull/6258)
  [(#6342)](https://github.com/PennyLaneAI/pennylane/pull/6342)

* PennyLane is now compatible with Jax 0.4.28.
  [(#6255)](https://github.com/PennyLaneAI/pennylane/pull/6255)

* The `diagonalize_measurements` transform now uses a more efficient method of diagonalization when possible, 
  based on the `pauli_rep` of the relevant observables.
  [(#6113)](https://github.com/PennyLaneAI/pennylane/pull/6113/)

* The `QuantumScript.copy` method now takes `operations`, `measurements`, `shots` and `trainable_params` 
  as keyword arguments. If any of these are passed when copying a tape, the specified attributes will 
  replace the copied attributes on the new tape.
  [(#6285)](https://github.com/PennyLaneAI/pennylane/pull/6285)
  [(#6363)](https://github.com/PennyLaneAI/pennylane/pull/6363)

* The `Hermitian` operator now has a `compute_sparse_matrix` implementation.
  [(#6225)](https://github.com/PennyLaneAI/pennylane/pull/6225)

* When an observable is repeated on a tape, `tape.diagonalizing_gates` no longer returns the 
  diagonalizing gates for each instance of the observable. Instead, the diagonalizing gates of
  each observable on the tape are included just once.
  [(#6288)](https://github.com/PennyLaneAI/pennylane/pull/6288)

* The number of diagonalizing gates returned in `qml.specs` now follows the `level` keyword argument 
  regarding whether the diagonalizing gates are modified by device, instead of always counting unprocessed 
  diagonalizing gates.
  [(#6290)](https://github.com/PennyLaneAI/pennylane/pull/6290)

* A more sensible error message is raised from a `RecursionError` encountered when accessing properties 
  and methods of a nested `CompositeOp` or `SProd`.
  [(#6375)](https://github.com/PennyLaneAI/pennylane/pull/6375)

* The performance of the decomposition of `qml.QFT` has been improved.
  [(#6434)](https://github.com/PennyLaneAI/pennylane/pull/6434)

<h3>Breaking changes 💔</h3>

* The `AllWires` validation in `QNode.construct` has been removed. 
  [(#6373)](https://github.com/PennyLaneAI/pennylane/pull/6373)

* The `simplify` argument in `qml.Hamiltonian` and `qml.ops.LinearCombination` has been removed.
  Instead, `qml.simplify()` can be called on the constructed operator.
  [(#6279)](https://github.com/PennyLaneAI/pennylane/pull/6279)

* The functions `qml.qinfo.classical_fisher` and `qml.qinfo.quantum_fisher` have been removed and migrated 
  to the `qml.gradients` module. `qml.gradients.classical_fisher` and `qml.gradients.quantum_fisher` 
  should be used instead.
  [(#5911)](https://github.com/PennyLaneAI/pennylane/pull/5911)

* Python 3.9 is no longer supported. Please update to 3.10 or newer.
  [(#6223)](https://github.com/PennyLaneAI/pennylane/pull/6223)

* `DefaultQubitTF`, `DefaultQubitTorch`, `DefaultQubitJax`, and `DefaultQubitAutograd` have been removed.
  Please use `default.qubit` for all interfaces.
  [(#6207)](https://github.com/PennyLaneAI/pennylane/pull/6207)
  [(#6208)](https://github.com/PennyLaneAI/pennylane/pull/6208)
  [(#6209)](https://github.com/PennyLaneAI/pennylane/pull/6209)
  [(#6210)](https://github.com/PennyLaneAI/pennylane/pull/6210)

* `expand_fn`, `max_expansion`, `override_shots`, and `device_batch_transform` have been removed from 
  the signature of `qml.execute`.
  [(#6203)](https://github.com/PennyLaneAI/pennylane/pull/6203)

* `max_expansion` and `expansion_strategy` have been removed from the QNode.
  [(#6203)](https://github.com/PennyLaneAI/pennylane/pull/6203)

* `expansion_strategy` has been removed from `qml.draw`, `qml.draw_mpl`, and `qml.specs`. `max_expansion` 
  has been removed from `qml.specs`, as it had no impact on the output.
  [(#6203)](https://github.com/PennyLaneAI/pennylane/pull/6203)

* `qml.transforms.hamiltonian_expand` and `qml.transforms.sum_expand` have been removed. Please use 
  `qml.transforms.split_non_commuting` instead.
  [(#6204)](https://github.com/PennyLaneAI/pennylane/pull/6204)

* The `decomp_depth` keyword argument to `qml.device` has been removed.
  [(#6234)](https://github.com/PennyLaneAI/pennylane/pull/6234)

* `Operator.expand` has been removed. Please use `qml.tape.QuantumScript(op.deocomposition())` instead.
  [(#6227)](https://github.com/PennyLaneAI/pennylane/pull/6227)

* The native folding method `qml.transforms.fold_global` for the `qml.transforms.mitiagte_with_zne`
  transform no longer expands the circuit automatically. Instead, the user should apply `qml.transforms.decompose` to
  decompose a circuit into a target gate set before applying `fold_global` or `mitigate_with_zne`.
  [(#6382)](https://github.com/PennyLaneAI/pennylane/pull/6382)

* The `LightningVJPs` class has been removed, as all lightning devices now follow the new device interface.
  [(#6420)](https://github.com/PennyLaneAI/pennylane/pull/6420)

<h3>Deprecations 👋</h3>

* The `expand_depth` and `max_expansion` arguments for `qml.transforms.compile` and
  `qml.transforms.decompositions.clifford_t_decomposition` respectively have been deprecated.
  [(#6404)](https://github.com/PennyLaneAI/pennylane/pull/6404)

* Legacy operator arithmetic has been deprecated. This includes `qml.ops.Hamiltonian`, `qml.operation.Tensor`,
  `qml.operation.enable_new_opmath`, `qml.operation.disable_new_opmath`, and `qml.operation.convert_to_legacy_H`.
  Note that when new operator arithmetic is enabled, `qml.Hamiltonian` will continue to dispatch to
  `qml.ops.LinearCombination`; this behaviour is not deprecated. For more information, check out the
  [updated operator troubleshooting page](https://docs.pennylane.ai/en/stable/news/new_opmath.html).
  [(#6287)](https://github.com/PennyLaneAI/pennylane/pull/6287)
  [(#6365)](https://github.com/PennyLaneAI/pennylane/pull/6365)

* `qml.pauli.PauliSentence.hamiltonian` and `qml.pauli.PauliWord.hamiltonian` have been deprecated. 
  Instead, please use `qml.pauli.PauliSentence.operation` and `qml.pauli.PauliWord.operation`, respectively.
  [(#6287)](https://github.com/PennyLaneAI/pennylane/pull/6287)

* `qml.pauli.simplify()` has been deprecated. Instead, please use `qml.simplify(op)` or `op.simplify()`.
  [(#6287)](https://github.com/PennyLaneAI/pennylane/pull/6287)

* The `qml.BasisStatePreparation` template has been deprecated. Instead, use `qml.BasisState`.
  [(#6021)](https://github.com/PennyLaneAI/pennylane/pull/6021)

* The `'ancilla'` argument for `qml.iterative_qpe` has been deprecated. Instead, use the `'aux_wire'` 
  argument.
  [(#6277)](https://github.com/PennyLaneAI/pennylane/pull/6277)

* `qml.shadows.shadow_expval` has been deprecated. Instead, use the `qml.shadow_expval` measurement
  process.
  [(#6277)](https://github.com/PennyLaneAI/pennylane/pull/6277)

* `qml.broadcast` has been deprecated. Please use Python `for` loops instead.
  [(#6277)](https://github.com/PennyLaneAI/pennylane/pull/6277)

* The `qml.QubitStateVector` template has been deprecated. Instead, use `qml.StatePrep`.
  [(#6172)](https://github.com/PennyLaneAI/pennylane/pull/6172)

* The `qml.qinfo` module has been deprecated. Please see the respective functions in the `qml.math` 
  and `qml.measurements` modules instead.
  [(#5911)](https://github.com/PennyLaneAI/pennylane/pull/5911)

* `Device`, `QubitDevice`, and `QutritDevice` will no longer be accessible via top-level import in v0.40.
  They will still be accessible as `qml.devices.LegacyDevice`, `qml.devices.QubitDevice`, and `qml.devices.QutritDevice`
  respectively.
  [(#6238)](https://github.com/PennyLaneAI/pennylane/pull/6238/)

* `QNode.gradient_fn` has been deprecated. Please use `QNode.diff_method` and `QNode.get_gradient_fn` 
  instead.
  [(#6244)](https://github.com/PennyLaneAI/pennylane/pull/6244)

<h3>Documentation 📝</h3>

* Updated `qml.spin` documentation.
  [(#6387)](https://github.com/PennyLaneAI/pennylane/pull/6387)

* Updated links to PennyLane.ai in the documentation to use the latest URL format, which excludes the `.html` prefix.
  [(#6412)](https://github.com/PennyLaneAI/pennylane/pull/6412)

* Update `qml.Qubitization` documentation based on new decomposition.
  [(#6276)](https://github.com/PennyLaneAI/pennylane/pull/6276)

* Fixed examples in the documentation of a few optimizers.
  [(#6303)](https://github.com/PennyLaneAI/pennylane/pull/6303)
  [(#6315)](https://github.com/PennyLaneAI/pennylane/pull/6315)

* Corrected examples in the documentation of `qml.jacobian`.
  [(#6283)](https://github.com/PennyLaneAI/pennylane/pull/6283)
  [(#6315)](https://github.com/PennyLaneAI/pennylane/pull/6315)

* Fixed spelling in a number of places across the documentation.
  [(#6280)](https://github.com/PennyLaneAI/pennylane/pull/6280)

* Add `work_wires` parameter to `qml.MultiControlledX` docstring signature.
  [(#6271)](https://github.com/PennyLaneAI/pennylane/pull/6271)

* Removed ambiguity in error raised by the `PauliRot` class.
  [(#6298)](https://github.com/PennyLaneAI/pennylane/pull/6298)

* Renamed an incorrectly named test in `test_pow_ops.py`.
  [(#6388)](https://github.com/PennyLaneAI/pennylane/pull/6388)

<h3>Bug fixes 🐛</h3>

* Fixes unnecessary call of `eigvals` in `qml.ops.op_math.decompositions.two_qubit_unitary.py` that was causing an error in VJP. Raises warnings to users if this essentially nondifferentiable module is used.
  [(#6437)](https://github.com/PennyLaneAI/pennylane/pull/6437)

* Patches the `math` module to function with autoray 0.7.0.
  [(#6429)](https://github.com/PennyLaneAI/pennylane/pull/6429)

* Fixes incorrect differentiation of `PrepSelPrep` when using `diff_method="parameter-shift"`. 
  [(#6423)](https://github.com/PennyLaneAI/pennylane/pull/6423)

* `default.tensor` can now handle mid circuit measurements via the deferred measurement principle.
  [(#6408)](https://github.com/PennyLaneAI/pennylane/pull/6408)

* The `validate_device_wires` transform now raises an error if abstract wires are provided.
  [(#6405)](https://github.com/PennyLaneAI/pennylane/pull/6405)

* Fixes `qml.math.expand_matrix` for qutrit and arbitrary qudit operators.
  [(#6398)](https://github.com/PennyLaneAI/pennylane/pull/6398/)

* `MeasurementValue` now raises an error when it is used as a boolean.
  [(#6386)](https://github.com/PennyLaneAI/pennylane/pull/6386)

* `default.qutrit` now returns integer samples.
  [(#6385)](https://github.com/PennyLaneAI/pennylane/pull/6385)

* `adjoint_metric_tensor` now works with circuits containing state preparation operations.
  [(#6358)](https://github.com/PennyLaneAI/pennylane/pull/6358)

* `quantum_fisher` now respects the classical Jacobian of QNodes.
  [(#6350)](https://github.com/PennyLaneAI/pennylane/pull/6350)

* `qml.map_wires` can now be applied to a batch of tapes.
  [(#6295)](https://github.com/PennyLaneAI/pennylane/pull/6295)

* Fix float-to-complex casting in various places across PennyLane.
 [(#6260)](https://github.com/PennyLaneAI/pennylane/pull/6260)
 [(#6268)](https://github.com/PennyLaneAI/pennylane/pull/6268)

* Fix a bug where zero-valued JVPs were calculated wrongly in the presence of shot vectors.
  [(#6219)](https://github.com/PennyLaneAI/pennylane/pull/6219)

* Fix `qml.PrepSelPrep` template to work with `torch`.
  [(#6191)](https://github.com/PennyLaneAI/pennylane/pull/6191)

* Now `qml.equal` compares correctly `qml.PrepSelPrep` operators.
  [(#6182)](https://github.com/PennyLaneAI/pennylane/pull/6182)

* The `qml.QSVT` template now orders the `projector` wires first and the `UA` wires second, which is the expected order of the decomposition.
  [(#6212)](https://github.com/PennyLaneAI/pennylane/pull/6212)

* The `qml.Qubitization` template now orders the `control` wires first and the `hamiltonian` wires second, which is the expected according to other templates.
  [(#6229)](https://github.com/PennyLaneAI/pennylane/pull/6229)

* Fixes a bug where a circuit using the `autograd` interface sometimes returns nested values that are not of the `autograd` interface.
  [(#6225)](https://github.com/PennyLaneAI/pennylane/pull/6225)

* Fixes a bug where a simple circuit with no parameters or only builtin/numpy arrays as parameters returns autograd tensors.
  [(#6225)](https://github.com/PennyLaneAI/pennylane/pull/6225)

* `qml.pauli.PauliVSpace` now uses a more stable SVD-based linear independence check to avoid running into `LinAlgError: Singular matrix`. This stabilizes the usage of `qml.lie_closure`. It also introduces normalization of the basis vector's internal representation `_M` to avoid exploding coefficients.
  [(#6232)](https://github.com/PennyLaneAI/pennylane/pull/6232)

* Fixes a bug where `csc_dot_product` is used during measurement for `Sum`/`Hamiltonian` that contains observables that does not define a sparse matrix.
  [(#6278)](https://github.com/PennyLaneAI/pennylane/pull/6278)
  [(#6310)](https://github.com/PennyLaneAI/pennylane/pull/6310)

* Fixes a bug where `None` was added to the wires in `qml.PhaseAdder`, `qml.Adder` and `qml.OutAdder`.
  [(#6360)](https://github.com/PennyLaneAI/pennylane/pull/6360)

* Fixes a test after updating to the nightly version of Catalyst.
  [(#6362)](https://github.com/PennyLaneAI/pennylane/pull/6362)

* Fixes a bug where `CommutingEvolution` with a trainable `Hamiltonian` cannot be differentiated using parameter shift.
  [(#6372)](https://github.com/PennyLaneAI/pennylane/pull/6372)

* Fixes a bug where `mitigate_with_zne` loses the `shots` information of the original tape.
  [(#6444)](https://github.com/PennyLaneAI/pennylane/pull/6444)

* Fixes a bug where `default.tensor` raises an error when applying `Identity`/`GlobalPhase` on no wires, and `PauliRot`/`MultiRZ` on a single wire.
  [(#6448)](https://github.com/PennyLaneAI/pennylane/pull/6448)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso,
Utkarsh Azad,
Oleksandr Borysenko,
Astral Cai,
Yushao Chen,
Isaac De Vlugt,
Diksha Dhawan,
Lillian M. A. Frederiksen,
Pietropaolo Frisoni,
Emiliano Godinez,
Anthony Hayes,
Austin Huang,
Soran Jahangiri,
Jacob Kitchen,
Korbinian Kottmann,
Christina Lee,
William Maxwell,
Erick Ochoa Lopez,
Lee J. O'Riordan,
Mudit Pandey,
Andrija Paurevic,
Alex Preciado,
Ashish Kanwar Singh,
David Wierichs,
