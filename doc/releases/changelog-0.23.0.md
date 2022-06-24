:orphan:

# Release 0.23.0

<h3>New features since last release</h3>

<h4> More powerful circuit cutting ✂️</h4>

* Quantum circuit cutting (running `N`-wire circuits on devices with fewer than
  `N` wires) is now supported for QNodes of finite-shots using the new
  `@qml.cut_circuit_mc` transform.
  [(#2313)](https://github.com/PennyLaneAI/pennylane/pull/2313)
  [(#2321)](https://github.com/PennyLaneAI/pennylane/pull/2321)
  [(#2332)](https://github.com/PennyLaneAI/pennylane/pull/2332)
  [(#2358)](https://github.com/PennyLaneAI/pennylane/pull/2358)
  [(#2382)](https://github.com/PennyLaneAI/pennylane/pull/2382)
  [(#2399)](https://github.com/PennyLaneAI/pennylane/pull/2399)
  [(#2407)](https://github.com/PennyLaneAI/pennylane/pull/2407)
  [(#2444)](https://github.com/PennyLaneAI/pennylane/pull/2444)

  With these new additions, samples from the original circuit can be
  simulated using a Monte Carlo method, using fewer qubits at the expense of
  more device executions. Additionally, this transform can take an optional
  classical processing function as an argument and return an expectation
  value.

  The following `3`-qubit circuit contains a `WireCut` operation and a `sample`
  measurement. When decorated with `@qml.cut_circuit_mc`, we can cut the circuit
  into two `2`-qubit fragments:

  ```python
  dev = qml.device("default.qubit", wires=2, shots=1000)

  @qml.cut_circuit_mc
  @qml.qnode(dev)
  def circuit(x):
      qml.RX(0.89, wires=0)
      qml.RY(0.5, wires=1)
      qml.RX(1.3, wires=2)

      qml.CNOT(wires=[0, 1])
      qml.WireCut(wires=1)
      qml.CNOT(wires=[1, 2])

      qml.RX(x, wires=0)
      qml.RY(0.7, wires=1)
      qml.RX(2.3, wires=2)
      return qml.sample(wires=[0, 2])
  ```

  we can then execute the circuit as usual by calling the QNode:

  ```pycon
  >>> x = 0.3
  >>> circuit(x)
  tensor([[1, 1],
          [0, 1],
          [0, 1],
          ...,
          [0, 1],
          [0, 1],
          [0, 1]], requires_grad=True)
  ```

  Furthermore, the number of shots can be temporarily altered when calling
  the QNode:

  ```pycon
  >>> results = circuit(x, shots=123)
  >>> results.shape
  (123, 2)
  ```

  The `cut_circuit_mc` transform also supports returning sample-based
  expectation values of observables using the `classical_processing_fn`
  argument. Refer to the `UsageDetails` section of the [transform
  documentation](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.cut_circuit_mc.html)
  for an example.

* The `cut_circuit` transform now supports automatic graph partitioning by
  specifying `auto_cutter=True` to cut arbitrary tape-converted graphs using
  the general purpose graph partitioning framework
  [KaHyPar](https://pypi.org/project/kahypar/).
  [(#2330)](https://github.com/PennyLaneAI/pennylane/pull/2330)
  [(#2428)](https://github.com/PennyLaneAI/pennylane/pull/2428)

  Note that `KaHyPar` needs to be installed separately with the `auto_cutter=True` option.

  For integration with the  existing low-level manual cut pipeline, refer to
  the [documentation of the
  function](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.transforms.qcut.find_and_place_cuts.html)
  .
  ```pycon
  @qml.cut_circuit(auto_cutter=True)
  @qml.qnode(dev)
  def circuit(x):
      qml.RX(x, wires=0)
      qml.RY(0.9, wires=1)
      qml.RX(0.3, wires=2)
      qml.CZ(wires=[0, 1])
      qml.RY(-0.4, wires=0)
      qml.CZ(wires=[1, 2])
      return qml.expval(qml.grouping.string_to_pauli_word("ZZZ"))
  ```
  ```pycon
  >>> x = np.array(0.531, requires_grad=True)
  >>> circuit(x)
  0.47165198882111165
  >>> qml.grad(circuit)(x)
  -0.276982865449393
  ```

<h4>Grand QChem unification ⚛️  🏰</h4>

* Quantum chemistry functionality --- previously split between an external
  `pennylane-qchem` package and internal `qml.hf` differentiable Hartree-Fock
  solver --- is now unified into a single, included, `qml.qchem` module.
  [(#2164)](https://github.com/PennyLaneAI/pennylane/pull/2164)
  [(#2385)](https://github.com/PennyLaneAI/pennylane/pull/2385)
  [(#2352)](https://github.com/PennyLaneAI/pennylane/pull/2352)
  [(#2420)](https://github.com/PennyLaneAI/pennylane/pull/2420)
  [(#2454)](https://github.com/PennyLaneAI/pennylane/pull/2454)  
  [(#2199)](https://github.com/PennyLaneAI/pennylane/pull/2199)
  [(#2371)](https://github.com/PennyLaneAI/pennylane/pull/2371)
  [(#2272)](https://github.com/PennyLaneAI/pennylane/pull/2272)
  [(#2230)](https://github.com/PennyLaneAI/pennylane/pull/2230)
  [(#2415)](https://github.com/PennyLaneAI/pennylane/pull/2415)
  [(#2426)](https://github.com/PennyLaneAI/pennylane/pull/2426)
  [(#2465)](https://github.com/PennyLaneAI/pennylane/pull/2465)

  The `qml.qchem` module provides a differentiable Hartree-Fock solver and the functionality to
  construct a fully-differentiable molecular Hamiltonian.
  
  For example, one can continue to generate molecular Hamiltonians using  
  `qml.qchem.molecular_hamiltonian`:

  ```python
  symbols = ["H", "H"]
  geometry = np.array([[0., 0., -0.66140414], [0., 0., 0.66140414]])
  hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, method="dhf")
  ```

  By default, this will use the differentiable Hartree-Fock solver; however, simply set
  `method="pyscf"` to continue to use PySCF for Hartree-Fock calculations.

* Functions are added for building a differentiable dipole moment observable. Functions for 
  computing multipole moment molecular integrals, needed for building the dipole moment observable, 
  are also added.
  [(#2173)](https://github.com/PennyLaneAI/pennylane/pull/2173)
  [(#2166)](https://github.com/PennyLaneAI/pennylane/pull/2166)

  The dipole moment observable can be constructed using `qml.qchem.dipole_moment`:

  ```python
  symbols  = ['H', 'H']
  geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
  mol = qml.qchem.Molecule(symbols, geometry)
  args = [geometry]
  D = qml.qchem.dipole_moment(mol)(*args)
  ```
  
* The efficiency of computing molecular integrals and Hamiltonian is improved. This has been done 
  by adding optimized functions for building fermionic and qubit observables and optimizing 
  the functions used for computing the electron repulsion integrals.
  [(#2316)](https://github.com/PennyLaneAI/pennylane/pull/2316)


* The `6-31G` basis set is added to the qchem basis set repo. This addition allows performing 
  differentiable Hartree-Fock calculations with basis sets beyond the minimal `sto-3g` basis set
  for atoms with atomic number 1-10.
  [(#2372)](https://github.com/PennyLaneAI/pennylane/pull/2372)

  The `6-31G` basis set can be used to construct a Hamiltonian as  

  ```python
  symbols = ["H", "H"]
  geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
  H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, basis="6-31g")
  ```

* External dependencies are replaced with local functions for spin and particle number observables. 
  [(#2197)](https://github.com/PennyLaneAI/pennylane/pull/2197)
  [(#2362)](https://github.com/PennyLaneAI/pennylane/pull/2362)

<h4>Pattern matching optimization 🔎 💎 </h4>

* Added an optimization transform that matches pieces of user-provided identity
  templates in a circuit and replaces them with an equivalent component.
  [(#2032)](https://github.com/PennyLaneAI/pennylane/pull/2032)

  For example, consider the following circuit where we want to
  replace sequence of two `pennylane.S` gates with a
  `pennylane.PauliZ` gate.

  ```python
  def circuit():
      qml.S(wires=0)
      qml.PauliZ(wires=0)
      qml.S(wires=1)
      qml.CZ(wires=[0, 1])
      qml.S(wires=1)
      qml.S(wires=2)
      qml.CZ(wires=[1, 2])
      qml.S(wires=2)
      return qml.expval(qml.PauliX(wires=0))
  ```

  We specify use the following pattern that implements the identity:

  ```python
  with qml.tape.QuantumTape() as pattern:
      qml.S(wires=0)
      qml.S(wires=0)
      qml.PauliZ(wires=0)
  ```

  To optimize the circuit with this identity pattern, we apply the
  `qml.transforms.pattern_matching` transform.

  ```pycon
  >>> dev = qml.device('default.qubit', wires=5)
  >>> qnode = qml.QNode(circuit, dev)
  >>> optimized_qfunc = qml.transforms.pattern_matching_optimization(pattern_tapes=[pattern])(circuit)
  >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
  >>> print(qml.draw(qnode)())
  0: ──S──Z─╭C──────────┤  <X>
  1: ──S────╰Z──S─╭C────┤
  2: ──S──────────╰Z──S─┤
  >>> print(qml.draw(optimized_qnode)())
  0: ──S⁻¹─╭C────┤  <X>
  1: ──Z───╰Z─╭C─┤
  2: ──Z──────╰Z─┤
  ```

  For more details on using pattern matching optimization you can check the
  [corresponding
  documentation](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.pattern_matching_optimization.html)
  and also the following [paper](https://dl.acm.org/doi/full/10.1145/3498325).

<h4>Measure the distance between two unitaries📏</h4>

* Added the `HilbertSchmidt` and the
  `LocalHilbertSchmidt` templates to be used for computing distance measures
  between unitaries.
  [(#2364)](https://github.com/PennyLaneAI/pennylane/pull/2364)

  Given a unitary `U`, `qml.HilberSchmidt` can be used to measure the distance
  between unitaries and to define a cost function (`cost_hst`) used for
  learning a unitary `V` that is equivalent to `U` up to a global phase:
  ```python
  # Represents unitary U
  with qml.tape.QuantumTape(do_queue=False) as u_tape:
      qml.Hadamard(wires=0)

  # Represents unitary V
  def v_function(params):
      qml.RZ(params[0], wires=1)

  @qml.qnode(dev)
  def hilbert_test(v_params, v_function, v_wires, u_tape):
      qml.HilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
      return qml.probs(u_tape.wires + v_wires)

  def cost_hst(parameters, v_function, v_wires, u_tape):
      return (1 - hilbert_test(v_params=parameters, v_function=v_function, v_wires=v_wires, u_tape=u_tape)[0])
  ```
  ```pycon
  >>> cost_hst(parameters=[0.1], v_function=v_function, v_wires=[1], u_tape=u_tape)
  tensor(0.999, requires_grad=True)
  ```
  For more information refer to the [documentation of
  qml.HilbertSchmidt](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.HilbertSchmidt.html).

<h4>More tensor network support 🕸️</h4>

* Adds the `qml.MERA` template for implementing quantum circuits with the shape
  of a multi-scale entanglement renormalization ansatz (MERA).
  [(#2418)](https://github.com/PennyLaneAI/pennylane/pull/2418)

  MERA follows the style of previous tensor network templates and is similar to
  [quantum convolutional neural networks](https://arxiv.org/abs/1810.03787).

  ```python
    def block(weights, wires):
        qml.CNOT(wires=[wires[0],wires[1]])
        qml.RY(weights[0], wires=wires[0])
        qml.RY(weights[1], wires=wires[1])

    n_wires = 4
    n_block_wires = 2
    n_params_block = 2
    n_blocks = qml.MERA.get_n_blocks(range(n_wires),n_block_wires)
    template_weights = [[0.1,-0.3]]*n_blocks

    dev= qml.device('default.qubit',wires=range(n_wires))
    @qml.qnode(dev)
    def circuit(template_weights):
        qml.MERA(range(n_wires),n_block_wires,block, n_params_block, template_weights)
        return qml.expval(qml.PauliZ(wires=1))
  ```
  It may be necessary to reorder the wires to see the MERA architecture clearly:
  ```pycon
  >>> print(qml.draw(circuit,expansion_strategy='device',wire_order=[2,0,1,3])(template_weights))
  2: ───────────────╭C──RY(0.10)──╭X──RY(-0.30)───────────────┤
  0: ─╭X──RY(-0.30)─│─────────────╰C──RY(0.10)──╭C──RY(0.10)──┤
  1: ─╰C──RY(0.10)──│─────────────╭X──RY(-0.30)─╰X──RY(-0.30)─┤  <Z>
  3: ───────────────╰X──RY(-0.30)─╰C──RY(0.10)────────────────┤
  ```

<h4>New transform for transpilation ⚙️ </h4>

* Added a swap based transpiler transform.
  [(#2118)](https://github.com/PennyLaneAI/pennylane/pull/2118)

  The transpile function takes a quantum function and a coupling map as inputs
  and compiles the circuit to ensure that it can be executed on corresponding
  hardware. The transform can be used as a decorator in the following way:

  ```python
  dev = qml.device('default.qubit', wires=4)

  @qml.qnode(dev)
  @qml.transforms.transpile(coupling_map=[(0, 1), (1, 2), (2, 3)])
  def circuit(param):
      qml.CNOT(wires=[0, 1])
      qml.CNOT(wires=[0, 2])
      qml.CNOT(wires=[0, 3])
      qml.PhaseShift(param, wires=0)
      return qml.probs(wires=[0, 1, 2, 3])
  ```
  ```pycon
  >>> print(qml.draw(circuit)(0.3))
  0: ─╭C───────╭C──────────╭C──Rϕ(0.30)─┤ ╭Probs
  1: ─╰X─╭SWAP─╰X────╭SWAP─╰X───────────┤ ├Probs
  2: ────╰SWAP─╭SWAP─╰SWAP──────────────┤ ├Probs
  3: ──────────╰SWAP────────────────────┤ ╰Probs
  ```

<h3>Improvements</h3>

* `QuantumTape` objects are now iterable, allowing iteration over the
  contained operations and measurements.
  [(#2342)](https://github.com/PennyLaneAI/pennylane/pull/2342)

  ```python
  with qml.tape.QuantumTape() as tape:
      qml.RX(0.432, wires=0)
      qml.RY(0.543, wires=0)
      qml.CNOT(wires=[0, 'a'])
      qml.RX(0.133, wires='a')
      qml.expval(qml.PauliZ(wires=[0]))
  ```

  Given a `QuantumTape` object the underlying quantum circuit can be iterated
  over using a `for` loop:

  ```pycon
  >>> for op in tape:
  ...     print(op)
  RX(0.432, wires=[0])
  RY(0.543, wires=[0])
  CNOT(wires=[0, 'a'])
  RX(0.133, wires=['a'])
  expval(PauliZ(wires=[0]))
  ```

  Indexing into the circuit is also allowed via `tape[i]`:

  ```pycon
  >>> tape[0]
  RX(0.432, wires=[0])
  ```

  A tape object can also be converted to a sequence (e.g., to a `list`) of
  operations and measurements:

  ```pycon
  >>> list(tape)
  [RX(0.432, wires=[0]),
   RY(0.543, wires=[0]),
   CNOT(wires=[0, 'a']),
   RX(0.133, wires=['a']),
   expval(PauliZ(wires=[0]))]
  ```

* Added the `QuantumTape.shape` method and `QuantumTape.numeric_type`
  attribute to allow extracting information about the shape and numeric type of
  the output returned by a quantum tape after execution.
  [(#2044)](https://github.com/PennyLaneAI/pennylane/pull/2044)

  ```python
  dev = qml.device("default.qubit", wires=2)
  a = np.array([0.1, 0.2, 0.3])

  def func(a):
      qml.RY(a[0], wires=0)
      qml.RX(a[1], wires=0)
      qml.RY(a[2], wires=0)

  with qml.tape.QuantumTape() as tape:
      func(a)
      qml.state()
  ```
  ```pycon
  >>> tape.shape(dev)
  (1, 4)
  >>> tape.numeric_type
  complex
  ```

* Defined a `MeasurementProcess.shape` method and a
  `MeasurementProcess.numeric_type` attribute to allow extracting information
  about the shape and numeric type of results obtained when evaluating QNodes
  using the specific measurement process.
  [(#2044)](https://github.com/PennyLaneAI/pennylane/pull/2044)

* The parameter-shift Hessian can now be computed for arbitrary
  operations that support the general parameter-shift rule for
  gradients, using `qml.gradients.param_shift_hessian`
  [(#2319)](https://github.com/XanaduAI/pennylane/pull/2319)

  Multiple ways to obtain the
  gradient recipe are supported, in the following order of preference:

  - A custom `grad_recipe`. It is iterated to obtain the shift rule for
    the second-order derivatives in the diagonal entries of the Hessian.

  - Custom `parameter_frequencies`. The second-order shift rule can
    directly be computed using them.

  - An operation's `generator`. Its eigenvalues will be used to obtain
    `parameter_frequencies`, if they are not given explicitly for an operation.

* The strategy for expanding a circuit can now be specified with the
  `qml.specs` transform, for example to calculate the specifications of the
  circuit that will actually be executed by the device
  (`expansion_strategy="device"`).
  [(#2395)](https://github.com/PennyLaneAI/pennylane/pull/2395)

* The `default.qubit` and `default.mixed` devices now skip over identity
  operators instead of performing matrix multiplication
  with the identity.
  [(#2356)](https://github.com/PennyLaneAI/pennylane/pull/2356)
  [(#2365)](https://github.com/PennyLaneAI/pennylane/pull/2365)

* The function `qml.eigvals` is modified to use the efficient `scipy.sparse.linalg.eigsh`
  method for obtaining the eigenvalues of a `SparseHamiltonian`. This `scipy` method is called
  to compute :math:`k` eigenvalues of a sparse :math:`N \times N` matrix if `k` is smaller
  than :math:`N-1`. If a larger :math:`k` is requested, the dense matrix representation of
  the Hamiltonian is constructed and the regular `qml.math.linalg.eigvalsh` is applied.
  [(#2333)](https://github.com/PennyLaneAI/pennylane/pull/2333)

* The function `qml.ctrl` was given the optional argument `control_values=None`.
  If overridden, `control_values` takes an integer or a list of integers corresponding to
  the binary value that each control value should take. The same change is reflected in
  `ControlledOperation`. Control values of `0` are implemented by `qml.PauliX` applied
  before and after the controlled operation
  [(#2288)](https://github.com/PennyLaneAI/pennylane/pull/2288)

* Operators now have a `has_matrix` property denoting whether or not the operator defines a matrix.
  [(#2331)](https://github.com/PennyLaneAI/pennylane/pull/2331)
  [(#2476)](https://github.com/PennyLaneAI/pennylane/pull/2476)

* Circuit cutting now performs expansion to search for wire cuts in contained operations or tapes.
  [(#2340)](https://github.com/PennyLaneAI/pennylane/pull/2340)

* The `qml.draw` and `qml.draw_mpl` transforms are now located in the `drawer` module. They can still be
  accessed via the top-level `qml` namespace.
  [(#2396)](https://github.com/PennyLaneAI/pennylane/pull/2396)

* Raise a warning where caching produces identical shot noise on execution results with finite shots.
  [(#2478)](https://github.com/PennyLaneAI/pennylane/pull/2478)

<h3>Deprecations</h3>

* The `ObservableReturnTypes` `Sample`, `Variance`, `Expectation`, `Probability`, `State`, and `MidMeasure`
  have been moved to `measurements` from `operation`.
  [(#2329)](https://github.com/PennyLaneAI/pennylane/pull/2329)
  [(#2481)](https://github.com/PennyLaneAI/pennylane/pull/2481)

<h3>Breaking changes</h3>

* The caching ability of devices has been removed. Using the caching on
  the QNode level is the recommended alternative going forward.
  [(#2443)](https://github.com/PennyLaneAI/pennylane/pull/2443)

  One way for replicating the removed `QubitDevice` caching behaviour is by
  creating a `cache` object (e.g., a dictionary) and passing it to the `QNode`:
  ```python
  n_wires = 4
  wires = range(n_wires)

  dev = qml.device('default.qubit', wires=n_wires)

  cache = {}

  @qml.qnode(dev, diff_method='parameter-shift', cache=cache)
  def expval_circuit(params):
      qml.templates.BasicEntanglerLayers(params, wires=wires, rotation=qml.RX)
      return qml.expval(qml.PauliZ(0) @ qml.PauliY(1) @ qml.PauliX(2) @ qml.PauliZ(3))

  shape = qml.templates.BasicEntanglerLayers.shape(5, n_wires)
  params = np.random.random(shape)
  ```
  ```pycon
  >>> expval_circuit(params)
  tensor(0.20598436, requires_grad=True)
  >>> dev.num_executions
  1
  >>> expval_circuit(params)
  tensor(0.20598436, requires_grad=True)
  >>> dev.num_executions
  1
  ```

* The `qml.finite_diff` function has been removed. Please use `qml.gradients.finite_diff` to compute
  the gradient of tapes of QNodes. Otherwise, manual implementation is required.
  [(#2464)](https://github.com/PennyLaneAI/pennylane/pull/2464)

* The `get_unitary_matrix` transform has been removed, please use
  `qml.matrix` instead.
  [(#2457)](https://github.com/PennyLaneAI/pennylane/pull/2457)

* The `update_stepsize` method has been removed from `GradientDescentOptimizer` and its child
  optimizers.  The `stepsize` property can be interacted with directly instead.
  [(#2370)](https://github.com/PennyLaneAI/pennylane/pull/2370)

* Most optimizers no longer flatten and unflatten arguments during computation. Due to this change, user
  provided gradient functions *must* return the same shape as `qml.grad`.
  [(#2381)](https://github.com/PennyLaneAI/pennylane/pull/2381)

* The old circuit text drawing infrastructure has been removed.
  [(#2310)](https://github.com/PennyLaneAI/pennylane/pull/2310)

  - `RepresentationResolver` was replaced by the `Operator.label` method.
  - `qml.drawer.CircuitDrawer` was replaced by `qml.drawer.tape_text`.
  - `qml.drawer.CHARSETS` was removed because unicode is assumed to be accessible.
  - `Grid` and `qml.drawer.drawable_grid` were removed because the custom data class was replaced
    by list of sets of operators or measurements.
  - `qml.transforms.draw_old` was replaced by `qml.draw`.
  - `qml.CircuitGraph.greedy_layers` was deleted, as it was no longer needed by the circuit drawer and
    did not seem to have uses outside of that situation.
  - `qml.CircuitGraph.draw` was deleted, as we draw tapes instead.
  - The tape method `qml.tape.QuantumTape.draw` now simply calls `qml.drawer.tape_text`.
  - In the new pathway, the `charset` keyword was deleted, the `max_length` keyword defaults to `100`, and
    the `decimals` and `show_matrices` keywords were added.

* The deprecated QNode, available via `qml.qnode_old.QNode`, has been removed. Please
  transition to using the standard `qml.QNode`.
  [(#2336)](https://github.com/PennyLaneAI/pennylane/pull/2336)
  [(#2337)](https://github.com/PennyLaneAI/pennylane/pull/2337)
  [(#2338)](https://github.com/PennyLaneAI/pennylane/pull/2338)

  In addition, several other components which powered the deprecated QNode have been removed:

  - The deprecated, non-batch compatible interfaces, have been removed.

  - The deprecated tape subclasses `QubitParamShiftTape`, `JacobianTape`, `CVParamShiftTape`, and
    `ReversibleTape` have been removed.

* The deprecated tape execution method `tape.execute(device)` has been removed. Please use
  `qml.execute([tape], device)` instead.
  [(#2339)](https://github.com/PennyLaneAI/pennylane/pull/2339)

<h3>Bug fixes</h3>

* Fixed a bug in the `qml.PauliRot` operation, where computing the generator was
  not taking into account the operation wires.
  [(#2466)](https://github.com/PennyLaneAI/pennylane/pull/2466)

* Fixed a bug where non-trainable arguments were shifted in the `NesterovMomentumOptimizer`
  if a trainable argument was after it in the argument list.
  [(#2466)](https://github.com/PennyLaneAI/pennylane/pull/2466)

* Fixed a bug with `@jax.jit` for grad when `diff_method="adjoint"` and `mode="backward"`.
  [(#2460)](https://github.com/PennyLaneAI/pennylane/pull/2460)

* Fixed a bug where `qml.DiagonalQubitUnitary` did not support `@jax.jit`
  and `@tf.function`.
  [(#2445)](https://github.com/PennyLaneAI/pennylane/pull/2445)

* Fixed a bug in the `qml.PauliRot` operation, where computing the generator was not taking into
  account the operation wires.
  [(#2442)](https://github.com/PennyLaneAI/pennylane/pull/2442)

* Fixed a bug with the padding capability of `AmplitudeEmbedding` where the
  inputs are on the GPU.
  [(#2431)](https://github.com/PennyLaneAI/pennylane/pull/2431)

* Fixed a bug by adding a comprehensible error message for calling `qml.probs`
  without passing wires or an observable.
  [(#2438)](https://github.com/PennyLaneAI/pennylane/pull/2438)

* The behaviour of `qml.about()` was modified to avoid warnings being emitted
  due to legacy behaviour of `pip`.
  [(#2422)](https://github.com/PennyLaneAI/pennylane/pull/2422)

* Fixed a bug where observables were not considered when determining the use of
  the `jax-jit` interface.
  [(#2427)](https://github.com/PennyLaneAI/pennylane/pull/2427)
  [(#2474)](https://github.com/PennyLaneAI/pennylane/pull/2474)

* Fixed a bug where computing statistics for a relatively few number of shots
  (e.g., `shots=10`), an error arose due to indexing into an array using a
  `DeviceArray`.
  [(#2427)](https://github.com/PennyLaneAI/pennylane/pull/2427)

* PennyLane Lightning version in Docker container is pulled from latest wheel-builds.
  [(#2416)](https://github.com/PennyLaneAI/pennylane/pull/2416)

* Optimizers only consider a variable trainable if they have `requires_grad = True`.
  [(#2381)](https://github.com/PennyLaneAI/pennylane/pull/2381)

* Fixed a bug with `qml.expval`, `qml.var`, `qml.state` and
  `qml.probs` (when `qml.probs` is the only measurement) where the `dtype`
  specified on the device did not match the `dtype` of the QNode output.
  [(#2367)](https://github.com/PennyLaneAI/pennylane/pull/2367)

* Fixed a bug where the output shapes from batch transforms are inconsistent
  with the QNode output shape.
  [(#2215)](https://github.com/PennyLaneAI/pennylane/pull/2215)

* Fixed a bug caused by the squeezing in `qml.gradients.param_shift_hessian`.
  [(#2215)](https://github.com/PennyLaneAI/pennylane/pull/2215)

* Fixed a bug in which the `expval`/`var` of a `Tensor(Observable)` would depend on the order
  in which the observable is defined:
  [(#2276)](https://github.com/PennyLaneAI/pennylane/pull/2276)
  ```pycon
  >>> @qml.qnode(dev)
  ... def circ(op):
  ...   qml.RX(0.12, wires=0)
  ...   qml.RX(1.34, wires=1)
  ...   qml.RX(3.67, wires=2)
  ...   return qml.expval(op)
  >>> op1 = qml.Identity(wires=0) @ qml.Identity(wires=1) @ qml.PauliZ(wires=2)
  >>> op2 = qml.PauliZ(wires=2) @ qml.Identity(wires=0) @ qml.Identity(wires=1)
  >>> print(circ(op1), circ(op2))
  -0.8636111153905662 -0.8636111153905662
  ```

* Fixed a bug where `qml.hf.transform_hf()` would fail due to missing wires in
  the qubit operator that is prepared for tapering the HF state.
  [(#2441)](https://github.com/PennyLaneAI/pennylane/pull/2441)

* Fixed a bug with custom device defined jacobians not being returned properly.
  [(#2485)](https://github.com/PennyLaneAI/pennylane/pull/2485)

<h3>Documentation</h3>

* The sections on adding operator and observable support in the "How to add a
  plugin" section of the plugins page have been updated.
  [(#2389)](https://github.com/PennyLaneAI/pennylane/pull/2389)

* The missing arXiv reference in the `LieAlgebra` optimizer has been fixed.
  [(#2325)](https://github.com/PennyLaneAI/pennylane/pull/2325)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Karim Alaa El-Din, Guillermo Alonso-Linaje, Juan Miguel Arrazola, Ali Asadi,
Utkarsh Azad, Sam Banning, Thomas Bromley, Alain Delgado, Isaac De Vlugt,
Olivia Di Matteo, Amintor Dusko, Anthony Hayes, David Ittah, Josh Izaac, Soran
Jahangiri, Nathan Killoran, Christina Lee, Angus Lowe, Romain Moyard, Zeyue
Niu, Matthew Silverman, Lee James O'Riordan, Maria Schuld, Jay Soni, Antal
Száva, Maurice Weber, David Wierichs.
