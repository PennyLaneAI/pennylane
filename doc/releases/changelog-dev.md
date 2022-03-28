:orphan:

# Release 0.23.0-dev (development release)

<h3>New features since last release</h3>

* Adds an optimization transform that matches pieces of user-provided identity templates in a circuit and replaces them with an equivalent component.
  [(#2032)](https://github.com/PennyLaneAI/pennylane/pull/2032)
  
  First let's consider the following circuit where we want to replace sequence of two ``pennylane.S`` gates with a
  ``pennylane.PauliZ`` gate.
  
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

  Therefore we use the following pattern that implements the identity:

  ```python
  with qml.tape.QuantumTape() as pattern:
      qml.S(wires=0)
      qml.S(wires=0)
      qml.PauliZ(wires=0)
  ```

  For optimizing the circuit given the given following template of CNOTs we apply the `pattern_matching`
  transform.
  
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

  For more details on using pattern matching optimization you can check the corresponding documentation and also the
  following [paper](https://dl.acm.org/doi/full/10.1145/3498325).

* Added a swap based transpiler transform.
  [(#2118)](https://github.com/PennyLaneAI/pennylane/pull/2118)

  The transpile function takes a quantum function and a coupling map as inputs and compiles the circuit to ensure that it can be 
  executed on corresponding hardware. The transform can be used as a decorator in the following way:

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

* A differentiable quantum chemistry module is added to `qml.qchem`. The new module inherits a 
  modified version of the differentiable Hartree-Fock solver from `qml.hf`, contains new functions
  for building a differentiable dipole moment observable and also contains modified functions for 
  building spin and particle number observables independent of external libraries.

  - New functions are added for computing multipole moment molecular integrals
    [(#2166)](https://github.com/PennyLaneAI/pennylane/pull/2166)
  - New functions are added for building a differentiable dipole moment observable
    [(#2173)](https://github.com/PennyLaneAI/pennylane/pull/2173)
  - External dependencies are replaced with local functions for spin and particle number observables
    [(#2197)](https://github.com/PennyLaneAI/pennylane/pull/2197)
    [(#2362)](https://github.com/PennyLaneAI/pennylane/pull/2362)
  - New functions are added for building fermionic and qubit observables
    [(#2230)](https://github.com/PennyLaneAI/pennylane/pull/2230)
  - A new module is created for hosting openfermion to pennylane observable conversion functions
    [(#2199)](https://github.com/PennyLaneAI/pennylane/pull/2199)
    [(#2371)](https://github.com/PennyLaneAI/pennylane/pull/2371)
  - Expressive names are used for the Hartree-Fock solver functions
    [(#2272)](https://github.com/PennyLaneAI/pennylane/pull/2272)
  - These new additions are added to a feature branch
    [(#2164)](https://github.com/PennyLaneAI/pennylane/pull/2164)
  - The efficiency of computing molecular integrals and Hamiltonian is improved
    [(#2316)](https://github.com/PennyLaneAI/pennylane/pull/2316)

* Development of a circuit-cutting compiler extension to circuits with sampling
  measurements has begun:

    - The existing `qcut.tape_to_graph()` method has been extended to convert a
    sample measurement without an observable specified to multiple single-qubit sample
    nodes.
    [(#2313)](https://github.com/PennyLaneAI/pennylane/pull/2313)
  - An automatic graph partitioning method `qcut.kahypar_cut()` has been implemented for cutting
    arbitrary tape-converted graphs using the general purpose graph partitioning framework
    [KaHyPar](https://pypi.org/project/kahypar/) which needs to be installed separately.
    To integrate with the existing manual cut pipeline, method `qcut.find_and_place_cuts()` and related
    utilities are implemented which uses `qcut.kahypar_cut()` as the default auto cutter.
    [(#2330)](https://github.com/PennyLaneAI/pennylane/pull/2330)

  - The existing `qcut.graph_to_tape()` method has been extended to convert
    graphs containing sample measurement nodes to tapes.
    [(#2321)](https://github.com/PennyLaneAI/pennylane/pull/2321)

  - A `qcut.expand_fragment_tapes_mc()` method has been added to expand fragment
    tapes to random configurations by replacing measure and prepare nodes with
    sampled Pauli measurements and state preparations.
    [(#2332)](https://github.com/PennyLaneAI/pennylane/pull/2332)

  - Postprocessing functions `qcut.qcut_processing_fn_sample()` and
    `qcut.qcut_processing_fn_mc()` have been added to return samples and expectation
    values, respectively, of recombined fragments using the Monte Carlo sampling
    approach.
    [(#2358)](https://github.com/PennyLaneAI/pennylane/pull/2358)

<h3>Improvements</h3>

* `default.qubit` and `default.mixed` now skip over identity operators instead of performing matrix multiplication
  with the identity.
  [(#2356)](https://github.com/PennyLaneAI/pennylane/pull/2356)
  [(#2365)](https://github.com/PennyLaneAI/pennylane/pull/2365)

* `QuantumTape` objects are now iterable and accessing the
  operations and measurements of the underlying quantum circuit is more
  seamless.
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
  
* Circuit cutting now performs expansion to search for wire cuts in contained operations or tapes.
  [(#2340)](https://github.com/PennyLaneAI/pennylane/pull/2340)

<h3>Deprecations</h3>

<h3>Breaking changes</h3>

* Most optimizers no longer flatten and unflatten arguments during computation. Due to this change, user
  provided gradient functions *must* return the same shape as `qml.grad`.
  [(#2381)](https://github.com/PennyLaneAI/pennylane/pull/2381)

* The old circuit text drawing infrastructure is being deleted.
  [(#2310)](https://github.com/PennyLaneAI/pennylane/pull/2310)

  - `qml.drawer.CircuitDrawer` is replaced by `qml.drawer.tape_text`.
  - `qml.drawer.CHARSETS` is deleted because we now assume everyone has access to unicode.
  - `Grid` and `qml.drawer.drawable_grid` are removed because the custom data class is replaced
      by list of sets of operators or measurements.
  - `RepresentationResolver` is replaced by the `Operator.label` method.
  - `qml.transforms.draw_old` is replaced by `qml.draw`.
  - `qml.CircuitGraph.greedy_layers` is deleted, as it is no longer needed by the circuit drawer and
      does not seem to have uses outside of that situation.
  - `qml.CircuitGraph.draw` has been deleted, as we draw tapes instead.

The tape method `qml.tape.QuantumTape.draw` now simply calls `qml.drawer.tape_text`. 
In the new pathway, the `charset` keyword is deleted, the `max_length` keyword defaults to `100`, and
the `decimals` and `show_matrices` keywords are added. `qml.drawer.tape_text(tape)`

* The `ObservableReturnTypes` `Sample`, `Variance`, `Expectation`, `Probability`, `State`, and `MidMeasure`
  have been moved to `measurements` from `operation`.
  [(#2329)](https://github.com/PennyLaneAI/pennylane/pull/2329)

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

* Optimizers only consider a variable trainable if they have `requires_grad = True`.
  [(#2381)](https://github.com/PennyLaneAI/pennylane/pull/2381)

* Fixes a bug with `qml.expval`, `qml.var`, `qml.state` and
  `qml.probs` (when `qml.probs` is the only measurement) where the `dtype`
  specified on the device did not match the `dtype` of the QNode output.
  [(#2367)](https://github.com/PennyLaneAI/pennylane/pull/2367)

* Fixes cases with `qml.measure` where unexpected operations were added to the
  circuit.
  [(#2328)](https://github.com/PennyLaneAI/pennylane/pull/2328)

* Fixes a bug in which the `expval`/`var` of a `Tensor(Observable)` would depend on the order 
  in which the observable is defined: 
  ```python
  @qml.qnode(dev)
  def circ(op):
    qml.RX(0.12, wires=0)
    qml.RX(1.34, wires=1)
    qml.RX(3.67, wires=2)
    
    return qml.expval(op)
  
  op1 = qml.Identity(wires=0) @ qml.Identity(wires=1) @ qml.PauliZ(wires=2)
  op2 = qml.PauliZ(wires=2) @ qml.Identity(wires=0) @ qml.Identity(wires=1)
  ```

  ```
  >>> print(circ(op1), circ(op2))
  -0.8636111153905662 -0.8636111153905662
  ```
  [(#2276)](https://github.com/PennyLaneAI/pennylane/pull/2276)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Karim Alaa El-Din, Guillermo Alonso-Linaje, Juan Miguel Arrazola, Thomas Bromley, Alain Delgado,
Anthony Hayes, David Ittah, Josh Izaac, Soran Jahangiri, Christina Lee, Romain Moyard, Zeyue Niu,
Jay Soni, Antal Száva, Maurice Weber.
