:orphan:

# Release 0.23.0-dev (development release)

<h3>New features since last release</h3>

* Added an optimization transform that matches pieces of user-provided identity templates in a circuit and replaces them with an equivalent component.
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
  
* Added two new templates the `HilbertSchmidt` template and the `LocalHilbertSchmidt` template.
  [(#2364)](https://github.com/PennyLaneAI/pennylane/pull/2364)
  
  ```python
  with qml.tape.QuantumTape(do_queue=False) as u_tape:
      qml.Hadamard(wires=0)

  def v_function(params):
      qml.RZ(params[0], wires=1)
  
  @qml.qnode(dev)
  def hilbert_test(v_params, v_function, v_wires, u_tape):
      qml.HilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
      return qml.probs(u_tape.wires + v_wires)

  def cost_hst(parameters, v_function, v_wires, u_tape):
      return (1 - hilbert_test(v_params=parameters, v_function=v_function, v_wires=v_wires, u_tape=u_tape)[0])
  
  cost = cost_hst(v_params=[0.1], v_function=v_function, v_wires=[1], u_tape=u_tape)
  ```

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
  - The qchem and new hf modules are merged
    [(#2385)](https://github.com/PennyLaneAI/pennylane/pull/2385)
  - The 6-31G basis set is added to the qchem basis set repo
    [(#2372)](https://github.com/PennyLaneAI/pennylane/pull/2372)
  - The dependency on openbabel is removed
    [(#2415)](https://github.com/PennyLaneAI/pennylane/pull/2415)
  - The tapering functions are added to qchem
    [(#2426)](https://github.com/PennyLaneAI/pennylane/pull/2426)
  - Differentiable and non-differentiable backends can be selected for building a Hamiltonian
    [(#2441)](https://github.com/PennyLaneAI/pennylane/pull/2441)
  - The quantum chemistry functionalities are unified
    [(#2420)](https://github.com/PennyLaneAI/pennylane/pull/2420)

* Adds a MERA template.
  [(#2418)](https://github.com/PennyLaneAI/pennylane/pull/2418)

  Quantum circuits with the shape
  of a multi-scale entanglement renormalization ansatz can now be easily implemented
  using the new `qml.MERA` template. This follows the style of previous 
  tensor network templates and is similar to
  [quantum convolutional neural networks](https://arxiv.org/abs/1810.03787).
  ```python
    import pennylane as qml
    import numpy as np

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

* <h4> Finite-shot circuit cutting ✂️</h4>

  * You can now run `N`-wire circuits containing sample-based measurements on
    devices with fewer than `N` wires by inserting `WireCut` operations into
    the circuit and decorating your QNode with `@qml.cut_circuit_mc`.
    With this, samples from the original circuit can be simulated using
    a Monte Carlo method,
    using fewer qubits at the expense of more device executions. Additionally,
    this transform
    can take an optional classical processing function as an argument
    and return an expectation value.
    [(#2313)](https://github.com/PennyLaneAI/pennylane/pull/2313)
    [(#2321)](https://github.com/PennyLaneAI/pennylane/pull/2321)
    [(#2332)](https://github.com/PennyLaneAI/pennylane/pull/2332)
    [(#2358)](https://github.com/PennyLaneAI/pennylane/pull/2358)
    [(#2382)](https://github.com/PennyLaneAI/pennylane/pull/2382)
    [(#2399)](https://github.com/PennyLaneAI/pennylane/pull/2399)
    [(#2407)](https://github.com/PennyLaneAI/pennylane/pull/2407)
    [(#2444)](https://github.com/PennyLaneAI/pennylane/pull/2444)

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

    Using the Monte Carlo approach of [Peng et. al](https://arxiv.org/abs/1904.00102), the
    `cut_circuit_mc` transform also supports returning sample-based expectation values of
    observables that are diagonal in the computational basis, as shown below for a `ZZ` measurement
    on wires `0` and `2`:

    ```python
    dev = qml.device("default.qubit", wires=2, shots=10000)

    def observable(bitstring):
        return (-1) ** np.sum(bitstring)

    @qml.cut_circuit_mc(classical_processing_fn=observable)
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

    We can now approximate the expectation value of the observable using

    ```pycon
    >>> circuit(x)
    tensor(-0.776, requires_grad=True)
    ```

  * An automatic graph partitioning method `qcut.kahypar_cut()` has been implemented for cutting
    arbitrary tape-converted graphs using the general purpose graph partitioning framework
    [KaHyPar](https://pypi.org/project/kahypar/) which needs to be installed separately.
    To integrate with the existing low-level manual cut pipeline, method `qcut.find_and_place_cuts()`,
    which uses `qcut.kahypar_cut()` as the default auto cutter, has been implemented.
    The automatic cutting feature is further integrated into the high-level interfaces
    `qcut.cut_circuit()` and `qcut.cut_circuit_mc()` for automatic execution of arbitrary
    circuits on smaller devices.
    [(#2330)](https://github.com/PennyLaneAI/pennylane/pull/2330)
    [(#2428)](https://github.com/PennyLaneAI/pennylane/pull/2428)

<h3>Improvements</h3>

* Added the `QuantumTape.shape` method and `QuantumTape.numeric_type`
  attribute to allow extracting information about the shape and numeric type of
  quantum tapes.
  [(#2044)](https://github.com/PennyLaneAI/pennylane/pull/2044)

* Defined a `MeasurementProcess.shape` method and a
  `MeasurementProcess.numeric_type` attribute.
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

* Most compilation transforms, and relevant subroutines, have been updated to
  support just-in-time compilation with `jax.jit`.
  [(#1894)](https://github.com/PennyLaneAI/pennylane/pull/1894/)

* The `qml.specs` transform now accepts an `expansion_strategy` keyword argument.
  [(#2395)](https://github.com/PennyLaneAI/pennylane/pull/2395)

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

* The `qml.draw` and `qml.draw_mpl` transforms are now located in the `drawer` module. They can still be
  accessed via the top-level `qml` namespace.
  [(#2396)](https://github.com/PennyLaneAI/pennylane/pull/2396)

<h3>Deprecations</h3>

<h3>Breaking changes</h3>

* The `get_unitary_matrix` transform has been removed, users should use
  `qml.matrix` instead.
  [(#2457)](https://github.com/PennyLaneAI/pennylane/pull/2457)

* The caching ability of `QubitDevice` has been removed, using the caching on
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

* The `update_stepsize` method is being deleted from `GradientDescentOptimizer` and its child
  optimizers.  The `stepsize` property can be interacted with directly instead.
  [(#2370)](https://github.com/PennyLaneAI/pennylane/pull/2370)

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

* Fixes a bug where `qml.DiagonalQubitUnitary` did not support `@jax.jit`
  and `@tf.function`.
  [(#2445)](https://github.com/PennyLaneAI/pennylane/pull/2445)

* Fixes a bug in the `qml.PauliRot` operation, where computing the generator was not taking into
  account the operation wires.
  [(#2442)](https://github.com/PennyLaneAI/pennylane/pull/2442)

* Fixes a bug with the padding capability of `AmplitudeEmbedding` where the
  inputs are on the GPU.
  [(#2431)](https://github.com/PennyLaneAI/pennylane/pull/2431)

* Fixes a bug by adding a comprehensible error message for calling `qml.probs`
  without passing wires or an observable.
  [(#2438)](https://github.com/PennyLaneAI/pennylane/pull/2438)

* Call `pip show` with the subprocess library to avoid outputting a common warning.
  [(#2422)](https://github.com/PennyLaneAI/pennylane/pull/2422)

* Fixes a bug where observables were not considered when determining the use of
  the `jax-jit` interface.
  [(#2427)](https://github.com/PennyLaneAI/pennylane/pull/2427)

* Fixes a bug where computing statistics for a relatively few number of shots
  (e.g., `shots=10`), an error arose due to indexing into an array using a
  `DeviceArray`.
  [(#2427)](https://github.com/PennyLaneAI/pennylane/pull/2427)

* PennyLane Lightning version in Docker container is pulled from latest wheel-builds.
  [(#2416)](https://github.com/PennyLaneAI/pennylane/pull/2416)

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

* Fixes a bug where `qml.hf.transform_hf()` would fail due to missing wires in
  the qubit operator that is prepared for tapering the HF state.  
  [(#2441)](https://github.com/PennyLaneAI/pennylane/pull/2441)

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Karim Alaa El-Din, Guillermo Alonso-Linaje, Juan Miguel Arrazola, Utkarsh Azad, Thomas Bromley, Alain Delgado,
Olivia Di Matteo, Anthony Hayes, David Ittah, Josh Izaac, Soran Jahangiri, Christina Lee, Romain Moyard, Zeyue Niu,
Matthew Silverman, Lee James O'Riordan, Jay Soni, Antal Száva, Maurice Weber, David Wierichs.
