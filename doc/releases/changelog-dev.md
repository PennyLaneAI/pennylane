:orphan:

# Release 0.34.0-dev (development release)

<h3>New features since last release</h3>

<h4>Statistics and drawings for mid-circuit measurements 🎨</h4>

* Mid-circuit measurements can now be visualized with the text-based `qml.draw()` and the 
  graphical `qml.draw_mpl()`.
  [(#4775)](https://github.com/PennyLaneAI/pennylane/pull/4775)
  [(#4803)](https://github.com/PennyLaneAI/pennylane/pull/4803)
  [(#4832)](https://github.com/PennyLaneAI/pennylane/pull/4832)
  [(#4901)](https://github.com/PennyLaneAI/pennylane/pull/4901)
  [(#4850)](https://github.com/PennyLaneAI/pennylane/pull/4850)
  [(#4917)](https://github.com/PennyLaneAI/pennylane/pull/4917)
  [(#4930)](https://github.com/PennyLaneAI/pennylane/pull/4930)

  Drawing of mid-circuit measurement capabilities including qubit reuse and reset,
  postselection, conditioning, and collecting statistics is supported.

  ```python
  import pennylane as qml

  def circuit():
      m0 = qml.measure(0, reset=True)
      m1 = qml.measure(1, postselect=1)
      qml.cond(m0 - m1 == 0, qml.S)(0)
      m2 = qml.measure(1)
      qml.cond(m0 + m1 == 2, qml.T)(0)
      qml.cond(m2, qml.PauliX)(1)
  ```
  
  The text-based drawer outputs:

  ```pycon
  >>> print(qml.draw(circuit)())
  0: ──┤↗│  │0⟩────────S───────T────┤  
  1: ───║────────┤↗₁├──║──┤↗├──║──X─┤  
        ╚═════════║════╬═══║═══╣  ║    
                  ╚════╩═══║═══╝  ║    
                           ╚══════╝    
  ```
  
  The graphical drawer outputs:

  ```pycon
  >>> print(qml.draw_mpl(circuit)())
  ```
  
  <img src="https://docs.pennylane.ai/en/latest/_images/mid-circuit-measurement.png" width=70%/>

* Users can now return statistics for multiple mid-circuit measurements.
  [(#4888)](https://github.com/PennyLaneAI/pennylane/pull/4888)

  There are two ways in which mid-circuit measurement statistics can be collected:

  * By using arithmetic/binary operators. This can be through unary or binary operators as such:

    ```python
    import pennylane as qml

    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def circuit(phi, theta):
        qml.RX(phi, wires=0)
        m0 = qml.measure(wires=0)
        qml.RY(theta, wires=1)
        m1 = qml.measure(wires=1)
        return qml.expval(~m0 + m1)

    print(circuit(1.23, 4.56))
    ```
    ```
    1.2430187928114291
    ```

  * By using a list of mid-circuit measurement values:

    ```python
    import pennylane as qml

    dev = qml.device("default.qubit")

    @qml.qnode(dev)
    def circuit(phi, theta):
        qml.RX(phi, wires=0)
        m0 = qml.measure(wires=0)
        qml.RY(theta, wires=1)
        m1 = qml.measure(wires=1)
        return qml.sample([m0, m1])

    print(circuit(1.23, 4.56, shots=5))
    ```
    ```
    [[0 1]
     [0 1]
     [0 0]
     [1 0]
     [0 1]]
    ```

  This feature is supported on `default.qubit`, `default.qubit.legacy`, and `default.mixed`. To
  learn more about which measurements and arithmetic operators are supported, refer to the
  [Measurements](https://docs.pennylane.ai/en/stable/introduction/measurements.html) page and the
  documentation for [`qml.measure`](https://docs.pennylane.ai/en/stable/code/api/pennylane.measure.html).

<h4>Catalyst is seamlessly integrated with PennyLane ⚗️</h4>

* Catalyst, our next-generation compilation framework, is now accessible within PennyLane,
  allowing you to more easily benefit from hybrid just-in-time (JIT) compilation.

  To access these features, simply install `pennylane-catalyst`:

  ```
  pip install pennylane-catalyst
  ```

  The [`qml.compiler`](https://docs.pennylane.ai/en/latest/code/qml_compiler.html) 
  module provides support for hybrid quantum-classical compilation.
  [(#4692)](https://github.com/PennyLaneAI/pennylane/pull/4692)

  Through the use of the `qml.qjit` decorator, entire workflows can be JIT
  compiled — including both quantum and classical processing — down to a machine binary on
  first-function execution. Subsequent calls to the compiled function will execute
  the previously-compiled binary, resulting in significant performance improvements.

  ```python
  import pennylane as qml

  dev = qml.device("lightning.qubit", wires=2)

  @qml.qjit
  @qml.qnode(dev)
  def circuit(theta):
      qml.Hadamard(wires=0)
      qml.RX(theta, wires=1)
      qml.CNOT(wires=[0,1])
      return qml.expval(qml.PauliZ(wires=1))
  ```

  ```pycon
  >>> circuit(0.5)  # the first call, compilation occurs here
  array(0.)
  >>> circuit(0.5)  # the precompiled quantum function is called
  array(0.)
  ```

  Currently, PennyLane supports the [Catalyst hybrid compiler](https://github.com/pennylaneai/catalyst)
  hybrid compiler with the `qml.qjit` decorator. A significant benefit of Catalyst
  is the ability to preserve complex control flow around quantum operations — such as
  if statements and for loops, and including measurement feedback — during compilation,
  while continuing to support end-to-end autodifferentiation. 


* The following functions can now be used with the `qml.qjit` decorator: `qml.grad`, 
  `qml.jacobian`, `qml.vjp`, `qml.jvp`, and `qml.adjoint`.
  [(#4709)](https://github.com/PennyLaneAI/pennylane/pull/4709)
  [(#4724)](https://github.com/PennyLaneAI/pennylane/pull/4724)
  [(#4725)](https://github.com/PennyLaneAI/pennylane/pull/4725)
  [(#4726)](https://github.com/PennyLaneAI/pennylane/pull/4726)

  When `qml.grad` or `qml.jacobian` are used with `@qml.qjit`, they are patched to
  [catalyst.grad](https://docs.pennylane.ai/projects/catalyst/en/stable/code/api/catalyst.grad.html) and
  [catalyst.jacobian](https://docs.pennylane.ai/projects/catalyst/en/stable/code/api/catalyst.jacobian.html), 
  respectively.

  ``` python
  dev = qml.device("lightning.qubit", wires=1)

  @qml.qjit
  def workflow(x):

      @qml.qnode(dev)
      def circuit(x):
          qml.RX(np.pi * x[0], wires=0)
          qml.RY(x[1], wires=0)
          return qml.probs()

      g = qml.jacobian(circuit)

      return g(x)
  ```

  ``` pycon
  >>> workflow(np.array([2.0, 1.0]))
  array([[-1.32116540e-07,  1.33781874e-07],
          [-4.20735506e-01,  4.20735506e-01]])
  ```

* JIT-compatible functionality for control flow has been added via `qml.for_loop`,
  `qml.while_loop`, and `qml.cond`.
  [(#4698)](https://github.com/PennyLaneAI/pennylane/pull/4698)

  ``` python
  dev = qml.device("lightning.qubit", wires=1)

  @qml.qjit
  @qml.qnode(dev)
  def circuit(n: int, x: float):

      @qml.for_loop(0, n, 1)
      def loop_rx(i, x):
          # perform some work and update (some of) the arguments
          qml.RX(x, wires=0)

          # update the value of x for the next iteration
          return jnp.sin(x)

      # apply the for loop
      final_x = loop_rx(x)

      return qml.expval(qml.PauliZ(0)), final_x
  ```

  ``` pycon
  >>> circuit(7, 1.6)
  (array(0.97926626), array(0.55395718))
  ```

<h4>Decompose circuits into the Clifford+T gateset 🧩</h4>

* The new `qml.clifford_t_decomposition()` transform provides an approximate breakdown 
  of an input circuit into the [Clifford+T](https://en.wikipedia.org/wiki/Clifford_gates) 
  gateset. Behind the scenes, this decomposition is enacted via the `sk_decomposition()` 
  function using the Solovay-Kitaev algorithm.
  [(#4801)](https://github.com/PennyLaneAI/pennylane/pull/4801)
  [(#4802)](https://github.com/PennyLaneAI/pennylane/pull/4802)

  Given a total circuit error `epsilon=0.001`, the following circuit can be decomposed:

  ```python
  import pennylane as qml

  with qml.tape.QuantumTape() as circuit:
      qml.RX(1.1, 0)
      qml.CNOT([0, 1])
      qml.RY(2.2, 0)

  (circuit,), _ = qml.clifford_t_decomposition(circuit, 0.001)
  ```

  The resource requirements of this circuit can also be evaluated.

  ```pycon
  >>> circuit.specs["resources"]
  wires: 2
  gates: 49770
  depth: 49770
  shots: Shots(total=None)
  gate_types:
  {'Adjoint(T)': 13647, 'Hadamard': 22468, 'T': 13651, 'CNOT': 1, 'Adjoint(S)': 1, 'S': 1, 'GlobalPhase': 1}
  gate_sizes:
  {1: 49768, 2: 1, 0: 1}
  ```

<h4>Use an iterative approach for quantum phase estimation 🔄</h4>

* Iterative Quantum Phase Estimation is now available from `qml.iterative_qpe`.
  [(#4804)](https://github.com/PennyLaneAI/pennylane/pull/4804)

  The subroutine can be used similarly to mid-circuit measurements:

  ```python

  import pennylane as qml

  dev = qml.device("default.qubit", shots = 5)

  @qml.qnode(dev)
  def circuit():

    # Initial state
    qml.PauliX(wires = [0])

    # Iterative QPE
    measurements = qml.iterative_qpe(qml.RZ(2., wires = [0]), ancilla = [1], iters = 3)

    return [qml.sample(op = meas) for meas in measurements]
  ```

  ```pycon
  >>> print(circuit())
  [array([0, 0, 0, 0, 0]), array([1, 0, 0, 0, 0]), array([0, 1, 1, 1, 1])]
  ```

  The i-th element in the list refers to the 5 samples generated by the i-th measurement of the algorithm.

<h3>Improvements 🛠</h3>

* Implemented the method `process_counts` in the `ProbabilityMP` class (internal assignment).
  [(#4952)](https://github.com/PennyLaneAI/pennylane/pull/4952)

<h4>Community contributions 🥳</h4>

* The `+=` operand can now be used with a `PauliSentence`, which has also provided
  a performance boost.
  [(#4662)](https://github.com/PennyLaneAI/pennylane/pull/4662)

* The Approximate Quantum Fourier Transform (AQFT) is now available with `qml.AQFT`.
  [(#4715)](https://github.com/PennyLaneAI/pennylane/pull/4715)

* `qml.draw` and `qml.draw_mpl` now render operator IDs.
  [(#4749)](https://github.com/PennyLaneAI/pennylane/pull/4749)

* Non-parametric operators such as `Barrier`, `Snapshot` and `Wirecut` have been grouped together and moved to `pennylane/ops/meta.py`.
  Additionally, the relevant tests have been organized and placed in a new file, `tests/ops/test_meta.py`.
  [(#4789)](https://github.com/PennyLaneAI/pennylane/pull/4789)

* `TRX`, `TRY`, and `TRZ` operators are now differentiable via backpropagation on `default.qutrit`.
  [(#4790)](https://github.com/PennyLaneAI/pennylane/pull/4790)

* The function `qml.equal` now supports `ControlledSequence` operators.
  [(#4829)](https://github.com/PennyLaneAI/pennylane/pull/4829)

* XZX decomposition has been added to the list of supported single-qubit unitary decompositions.
  [(#4862)](https://github.com/PennyLaneAI/pennylane/pull/4862)

* `==` and `!=` operands can now be used with `TransformProgram` and `TransformContainers` instances.
  [(#4858)](https://github.com/PennyLaneAI/pennylane/pull/4858)

* `qml.equal` now supports comparison of `QuantumScript` and `BasisRotation` objects
  [(#4902)](https://github.com/PennyLaneAI/pennylane/pull/4902)
  [(#4919)](https://github.com/PennyLaneAI/pennylane/pull/4919)

* The function ``qml.Snapshot`` now supports arbitrary measurements of type ``StateMeasurement``.
  [(#4876)](https://github.com/PennyLaneAI/pennylane/pull/4908)

<h4>Better support for batching</h4>

* `default.qubit` now can evolve already batched states with `ParametrizedEvolution`
  [(#4863)](https://github.com/PennyLaneAI/pennylane/pull/4863)

* `AmplitudeEmbedding` now supports batching when used with Tensorflow.
  [(#4818)](https://github.com/PennyLaneAI/pennylane/pull/4818)

* `qml.ArbitraryUnitary` now supports batching.
  [(#4745)](https://github.com/PennyLaneAI/pennylane/pull/4745)

* Operator and tape batch sizes are evaluated lazily.
  [(#4911)](https://github.com/PennyLaneAI/pennylane/pull/4911)

<h4>Performance improvements and benchmarking</h4>

* Autograd, PyTorch, and JAX (non-jit) can now use VJPs provided by the device from the new device API. If a device provides
  a vector-Jacobian product, this can be selected by providing `device_vjp=True` to
  `qml.QNode` or `qml.execute`.
  [(#4557)](https://github.com/PennyLaneAI/pennylane/pull/4557)
  [(#4654)](https://github.com/PennyLaneAI/pennylane/pull/4654)
  [(#4878)](https://github.com/PennyLaneAI/pennylane/pull/4878)
  [(#4841)](https://github.com/PennyLaneAI/pennylane/pull/4841)

  ```pycon
  >>> dev = qml.device('default.qubit')
  >>> @qml.qnode(dev, diff_method="adjoint", device_vjp=True)
  >>> def circuit(x):
  ...     qml.RX(x, wires=0)
  ...     return qml.expval(qml.PauliZ(0))
  >>> with dev.tracker:
  ...     g = qml.grad(circuit)(qml.numpy.array(0.1))
  >>> dev.tracker.totals
  {'batches': 1, 'simulations': 1, 'executions': 1, 'vjp_batches': 1, 'vjps': 1}
  >>> g
  -0.09983341664682815
  ```

* `qml.expval` with large `Hamiltonian` objects is now faster and has a significantly lower memory footprint (and constant with respect to the number of `Hamiltonian` terms) when the `Hamiltonian` is a `PauliSentence`. This is due to the introduction of a specialized `dot` method in the `PauliSentence` class which performs `PauliSentence`-`state` products.
  [(#4839)](https://github.com/PennyLaneAI/pennylane/pull/4839)

* `default.qubit` no longer uses a dense matrix for `MultiControlledX` for more than 8 operation wires.
  [(#4673)](https://github.com/PennyLaneAI/pennylane/pull/4673)

* Some relevant Pytests have been updated to enable its use as a suite of benchmarks.
  [(#4703)](https://github.com/PennyLaneAI/pennylane/pull/4703)

* `default.qubit` now applies `GroverOperator` faster by not using its matrix representation but a
  custom rule for `apply_operation`. Also, the matrix representation of `GroverOperator` now runs faster.
  [(#4666)](https://github.com/PennyLaneAI/pennylane/pull/4666)

* A new pipeline to run benchmarks and plot graphs comparing with a fixed reference has been added. This pipeline will run on a schedule and can be activated on a PR with the label `ci:run_benchmarks`.
  [(#4741)](https://github.com/PennyLaneAI/pennylane/pull/4741)

* The benchmarks pipeline has been expanded to export all benchmark data to a single JSON file and a CSV file with runtimes. This includes all references and local benchmarks.
  [(#4873)](https://github.com/PennyLaneAI/pennylane/pull/4873)

<h4>Other improvements</h4>

* `qml.quantum_monte_carlo` now uses the new transform system.
  [(#4708)](https://github.com/PennyLaneAI/pennylane/pull/4708/)

* `qml.simplify` now uses the new transforms API.
  [(#4949)](https://github.com/PennyLaneAI/pennylane/pull/4949)

* The formal requirement that type hinting be providing when using
  the `qml.transform` decorator has been removed. Type hinting can still
  be used, but is now optional. Please use a type checker such as
  [mypy](https://github.com/python/mypy) if you wish to ensure types are
  being passed correctly.
  [(#4942)](https://github.com/PennyLaneAI/pennylane/pull/4942/)

* `SampleMeasurement` now has an optional method `process_counts` for computing the measurement results from a counts
  dictionary.
  [(#4941)](https://github.com/PennyLaneAI/pennylane/pull/4941/)

* A new function called `ops.functions.assert_valid` has been added for checking if an `Operator` class is defined correctly.
  [(#4764)](https://github.com/PennyLaneAI/pennylane/pull/4764)

* `Shots` can now be scaled with `*` via the `__mul__` and `__rmul__` dunders.
  [(#4913)](https://github.com/PennyLaneAI/pennylane/pull/4913)

* `GlobalPhase` now decomposes to nothing in case devices do not support global phases.
  [(#4855)](https://github.com/PennyLaneAI/pennylane/pull/4855)

* Custom operations can now provide their matrix directly through the `Operator.matrix()` method
  without needing to update the `has_matrix` property. `has_matrix` will now automatically be
  `True` if `Operator.matrix` is overridden, even if
  `Operator.compute_matrix` is not.
  [(#4844)](https://github.com/PennyLaneAI/pennylane/pull/4844)

* The logic for re-arranging states before returning them has been improved.
  [(#4817)](https://github.com/PennyLaneAI/pennylane/pull/4817)

* When multiplying `SparseHamiltonian`s by a scalar value, the result now stays as a
  `SparseHamiltonian`.
  [(#4828)](https://github.com/PennyLaneAI/pennylane/pull/4828)

* `trainable_params` can now be set on initialization of `QuantumScript` instead of having to set the
  parameter after initialization.
  [(#4877)](https://github.com/PennyLaneAI/pennylane/pull/4877)

* `default.qubit` now calculates the expectation value of `Hermitian` operators in a differentiable manner.
  [(#4866)](https://github.com/PennyLaneAI/pennylane/pull/4866)

* The `rot` decomposition now has support for returning a global phase.
  [(#4869)](https://github.com/PennyLaneAI/pennylane/pull/4869)

* The `"pennylane_sketch"` MPL-drawer style has been added. This is the same as the `"pennylane"`
  style, but with sketch-style lines.
  [(#4880)](https://github.com/PennyLaneAI/pennylane/pull/4880)

* `Conditional` and `MeasurementValue` objects now implement `map_wires`.
  [(#4884)](https://github.com/PennyLaneAI/pennylane/pull/4884)

* Operators now define a `pauli_rep` property, an instance of `PauliSentence`, defaulting
  to `None` if the operator has not defined it (or has no definition in the pauli basis).
  [(#4915)](https://github.com/PennyLaneAI/pennylane/pull/4915)

* `qml.ShotAdaptiveOptimizer` can now use a multinomial distribution for spreading shots across
  the terms of a Hamiltonian measured in a QNode. Note that this is equivalent to what can be
  done with `qml.ExpvalCost`, but this is the preferred method because `ExpvalCost` is deprecated.
  [(#4896)](https://github.com/PennyLaneAI/pennylane/pull/4896)

* Decomposition of `qml.PhaseShift` now uses `qml.GlobalPhase` for retaining the global phase information. 
  [(#4657)](https://github.com/PennyLaneAI/pennylane/pull/4657)
  [(#4947)](https://github.com/PennyLaneAI/pennylane/pull/4947)

* `qml.equal` for `Controlled` operators no longer returns `False` when equivalent but 
  differently-ordered sets of control wires and control values are compared.
  [(#4944)](https://github.com/PennyLaneAI/pennylane/pull/4944)

* All PennyLane `Operator` subclasses are automatically tested by `ops.functions.assert_valid` to ensure
  that they follow PennyLane `Operator` standards.
  [(#4922)](https://github.com/PennyLaneAI/pennylane/pull/4922)

<h3>Breaking changes 💔</h3>

* The functions `qml.transforms.one_qubit_decomposition`, `qml.transforms.two_qubit_decomposition`, 
  `qml.transforms.sk_decomposition` were moved to respectively, `qml.ops.one_qubit_decomposition`, `qml.ops.two_qubit_decomposition`, 
  `qml.ops.sk_decomposition`.
  [(#4906)](https://github.com/PennyLaneAI/pennylane/pull/4906)

* The function `qml.transforms.classical_jacobian` has been moved to the gradients module
  and is now accessible as `qml.gradients.classical_jacobian`.
  [(#4900)](https://github.com/PennyLaneAI/pennylane/pull/4900)

* The transforms submodule `qml.transforms.qcut` is now its own module: `qml.qcut`.
  [(#4819)](https://github.com/PennyLaneAI/pennylane/pull/4819)

* The decomposition of `GroverOperator` now has an additional global phase operation.
  [(#4666)](https://github.com/PennyLaneAI/pennylane/pull/4666)

* `qml.cond` and the `Conditional` operation have been moved from the `transforms` folder to the `ops/op_math` folder.
  `qml.transforms.Conditional` will now be available as `qml.ops.Conditional`.
  [(#4860)](https://github.com/PennyLaneAI/pennylane/pull/4860)

* The `prep` keyword argument has been removed from `QuantumScript` and `QuantumTape`.
  `StatePrepBase` operations should be placed at the beginning of the `ops` list instead.
  [(#4756)](https://github.com/PennyLaneAI/pennylane/pull/4756)

* `qml.gradients.pulse_generator` is now named `qml.gradients.pulse_odegen` to adhere to paper naming conventions.
  [(#4769)](https://github.com/PennyLaneAI/pennylane/pull/4769)

* Specifying `control_values` passed to `qml.ctrl` as a string is no longer supported.
  [(#4816)](https://github.com/PennyLaneAI/pennylane/pull/4816)

* The `rot` decomposition will now normalize its rotation angles to the range `[0, 4pi]` for consistency
  [(#4869)](https://github.com/PennyLaneAI/pennylane/pull/4869)

* `QuantumScript.graph` is now built using `tape.measurements` instead of `tape.observables`
  because it depended on the now-deprecated `Observable.return_type` property.
  [(#4762)](https://github.com/PennyLaneAI/pennylane/pull/4762)

* The `"pennylane"` MPL-drawer style now draws straight lines instead of sketch-style lines.
  [(#4880)](https://github.com/PennyLaneAI/pennylane/pull/4880)

* The default value for the `term_sampling` argument of `ShotAdaptiveOptimizer` is now
  `None` instead of `"weighted_random_sampling"`.
  [(#4896)](https://github.com/PennyLaneAI/pennylane/pull/4896)

<h3>Deprecations 👋</h3>

* `single_tape_transform`, `batch_transform`, `qfunc_transform`, and `op_transform` are deprecated.
  Use the new `qml.transform` function instead.
  [(#4774)](https://github.com/PennyLaneAI/pennylane/pull/4774)

* `Observable.return_type` is deprecated. Instead, you should inspect the type
  of the surrounding measurement process.
  [(#4762)](https://github.com/PennyLaneAI/pennylane/pull/4762)
  [(#4798)](https://github.com/PennyLaneAI/pennylane/pull/4798)

* All deprecations now raise a `qml.PennyLaneDeprecationWarning` instead of a `UserWarning`.
  [(#4814)](https://github.com/PennyLaneAI/pennylane/pull/4814)

* `QuantumScript.is_sampled` and `QuantumScript.all_sampled` are deprecated.
  Users should now validate these properties manually.
  [(#4773)](https://github.com/PennyLaneAI/pennylane/pull/4773)

<h3>Documentation 📝</h3>

* Documentation for unitaries and operations decompositions was moved from `qml.transforms` to `qml.ops.ops_math`.
  [(#4906)](https://github.com/PennyLaneAI/pennylane/pull/4906)

* Documentation for `qml.metric_tensor` and `qml.adjoint_metric_tensor` and `qml.transforms.classical_jacobian`
  are now accessible via the gradients API page `qml.gradients` in the documentation.
  [(#4900)](https://github.com/PennyLaneAI/pennylane/pull/4900)

* Documentation for `qml.specs` was moved to the resource module.
  [(#4904)](https://github.com/PennyLaneAI/pennylane/pull/4904)

* Documentation for QCut has moved to its own API page `qml.qcut`.
  [(#4819)](https://github.com/PennyLaneAI/pennylane/pull/4819)

* The documentation page for `qml.measurements` now links top-level accessible functions (e.g., `qml.expval`) 
  to their top-level pages rather than their module-level pages (e.g., `qml.measurements.expval`).
  [(#4750)](https://github.com/PennyLaneAI/pennylane/pull/4750)

* Information to the documentation for `qml.matrix` about wire ordering has been added for using `qml.matrix` on a
  `QNode` which uses a device with `device.wires=None`.
  [(#4874)](https://github.com/PennyLaneAI/pennylane/pull/4874)

<h3>Bug fixes 🐛</h3>

* `qml.cond` no longer incorrectly queues operators used as qfunc arguments.
  [(#4948)](https://github.com/PennyLaneAI/pennylane/pull/4948)

* `Attribute` objects now return `False` instead of raising a `TypeError` when checking if an object is inside
  the set.
  [(#4933)](https://github.com/PennyLaneAI/pennylane/pull/4933)

* Fixed a bug where the parameter-shift rule of `qml.ctrl(op)` was wrong if `op` had a generator
  that has two or more eigenvalues and is stored as a `SparseHamiltonian`.
  [(#4899)](https://github.com/PennyLaneAI/pennylane/pull/4899)

* Fixed a bug where trainable parameters in the post-processing of finite-differences were incorrect for JAX when applying
  the transform directly on a QNode.
  [(#4879)](https://github.com/PennyLaneAI/pennylane/pull/4879)

* `qml.grad` and `qml.jacobian` now explicitly raise errors if trainable parameters are integers.
  [(#4836)](https://github.com/PennyLaneAI/pennylane/pull/4836)

* JAX-JIT now works with shot vectors.
  [(#4772)](https://github.com/PennyLaneAI/pennylane/pull/4772/)

* JAX can now differentiate a batch of circuits where one tape does not have trainable parameters.
  [(#4837)](https://github.com/PennyLaneAI/pennylane/pull/4837)

* The decomposition of `GroverOperator` now has the same global phase as its matrix.
  [(#4666)](https://github.com/PennyLaneAI/pennylane/pull/4666)

* The `tape.to_openqasm` method no longer mistakenly includes interface information in the parameter
  string when converting tapes using non-NumPy interfaces.
  [(#4849)](https://github.com/PennyLaneAI/pennylane/pull/4849)

* `qml.defer_measurements` now correctly transforms circuits when terminal measurements include wires
  used in mid-circuit measurements.
  [(#4787)](https://github.com/PennyLaneAI/pennylane/pull/4787)

* Fixed a bug where the adjoint differentiation method would fail if
  an operation that has a parameter with `grad_method=None` is present.
  [(#4820)](https://github.com/PennyLaneAI/pennylane/pull/4820)

* `MottonenStatePreparation` now raises an error if decomposing a broadcasted state vector.
  [(#4767)](https://github.com/PennyLaneAI/pennylane/pull/4767)

* `BasisStatePreparation` now raises an error if decomposing a broadcasted state vector.
  [(#4767)](https://github.com/PennyLaneAI/pennylane/pull/4767)

* Gradient transforms now work with overridden shot vectors and default qubit.
  [(#4795)](https://github.com/PennyLaneAI/pennylane/pull/4795)

* Any `ScalarSymbolicOp`, like `Evolution`, now states that it has a matrix if the target
  is a `Hamiltonian`.
  [(#4768)](https://github.com/PennyLaneAI/pennylane/pull/4768)

* In `default.qubit`, initial states are now initialized with the simulator's wire order, not the circuit's
  wire order.
  [(#4781)](https://github.com/PennyLaneAI/pennylane/pull/4781)

* `transpile` can now handle measurements that are broadcasted onto all wires.
  [(#4793)](https://github.com/PennyLaneAI/pennylane/pull/4793)

* Parametrized circuits whose operators do not act on all wires return PennyLane tensors instead of NumPy arrays, as
  expected.
  [(#4811)](https://github.com/PennyLaneAI/pennylane/pull/4811)
  [(#4817)](https://github.com/PennyLaneAI/pennylane/pull/4817)

* `merge_amplitude_embeddings` no longer depends on queuing, allowing it to work as expected
  with QNodes.
  [(#4831)](https://github.com/PennyLaneAI/pennylane/pull/4831)

* `qml.pow(op)` and `qml.QubitUnitary.pow()` now also work with Tensorflow data raised to an
  integer power.
  [(#4827)](https://github.com/PennyLaneAI/pennylane/pull/4827)

* The text drawer has been fixed to correctly label `qinfo` measurements, as well as `qml.classical_shadow`
  `qml.shadow_expval`.
  [(#4803)](https://github.com/PennyLaneAI/pennylane/pull/4803)

* Removed an implicit assumption that an empty `PauliSentence` gets treated as identity under 
  multiplication.
  [(#4887)](https://github.com/PennyLaneAI/pennylane/pull/4887)

* Using a `CNOT` or `PauliZ` operation with large batched states and the Tensorflow
  interface no longer raises an unexpected error.
  [(#4889)](https://github.com/PennyLaneAI/pennylane/pull/4889)

* `qml.map_wires` no longer fails when mapping nested quantum tapes.
  [(#4901)](https://github.com/PennyLaneAI/pennylane/pull/4901)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Guillermo Alonso,
Ali Asadi,
Utkarsh Azad,
Gabriel Bottrill,
Thomas Bromley,
Astral Cai,
Minh Chau,
Isaac De Vlugt,
Amintor Dusko,
Lillian Frederiksen,
Josh Izaac,
Juan Giraldo,
Emiliano Godinez Ramirez,
Ankit Khandelwal,
Christina Lee,
Vincent Michaud-Rioux,
Anurav Modak,
Romain Moyard,
Mudit Pandey,
Matthew Silverman,
Jay Soni,
David Wierichs,
Justin Woodring.
