
# Release 0.34.0

<h3>New features since last release</h3>

<h4>Statistics and drawing for mid-circuit measurements 🎨</h4>

* It is now possible to return statistics of composite mid-circuit measurements.
  [(#4888)](https://github.com/PennyLaneAI/pennylane/pull/4888)

  Mid-circuit measurement results can be composed using basic arithmetic operations and then
  statistics can be calculated by putting the result within a PennyLane
  [measurement](https://docs.pennylane.ai/en/stable/introduction/measurements.html) like
  `qp.expval()`. For example:

  ```python
  import pennylane as qp

  dev = qp.device("default.qubit")

  @qp.qnode(dev)
  def circuit(phi, theta):
      qp.RX(phi, wires=0)
      m0 = qp.measure(wires=0)
      qp.RY(theta, wires=1)
      m1 = qp.measure(wires=1)
      return qp.expval(~m0 + m1)

  print(circuit(1.23, 4.56))
  ```
  ```
  1.2430187928114291
  ```

  Another option, for ease-of-use when using `qp.sample()`, `qp.probs()`, or `qp.counts()`, is to
  provide a simple list of mid-circuit measurement results:

  ```python
  dev = qp.device("default.qubit")

  @qp.qnode(dev)
  def circuit(phi, theta):
      qp.RX(phi, wires=0)
      m0 = qp.measure(wires=0)
      qp.RY(theta, wires=1)
      m1 = qp.measure(wires=1)
      return qp.sample(op=[m0, m1])

  print(circuit(1.23, 4.56, shots=5))
  ```

  ```
  [[0 1]
   [0 1]
   [0 0]
   [1 0]
   [0 1]]
  ```

  Composite mid-circuit measurement statistics are supported on `default.qubit` and `default.mixed`.
  To learn more about which measurements and arithmetic operators are supported,
  [refer to the measurements page](https://docs.pennylane.ai/en/stable/introduction/measurements.html) and the
  [documentation for qp.measure](https://docs.pennylane.ai/en/stable/code/api/pennylane.measure.html).

* Mid-circuit measurements can now be visualized with the text-based `qp.draw()` and the 
  graphical `qp.draw_mpl()` methods.
  [(#4775)](https://github.com/PennyLaneAI/pennylane/pull/4775)
  [(#4803)](https://github.com/PennyLaneAI/pennylane/pull/4803)
  [(#4832)](https://github.com/PennyLaneAI/pennylane/pull/4832)
  [(#4901)](https://github.com/PennyLaneAI/pennylane/pull/4901)
  [(#4850)](https://github.com/PennyLaneAI/pennylane/pull/4850)
  [(#4917)](https://github.com/PennyLaneAI/pennylane/pull/4917)
  [(#4930)](https://github.com/PennyLaneAI/pennylane/pull/4930)
  [(#4957)](https://github.com/PennyLaneAI/pennylane/pull/4957)

  Drawing of mid-circuit measurement capabilities including qubit reuse and reset,
  postselection, conditioning, and collecting statistics is now supported. Here 
  is an all-encompassing example:

  ```python
  def circuit():
      m0 = qp.measure(0, reset=True)
      m1 = qp.measure(1, postselect=1)
      qp.cond(m0 - m1 == 0, qp.S)(0)
      m2 = qp.measure(1)
      qp.cond(m0 + m1 == 2, qp.T)(0)
      qp.cond(m2, qp.PauliX)(1)
  ```
  
  The text-based drawer outputs:

  ```pycon
  >>> print(qp.draw(circuit)())
  0: ──┤↗│  │0⟩────────S───────T────┤  
  1: ───║────────┤↗₁├──║──┤↗├──║──X─┤  
        ╚═════════║════╬═══║═══╣  ║    
                  ╚════╩═══║═══╝  ║    
                           ╚══════╝    
  ```
  
  The graphical drawer outputs:

  ```pycon
  >>> print(qp.draw_mpl(circuit)())
  ```
  
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/mid-circuit-measurement.png" width=70%/>

<h4>Catalyst is seamlessly integrated with PennyLane ⚗️</h4>

* Catalyst, our next-generation compilation framework, is now accessible within PennyLane,
  allowing you to more easily benefit from hybrid just-in-time (JIT) compilation.

  To access these features, simply install `pennylane-catalyst`:

  ```
  pip install pennylane-catalyst
  ```

  The [qp.compiler](https://docs.pennylane.ai/en/latest/code/qml_compiler.html) 
  module provides support for hybrid quantum-classical compilation. 
  [(#4692)](https://github.com/PennyLaneAI/pennylane/pull/4692)
  [(#4979)](https://github.com/PennyLaneAI/pennylane/pull/4979)

  Through the use of the `qp.qjit` decorator, entire workflows can be JIT
  compiled — including both quantum and classical processing — down to a machine binary on
  first-function execution. Subsequent calls to the compiled function will execute
  the previously-compiled binary, resulting in significant performance improvements.

  ```python
  import pennylane as qp

  dev = qp.device("lightning.qubit", wires=2)

  @qp.qjit
  @qp.qnode(dev)
  def circuit(theta):
      qp.Hadamard(wires=0)
      qp.RX(theta, wires=1)
      qp.CNOT(wires=[0,1])
      return qp.expval(qp.PauliZ(wires=1))
  ```

  ```pycon
  >>> circuit(0.5)  # the first call, compilation occurs here
  array(0.)
  >>> circuit(0.5)  # the precompiled quantum function is called
  array(0.)
  ```

  Currently, PennyLane supports the [Catalyst hybrid compiler](https://github.com/pennylaneai/catalyst)
  with the `qp.qjit` decorator. A significant benefit of Catalyst
  is the ability to preserve complex control flow around quantum operations — such as
  `if` statements and `for` loops, and including measurement feedback — during compilation,
  while continuing to support end-to-end autodifferentiation. 

* The following functions can now be used with the `qp.qjit` decorator: `qp.grad`, 
  `qp.jacobian`, `qp.vjp`, `qp.jvp`, and `qp.adjoint`.
  [(#4709)](https://github.com/PennyLaneAI/pennylane/pull/4709)
  [(#4724)](https://github.com/PennyLaneAI/pennylane/pull/4724)
  [(#4725)](https://github.com/PennyLaneAI/pennylane/pull/4725)
  [(#4726)](https://github.com/PennyLaneAI/pennylane/pull/4726)

  When `qp.grad` or `qp.jacobian` are used with `@qp.qjit`, they are patched to
  [catalyst.grad](https://docs.pennylane.ai/projects/catalyst/en/stable/code/api/catalyst.grad.html) and
  [catalyst.jacobian](https://docs.pennylane.ai/projects/catalyst/en/stable/code/api/catalyst.jacobian.html), 
  respectively.

  ``` python
  dev = qp.device("lightning.qubit", wires=1)

  @qp.qjit
  def workflow(x):

      @qp.qnode(dev)
      def circuit(x):
          qp.RX(np.pi * x[0], wires=0)
          qp.RY(x[1], wires=0)
          return qp.probs()

      g = qp.jacobian(circuit)

      return g(x)
  ```

  ``` pycon
  >>> workflow(np.array([2.0, 1.0]))
  array([[ 3.48786850e-16, -4.20735492e-01],
         [-8.71967125e-17,  4.20735492e-01]])
  ```

* JIT-compatible functionality for control flow has been added via `qp.for_loop`,
  `qp.while_loop`, and `qp.cond`.
  [(#4698)](https://github.com/PennyLaneAI/pennylane/pull/4698)
  [(#5006)](https://github.com/PennyLaneAI/pennylane/pull/5006)

  `qp.for_loop` and `qp.while_loop` can be deployed as decorators on functions that are the 
  body of the loop. The arguments to both follow typical conventions: 

  ```
  @qp.for_loop(lower_bound, upper_bound, step)
  ```

  ```
  @qp.while_loop(cond_function)
  ```

  Here is a concrete example with `qp.for_loop`:

  `qp.for_loop` and `qp.while_loop` can be deployed as decorators on functions that are the 
  body of the loop. The arguments to both follow typical conventions: 

  ```
  @qp.for_loop(lower_bound, upper_bound, step)
  ```

  ```
  @qp.while_loop(cond_function)
  ```

  Here is a concrete example with `qp.for_loop`:

  ``` python
  dev = qp.device("lightning.qubit", wires=1)

  @qp.qjit
  @qp.qnode(dev)
  def circuit(n: int, x: float):

      @qp.for_loop(0, n, 1)
      def loop_rx(i, x):
          # perform some work and update (some of) the arguments
          qp.RX(x, wires=0)

          # update the value of x for the next iteration
          return jnp.sin(x)

      # apply the for loop
      final_x = loop_rx(x)

      return qp.expval(qp.PauliZ(0)), final_x
  ```

  ``` pycon
  >>> circuit(7, 1.6)
  (array(0.97926626), array(0.55395718))
  ```

<h4>Decompose circuits into the Clifford+T gateset 🧩</h4>

* The new `qp.clifford_t_decomposition()` transform provides an approximate breakdown 
  of an input circuit into the [Clifford+T gateset](https://en.wikipedia.org/wiki/Clifford_gates).
  Behind the scenes, this decomposition is enacted via the `sk_decomposition()` 
  function using the Solovay-Kitaev algorithm.
  [(#4801)](https://github.com/PennyLaneAI/pennylane/pull/4801)
  [(#4802)](https://github.com/PennyLaneAI/pennylane/pull/4802)

  The Solovay-Kitaev algorithm *approximately* decomposes a quantum circuit into the Clifford+T
  gateset. To account for this, a desired total circuit decomposition error, `epsilon`, must be 
  specified when using `qp.clifford_t_decomposition`:

  ``` python
  dev = qp.device("default.qubit")

  @qp.qnode(dev)
  def circuit():
      qp.RX(1.1, 0)
      return qp.state()

  circuit = qp.clifford_t_decomposition(circuit, epsilon=0.1)
  ```
  
  ``` pycon
  >>> print(qp.draw(circuit)())
  0: ──T†──H──T†──H──T──H──T──H──T──H──T──H──T†──H──T†──T†──H──T†──H──T──H──T──H──T──H──T──H──T†──H

  ───T†──H──T──H──GlobalPhase(0.39)─┤
  ```

  The resource requirements of this circuit can also be evaluated:

  ```pycon
  >>> with qp.Tracker(dev) as tracker:
  ...     circuit()
  >>> resources_lst = tracker.history["resources"]
  >>> resources_lst[0]
  wires: 1
  gates: 34
  depth: 34
  shots: Shots(total=None)
  gate_types:
  {'Adjoint(T)': 8, 'Hadamard': 16, 'T': 9, 'GlobalPhase': 1}
  gate_sizes:
  {1: 33, 0: 1}
  ```

<h4>Use an iterative approach for quantum phase estimation 🔄</h4>

* [Iterative Quantum Phase Estimation](https://arxiv.org/pdf/quant-ph/0610214.pdf)
  is now available with `qp.iterative_qpe`.
  [(#4804)](https://github.com/PennyLaneAI/pennylane/pull/4804)

  The subroutine can be used similarly to mid-circuit measurements:

  ```python
  import pennylane as qp

  dev = qp.device("default.qubit", shots=5)

  @qp.qnode(dev)
  def circuit():

    # Initial state
    qp.PauliX(wires=[0])

    # Iterative QPE
    measurements = qp.iterative_qpe(qp.RZ(2., wires=[0]), ancilla=[1], iters=3)

    return [qp.sample(op=meas) for meas in measurements]
  ```

  ```pycon
  >>> print(circuit())
  [array([0, 0, 0, 0, 0]), array([1, 0, 0, 0, 0]), array([0, 1, 1, 1, 1])]
  ```

  The :math:`i`-th element in the list refers to the 5 samples generated by the :math:`i`-th measurement of the algorithm.

<h3>Improvements 🛠</h3>

<h4>Community contributions 🥳</h4>

* The `+=` operand can now be used with a `PauliSentence`, which has also provides
  a performance boost.
  [(#4662)](https://github.com/PennyLaneAI/pennylane/pull/4662)

* The Approximate Quantum Fourier Transform (AQFT) is now available with `qp.AQFT`.
  [(#4715)](https://github.com/PennyLaneAI/pennylane/pull/4715)

* `qp.draw` and `qp.draw_mpl` now render operator IDs.
  [(#4749)](https://github.com/PennyLaneAI/pennylane/pull/4749)

  The ID can be specified as a keyword argument when instantiating an operator:

  ```pycon
  >>> def circuit():
  ...     qp.RX(0.123, id="data", wires=0)
  >>> print(qp.draw(circuit)())
  0: ──RX(0.12,"data")─┤  
  ```

* Non-parametric operators such as `Barrier`, `Snapshot`, and `Wirecut` have been grouped together and moved to `pennylane/ops/meta.py`.
  Additionally, the relevant tests have been organized and placed in a new file, `tests/ops/test_meta.py`.
  [(#4789)](https://github.com/PennyLaneAI/pennylane/pull/4789)

* The `TRX`, `TRY`, and `TRZ` operators are now differentiable via backpropagation on `default.qutrit`.
  [(#4790)](https://github.com/PennyLaneAI/pennylane/pull/4790)

* The function `qp.equal` now supports `ControlledSequence` operators.
  [(#4829)](https://github.com/PennyLaneAI/pennylane/pull/4829)

* XZX decomposition has been added to the list of supported single-qubit unitary decompositions.
  [(#4862)](https://github.com/PennyLaneAI/pennylane/pull/4862)

* `==` and `!=` operands can now be used with `TransformProgram` and `TransformContainers` instances.
  [(#4858)](https://github.com/PennyLaneAI/pennylane/pull/4858)

* A `qutrit_mixed` module has been added to `qp.devices` to store helper functions for a future qutrit 
  mixed-state device. A function called `create_initial_state` has been added to this module that creates 
  device-compatible initial states.
  [(#4861)](https://github.com/PennyLaneAI/pennylane/pull/4861)

* The function `qp.Snapshot` now supports arbitrary state-based measurements (i.e., measurements of type `StateMeasurement`).
  [(#4876)](https://github.com/PennyLaneAI/pennylane/pull/4908)

* `qp.equal` now supports the comparison of `QuantumScript` and `BasisRotation` objects.
  [(#4902)](https://github.com/PennyLaneAI/pennylane/pull/4902)
  [(#4919)](https://github.com/PennyLaneAI/pennylane/pull/4919)

* The function `qp.draw_mpl` now accept a keyword argument `fig` to specify the output figure window.
  [(#4956)](https://github.com/PennyLaneAI/pennylane/pull/4956)

<h4>Better support for batching</h4>

* `qp.AmplitudeEmbedding` now supports batching when used with Tensorflow.
  [(#4818)](https://github.com/PennyLaneAI/pennylane/pull/4818)

* `default.qubit` can now evolve already batched states with `qp.pulse.ParametrizedEvolution`.
  [(#4863)](https://github.com/PennyLaneAI/pennylane/pull/4863)

* `qp.ArbitraryUnitary` now supports batching.
  [(#4745)](https://github.com/PennyLaneAI/pennylane/pull/4745)

* Operator and tape batch sizes are evaluated lazily, helping run expensive computations less frequently
  and an issue with Tensorflow pre-computing batch sizes.
  [(#4911)](https://github.com/PennyLaneAI/pennylane/pull/4911)

<h4>Performance improvements and benchmarking</h4>

* Autograd, PyTorch, and JAX can now use vector-Jacobian products (VJPs) provided by the device from the new device API. If a device provides
  a VJP, this can be selected by providing `device_vjp=True` to a QNode or `qp.execute`.
  [(#4935)](https://github.com/PennyLaneAI/pennylane/pull/4935)
  [(#4557)](https://github.com/PennyLaneAI/pennylane/pull/4557)
  [(#4654)](https://github.com/PennyLaneAI/pennylane/pull/4654)
  [(#4878)](https://github.com/PennyLaneAI/pennylane/pull/4878)
  [(#4841)](https://github.com/PennyLaneAI/pennylane/pull/4841)

  ```pycon
  >>> dev = qp.device('default.qubit')
  >>> @qp.qnode(dev, diff_method="adjoint", device_vjp=True)
  >>> def circuit(x):
  ...     qp.RX(x, wires=0)
  ...     return qp.expval(qp.PauliZ(0))
  >>> with dev.tracker:
  ...     g = qp.grad(circuit)(qp.numpy.array(0.1))
  >>> dev.tracker.totals
  {'batches': 1, 'simulations': 1, 'executions': 1, 'vjp_batches': 1, 'vjps': 1}
  >>> g
  -0.09983341664682815
  ```

* `qp.expval` with large `Hamiltonian` objects is now faster and has a significantly lower memory footprint (and constant with respect to the number of `Hamiltonian` terms) when the `Hamiltonian` is a `PauliSentence`. This is due to the introduction of a specialized `dot` method in the `PauliSentence` class which performs `PauliSentence`-`state` products.
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

* `default.qubit` now supports adjoint differentiation for arbitrary diagonal state-based measurements.
  [(#4865)](https://github.com/PennyLaneAI/pennylane/pull/4865)

* The benchmarks pipeline has been expanded to export all benchmark data to a single JSON file and a CSV file with runtimes. This includes all references and local benchmarks.
  [(#4873)](https://github.com/PennyLaneAI/pennylane/pull/4873)

<h4>Final phase of updates to transforms</h4>

* `qp.quantum_monte_carlo` and `qp.simplify` now use the new transform system.
  [(#4708)](https://github.com/PennyLaneAI/pennylane/pull/4708/)
  [(#4949)](https://github.com/PennyLaneAI/pennylane/pull/4949)

* The formal requirement that type hinting be provided when using
  the `qp.transform` decorator has been removed. Type hinting can still
  be used, but is now optional. Please use a type checker such as
  [mypy](https://github.com/python/mypy) if you wish to ensure types are
  being passed correctly.
  [(#4942)](https://github.com/PennyLaneAI/pennylane/pull/4942/)

<h4>Other improvements</h4>

* Add PyTree-serialization interface for the `Wires` class.
  [(#4998)](https://github.com/PennyLaneAI/pennylane/pull/4998)

* PennyLane now supports Python 3.12.
  [(#4985)](https://github.com/PennyLaneAI/pennylane/pull/4985)

* `SampleMeasurement` now has an optional method `process_counts` for computing the measurement results from a counts
  dictionary.
  [(#4941)](https://github.com/PennyLaneAI/pennylane/pull/4941/)

* A new function called `ops.functions.assert_valid` has been added for checking if an `Operator` class is defined correctly.
  [(#4764)](https://github.com/PennyLaneAI/pennylane/pull/4764)

* `Shots` objects can now be multiplied by scalar values.
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

* `trainable_params` can now be set upon initialization of a `QuantumScript` instead of having to set the
  parameter after initialization.
  [(#4877)](https://github.com/PennyLaneAI/pennylane/pull/4877)

* `default.qubit` now calculates the expectation value of `Hermitian` operators in a differentiable manner.
  [(#4866)](https://github.com/PennyLaneAI/pennylane/pull/4866)

* The `rot` decomposition now has support for returning a global phase.
  [(#4869)](https://github.com/PennyLaneAI/pennylane/pull/4869)

* The `"pennylane_sketch"` MPL-drawer style has been added. This is the same as the `"pennylane"`
  style, but with sketch-style lines.
  [(#4880)](https://github.com/PennyLaneAI/pennylane/pull/4880)

* Operators now define a `pauli_rep` property, an instance of `PauliSentence`, defaulting
  to `None` if the operator has not defined it (or has no definition in the Pauli basis).
  [(#4915)](https://github.com/PennyLaneAI/pennylane/pull/4915)

* `qp.ShotAdaptiveOptimizer` can now use a multinomial distribution for spreading shots across
  the terms of a Hamiltonian measured in a QNode. Note that this is equivalent to what can be
  done with `qp.ExpvalCost`, but this is the preferred method because `ExpvalCost` is deprecated.
  [(#4896)](https://github.com/PennyLaneAI/pennylane/pull/4896)

* Decomposition of `qp.PhaseShift` now uses `qp.GlobalPhase` for retaining the global phase information. 
  [(#4657)](https://github.com/PennyLaneAI/pennylane/pull/4657)
  [(#4947)](https://github.com/PennyLaneAI/pennylane/pull/4947)

* `qp.equal` for `Controlled` operators no longer returns `False` when equivalent but 
  differently-ordered sets of control wires and control values are compared.
  [(#4944)](https://github.com/PennyLaneAI/pennylane/pull/4944)

* All PennyLane `Operator` subclasses are automatically tested by `ops.functions.assert_valid` to ensure
  that they follow PennyLane `Operator` standards.
  [(#4922)](https://github.com/PennyLaneAI/pennylane/pull/4922)

* Probability measurements can now be calculated from a `counts` dictionary with the addition of a 
  `process_counts` method in the `ProbabilityMP` class.
  [(#4952)](https://github.com/PennyLaneAI/pennylane/pull/4952)

* `ClassicalShadow.entropy` now uses the algorithm outlined in 
  [1106.5458](https://arxiv.org/abs/1106.5458) to project the approximate density matrix
  (with potentially negative eigenvalues) onto the closest valid density matrix.
  [(#4959)](https://github.com/PennyLaneAI/pennylane/pull/4959)

* The `ControlledSequence.compute_decomposition` default now decomposes the `Pow` operators, 
  improving compatibility with machine learning interfaces. 
  [(#4995)](https://github.com/PennyLaneAI/pennylane/pull/4995)

<h3>Breaking changes 💔</h3>

* The function `qp.transforms.classical_jacobian` has been moved to the gradients module
  and is now accessible as `qp.gradients.classical_jacobian`.
  [(#4900)](https://github.com/PennyLaneAI/pennylane/pull/4900)

* The transforms submodule `qp.transforms.qcut` is now its own module: `qp.qcut`.
  [(#4819)](https://github.com/PennyLaneAI/pennylane/pull/4819)

* The decomposition of `GroverOperator` now has an additional global phase operation.
  [(#4666)](https://github.com/PennyLaneAI/pennylane/pull/4666)

* `qp.cond` and the `Conditional` operation have been moved from the `transforms` folder to the `ops/op_math` folder.
  `qp.transforms.Conditional` will now be available as `qp.ops.Conditional`.
  [(#4860)](https://github.com/PennyLaneAI/pennylane/pull/4860)

* The `prep` keyword argument has been removed from `QuantumScript` and `QuantumTape`.
  `StatePrepBase` operations should be placed at the beginning of the `ops` list instead.
  [(#4756)](https://github.com/PennyLaneAI/pennylane/pull/4756)

* `qp.gradients.pulse_generator` is now named `qp.gradients.pulse_odegen` to adhere to paper naming conventions.
  [(#4769)](https://github.com/PennyLaneAI/pennylane/pull/4769)

* Specifying `control_values` passed to `qp.ctrl` as a string is no longer supported.
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
  Use the new `qp.transform` function instead.
  [(#4774)](https://github.com/PennyLaneAI/pennylane/pull/4774)

* `Observable.return_type` is deprecated. Instead, you should inspect the type
  of the surrounding measurement process.
  [(#4762)](https://github.com/PennyLaneAI/pennylane/pull/4762)
  [(#4798)](https://github.com/PennyLaneAI/pennylane/pull/4798)

* All deprecations now raise a `qp.PennyLaneDeprecationWarning` instead of a `UserWarning`.
  [(#4814)](https://github.com/PennyLaneAI/pennylane/pull/4814)

* `QuantumScript.is_sampled` and `QuantumScript.all_sampled` are deprecated.
  Users should now validate these properties manually.
  [(#4773)](https://github.com/PennyLaneAI/pennylane/pull/4773)

* With an algorithmic improvement to `ClassicalShadow.entropy`, the keyword `atol`
  becomes obsolete and will be removed in v0.35.
  [(#4959)](https://github.com/PennyLaneAI/pennylane/pull/4959)

<h3>Documentation 📝</h3>

* Documentation for unitaries and operations' decompositions has been moved from `qp.transforms` to `qp.ops.ops_math`.
  [(#4906)](https://github.com/PennyLaneAI/pennylane/pull/4906)

* Documentation for `qp.metric_tensor` and `qp.adjoint_metric_tensor` and `qp.transforms.classical_jacobian`
  is now accessible via the gradients API page `qp.gradients` in the documentation.
  [(#4900)](https://github.com/PennyLaneAI/pennylane/pull/4900)

* Documentation for `qp.specs` has been moved to the `resource` module.
  [(#4904)](https://github.com/PennyLaneAI/pennylane/pull/4904)

* Documentation for QCut has been moved to its own API page: `qp.qcut`.
  [(#4819)](https://github.com/PennyLaneAI/pennylane/pull/4819)

* The documentation page for `qp.measurements` now links top-level accessible functions (e.g., `qp.expval`) 
  to their top-level pages rather than their module-level pages (e.g., `qp.measurements.expval`).
  [(#4750)](https://github.com/PennyLaneAI/pennylane/pull/4750)

* Information for the documentation of `qp.matrix` about wire ordering has been added for using `qp.matrix` on a
  QNode which uses a device with `device.wires=None`.
  [(#4874)](https://github.com/PennyLaneAI/pennylane/pull/4874)

<h3>Bug fixes 🐛</h3>

* `TransformDispatcher` now stops queuing when performing the transform when applying it to a qfunc.
  Only the output of the transform will be queued.
  [(#4983)](https://github.com/PennyLaneAI/pennylane/pull/4983)

* `qp.map_wires` now works properly with `qp.cond` and `qp.measure`.
  [(#4884)](https://github.com/PennyLaneAI/pennylane/pull/4884)

* `Pow` operators are now picklable.
  [(#4966)](https://github.com/PennyLaneAI/pennylane/pull/4966)

* Finite differences and SPSA can now be used with tensorflow-autograph on setups that were seeing a bus error.
  [(#4961)](https://github.com/PennyLaneAI/pennylane/pull/4961)

* `qp.cond` no longer incorrectly queues operators used arguments.
  [(#4948)](https://github.com/PennyLaneAI/pennylane/pull/4948)

* `Attribute` objects now return `False` instead of raising a `TypeError` when checking if an object is inside
  the set.
  [(#4933)](https://github.com/PennyLaneAI/pennylane/pull/4933)

* Fixed a bug where the parameter-shift rule of `qp.ctrl(op)` was wrong if `op` had a generator
  that has two or more eigenvalues and is stored as a `SparseHamiltonian`.
  [(#4899)](https://github.com/PennyLaneAI/pennylane/pull/4899)

* Fixed a bug where trainable parameters in the post-processing of finite-differences were incorrect for JAX when applying
  the transform directly on a QNode.
  [(#4879)](https://github.com/PennyLaneAI/pennylane/pull/4879)

* `qp.grad` and `qp.jacobian` now explicitly raise errors if trainable parameters are integers.
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

* `qp.defer_measurements` now correctly transforms circuits when terminal measurements include wires
  used in mid-circuit measurements.
  [(#4787)](https://github.com/PennyLaneAI/pennylane/pull/4787)

* Fixed a bug where the adjoint differentiation method would fail if
  an operation that has a parameter with `grad_method=None` is present.
  [(#4820)](https://github.com/PennyLaneAI/pennylane/pull/4820)

* `MottonenStatePreparation` and `BasisStatePreparation` now raise an error when decomposing a broadcasted state vector.
  [(#4767)](https://github.com/PennyLaneAI/pennylane/pull/4767)

* Gradient transforms now work with overridden shot vectors and `default.qubit`.
  [(#4795)](https://github.com/PennyLaneAI/pennylane/pull/4795)

* Any `ScalarSymbolicOp`, like `Evolution`, now states that it has a matrix if the target
  is a `Hamiltonian`.
  [(#4768)](https://github.com/PennyLaneAI/pennylane/pull/4768)

* In `default.qubit`, initial states are now initialized with the simulator's wire order, not the circuit's
  wire order.
  [(#4781)](https://github.com/PennyLaneAI/pennylane/pull/4781)

* `qp.compile` will now always decompose to `expand_depth`, even if a target basis set is not specified.
  [(#4800)](https://github.com/PennyLaneAI/pennylane/pull/4800)

* `qp.transforms.transpile` can now handle measurements that are broadcasted onto all wires.
  [(#4793)](https://github.com/PennyLaneAI/pennylane/pull/4793)

* Parametrized circuits whose operators do not act on all wires return PennyLane tensors instead of NumPy arrays, as
  expected.
  [(#4811)](https://github.com/PennyLaneAI/pennylane/pull/4811)
  [(#4817)](https://github.com/PennyLaneAI/pennylane/pull/4817)

* `qp.transforms.merge_amplitude_embedding` no longer depends on queuing, allowing it to work as expected
  with QNodes.
  [(#4831)](https://github.com/PennyLaneAI/pennylane/pull/4831)

* `qp.pow(op)` and `qp.QubitUnitary.pow()` now also work with Tensorflow data raised to an
  integer power.
  [(#4827)](https://github.com/PennyLaneAI/pennylane/pull/4827)

* The text drawer has been fixed to correctly label `qp.qinfo` measurements, as well as `qp.classical_shadow`
  `qp.shadow_expval`.
  [(#4803)](https://github.com/PennyLaneAI/pennylane/pull/4803)

* Removed an implicit assumption that an empty `PauliSentence` gets treated as identity under 
  multiplication.
  [(#4887)](https://github.com/PennyLaneAI/pennylane/pull/4887)

* Using a `CNOT` or `PauliZ` operation with large batched states and the Tensorflow
  interface no longer raises an unexpected error.
  [(#4889)](https://github.com/PennyLaneAI/pennylane/pull/4889)

* `qp.map_wires` no longer fails when mapping nested quantum tapes.
  [(#4901)](https://github.com/PennyLaneAI/pennylane/pull/4901)

* Conversion of circuits to openqasm now decomposes to a depth of 10, allowing support 
  for operators requiring more than 2 iterations of decomposition, such as the `ApproxTimeEvolution` gate.
  [(#4951)](https://github.com/PennyLaneAI/pennylane/pull/4951)

* `MPLDrawer` does not add the bonus space for classical wires when no classical wires are present.
  [(#4987)](https://github.com/PennyLaneAI/pennylane/pull/4987)

* `Projector` now works with parameter-broadcasting.
  [(#4993)](https://github.com/PennyLaneAI/pennylane/pull/4993)
  
* The jax-jit interface can now be used with float32 mode.
  [(#4990)](https://github.com/PennyLaneAI/pennylane/pull/4990)

* Keras models with a `qnn.KerasLayer` no longer fail to save and load weights
  properly when they are named "weights".
  [(#5008)](https://github.com/PennyLaneAI/pennylane/pull/5008)

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
Pieter Eendebak,
Lillian Frederiksen,
Pietropaolo Frisoni,
Josh Izaac,
Juan Giraldo,
Emiliano Godinez Ramirez,
Ankit Khandelwal,
Korbinian Kottmann,
Christina Lee,
Vincent Michaud-Rioux,
Anurav Modak,
Romain Moyard,
Mudit Pandey,
Matthew Silverman,
Jay Soni,
David Wierichs,
Justin Woodring,
Sergei Mironov.
