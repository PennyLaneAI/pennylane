:orphan:

# Release 0.38.0-dev (development release)

<h3>New features since last release</h3>

* Resolved the bug in `qml.ThermalRelaxationError` where there was a typo from `tq` to `tg`.
  [(#5988)](https://github.com/PennyLaneAI/pennylane/issues/5988)

* A new method `process_density_matrix` has been added to the `ProbabilityMP` and `DensityMatrixMP` classes, allowing for more efficient handling of quantum density matrices, particularly with batch processing support. This method simplifies the calculation of probabilities from quantum states represented as density matrices.
  [(#5830)](https://github.com/PennyLaneAI/pennylane/pull/5830)

* The `qml.PrepSelPrep` template is added. The template implements a block-encoding of a linear
  combination of unitaries.
  [(#5756)](https://github.com/PennyLaneAI/pennylane/pull/5756)
  [(#5987)](https://github.com/PennyLaneAI/pennylane/pull/5987)

* The `split_to_single_terms` transform is added. This transform splits expectation values of sums
  into multiple single-term measurements on a single tape, providing better support for simulators
  that can handle non-commuting observables but don't natively support multi-term observables.
  [(#5884)](https://github.com/PennyLaneAI/pennylane/pull/5884)

* `SProd.terms` now flattens out the terms if the base is a multi-term observable.
  [(#5885)](https://github.com/PennyLaneAI/pennylane/pull/5885)

* A new method `to_mat` has been added to the `FermiWord` and `FermiSentence` classes, which allows
  computing the matrix representation of these Fermi operators.
  [(#5920)](https://github.com/PennyLaneAI/pennylane/pull/5920)

* New functionality has been added to natively support exponential extrapolation when using the `mitigate_with_zne`. This allows
  users to have more control over the error mitigation protocol without needing to add further dependencies.
  [(#5972)](https://github.com/PennyLaneAI/pennylane/pull/5972)

<h3>Improvements 🛠</h3>

* `StateMP.process_state` defines rules in `cast_to_complex` for complex casting, avoiding a superfluous state vector copy in Lightning simulations
  [(#5995)](https://github.com/PennyLaneAI/pennylane/pull/5995)

* Port the fast `apply_operation` implementation of `PauliZ` to `PhaseShift`, `S` and `T`.
  [(#5876)](https://github.com/PennyLaneAI/pennylane/pull/5876)

* `qml.UCCSD` now accepts an additional optional argument, `n_repeats`, which defines the number of
  times the UCCSD template is repeated. This can improve the accuracy of the template by reducing
  the Trotter error but would result in deeper circuits.
  [(#5801)](https://github.com/PennyLaneAI/pennylane/pull/5801)

* `QuantumScript.hash` is now cached, leading to performance improvements.
  [(#5919)](https://github.com/PennyLaneAI/pennylane/pull/5919)

* The representation for `Wires` has now changed to be more copy-paste friendly.
  [(#5958)](https://github.com/PennyLaneAI/pennylane/pull/5958)

* Observable validation for `default.qubit` is now based on execution mode (analytic vs. finite shots) and measurement type (sample measurement vs. state measurement).
  [(#5890)](https://github.com/PennyLaneAI/pennylane/pull/5890)

* Molecules and Hamiltonians can now be constructed for all the elements present in the periodic table.
  [(#5821)](https://github.com/PennyLaneAI/pennylane/pull/5821)

* `qml.for_loop` and `qml.while_loop` now fallback to standard Python control
  flow if `@qjit` is not present, allowing the same code to work with and without
  `@qjit` without any rewrites.
  [(#6014)](https://github.com/PennyLaneAI/pennylane/pull/6014)

  ```python
  dev = qml.device("lightning.qubit", wires=3)

  @qml.qnode(dev)
  def circuit(x, n):

      @qml.for_loop(0, n, 1)
      def init_state(i):
          qml.Hadamard(wires=i)

      init_state()

      @qml.for_loop(0, n, 1)
      def apply_operations(i, x):
          qml.RX(x, wires=i)

          @qml.for_loop(i + 1, n, 1)
          def inner(j):
              qml.CRY(x**2, [i, j])

          inner()
          return jnp.sin(x)

      apply_operations(x)
      return qml.probs()
  ```

  ```pycon
  >>> print(qml.draw(circuit)(0.5, 3))
  0: ──H──RX(0.50)─╭●────────╭●──────────────────────────────────────┤  Probs
  1: ──H───────────╰RY(0.25)─│──────────RX(0.48)─╭●──────────────────┤  Probs
  2: ──H─────────────────────╰RY(0.25)───────────╰RY(0.23)──RX(0.46)─┤  Probs
  >>> circuit(0.5, 3)
  array([0.125     , 0.125     , 0.09949758, 0.15050242, 0.07594666,
       0.11917543, 0.08942104, 0.21545687])
  >>> qml.qjit(circuit)(0.5, 3)
  Array([0.125     , 0.125     , 0.09949758, 0.15050242, 0.07594666,
       0.11917543, 0.08942104, 0.21545687], dtype=float64)
  ```

* The `qubit_observable` function is modified to return an ascending wire order for molecular 
  Hamiltonians.
  [(#5950)](https://github.com/PennyLaneAI/pennylane/pull/5950)

<h4>Community contributions 🥳</h4>

* `DefaultQutritMixed` readout error has been added using parameters `readout_relaxation_probs` and 
  `readout_misclassification_probs` on the `default.qutrit.mixed` device. These parameters add a `~.QutritAmplitudeDamping`  and a `~.TritFlip` channel, respectively,
  after measurement diagonalization. The amplitude damping error represents the potential for
  relaxation to occur during longer measurements. The trit flip error represents misclassification during readout.
  [(#5842)](https://github.com/PennyLaneAI/pennylane/pull/5842)

<h3>Breaking changes 💔</h3>

* The `CircuitGraph.graph` rustworkx graph now stores indices into the circuit as the node labels,
  instead of the operator/ measurement itself.  This allows the same operator to occur multiple times in
  the circuit.
  [(#5907)](https://github.com/PennyLaneAI/pennylane/pull/5907)

* `queue_idx` attribute has been removed from the `Operator`, `CompositeOp`, and `SymboliOp` classes.
  [(#6005)](https://github.com/PennyLaneAI/pennylane/pull/6005)

* `qml.from_qasm` no longer removes measurements from the QASM code. Use 
  `measurements=[]` to remove measurements from the original circuit.
  [(#5982)](https://github.com/PennyLaneAI/pennylane/pull/5982)

* `qml.transforms.map_batch_transform` has been removed, since transforms can be applied directly to a batch of tapes.
  See :func:`~.pennylane.transform` for more information.
  [(#5981)](https://github.com/PennyLaneAI/pennylane/pull/5981)

* `QuantumScript.interface` has been removed.
  [(#5980)](https://github.com/PennyLaneAI/pennylane/pull/5980)

<h3>Deprecations 👋</h3>

* The `decomp_depth` argument in `qml.device` has been deprecated.
  [(#6026)](https://github.com/PennyLaneAI/pennylane/pull/6026)

* The `max_expansion` argument in `qml.QNode` has been deprecated.
  [(#6026)](https://github.com/PennyLaneAI/pennylane/pull/6026)

* The `expansion_strategy` attribute in the `QNode` class is deprecated.
  [(#5989)](https://github.com/PennyLaneAI/pennylane/pull/5989)

* The `expansion_strategy` argument has been deprecated in all of `qml.draw`, `qml.draw_mpl`, and `qml.specs`.
  The `level` argument should be used instead.
  [(#5989)](https://github.com/PennyLaneAI/pennylane/pull/5989)

* `Operator.expand` has been deprecated. Users should simply use `qml.tape.QuantumScript(op.decomposition())`
  for equivalent behaviour.
  [(#5994)](https://github.com/PennyLaneAI/pennylane/pull/5994)

* `pennylane.transforms.sum_expand` and `pennylane.transforms.hamiltonian_expand` have been deprecated.
  Users should instead use `pennylane.transforms.split_non_commuting` for equivalent behaviour.
  [(#6003)](https://github.com/PennyLaneAI/pennylane/pull/6003)

* The `expand_fn` argument in `qml.execute` has been deprecated.
  Instead, please create a `qml.transforms.core.TransformProgram` with the desired preprocessing and pass it to the `transform_program` argument of `qml.execute`.
  [(#5984)](https://github.com/PennyLaneAI/pennylane/pull/5984)

* The `max_expansion` argument in `qml.execute` has been deprecated.
  Instead, please use `qml.devices.preprocess.decompose` with the desired expansion level, add it to a `TransformProgram` and pass it to the `transform_program` argument of `qml.execute`.
  [(#5984)](https://github.com/PennyLaneAI/pennylane/pull/5984)

* The `override_shots` argument in `qml.execute` is deprecated.
  Instead, please add the shots to the `QuantumTape`'s to be executed.
  [(#5984)](https://github.com/PennyLaneAI/pennylane/pull/5984)

* The `device_batch_transform` argument in `qml.execute` is deprecated.
  Instead, please create a `qml.transforms.core.TransformProgram` with the desired preprocessing and pass it to the `transform_program` argument of `qml.execute`.
  [(#5984)](https://github.com/PennyLaneAI/pennylane/pull/5984)

* `pennylane.qinfo.classical_fisher` and `pennylane.qinfo.quantum_fisher` have been deprecated.
  Instead, use `pennylane.gradients.classical_fisher` and `pennylane.gradients.quantum_fisher`.
  [(#5985)](https://github.com/PennyLaneAI/pennylane/pull/5985)

<h3>Documentation 📝</h3>

* Improves the docstring for `QuantumScript.expand` and `qml.tape.tape.expand_tape`.
  [(#5974)](https://github.com/PennyLaneAI/pennylane/pull/5974)

<h3>Bug fixes 🐛</h3>

* Fixed a bug in `qml.SPSAOptimizer` that ignored keyword arguments in the objective function.
  [(#6027)](https://github.com/PennyLaneAI/pennylane/pull/6027)

* `dynamic_one_shot` was broken for old-API devices since `override_shots` was deprecated.
  [(#6024)](https://github.com/PennyLaneAI/pennylane/pull/6024)

* `CircuitGraph` can now handle circuits with the same operation instance occuring multiple times.
  [(#5907)](https://github.com/PennyLaneAI/pennylane/pull/5907)

* `qml.QSVT` is updated to store wire order correctly.
  [(#5959)](https://github.com/PennyLaneAI/pennylane/pull/5959)

* `qml.devices.qubit.measure_with_samples` now returns the correct result if the provided measurements
  contain sum of operators acting on the same wire.
  [(#5978)](https://github.com/PennyLaneAI/pennylane/pull/5978)

* `qml.AmplitudeEmbedding` has better support for features using low precision integer data types.
[(#5969)](https://github.com/PennyLaneAI/pennylane/pull/5969)


<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
Guillermo Alonso,
Utkarsh Azad
Astral Cai,
Yushao Chen,
Gabriel Bottrill,
Ahmed Darwish,
Lillian M. A. Frederiksen,
Pietropaolo Frisoni,
Emiliano Godinez,
Renke Huang,
Josh Izaac,
Soran Jahangiri,
Christina Lee,
Austin Huang,
Christina Lee,
William Maxwell,
Vincent Michaud-Rioux,
Mudit Pandey,
Erik Schultheis,
nate stemen.
