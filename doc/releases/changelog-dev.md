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

* The wires for the `default.tensor` device are selected at runtime if they are not provided by user.
  [(#5744)](https://github.com/PennyLaneAI/pennylane/pull/5744)

* Added `packaging` in the required list of packages.
  [(#5769)](https://github.com/PennyLaneAI/pennylane/pull/5769).

* Logging now allows for an easier opt-in across the stack, and also extends control support to `catalyst`.
  [(#5528)](https://github.com/PennyLaneAI/pennylane/pull/5528).

* A number of templates have been updated to be valid pytrees and PennyLane operations.
  [(#5698)](https://github.com/PennyLaneAI/pennylane/pull/5698)

* `ctrl` now works with tuple-valued `control_values` when applied to any already controlled operation.
  [(#5725)](https://github.com/PennyLaneAI/pennylane/pull/5725)

* Add support for 3 new pytest markers: `unit`, `integration` and `system`.
  [(#5517)](https://github.com/PennyLaneAI/pennylane/pull/5517)

* The sorting order of parameter-shift terms is now guaranteed to resolve ties in the absolute value with the sign of the shifts.
  [(#5582)](https://github.com/PennyLaneAI/pennylane/pull/5582)

* Added compute_sparse_matrix for BasisStateProjector.
  [(#5790)](https://github.com/PennyLaneAI/pennylane/pull/5790)

* `qml.transforms.split_non_commuting` can now handle circuits containing measurements of multi-term observables.
  [(#5729)](https://github.com/PennyLaneAI/pennylane/pull/5729)

<h4>Mid-circuit measurements and dynamic circuits</h4>

* The `dynamic_one_shot` transform uses a single auxiliary tape with a shot vector and `default.qubit` implements the loop over shots with `jax.vmap`.
  [(#5617)](https://github.com/PennyLaneAI/pennylane/pull/5617)
  
* The `dynamic_one_shot` transform can be compiled with `jax.jit`.
  [(#5557)](https://github.com/PennyLaneAI/pennylane/pull/5557)
  
* When using `defer_measurements` with postselecting mid-circuit measurements, operations
  that will never be active due to the postselected state are skipped in the transformed
  quantum circuit. In addition, postselected controls are skipped, as they are evaluated
  at transform time. This optimization feature can be turned off by setting `reduce_postselected=False`
  [(#5558)](https://github.com/PennyLaneAI/pennylane/pull/5558)

  Consider a simple circuit with three mid-circuit measurements, two of which are postselecting,
  and a single gate conditioned on those measurements:

  ```python
  @qml.qnode(qml.device("default.qubit"))
  def node(x):
      qml.RX(x, 0)
      qml.RX(x, 1)
      qml.RX(x, 2)
      mcm0 = qml.measure(0, postselect=0, reset=False)
      mcm1 = qml.measure(1, postselect=None, reset=True)
      mcm2 = qml.measure(2, postselect=1, reset=False)
      qml.cond(mcm0+mcm1+mcm2==1, qml.RX)(0.5, 3)
      return qml.expval(qml.Z(0) @ qml.Z(3))
  ```

  Without the new optimization, we obtain three gates, each controlled on the three measured
  qubits. They correspond to the combinations of controls that satisfy the condition
  `mcm0+mcm1+mcm2==1`:

  ```pycon
  >>> print(qml.draw(qml.defer_measurements(node, reduce_postselected=False))(0.6))
  0: ──RX(0.60)──|0⟩⟨0|─╭●─────────────────────────────────────────────┤ ╭<Z@Z>
  1: ──RX(0.60)─────────│──╭●─╭X───────────────────────────────────────┤ │
  2: ──RX(0.60)─────────│──│──│───|1⟩⟨1|─╭○────────╭○────────╭●────────┤ │
  3: ───────────────────│──│──│──────────├RX(0.50)─├RX(0.50)─├RX(0.50)─┤ ╰<Z@Z>
  4: ───────────────────╰X─│──│──────────├○────────├●────────├○────────┤
  5: ──────────────────────╰X─╰●─────────╰●────────╰○────────╰○────────┤
  ```

  If we do not explicitly deactivate the optimization, we obtain a much simpler circuit:

  ```pycon
  >>> print(qml.draw(qml.defer_measurements(node))(0.6))
  0: ──RX(0.60)──|0⟩⟨0|─╭●─────────────────┤ ╭<Z@Z>
  1: ──RX(0.60)─────────│──╭●─╭X───────────┤ │
  2: ──RX(0.60)─────────│──│──│───|1⟩⟨1|───┤ │
  3: ───────────────────│──│──│──╭RX(0.50)─┤ ╰<Z@Z>
  4: ───────────────────╰X─│──│──│─────────┤
  5: ──────────────────────╰X─╰●─╰○────────┤
  ```

  There is only one controlled gate with only one control wire.

* `qml.devices.LegacyDevice` is now an alias for `qml.Device`, so it is easier to distinguish it from
  `qml.devices.Device`, which follows the new device API.
  [(#5581)](https://github.com/PennyLaneAI/pennylane/pull/5581)

* The `dtype` for `eigvals` of `X`, `Y`, `Z` and `Hadamard` is changed from `int` to `float`, making them 
  consistent with the other observables. The `dtype` of the returned values when sampling these observables 
  (e.g. `qml.sample(X(0))`) is also changed to `float`. 
  [(#5607)](https://github.com/PennyLaneAI/pennylane/pull/5607)

* Sets up the framework for the development of an `assert_equal` function for testing operator comparison.
  [(#5634)](https://github.com/PennyLaneAI/pennylane/pull/5634)

* `qml.sample` can now be used on Boolean values representing mid-circuit measurement results in
  traced quantum functions. This feature is used with Catalyst to enable the pattern
  `m = measure(0); qml.sample(m)`.
  [(#5673)](https://github.com/PennyLaneAI/pennylane/pull/5673)

* PennyLane operators, measurements, and QNodes can now automatically be captured as instructions in JAXPR.
  [(#5564)](https://github.com/PennyLaneAI/pennylane/pull/5564)
  [(#5511)](https://github.com/PennyLaneAI/pennylane/pull/5511)
  [(#5708)](https://github.com/PennyLaneAI/pennylane/pull/5708)
  [(#5523)](https://github.com/PennyLaneAI/pennylane/pull/5523)

* The `decompose` transform has an `error` kwarg to specify the type of error that should be raised, 
  allowing error types to be more consistent with the context the `decompose` function is used in.
  [(#5669)](https://github.com/PennyLaneAI/pennylane/pull/5669)

* The `qml.pytrees` module now has `flatten` and `unflatten` methods for serializing pytrees.
  [(#5701)](https://github.com/PennyLaneAI/pennylane/pull/5701)

* Empty initialization of `PauliVSpace` is permitted.
  [(#5675)](https://github.com/PennyLaneAI/pennylane/pull/5675)

* `QuantumScript` properties are only calculated when needed, instead of on initialization. This decreases the classical overhead by >20%.
  `par_info`, `obs_sharing_wires`, and `obs_sharing_wires_id` are now public attributes.
  [(#5696)](https://github.com/PennyLaneAI/pennylane/pull/5696)

* The `qml.qchem.Molecule` object is now the central object used by all qchem functions.
  [(#5571)](https://github.com/PennyLaneAI/pennylane/pull/5571)

* The `qml.qchem.Molecule` class now supports Angstrom as a unit.
  [(#5694)](https://github.com/PennyLaneAI/pennylane/pull/5694)

* The `qml.qchem.Molecule` class now supports open-shell systems.
  [(#5655)](https://github.com/PennyLaneAI/pennylane/pull/5655)

* The `qml.qchem.molecular_hamiltonian` function now supports parity and Bravyi-Kitaev mappings.
  [(#5657)](https://github.com/PennyLaneAI/pennylane/pull/5657/)

* The qchem docs are updated with the new qchem improvements.
  [(#5758)](https://github.com/PennyLaneAI/pennylane/pull/5758/)
  [(#5638)](https://github.com/PennyLaneAI/pennylane/pull/5638/)

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
Soran Jahangiri,
Christina Lee,
Austin Huang,
William Maxwell,
Vincent Michaud-Rioux,
Anurav Modak,
Mudit Pandey,
Erik Schultheis,
nate stemen.
