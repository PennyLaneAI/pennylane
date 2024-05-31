:orphan:

# Release 0.37.0-dev (development release)

<h3>New features since last release</h3>

* The `default.tensor` device is introduced to perform tensor network simulation of a quantum circuit.
  [(#5699)](https://github.com/PennyLaneAI/pennylane/pull/5699)

<h3>Improvements 🛠</h3>

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

* PennyLane operators and measurements can now automatically be captured as instructions in JAXPR.
  [(#5564)](https://github.com/PennyLaneAI/pennylane/pull/5564)
  [(#5511)](https://github.com/PennyLaneAI/pennylane/pull/5511)
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

<h4>Community contributions 🥳</h4>

* Implemented kwargs (`check_interface`, `check_trainability`, `rtol` and `atol`) support in `qml.equal` for the operators `Pow`, `Adjoint`, `Exp`, and `SProd`.
  [(#5668)](https://github.com/PennyLaneAI/pennylane/issues/5668)
  
* ``qml.QutritDepolarizingChannel`` has been added, allowing for depolarizing noise to be simulated on the `default.qutrit.mixed` device.
  [(#5502)](https://github.com/PennyLaneAI/pennylane/pull/5502)

<h3>Breaking changes 💔</h3>

* A custom decomposition can no longer be provided to `QDrift`. Instead, apply the operations in your custom
  operation directly with `qml.apply`.
  [(#5698)](https://github.com/PennyLaneAI/pennylane/pull/5698)

* Sampling observables composed of `X`, `Y`, `Z` and `Hadamard` now returns values of type `float` instead of `int`.
  [(#5607)](https://github.com/PennyLaneAI/pennylane/pull/5607)

* `qml.is_commuting` no longer accepts the `wire_map` argument, which does not bring any functionality.
  [(#5660)](https://github.com/PennyLaneAI/pennylane/pull/5660)

* `qml.from_qasm_file` has been removed. The user can open files and load their content using `qml.from_qasm`.
  [(#5659)](https://github.com/PennyLaneAI/pennylane/pull/5659)

* `qml.load` has been removed in favour of more specific functions, such as `qml.from_qiskit`, etc.
  [(#5654)](https://github.com/PennyLaneAI/pennylane/pull/5654)

* `qml.transforms.convert_to_numpy_parameters` is now a proper transform and its output signature has changed,
  returning a list of `QuantumTape`s and a post-processing function instead of simply the transformed circuit.
  [(#5693)](https://github.com/PennyLaneAI/pennylane/pull/5693)

* `Controlled.wires` does not include `self.work_wires` anymore. That can be accessed separately through `Controlled.work_wires`.
  Consequently, `Controlled.active_wires` has been removed in favour of the more common `Controlled.wires`.
  [(#5728)](https://github.com/PennyLaneAI/pennylane/pull/5728)
  
* `qml.QutritAmplitudeDamping` channel has been added, allowing for noise processes modelled by amplitude damping to be simulated on the `default.qutrit.mixed` device.
  [(#5503)](https://github.com/PennyLaneAI/pennylane/pull/5503)

<h3>Deprecations 👋</h3>

* The `simplify` argument in `qml.Hamiltonian` and `qml.ops.LinearCombination` is deprecated. 
  Instead, `qml.simplify()` can be called on the constructed operator.
  [(#5677)](https://github.com/PennyLaneAI/pennylane/pull/5677)

* `qml.transforms.map_batch_transform` is deprecated, since a transform can be applied directly to a batch of tapes.
  [(#5676)](https://github.com/PennyLaneAI/pennylane/pull/5676)

<h3>Documentation 📝</h3>

* The documentation for the `default.tensor` device has been added.
  [(#5719)](https://github.com/PennyLaneAI/pennylane/pull/5719)

* A small typo was fixed in the docstring for `qml.sample`.
  [(#5685)](https://github.com/PennyLaneAI/pennylane/pull/5685)

<h3>Bug fixes 🐛</h3>

* The validation of the adjoint method in `DefaultQubit` correctly handles device wires now.
  [(#5761)](https://github.com/PennyLaneAI/pennylane/pull/5761)

* `QuantumPhaseEstimation.map_wires` on longer modifies the original operation instance.
  [(#5698)](https://github.com/PennyLaneAI/pennylane/pull/5698)

* The decomposition of `AmplitudeAmplification` now correctly queues all operations.
  [(#5698)](https://github.com/PennyLaneAI/pennylane/pull/5698)

* Replaced `semantic_version` with `packaging.version.Version`, since the former cannot
  handle the metadata `.post` in the version string.
  [(#5754)](https://github.com/PennyLaneAI/pennylane/pull/5754)

* The `dynamic_one_shot` transform now has expanded support for the `jax` and `torch` interfaces.
  [(#5672)](https://github.com/PennyLaneAI/pennylane/pull/5672)

* The decomposition of `StronglyEntanglingLayers` is now compatible with broadcasting.
  [(#5716)](https://github.com/PennyLaneAI/pennylane/pull/5716)

* `qml.cond` can now be applied to `ControlledOp` operations when deferring measurements.
  [(#5725)](https://github.com/PennyLaneAI/pennylane/pull/5725)

* The legacy `Tensor` class can now handle a `Projector` with abstract tracer input.
  [(#5720)](https://github.com/PennyLaneAI/pennylane/pull/5720)

* Fixed a bug that raised an error regarding expected vs actual `dtype` when using `JAX-JIT` on a circuit that 
  returned samples of observables containing the `qml.Identity` operator.
  [(#5607)](https://github.com/PennyLaneAI/pennylane/pull/5607)

* The signature of `CaptureMeta` objects (like `Operator`) now match the signature of the `__init__` call.
  [(#5727)](https://github.com/PennyLaneAI/pennylane/pull/5727)

* Use vanilla NumPy arrays in `test_projector_expectation` to avoid differentiating `qml.Projector` with respect to the state attribute.
  [(#5683)](https://github.com/PennyLaneAI/pennylane/pull/5683)

* `qml.Projector` is now compatible with jax-jit.
  [(#5595)](https://github.com/PennyLaneAI/pennylane/pull/5595)

* Finite shot circuits with a `qml.probs` measurement, both with a `wires` or `op` argument, can now be compiled with `jax.jit`.
  [(#5619)](https://github.com/PennyLaneAI/pennylane/pull/5619)
  
* `param_shift`, `finite_diff`, `compile`, `insert`, `merge_rotations`, and `transpile` now
  all work with circuits with non-commuting measurements.
  [(#5424)](https://github.com/PennyLaneAI/pennylane/pull/5424)
  [(#5681)](https://github.com/PennyLaneAI/pennylane/pull/5681)

* A correction is added to `bravyi_kitaev` to call the correct function for a FermiSentence input.
  [(#5671)](https://github.com/PennyLaneAI/pennylane/pull/5671)

* Fixes a bug where `sum_expand` produces incorrect result dimensions when combining shot vectors, 
  multiple measurements, and parameter broadcasting.
  [(#5702)](https://github.com/PennyLaneAI/pennylane/pull/5702)

* Fixes a bug in `qml.math.dot` that raises an error when only one of the operands is a scalar.
  [(#5702)](https://github.com/PennyLaneAI/pennylane/pull/5702)

* `qml.matrix` is now compatible with qnodes compiled by catalyst.qjit.
  [(#5753)](https://github.com/PennyLaneAI/pennylane/pull/5753)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Lillian M. A. Frederiksen,
Gabriel Bottrill,
Astral Cai,
Ahmed Darwish,
Isaac De Vlugt,
Diksha Dhawan,
Pietropaolo Frisoni,
Emiliano Godinez,
David Ittah,
Soran Jahangiri,
Korbinian Kottmann,
Christina Lee,
Vincent Michaud-Rioux,
Lee James O'Riordan,
Mudit Pandey,
Kenya Sakka,
Haochen Paul Wang,
David Wierichs.
