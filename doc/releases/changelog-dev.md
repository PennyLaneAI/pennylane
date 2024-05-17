:orphan:

# Release 0.37.0-dev (development release)

<h3>New features since last release</h3>

* A new `qml.noise` module which contains utililty function for building `NoiseModels`.
  [(#5674)](https://github.com/PennyLaneAI/pennylane/pull/5674)
  [(#5684)](https://github.com/PennyLaneAI/pennylane/pull/5684)

  ```python
  fcond = qml.noise.op_eq(qml.X) | qml.noise.op_eq(qml.Y)
  noise = qml.noise.partial_wires(qml.AmplitudeDamping, 0.4)
  ```

  ```pycon
  >>> qml.NoiseModel({fcond: noise}, t1=0.04)
  NoiseModel({
    OpEq(PauliX) | OpEq(PauliY) = AmplitudeDamping(gamma=0.4)
  }, t1 = 0.04)
  ```

<h3>Improvements 🛠</h3>

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

* Sets up the framework for the development of an `assert_equal` function for testing operator comparison.
  [(#5634)](https://github.com/PennyLaneAI/pennylane/pull/5634)

* PennyLane operators can now automatically be captured as instructions in JAXPR. See the experimental
  `capture` module for more information.
  [(#5511)](https://github.com/PennyLaneAI/pennylane/pull/5511)

* The `decompose` transform has an `error` kwarg to specify the type of error that should be raised, 
  allowing error types to be more consistent with the context the `decompose` function is used in.
  [(#5669)](https://github.com/PennyLaneAI/pennylane/pull/5669)

* Empty initialization of `PauliVSpace` is permitted.
  [(#5675)](https://github.com/PennyLaneAI/pennylane/pull/5675)

<h4>Community contributions 🥳</h4>

* Implemented kwargs (`check_interface`, `check_trainability`, `rtol` and `atol`) support in `qml.equal` for the operators `Pow`, `Adjoint`, `Exp`, and `SProd`.
  [(#5668)](https://github.com/PennyLaneAI/pennylane/issues/5668)
  
* ``qml.QutritDepolarizingChannel`` has been added, allowing for depolarizing noise to be simulated on the `default.qutrit.mixed` device.
  [(#5502)](https://github.com/PennyLaneAI/pennylane/pull/5502)

<h3>Breaking changes 💔</h3>

* `qml.is_commuting` no longer accepts the `wire_map` argument, which does not bring any functionality.
  [(#5660)](https://github.com/PennyLaneAI/pennylane/pull/5660)

* `qml.from_qasm_file` has been removed. The user can open files and load their content using `qml.from_qasm`.
  [(#5659)](https://github.com/PennyLaneAI/pennylane/pull/5659)

* `qml.load` has been removed in favour of more specific functions, such as `qml.from_qiskit`, etc.
  [(#5654)](https://github.com/PennyLaneAI/pennylane/pull/5654)

* `qml.transforms.convert_to_numpy_parameters` is now a proper transform and its output signature has changed,
  returning a list of `QuantumTape`s and a post-processing function instead of simply the transformed circuit.
  [(#5693)](https://github.com/PennyLaneAI/pennylane/pull/5693)

<h3>Deprecations 👋</h3>

* The `simplify` argument in `qml.Hamiltonian` and `qml.ops.LinearCombination` is deprecated. 
  Instead, `qml.simplify()` can be called on the constructed operator.
  [(#5677)](https://github.com/PennyLaneAI/pennylane/pull/5677)

* `qml.transforms.map_batch_transform` is deprecated, since a transform can be applied directly to a batch of tapes.
  [(#5676)](https://github.com/PennyLaneAI/pennylane/pull/5676)

<h3>Documentation 📝</h3>

* A small typo was fixed in the docstring for `qml.sample`.
  [(#5685)](https://github.com/PennyLaneAI/pennylane/pull/5685)

<h3>Bug fixes 🐛</h3>

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

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Utkarsh Azad,
Ahmed Darwish,
Gabriel Bottrill,
Isaac De Vlugt,
Pietropaolo Frisoni,
Soran Jahangiri,
Korbinian Kottmann,
Christina Lee,
Vincent Michaud-Rioux,
Kenya Sakka,
David Wierichs.
