:orphan:

# Release 0.33.0-dev (development release)

<h3>New features since last release</h3>

* Operator transforms `qml.matrix`, `qml.eigvals`, `qml.generator`, and `qml.transforms.to_zx` are updated
  to the new transform program system.
  [(#4573)](https://github.com/PennyLaneAI/pennylane/pull/4573)

* All quantum functions transforms are update to the new transform program system.
 [(#4439)](https://github.com/PennyLaneAI/pennylane/pull/4439)

* All batch transforms are updated to the new transform program system.
  [(#4440)](https://github.com/PennyLaneAI/pennylane/pull/4440)

* Quantum information transforms are updated to the new transform program system.
  [(#4569)](https://github.com/PennyLaneAI/pennylane/pull/4569)

* `qml.devices.DefaultQubit` now implements the new device API. The old version of `default.qubit`
  is still accessible via `qml.devices.DefaultQubitLegacy`, or via short name `default.qubit.legacy`.
  [(#4594)](https://github.com/PennyLaneAI/pennylane/pull/4594)

<h3>Improvements 🛠</h3>

* Tensor-network template `qml.MPS` now supports changing `offset` between subsequent blocks for more flexibility.
 [(#4531)](https://github.com/PennyLaneAI/pennylane/pull/4531)

* The qchem ``fermionic_dipole`` and ``particle_number`` functions are updated to use a
  ``FermiSentence``. The deprecated features for using tuples to represent fermionic operations are
  removed.
  [(#4546)](https://github.com/PennyLaneAI/pennylane/pull/4546)
  [(#4556)](https://github.com/PennyLaneAI/pennylane/pull/4556)

* Add the method ``add_transform`` and ``insert_front_transform`` transform in the ``TransformProgram``.
  [(#4559)](https://github.com/PennyLaneAI/pennylane/pull/4559)

* Dunder ``__add__`` method is added to the ``TransformProgram`` class, therefore two programs can be added using ``+`` .
  [(#4549)](https://github.com/PennyLaneAI/pennylane/pull/4549)

* `qml.sample()` in the new device API now returns a `np.int64` array instead of `np.bool8`.
  [(#4539)](https://github.com/PennyLaneAI/pennylane/pull/4539)

* Wires can be provided to the new device API.
  [(#4538)](https://github.com/PennyLaneAI/pennylane/pull/4538)
  [(#4562)](https://github.com/PennyLaneAI/pennylane/pull/4562)

* The new device API now has a `repr()`
  [(#4562)](https://github.com/PennyLaneAI/pennylane/pull/4562)

* The density matrix aspects of `StateMP` have been split into their own measurement
  process, `DensityMatrixMP`.
  [(#4558)](https://github.com/PennyLaneAI/pennylane/pull/4558)

* `qml.exp` returns a more informative error message when decomposition is unavailable for non-unitary operator.
  [(#4571)](https://github.com/PennyLaneAI/pennylane/pull/4571)

* The `StateMP` measurement now accepts a wire order (eg. a device wire order). The `process_state`
  method will re-order the given state to go from the inputted wire-order to the process's wire-order.
  If the process's wire-order contains extra wires, it will assume those are in the zero-state.
  [(#4570)](https://github.com/PennyLaneAI/pennylane/pull/4570)
  [(#4602)](https://github.com/PennyLaneAI/pennylane/pull/4602)

* Improve builtin types support with `qml.pauli_decompose`.
  [(#4577)](https://github.com/PennyLaneAI/pennylane/pull/4577)

* Various changes to measurements to improve feature parity between the legacy `default.qubit` and
  the new `DefaultQubit2`. This includes not trying to squeeze batched `CountsMP` results and implementing
  `MutualInfoMP.map_wires`.
  [(#4574)](https://github.com/PennyLaneAI/pennylane/pull/4574)

* `devices.qubit.simulate` now accepts an interface keyword argument. If a QNode with `DefaultQubit2`
  specifies an interface, the result will be computed with that interface.
  [(#4582)](https://github.com/PennyLaneAI/pennylane/pull/4582)

* `DefaultQubit2` now works as expected with measurement processes that don't specify wires.
  [(#4580)](https://github.com/PennyLaneAI/pennylane/pull/4580)

* `AmplitudeEmbedding` now inherits from `StatePrep`, allowing for it to not be decomposed
  when at the beginning of a circuit, thus behaving like `StatePrep`.
  [(#4583)](https://github.com/PennyLaneAI/pennylane/pull/4583)

* `StateMeasurement.process_state` now assumes the input is flat. `ProbabilityMP.process_state` has
  been updated to reflect this assumption and avoid redundant reshaping.
  [(#4602)](https://github.com/PennyLaneAI/pennylane/pull/4602)

<h3>Breaking changes 💔</h3>

* The `__eq__` and `__hash__` methods of `Operator` and `MeasurementProcess` no longer rely on the
  object's address is memory. Using `==` with operators and measurement processes will now behave the
  same as `qml.equal`, and objects of the same type with the same data and hyperparameters will have
  the same hash.
  [(#4536)](https://github.com/PennyLaneAI/pennylane/pull/4536)

  In the following scenario, the second and third code blocks show the previous and current behaviour
  of operator and measurement process equality, determined by the `__eq__` dunder method:

  ```python
  op1 = qml.PauliX(0)
  op2 = qml.PauliX(0)
  op3 = op1
  ```
  Old behaviour:
  ```pycon
  >>> op1 == op2
  False
  >>> op1 == op3
  True
  ```
  New behaviour:
  ```pycon
  >>> op1 == op2
  True
  >>> op1 == op3
  True
  ```

  The `__hash__` dunder method defines the hash of an object. The default hash of an object
  is determined by the objects memory address. However, the new hash is determined by the
  properties and attributes of operators and measurement processes. Consider the scenario below.
  The second and third code blocks show the previous and current behaviour.

  ```python
  op1 = qml.PauliX(0)
  op2 = qml.PauliX(0)
  ```
  Old behaviour:
  ```pycon
  >>> print({op1, op2})
  {PauliX(wires=[0]), PauliX(wires=[0])}
  ```
  New behaviour:
  ```pycon
  >>> print({op1, op2})
  {PauliX(wires=[0])}
  ```

* The old return type and associated functions ``qml.enable_return`` and ``qml.disable_return`` are removed.
  [(#4503)](https://github.com/PennyLaneAI/pennylane/pull/4503)

* The ``mode`` keyword argument in ``QNode`` is removed. Please use ``grad_on_execution`` instead.
  [(#4503)](https://github.com/PennyLaneAI/pennylane/pull/4503)

* The CV observables ``qml.X`` and ``qml.P`` are removed. Please use ``qml.QuadX`` and ``qml.QuadP`` instead.
  [(#4533)](https://github.com/PennyLaneAI/pennylane/pull/4533)

* The method ``tape.unwrap()`` and corresponding ``UnwrapTape`` and ``Unwrap`` classes are removed.
  Instead of ``tape.unwrap()``, use :func:`~.transforms.convert_to_numpy_parameters`.
  [(#4535)](https://github.com/PennyLaneAI/pennylane/pull/4535)

* The ``RandomLayers.compute_decomposition`` keyword argument ``ratio_imprivitive`` has been changed to
  ``ratio_imprim`` to match the call signature of the operation.
  [(#4552)](https://github.com/PennyLaneAI/pennylane/pull/4552)

* The ``sampler_seed`` argument of ``qml.gradients.spsa_grad`` has been removed.
  Instead, the ``sampler_rng`` argument should be set, either to an integer value, which will be used
  to create a PRNG internally, or to a NumPy pseudo-random number generator (PRNG) created via
  ``np.random.default_rng(seed)``.
  [(#4550)](https://github.com/PennyLaneAI/pennylane/pull/4550)

* The ``QuantumScript.set_parameters`` method and the ``QuantumScript.data`` setter have
  been removed. Please use ``QuantumScript.bind_new_parameters`` instead.
  [(#4548)](https://github.com/PennyLaneAI/pennylane/pull/4548)

<h3>Deprecations 👋</h3>

* The ``prep`` keyword argument in ``QuantumScript`` is deprecated and will be removed from `QuantumScript`.
  ``StatePrepBase`` operations should be placed at the beginning of the `ops` list instead.
  [(#4554)](https://github.com/PennyLaneAI/pennylane/pull/4554)

* The following decorator syntax for transforms has been deprecated and will raise a warning:
  ```python
  @transform_fn(**transform_kwargs)
  @qml.qnode(dev)
  def circuit():
      ...
  ```
  If you are using a transform that has supporting `transform_kwargs`, please call the
  transform directly using `circuit = transform_fn(circuit, **transform_kwargs)`,
  or use `functools.partial`:
  ```python
  @functools.partial(transform_fn, **transform_kwargs)
  @qml.qnode(dev)
  def circuit():
      ...
  ```
  [(#4457)](https://github.com/PennyLaneAI/pennylane/pull/4457/)

<h3>Documentation 📝</h3>

* Minor documentation improvements to the new device API. The documentation now correctly states that interface-specific
  parameters are only passed to the device for backpropagation derivatives. 
  [(#4542)](https://github.com/PennyLaneAI/pennylane/pull/4542)

* Add functions for qubit-simulation to the `qml.devices` sub-page of the "Internal" section.
  Note that these functions are unstable while device upgrades are underway.
  [(#4555)](https://github.com/PennyLaneAI/pennylane/pull/4555)

<h3>Bug fixes 🐛</h3>

* Fixed issue where `__copy__` method of the `qml.Select()` operator attempted to access un-initialized data.
[(#4551)](https://github.com/PennyLaneAI/pennylane/pull/4551)

* Fix `skip_first` option in `expand_tape_state_prep`.
  [(#4564)](https://github.com/PennyLaneAI/pennylane/pull/4564)

* `convert_to_numpy_parameters` now uses `qml.ops.functions.bind_new_parameters`. This reinitializes the operation and
  makes sure everything references the new numpy parameters.

* `tf.function` no longer breaks `ProbabilityMP.process_state` which is needed by new devices.
  [(#4470)](https://github.com/PennyLaneAI/pennylane/pull/4470)

* Fix mocking in the unit tests for `qml.qchem.mol_data`.
  [(#4591)](https://github.com/PennyLaneAI/pennylane/pull/4591)

* Fix `ProbabilityMP.process_state` so it allows for proper Autograph compilation. Without this,
  decorating a QNode that returns an `expval` with `tf.function` would fail when computing the
  expectation.
  [(#4590)](https://github.com/PennyLaneAI/pennylane/pull/4590)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Utkarsh Azad,
Diego Guala,
Soran Jahangiri,
Lillian M. A. Frederiksen,
Vincent Michaud-Rioux,
Romain Moyard,
Mudit Pandey,
Matthew Silverman,
Jay Soni,
