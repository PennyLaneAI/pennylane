:orphan:

# Release 0.34.0-dev (development release)

<h3>New features since last release</h3>

* Approximate Quantum Fourier Transform (AQFT) is now available from `qml.AQFT`.
  [(#4656)](https://github.com/PennyLaneAI/pennylane/pull/4656)

<h3>Improvements 🛠</h3>

* `qml.draw` now supports drawing mid-circuit measurements.
  [(#4775)](https://github.com/PennyLaneAI/pennylane/pull/4775)

* Autograd can now use vjps provided by the device from the new device API. If a device provides
  a vector Jacobian product, this can be selected by providing `device_vjp=True` to
  `qml.execute`.
  [(#4557)](https://github.com/PennyLaneAI/pennylane/pull/4557)

* Updates to some relevant Pytests to enable its use as a suite of benchmarks.
  [(#4703)](https://github.com/PennyLaneAI/pennylane/pull/4703)

* Added `__iadd__` method to PauliSentence, which enables inplace-addition using `+=`, we no longer need to perform a copy, leading to performance improvements.
[(#4662)](https://github.com/PennyLaneAI/pennylane/pull/4662) 

* `qml.ArbitraryUnitary` now supports batching.
  [(#4745)](https://github.com/PennyLaneAI/pennylane/pull/4745)

* `qml.draw` and `qml.draw_mpl` now render operator ids.
  [(#4749)](https://github.com/PennyLaneAI/pennylane/pull/4749)

<h3>Breaking changes 💔</h3>

* The `prep` keyword argument has been removed from `QuantumScript` and `QuantumTape`.
  `StatePrepBase` operations should be placed at the beginning of the `ops` list instead.
  [(#4756)](https://github.com/PennyLaneAI/pennylane/pull/4756)

* `qml.gradients.pulse_generator` has become `qml.gradients.pulse_odegen` to adhere to paper naming conventions.
  [(#4769)](https://github.com/PennyLaneAI/pennylane/pull/4769)

* Non-parametric-ops such as `Barrier`, `Snapshot` and `Wirecut` have been grouped together and moved to `pennylane/ops/meta.py`.
  Additionally, the relevant tests have been organized and placed in a new file, `tests/ops/test_meta.py` .
  [(#4789)](https://github.com/PennyLaneAI/pennylane/pull/4789)
  
* `QuantumScript.graph` is now built using `tape.measurements` instead of `tape.observables`
  because it depended on the now-deprecated `Observable.return_type` property.
  [(#4762)](https://github.com/PennyLaneAI/pennylane/pull/4762)

<h3>Deprecations 👋</h3>

* All deprecations now raise a `qml.PennyLaneDeprecationWarning` instead of a `UserWarning`.
  [(#4814)](https://github.com/PennyLaneAI/pennylane/pull/4814)

* `QuantumScript.is_sampled` and `QuantumScript.all_sampled` are deprecated.
  Users should now validate these properties manually.
  [(#4773)](https://github.com/PennyLaneAI/pennylane/pull/4773)

* `single_tape_transform`, `batch_transform`, `qfunc_transform`, and `op_transform` are deprecated.
  Instead switch to using the new `qml.transform` function.
  [(#4774)](https://github.com/PennyLaneAI/pennylane/pull/4774)

* `Observable.return_type` is deprecated. Instead, you should inspect the type
  of the surrounding measurement process.
  [(#4762)](https://github.com/PennyLaneAI/pennylane/pull/4762)
  [(#4798)](https://github.com/PennyLaneAI/pennylane/pull/4798)

<h3>Documentation 📝</h3>

* Documentation page for `qml.measurements` now links top-level accessible functions (e.g. `qml.expval`) 
  to their top-level pages (rather than their module-level pages, eg. `qml.measurements.expval`).
  [(#4750)](https://github.com/PennyLaneAI/pennylane/pull/4750)

<h3>Bug fixes 🐛</h3>

* Fixes a bug where the adjoint method differentiation would fail if
  an operation with `grad_method=None` that has a parameter is present.
  [(#4820)](https://github.com/PennyLaneAI/pennylane/pull/4820)
  
* `MottonenStatePreparation` now raises an error if decomposing a broadcasted state vector.
  [(#4767)](https://github.com/PennyLaneAI/pennylane/pull/4767)

* `BasisStatePreparation` now raises an error if decomposing a broadcasted state vector.
  [(#4767)](https://github.com/PennyLaneAI/pennylane/pull/4767)

* Gradient transforms now work with overridden shot vectors and default qubit.
  [(#4795)](https://github.com/PennyLaneAI/pennylane/pull/4795)

* `qml.defer_measurements` now correctly transforms circuits when terminal measurements include wires
  used in mid-circuit measurements.
  [(#4787)](https://github.com/PennyLaneAI/pennylane/pull/4787)

* Jax jit now works with shot vectors.
  [(#4772)](https://github.com/PennyLaneAI/pennylane/pull/4772/)

* Any `ScalarSymbolicOp`, like `Evolution`, now states that it has a matrix if the target
  is a `Hamiltonian`.
  [(#4768)](https://github.com/PennyLaneAI/pennylane/pull/4768)

* In `default.qubit`, initial states are now initialized with the simulator's wire order, not the circuit's
  wire order.
  [(#4781)](https://github.com/PennyLaneAI/pennylane/pull/4781)

* `transpile` can now handle measurements that are broadcasted onto all wires.
  [(#4793)](https://github.com/PennyLaneAI/pennylane/pull/4793)

* Parametrized circuits whose operators do not act on all wires return pennylane tensors as
  expected, instead of numpy arrays.
  [(#4811)](https://github.com/PennyLaneAI/pennylane/pull/4811)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Amintor Dusko,
Lillian Frederiksen,
Ankit Khandelwal,
Christina Lee,
Anurav Modak,
Mudit Pandey,
Matthew Silverman,
David Wierichs,
Justin Woodring,
