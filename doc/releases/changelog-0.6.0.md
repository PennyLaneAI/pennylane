
# Release 0.6.0

<h3>New features since last release</h3>

* The devices `default.qubit` and `default.gaussian` have a new initialization parameter
  `analytic` that indicates if expectation values and variances should be calculated
  analytically and not be estimated from data.
  [(#317)](https://github.com/XanaduAI/pennylane/pull/317)

* Added C-SWAP gate to the set of qubit operations
  [(#330)](https://github.com/XanaduAI/pennylane/pull/330)

* The TensorFlow interface has been renamed from `"tfe"` to `"tf"`, and
  now supports TensorFlow 2.0.
  [(#337)](https://github.com/XanaduAI/pennylane/pull/337)

* Added the S and T gates to the set of qubit operations.
  [(#343)](https://github.com/XanaduAI/pennylane/pull/343)

* Tensor observables are now supported within the `expval`,
  `var`, and `sample` functions, by using the `@` operator.
  [(#267)](https://github.com/XanaduAI/pennylane/pull/267)


<h3>Breaking changes</h3>

* The argument `n` specifying the number of samples in the method `Device.sample` was removed.
  Instead, the method will always return `Device.shots` many samples.
  [(#317)](https://github.com/XanaduAI/pennylane/pull/317)

<h3>Improvements</h3>

* The number of shots / random samples used to estimate expectation values and variances, `Device.shots`,
  can now be changed after device creation.
  [(#317)](https://github.com/XanaduAI/pennylane/pull/317)

* Unified import shortcuts to be under qml in qnode.py
  and test_operation.py
  [(#329)](https://github.com/XanaduAI/pennylane/pull/329)

* The quantum natural gradient now uses `scipy.linalg.pinvh` which is more efficient for symmetric matrices
  than the previously used `scipy.linalg.pinv`.
  [(#331)](https://github.com/XanaduAI/pennylane/pull/331)

* The deprecated `qp.expval.Observable` syntax has been removed.
  [(#267)](https://github.com/XanaduAI/pennylane/pull/267)

* Remainder of the unittest-style tests were ported to pytest.
  [(#310)](https://github.com/XanaduAI/pennylane/pull/310)

* The `do_queue` argument for operations now only takes effect
  within QNodes. Outside of QNodes, operations can now be instantiated
  without needing to specify `do_queue`.
  [(#359)](https://github.com/XanaduAI/pennylane/pull/359)

<h3>Documentation</h3>

* The docs are rewritten and restructured to contain a code introduction section as well as an API section.
  [(#314)](https://github.com/XanaduAI/pennylane/pull/275)

* Added Ising model example to the tutorials
  [(#319)](https://github.com/XanaduAI/pennylane/pull/319)

* Added tutorial for QAOA on MaxCut problem
  [(#328)](https://github.com/XanaduAI/pennylane/pull/328)

* Added QGAN flow chart figure to its tutorial
  [(#333)](https://github.com/XanaduAI/pennylane/pull/333)

* Added missing figures for gallery thumbnails of state-preparation
  and QGAN tutorials
  [(#326)](https://github.com/XanaduAI/pennylane/pull/326)

* Fixed typos in the state preparation tutorial
  [(#321)](https://github.com/XanaduAI/pennylane/pull/321)

* Fixed bug in VQE tutorial 3D plots
  [(#327)](https://github.com/XanaduAI/pennylane/pull/327)

<h3>Bug fixes</h3>

* Fixed typo in measurement type error message in qnode.py
  [(#341)](https://github.com/XanaduAI/pennylane/pull/341)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Shahnawaz Ahmed, Ville Bergholm, Aroosa Ijaz, Josh Izaac, Nathan Killoran, Angus Lowe,
Johannes Jakob Meyer, Maria Schuld, Antal Száva, Roeland Wiersema.
