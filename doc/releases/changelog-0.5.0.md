
# Release 0.5.0

<h3>New features since last release</h3>

* Adds a new optimizer, `qp.QNGOptimizer`, which optimizes QNodes using
  quantum natural gradient descent. See https://arxiv.org/abs/1909.02108
  for more details.
  [(#295)](https://github.com/XanaduAI/pennylane/pull/295)
  [(#311)](https://github.com/XanaduAI/pennylane/pull/311)

* Adds a new QNode method, `QNode.metric_tensor()`,
  which returns the block-diagonal approximation to the Fubini-Study
  metric tensor evaluated on the attached device.
  [(#295)](https://github.com/XanaduAI/pennylane/pull/295)

* Sampling support: QNodes can now return a specified number of samples
  from a given observable via the top-level `pennylane.sample()` function.
  To support this on plugin devices, there is a new `Device.sample` method.

  Calculating gradients of QNodes that involve sampling is not possible.
  [(#256)](https://github.com/XanaduAI/pennylane/pull/256)

* `default.qubit` has been updated to provide support for sampling.
  [(#256)](https://github.com/XanaduAI/pennylane/pull/256)

* Added controlled rotation gates to PennyLane operations and `default.qubit` plugin.
  [(#251)](https://github.com/XanaduAI/pennylane/pull/251)

<h3>Breaking changes</h3>

* The method `Device.supported` was removed, and replaced with the methods
  `Device.supports_observable` and `Device.supports_operation`.
  Both methods can be called with string arguments (`dev.supports_observable('PauliX')`) and
  class arguments (`dev.supports_observable(qp.PauliX)`).
  [(#276)](https://github.com/XanaduAI/pennylane/pull/276)

* The following CV observables were renamed to comply with the new Operation/Observable
  scheme: `MeanPhoton` to `NumberOperator`, `Homodyne` to `QuadOperator` and `NumberState` to `FockStateProjector`.
  [(#254)](https://github.com/XanaduAI/pennylane/pull/254)

<h3>Improvements</h3>

* The `AmplitudeEmbedding` function now provides options to normalize and
  pad features to ensure a valid state vector is prepared.
  [(#275)](https://github.com/XanaduAI/pennylane/pull/275)

* Operations can now optionally specify generators, either as existing PennyLane
  operations, or by providing a NumPy array.
  [(#295)](https://github.com/XanaduAI/pennylane/pull/295)
  [(#313)](https://github.com/XanaduAI/pennylane/pull/313)

* Adds a `Device.parameters` property, so that devices can view a dictionary mapping free
  parameters to operation parameters. This will allow plugin devices to take advantage
  of parametric compilation.
  [(#283)](https://github.com/XanaduAI/pennylane/pull/283)

* Introduces two enumerations: `Any` and `All`, representing any number of wires
  and all wires in the system respectively. They can be imported from
  `pennylane.operation`, and can be used when defining the `Operation.num_wires`
  class attribute of operations.
  [(#277)](https://github.com/XanaduAI/pennylane/pull/277)

  As part of this change:

  - `All` is equivalent to the integer 0, for backwards compatibility with the
    existing test suite

  - `Any` is equivalent to the integer -1 to allow numeric comparison
    operators to continue working

  - An additional validation is now added to the `Operation` class,
    which will alert the user that an operation with `num_wires = All`
    is being incorrectly.

* The one-qubit rotations in `pennylane.plugins.default_qubit` no longer depend on Scipy's `expm`. Instead
  they are calculated with Euler's formula.
  [(#292)](https://github.com/XanaduAI/pennylane/pull/292)

* Creates an `ObservableReturnTypes` enumeration class containing `Sample`,
  `Variance` and `Expectation`. These new values can be assigned to the `return_type`
  attribute of an `Observable`.
  [(#290)](https://github.com/XanaduAI/pennylane/pull/290)

* Changed the signature of the `RandomLayer` and `RandomLayers` templates to have a fixed seed by default.
  [(#258)](https://github.com/XanaduAI/pennylane/pull/258)

* `setup.py` has been cleaned up, removing the non-working shebang,
  and removing unused imports.
  [(#262)](https://github.com/XanaduAI/pennylane/pull/262)

<h3>Documentation</h3>

* A documentation refactor to simplify the tutorials and
  include Sphinx-Gallery.
  [(#291)](https://github.com/XanaduAI/pennylane/pull/291)

  - Examples and tutorials previously split across the `examples/`
    and `doc/tutorials/` directories, in a mixture of ReST and Jupyter notebooks,
    have been rewritten as Python scripts with ReST comments in a single location,
    the `examples/` folder.

  - Sphinx-Gallery is used to automatically build and run the tutorials.
    Rendered output is displayed in the Sphinx documentation.

  - Links are provided at the top of every tutorial page for downloading the
    tutorial as an executable python script, downloading the tutorial
    as a Jupyter notebook, or viewing the notebook on GitHub.

  - The tutorials table of contents have been moved to a single quick start page.

* Fixed a typo in `QubitStateVector`.
  [(#296)](https://github.com/XanaduAI/pennylane/pull/296)

* Fixed a typo in the `default_gaussian.gaussian_state` function.
  [(#293)](https://github.com/XanaduAI/pennylane/pull/293)

* Fixed a typo in the gradient recipe within the `RX`, `RY`, `RZ`
  operation docstrings.
  [(#248)](https://github.com/XanaduAI/pennylane/pull/248)

* Fixed a broken link in the tutorial documentation, as a
  result of the `qp.expval.Observable` deprecation.
  [(#246)](https://github.com/XanaduAI/pennylane/pull/246)

<h3>Bug fixes</h3>

* Fixed a bug where a `PolyXP` observable would fail if applied to subsets
  of wires on `default.gaussian`.
  [(#277)](https://github.com/XanaduAI/pennylane/pull/277)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Simon Cross, Aroosa Ijaz, Josh Izaac, Nathan Killoran, Johannes Jakob Meyer,
Rohit Midha, Nicolás Quesada, Maria Schuld, Antal Száva, Roeland Wiersema.
