# Release 0.5.0-dev

### New features since last release

* Adds a `Device.parameters` property, so that devices can view a dictionary mapping free
  parameters to operation parameters. This will allow plugin devices to take advantage
  of parametric compilation.
  [#283](https://github.com/XanaduAI/pennylane/pull/283)

* Sampling support: QNodes can now return a specified number of samples
  from a given observable via the top-level `pennylane.sample()` function.
  To support this on plugin devices, there is a new `Device.sample` method.

  Calculating gradients of QNodes that involve sampling is not possible.
  [#256](https://github.com/XanaduAI/pennylane/pull/256)

* Added controlled rotation gates to PennyLane operations and `default.qubit` plugin.
  [#251](https://github.com/XanaduAI/pennylane/pull/251)

### Breaking changes

* The method `Device.supported` was removed.
  [#276](https://github.com/XanaduAI/pennylane/pull/276)

* The following CV observables were renamed to comply with the new Operation/Observable
  scheme: `MeanPhoton` to `NumberOperator`, `Homodyne` to `QuadOperator` and `NumberState` to `FockStateProjector`.
  [#243](https://github.com/XanaduAI/pennylane/pull/243)

### Improvements

* Introduces two enumerations: `Any` and `All`, representing any number of wires
  and all wires in the system respectively. They can be imported from
  `pennylane.operation`, and can be used when defining the `Operation.num_wires`
  class attribute of operations.

  As part of this change:

  - `All` is equivalent to the integer 0, for backwards compatibility with the
    existing test suite

  - `Any` is equivalent to the integer -1 to allow numeric comparison
    operators to continue working

  - An additional validation is now added to the `Operation` class,
    which will alert the user that an operation with `num_wires = All`
    is being incorrectly.

  [#277](https://github.com/XanaduAI/pennylane/pull/277)

* The method `Device.supported` that listed all the supported operations and observables
  was replaced with two separate methods `Device.supports_observable` and `Device.supports_operation`.
  The methods can now be called with string arguments (`dev.supports_observable('PauliX')`) and with
  class information arguments (`dev.supports_observable(qml.PauliX)`).
  [#276](https://github.com/XanaduAI/pennylane/pull/276)

* The one-qubit rotations in `pennylane.plugins.default_qubit` no longer depend on Scipy's `expm`. Instead 
  they are calculated with Euler's formula.
  [#292](https://github.com/XanaduAI/pennylane/pull/292)

### Bug fixes

* Fixed a bug where a `PolyXP` observable would fail if applied to subsets
  of wires on `default.gaussian`.
  [#277](https://github.com/XanaduAI/pennylane/pull/277)

### Contributors

This release contains contributions from (in alphabetical order):

Aroosa Ijaz, Josh Izaac, Johannes Jakob Meyer, Roeland Wiersema.

---

# Release 0.4.0

### New features since last release

* `pennylane.expval()` is now a top-level *function*, and is no longer
  a package of classes. For now, the existing `pennylane.expval.Observable`
  interface continues to work, but will raise a deprecation warning.
  [#232](https://github.com/XanaduAI/pennylane/pull/232)

* Variance support: QNodes can now return the variance of observables,
  via the top-level `pennylane.var()` function. To support this on
  plugin devices, there is a new `Device.var` method.

  The following observables support analytic gradients of variances:

  - All qubit observables (requiring 3 circuit evaluations for involutory
    observables such as `Identity`, `X`, `Y`, `Z`; and 5 circuit evals for
    non-involutary observables, currently only `qml.Hermitian`)

  - First-order CV observables (requiring 5 circuit evaluations)

  Second-order CV observables support numerical variance gradients.

* `pennylane.about()` function added, providing details
  on current PennyLane version, installed plugins, Python,
  platform, and NumPy versions [#186](https://github.com/XanaduAI/pennylane/pull/186)

* Removed the logic that allowed `wires` to be passed as a positional
  argument in quantum operations. This allows us to raise more useful
  error messages for the user if incorrect syntax is used.
  [#188](https://github.com/XanaduAI/pennylane/pull/188)

* Adds support for multi-qubit expectation values of the `pennylane.Hermitian()`
  observable [#192](https://github.com/XanaduAI/pennylane/pull/192)

* Adds support for multi-qubit expectation values in `default.qubit`.
  [#202](https://github.com/XanaduAI/pennylane/pull/202)

* Organize templates into submodules [#195](https://github.com/XanaduAI/pennylane/pull/195).
  This included the following improvements:

  - Distinguish embedding templates from layer templates.

  - New random initialization functions supporting the templates available
    in the new submodule `pennylane.init`.

  - Added a random circuit template (`RandomLayers()`), in which rotations and 2-qubit gates are randomly
    distributed over the wires

  - Add various embedding strategies

### Breaking changes

* The `Device` methods `expectations`, `pre_expval`, and `post_expval` have been
  renamed to `observables`, `pre_measure`, and `post_measure` respectively.
  [#232](https://github.com/XanaduAI/pennylane/pull/232)

### Improvements

* `default.qubit` plugin now uses `np.tensordot` when applying quantum operations
  and evaluating expectations, resulting in significant speedup [#239](https://github.com/XanaduAI/pennylane/pull/239), [#241](https://github.com/XanaduAI/pennylane/pull/241)

* PennyLane now allows division of quantum operation parameters by a constant [#179](https://github.com/XanaduAI/pennylane/pull/179)

* Portions of the test suite are in the process of being ported to pytest.
  Note: this is still a work in progress.

  Ported tests include:

  - `test_ops.py`
  - `test_about.py`
  - `test_classical_gradients.py`
  - `test_observables.py`
  - `test_measure.py`
  - `test_init.py`
  - `test_templates*.py`
  - `test_ops.py`
  - `test_variable.py`
  - `test_qnode.py` (partial)

### Bug fixes

* Fixed a bug in `Device.supported`, which would incorrectly
  mark an operation as supported if it shared a name with an
  observable [#203](https://github.com/XanaduAI/pennylane/pull/203)

* Fixed a bug in `Operation.wires`, by explicitly casting the
  type of each wire to an integer [#206](https://github.com/XanaduAI/pennylane/pull/206)

* Removed code in PennyLane which configured the logger,
  as this would clash with users' configurations
  [#208](https://github.com/XanaduAI/pennylane/pull/208)

* Fixed a bug in `default.qubit`, in which `QubitStateVector` operations
  were accidentally being cast to `np.float` instead of `np.complex`.
  [#211](https://github.com/XanaduAI/pennylane/pull/211)


### Contributors

This release contains contributions from:

Shahnawaz Ahmed, riveSunder, Aroosa Ijaz, Josh Izaac, Nathan Killoran, Maria Schuld.

# Release 0.3.1

### Bug fixes

* Fixed a bug where the interfaces submodule was not correctly being packaged via setup.py

# Release 0.3.0

### New features since last release

* PennyLane now includes a new `interfaces` submodule, which enables QNode integration with additional machine learning libraries.
* Adds support for an experimental PyTorch interface for QNodes
* Adds support for an experimental TensorFlow eager execution interface for QNodes
* Adds a PyTorch+GPU+QPU tutorial to the documentation
* Documentation now includes links and tutorials including the new [PennyLane-Forest](https://github.com/rigetti/pennylane-forest) plugin.

### Improvements

* Printing a QNode object, via `print(qnode)` or in an interactive terminal, now displays more useful information regarding the QNode,
  including the device it runs on, the number of wires, it's interface, and the quantum function it uses:

  ```python
  >>> print(qnode)
  <QNode: device='default.qubit', func=circuit, wires=2, interface=PyTorch>
  ```

### Contributors

This release contains contributions from:

Josh Izaac and Nathan Killoran.


# Release 0.2.0

### New features since last release

* Added the `Identity` expectation value for both CV and qubit models (#135)
* Added the `templates.py` submodule, containing some commonly used QML models to be used as ansatz in QNodes (#133)
* Added the `qml.Interferometer` CV operation (#152)
* Wires are now supported as free QNode parameters (#151)
* Added ability to update stepsizes of the optimizers (#159)

### Improvements

* Removed use of hardcoded values in the optimizers, made them parameters (see #131 and #132)
* Created the new `PlaceholderExpectation`, to be used when both CV and qubit expval modules contain expectations with the same name
* Provide the plugins a way to view the operation queue _before_ applying operations. This allows for on-the-fly modifications of
  the queue, allowing hardware-based plugins to support the full range of qubit expectation values. (#143)
* QNode return values now support _any_ form of sequence, such as lists, sets, etc. (#144)
* CV analytic gradient calculation is now more robust, allowing for operations which may not themselves be differentiated, but have a
  well defined `_heisenberg_rep` method, and so may succeed operations that are analytically differentiable (#152)

### Bug fixes

* Fixed a bug where the variational classifier example was not batching when learning parity (see #128 and #129)
* Fixed an inconsistency where some initial state operations were documented as accepting complex parameters - all operations
  now accept real values (#146)

### Contributors

This release contains contributions from:

Christian Gogolin, Josh Izaac, Nathan Killoran, and Maria Schuld.


# Release 0.1.0

Initial public release.

### Contributors
This release contains contributions from:

Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, and Nathan Killoran.
