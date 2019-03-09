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
