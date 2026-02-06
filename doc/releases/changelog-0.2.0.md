
# Release 0.2.0

<h3>New features since last release</h3>

* Added the `Identity` expectation value for both CV and qubit models [(#135)](https://github.com/XanaduAI/pennylane/pull/135)
* Added the `templates.py` submodule, containing some commonly used QML models to be used as ansatz in QNodes [(#133)](https://github.com/XanaduAI/pennylane/pull/133)
* Added the `qp.Interferometer` CV operation [(#152)](https://github.com/XanaduAI/pennylane/pull/152)
* Wires are now supported as free QNode parameters [(#151)](https://github.com/XanaduAI/pennylane/pull/151)
* Added ability to update stepsizes of the optimizers [(#159)](https://github.com/XanaduAI/pennylane/pull/159)

<h3>Improvements</h3>

* Removed use of hardcoded values in the optimizers, made them parameters (see [#131](https://github.com/XanaduAI/pennylane/pull/131) and [#132](https://github.com/XanaduAI/pennylane/pull/132))
* Created the new `PlaceholderExpectation`, to be used when both CV and qubit expval modules contain expectations with the same name
* Provide a way for plugins to view the operation queue _before_ applying operations. This allows for on-the-fly modifications of
  the queue, allowing hardware-based plugins to support the full range of qubit expectation values. [(#143)](https://github.com/XanaduAI/pennylane/pull/143)
* QNode return values now support _any_ form of sequence, such as lists, sets, etc. [(#144)](https://github.com/XanaduAI/pennylane/pull/144)
* CV analytic gradient calculation is now more robust, allowing for operations which may not themselves be differentiated, but have a
  well defined `_heisenberg_rep` method, and so may succeed operations that are analytically differentiable [(#152)](https://github.com/XanaduAI/pennylane/pull/152)

<h3>Bug fixes</h3>

* Fixed a bug where the variational classifier example was not batching when learning parity (see [#128](https://github.com/XanaduAI/pennylane/pull/128) and [#129](https://github.com/XanaduAI/pennylane/pull/129))
* Fixed an inconsistency where some initial state operations were documented as accepting complex parameters - all operations
  now accept real values [(#146)](https://github.com/XanaduAI/pennylane/pull/146)

<h3>Contributors</h3>

This release contains contributions from:

Christian Gogolin, Josh Izaac, Nathan Killoran, and Maria Schuld.


