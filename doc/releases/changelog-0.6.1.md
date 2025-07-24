
# Release 0.6.1

<h3>New features since last release</h3>

* Added a `print_applied` method to QNodes, allowing the operation
  and observable queue to be printed as last constructed.
  [(#378)](https://github.com/XanaduAI/pennylane/pull/378)

<h3>Improvements</h3>

* A new `Operator` base class is introduced, which is inherited by both the
  `Observable` class and the `Operation` class.
  [(#355)](https://github.com/XanaduAI/pennylane/pull/355)

* Removed deprecated `@abstractproperty` decorators
  in `_device.py`.
  [(#374)](https://github.com/XanaduAI/pennylane/pull/374)

* The `CircuitGraph` class is updated to deal with `Operation` instances directly.
  [(#344)](https://github.com/XanaduAI/pennylane/pull/344)

* Comprehensive gradient tests have been added for the interfaces.
  [(#381)](https://github.com/XanaduAI/pennylane/pull/381)

<h3>Documentation</h3>

* The new restructured documentation has been polished and updated.
  [(#387)](https://github.com/XanaduAI/pennylane/pull/387)
  [(#375)](https://github.com/XanaduAI/pennylane/pull/375)
  [(#372)](https://github.com/XanaduAI/pennylane/pull/372)
  [(#370)](https://github.com/XanaduAI/pennylane/pull/370)
  [(#369)](https://github.com/XanaduAI/pennylane/pull/369)
  [(#367)](https://github.com/XanaduAI/pennylane/pull/367)
  [(#364)](https://github.com/XanaduAI/pennylane/pull/364)

* Updated the development guides.
  [(#382)](https://github.com/XanaduAI/pennylane/pull/382)
  [(#379)](https://github.com/XanaduAI/pennylane/pull/379)

* Added all modules, classes, and functions to the API section
  in the documentation.
  [(#373)](https://github.com/XanaduAI/pennylane/pull/373)

<h3>Bug fixes</h3>

* Replaces the existing `np.linalg.norm` normalization with hand-coded
  normalization, allowing `AmplitudeEmbedding` to be used with differentiable
  parameters. AmplitudeEmbedding tests have been added and improved.
  [(#376)](https://github.com/XanaduAI/pennylane/pull/376)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Ville Bergholm, Josh Izaac, Nathan Killoran, Maria Schuld, Antal Száva
