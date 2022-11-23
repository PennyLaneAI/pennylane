:orphan:

# Release 0.28.0-dev (development release)

<h3>New features since last release</h3>

* New gradient transform `qml.gradients.spsa` based on the idea of SPSA.
  [#3366](https://github.com/PennyLaneAI/pennylane/pull/3366)

  This new transform allows to compute a single estimate of a quantum gradient
  using simultaneous perturbation of parameters and a stochastic approximation.
  Given some QNode `circuit` that takes, say, an argument `x`, the approximate
  gradient can be computed via

  >>> grad_fn = qml.gradients.spsa(circuit, h=0.1, num_samples=1
  >>> grad = grad_fn(x)

  The argument `num_samples` determines how many directions of simultaneous
  perturbation are used and therefore the number of circuit evaluations, up
  to a prefactor. See the
  [spsa gradient transform documentation](
  https://docs.pennylane.ai/en/stable/code/api/pennylane.gradients.spsa.html
  ) for details.
  Note: The full SPSA optimization method already is available as `SPSAOptimizer`.

* New basis sets, `6-311g` and `CC-PVDZ`, are added to the qchem basis set repo.
  [#3279](https://github.com/PennyLaneAI/pennylane/pull/3279)

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Bug fixes</h3>

* Small fix of `MeasurementProcess.map_wires`, where both the `self.obs` and `self._wires`
  attributes were modified.
  [#3292](https://github.com/PennyLaneAI/pennylane/pull/3292)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

David Wierichs
