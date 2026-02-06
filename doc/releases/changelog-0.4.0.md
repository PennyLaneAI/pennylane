
# Release 0.4.0

<h3>New features since last release</h3>

* `pennylane.expval()` is now a top-level *function*, and is no longer
  a package of classes. For now, the existing `pennylane.expval.Observable`
  interface continues to work, but will raise a deprecation warning.
  [(#232)](https://github.com/XanaduAI/pennylane/pull/232)

* Variance support: QNodes can now return the variance of observables,
  via the top-level `pennylane.var()` function. To support this on
  plugin devices, there is a new `Device.var` method.

  The following observables support analytic gradients of variances:

  - All qubit observables (requiring 3 circuit evaluations for involutory
    observables such as `Identity`, `X`, `Y`, `Z`; and 5 circuit evals for
    non-involutary observables, currently only `qp.Hermitian`)

  - First-order CV observables (requiring 5 circuit evaluations)

  Second-order CV observables support numerical variance gradients.

* `pennylane.about()` function added, providing details
  on current PennyLane version, installed plugins, Python,
  platform, and NumPy versions [(#186)](https://github.com/XanaduAI/pennylane/pull/186)

* Removed the logic that allowed `wires` to be passed as a positional
  argument in quantum operations. This allows us to raise more useful
  error messages for the user if incorrect syntax is used.
  [(#188)](https://github.com/XanaduAI/pennylane/pull/188)

* Adds support for multi-qubit expectation values of the `pennylane.Hermitian()`
  observable [(#192)](https://github.com/XanaduAI/pennylane/pull/192)

* Adds support for multi-qubit expectation values in `default.qubit`.
  [(#202)](https://github.com/XanaduAI/pennylane/pull/202)

* Organize templates into submodules [(#195)](https://github.com/XanaduAI/pennylane/pull/195).
  This included the following improvements:

  - Distinguish embedding templates from layer templates.

  - New random initialization functions supporting the templates available
    in the new submodule `pennylane.init`.

  - Added a random circuit template (`RandomLayers()`), in which rotations and 2-qubit gates are randomly
    distributed over the wires

  - Add various embedding strategies

<h3>Breaking changes</h3>

* The `Device` methods `expectations`, `pre_expval`, and `post_expval` have been
  renamed to `observables`, `pre_measure`, and `post_measure` respectively.
  [(#232)](https://github.com/XanaduAI/pennylane/pull/232)

<h3>Improvements</h3>

* `default.qubit` plugin now uses `np.tensordot` when applying quantum operations
  and evaluating expectations, resulting in significant speedup
  [(#239)](https://github.com/XanaduAI/pennylane/pull/239),
  [(#241)](https://github.com/XanaduAI/pennylane/pull/241)

* PennyLane now allows division of quantum operation parameters by a constant
  [(#179)](https://github.com/XanaduAI/pennylane/pull/179)

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

<h3>Bug fixes</h3>

* Fixed a bug in `Device.supported`, which would incorrectly
  mark an operation as supported if it shared a name with an
  observable [(#203)](https://github.com/XanaduAI/pennylane/pull/203)

* Fixed a bug in `Operation.wires`, by explicitly casting the
  type of each wire to an integer [(#206)](https://github.com/XanaduAI/pennylane/pull/206)

* Removed code in PennyLane which configured the logger,
  as this would clash with users' configurations
  [(#208)](https://github.com/XanaduAI/pennylane/pull/208)

* Fixed a bug in `default.qubit`, in which `QubitStateVector` operations
  were accidentally being cast to `np.float` instead of `np.complex`.
  [(#211)](https://github.com/XanaduAI/pennylane/pull/211)


<h3>Contributors</h3>

This release contains contributions from:

Shahnawaz Ahmed, riveSunder, Aroosa Ijaz, Josh Izaac, Nathan Killoran, Maria Schuld.

