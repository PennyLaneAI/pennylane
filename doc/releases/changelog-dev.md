:orphan:

# Release 0.33.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>
* The qchem ``fermionic_dipole`` and ``particle_number`` functions are updated to use fermi 
  sentence.
  [#4546](https://github.com/PennyLaneAI/pennylane/pull/4546)

* `qml.sample()` in the new device API now returns a `np.int64` array instead of `np.bool8`.
  [(#4539)](https://github.com/PennyLaneAI/pennylane/pull/4539)

* Wires can be provided to the new device API.
  [(#4538)](https://github.com/PennyLaneAI/pennylane/pull/4538)

<h3>Breaking changes 💔</h3>

* The old return type and associated functions ``qml.enable_return`` and ``qml.disable_return`` are removed.
  [(#4503)](https://github.com/PennyLaneAI/pennylane/pull/4503)

* The ``mode`` keyword argument in ``QNode`` is removed. Please use ``grad_on_execution`` instead.
  [(#4503)](https://github.com/PennyLaneAI/pennylane/pull/4503)

* The CV observables ``qml.X`` and ``qml.P`` are removed. Please use ``qml.QuadX`` and ``qml.QuadP`` instead.
  [#4533](https://github.com/PennyLaneAI/pennylane/pull/4533)

* The method ``tape.unwrap()`` and corresponding ``UnwrapTape`` and ``Unwrap`` classes are removed.
  Instead of ``tape.unwrap()``, use :func:`~.transforms.convert_to_numpy_parameters`.
  [#4535](https://github.com/PennyLaneAI/pennylane/pull/4535)

* The ``sampler_seed`` argument of ``qml.gradients.spsa_grad`` has been removed.
  Instead, the ``sampler_rng`` argument should be set, either to an integer value, which will be used
  to create a PRNG internally, or to a NumPy pseudo-random number generator (PRNG) created via
  ``np.random.default_rng(seed)``.
  [(#4550)](https://github.com/PennyLaneAI/pennylane/pull/4550)

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* `tf.function` no longer breaks `ProbabilityMP.process_state` which is needed by new devices.
  [(#4470)](https://github.com/PennyLaneAI/pennylane/pull/4470)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):


Soran Jahangiri,
Romain Moyard,
Matthew Silverman

