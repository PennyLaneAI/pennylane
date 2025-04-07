:orphan:

# Release 0.42.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* Adding `HOState` and `VibronicHO` classes for representing harmonic oscillator states.
  [(#7035)](https://github.com/PennyLaneAI/pennylane/pull/7035)

* Adding base classes for Trotter error estimation on Realspace Hamiltonians: ``RealspaceOperator``, ``RealspaceSum``, ``RealspaceCoeffs``, and ``RealspaceMatrix``
  [(#7034)](https://github.com/PennyLaneAI/pennylane/pull/7034)

* Adding functions for Trotter error estimation and Hamiltonian fragment generation: ``trotter_error``, ``perturbation_error``, ``vibrational_fragments``, ``vibronic_fragments``, and ``generic_fragments``.
  [(#7036)](https://github.com/PennyLaneAI/pennylane/pull/7036)

  As an example we compute the peruturbation error of a vibrational Hamiltonian.
  First we generate random harmonic frequences and Taylor coefficients to iniitialize the vibrational Hamiltonian.

  ```pycon
  >>> from pennylane.labs.trotter_error import HOState, vibrational_fragments, perturbation_error
  >>> import numpy as np
  >>> n_modes = 2
  >>> r_state = np.random.RandomState(42)
  >>> freqs = r_state.random(n_modes)
  >>> taylor_coeffs = [
  >>>     np.array(0),
  >>>     r_state.random(size=(n_modes, )),
  >>>     r_state.random(size=(n_modes, n_modes)),
  >>>     r_state.random(size=(n_modes, n_modes, n_modes))
  >>> ]
  ```
    
  We call ``vibrational_fragments`` to get the harmonic and anharmonic fragments of the vibrational Hamiltonian.
  ```pycon
  >>> frags = vibrational_fragments(n_modes, freqs, taylor_coeffs)
  ```

  We build state vectors in the harmonic oscilator basis with the ``HOState`` class. 

  ```pycon
  >>> gridpoints = 5
  >>> state1 = HOState(n_modes, gridpoints, {(0, 0): 1})
  >>> state2 = HOState(n_modes, gridpoints, {(1, 1): 1})
  ```

  Finally, we compute the error by calling ``perturbation_error``.

  ```pycon
  >>> perturbation_error(frags, [state1, state2])
  [(-0.9189251160920879+0j), (-4.797716682426851+0j)]
  ```

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):
