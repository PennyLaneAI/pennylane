:orphan:

# Release 0.36.0-dev (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

* `qml.expval`, `qml.var`, `qml.sample`, `qml.probs`, and `qml.counts` now construct measurement processes
  based on the type of the arguments rather than their name.
  [(#5224)](https://github.com/PennyLaneAI/pennylane/pull/5224)

  This allows users to pass in any argument as a positional argument without worrying about
  setting the arguments incorrectly. For example, previously, `qml.probs(qml.PauliZ(0))` would
  set the `wires` attribute to be `qml.PauliZ(0)` as the first argument of `probs` was `wires`.
  However, now, `qml.PauliZ(0)` is correctly set as the observable.

  Moreover, the type of the argument now supercedes the name. If the name and type of an
  argument don't match, a warning is raised. For example:

  ```pycon
  >>> qml.probs(wires=qml.PauliZ(0))
  UserWarning: probs got argument 'wires' of type <class 'pennylane.ops.qubit.non_parametric_ops.PauliZ'>. Using argument as op
    warn(
  probs(Z(0))
  ```

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* We no longer perform unwanted dtype promotion in the `pauli_rep` of `SProd` instances when using tensorflow.
  [(#5246)](https://github.com/PennyLaneAI/pennylane/pull/5246)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Korbinian Kottmann