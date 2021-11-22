:orphan:

# Release 0.19.1 (current release)

<h3>Bug fixes</h3>

* Fixes a bug where using JAX's jit function on certain QNodes that contain
  the `qml.QubitStateVector` operation raised an error with earlier JAX
  versions (e.g., `jax==0.2.10` and `jaxlib==0.1.64`).
  [(#1924)](https://github.com/PennyLaneAI/pennylane/pull/1924)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Josh Izaac, Romain Moyard, Antal Száva.
