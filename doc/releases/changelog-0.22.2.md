:orphan:

# Release 0.22.2

<h3>Bug fixes</h3>

* Most compilation transforms, and relevant subroutines, have been updated to
  support just-in-time compilation with jax.jit. This fix was intended to be
  included in `v0.22.0`, but due to a bug was incomplete.
  [(#2397)](https://github.com/PennyLaneAI/pennylane/pull/2397)

<h3>Documentation</h3>

* The documentation run has been updated to require `jinja2==3.0.3` due to an
  issue that arises with `jinja2` `v3.1.0` and `sphinx` `v3.5.3`.
  [(#2378)](https://github.com/PennyLaneAI/pennylane/pull/2378)

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Olivia Di Matteo, Christina Lee, Romain Moyard, Antal Száva.
