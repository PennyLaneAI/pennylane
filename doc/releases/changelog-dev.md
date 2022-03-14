:orphan:

# Release 0.23.0-dev (development release)

<h3>New features since last release</h3>

* Development of a circuit-cutting compiler extension to circuits with sampling
  measurements has begun:

  - The existing `qcut.tape_to_graph()` method has been extended to convert a
    sample measurement without an observable specified to multiple single-qubit sample
    nodes.
    [(#2313)](https://github.com/PennyLaneAI/pennylane/pull/2313)

  - The existing `qcut.graph_to_tape()` method has been extended to convert
    graphs containing sample measurement nodes to tapes.
    [(#2321)](https://github.com/PennyLaneAI/pennylane/pull/2321)

<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):

Anthony Hayes