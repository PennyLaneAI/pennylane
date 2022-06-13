:orphan:

# Release 0.25.0-dev (development release)

<h3>New features since last release</h3>

* `DefaultQubit` devices now natively support parameter broadcasting.
  [(#2627)](https://github.com/PennyLaneAI/pennylane/pull/2627)
  
  Instead of utilizing the `broadcast_expand` transform, `DefaultQubit`
  devices now are able to directly execute broadcasted circuits, providing
  a faster way of executing the same circuit at varied parameter positions.
  
<h3>Improvements</h3>

<h3>Breaking changes</h3>

<h3>Deprecations</h3>

<h3>Documentation</h3>

<h3>Contributors</h3>

This release contains contributions from (in alphabetical order):
