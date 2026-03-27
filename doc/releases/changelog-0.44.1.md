# Release 0.44.1 (current release)

<h3>Bug fixes 🐛</h3>

* The ``gast`` package is now an explicit dependency in PennyLane. The ``gast`` package was previously
  pulled in transitively by ``diastatic-malt``, but ``diastatic-malt==2.15.3`` dropped ``gast`` as a dependency, which caused an error when importing PennyLane.
  [(#9160)](https://github.com/PennyLaneAI/pennylane/pull/9160)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Yushao Chen,
Andrija Paurević,
David Wierichs
