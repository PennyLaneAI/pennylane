# Release 0.44.1 (current release)

<h3>Bug fixes 🐛</h3>

* Add `gast` as an explicit dependency to fix `ModuleNotFoundError: No module named 'gast'`
  on import when `diastatic-malt>=2.15.3` is installed. The `gast` package was previously
  pulled in transitively by `diastatic-malt`, but version 2.15.3 dropped it as a dependency.
  [(#9160)](https://github.com/PennyLaneAI/pennylane/pull/9160)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Yushao Chen,
Andrija Paurević,
David Wierichs
