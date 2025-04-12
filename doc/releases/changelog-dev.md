:orphan:

# Release 0.42.0-dev (development release)

<h3>New features since last release</h3>
Added support for Keras 3 with Tensorflow (>=2.0) and Pytorch backends. Acheived by:
<ul>
  <li> Replacing all `tf.*` calls in the original KerasLayer the multiplatform `ops.*` call from  Keras 3 .</li>
  <li> Added backend check at the top of Keras 3 to allow for backend specific changes</li>
  <li>Modified the `construct()` class in the Keras Layer class to utlize the TorchLayer circuit construct if the backend is torch.</li>
</ul>

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Fixes a bug where the global phase was not being added in the ``QubitUnitary`` decomposition.  
  [(#7244)](https://github.com/PennyLaneAI/pennylane/pull/7244)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

- Guillermo Alonso-Linaje
- Vinayak Sharma