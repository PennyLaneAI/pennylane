# Release 0.46.0 (development release)

<h3>New features since last release</h3>

<h3>Improvements 🛠</h3>

<h3>Labs: a place for unified and rapid prototyping of research software 🧪</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Internal changes ⚙️</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

* Fixed a bug where `qp.qnn.TorchLayer` produced incorrect output shape `(n_measurements, batch, 1)`
  instead of `(batch, n_measurements)` when the wrapped QNode returns multiple measurements as a tuple
  (e.g., `return qp.expval(qp.Z(0)), qp.expval(qp.Z(1))`) and receives batched inputs. This
  previously caused shape mismatch errors when feeding the output into downstream `torch.nn.Linear`
  layers.
  [(#9284)](https://github.com/PennyLaneAI/pennylane/pull/9284)

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Daniel Casota.