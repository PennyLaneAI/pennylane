:orphan:

# Release 0.40.0-dev (development release)

<h3>New features since last release</h3>

* A `DeviceCapabilities` data class is defined to contain all capabilities of the device's execution interface (i.e. its implementation of `Device.execute`). A TOML file can be used to define the capabilities of a device, and it can be loaded into a `DeviceCapabilities` object.
  [(#6407)](https://github.com/PennyLaneAI/pennylane/pull/6407)

  ```pycon
  >>> from pennylane.devices.capabilities import load_toml_file, parse_toml_document, DeviceCapabilities
  >>> document = load_toml_file("my_device.toml")
  >>> capabilities = parse_toml_document(document)
  >>> isinstance(capabilities, DeviceCapabilities)
  True
  ```

<h3>Improvements 🛠</h3>

<h3>Breaking changes 💔</h3>

<h3>Deprecations 👋</h3>

<h3>Documentation 📝</h3>

<h3>Bug fixes 🐛</h3>

<h3>Contributors ✍️</h3>

This release contains contributions from (in alphabetical order):

Astral Cai,
