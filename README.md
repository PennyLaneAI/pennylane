<p align="center">
  <!-- Tests (GitHub actions) -->
  <a href="https://github.com/PennyLaneAI/pennylane/actions?query=workflow%3ATests">
    <img src="https://img.shields.io/github/workflow/status/PennyLaneAI/pennylane/Tests/master?logo=github&style=flat-square" />
  </a>
  <!-- CodeCov -->
  <a href="https://codecov.io/gh/PennyLaneAI/pennylane">
    <img src="https://img.shields.io/codecov/c/github/PennyLaneAI/pennylane/master.svg?logo=codecov&style=flat-square" />
  </a>
  <!-- ReadTheDocs -->
  <a href="https://docs.pennylane.ai/en/latest">
    <img src="https://readthedocs.com/projects/xanaduai-pennylane/badge/?version=latest&style=flat-square" />
  </a>
  <!-- PyPI -->
  <a href="https://pypi.org/project/PennyLane">
    <img src="https://img.shields.io/pypi/v/PennyLane.svg?style=flat-square" />
  </a>
  <!-- Forum -->
  <a href="https://discuss.pennylane.ai">
    <img src="https://img.shields.io/discourse/https/discuss.pennylane.ai/posts.svg?logo=discourse&style=flat-square" />
  </a>
  <!-- License -->
  <a href="https://www.apache.org/licenses/LICENSE-2.0">
    <img src="https://img.shields.io/pypi/l/PennyLane.svg?logo=apache&style=flat-square" />
  </a>
</p>

<p align="center">
  <a href="https://pennylane.ai">PennyLane</a> is a cross-platform Python library for <a
  href="https://en.wikipedia.org/wiki/Differentiable_programming">differentiable
  programming</a> of quantum computers.
</p>

<p align="center">
  <strong>Train a quantum computer the same way as a neural network.</strong>
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/header.png#gh-light-mode-only" width="700px">
    <!--
    Use a relative import for the dark mode image. When loading on PyPI, this
    will fail automatically and show nothing.
    -->
    <img src="./doc/_static/header-dark-mode.png#gh-dark-mode-only" width="700px" onerror="this.style.display='none'" alt=""/>
</p>

## Key Features

<img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/code.png" width="400px" align="right">

- *Machine learning on quantum hardware*. Connect to quantum hardware using **PyTorch**, **TensorFlow**, **JAX**, **Keras**, or **NumPy**. Build rich and flexible hybrid quantum-classical models.

- *Device-independent*. Run the same quantum circuit on different quantum backends. Install
  [plugins](https://pennylane.ai/plugins.html) to access even more devices, including **Strawberry
  Fields**, **Amazon Braket**, **IBM Q**, **Google Cirq**, **Rigetti Forest**, **Qulacs**, **Pasqal**, **Honeywell**, and more.

- *Follow the gradient*. Hardware-friendly **automatic differentiation** of quantum circuits.

- *Batteries included*. Built-in tools for **quantum machine learning**, **optimization**, and
  **quantum chemistry**. Rapidly prototype using built-in quantum simulators with
  backpropagation support.

## Installation

PennyLane requires Python version 3.7 and above. Installation of PennyLane, as well as all
dependencies, can be done using pip:

```console
python -m pip install pennylane
```

## Docker support

**Docker** support exists for building using **CPU** and **GPU** (Nvidia CUDA
11.1+) images. [See a more detailed description
here](https://pennylane.readthedocs.io/en/stable/development/guide/installation.html#installation).

## Getting started

For an introduction to quantum machine learning, guides and resources are available on
PennyLane's [quantum machine learning hub](https://pennylane.ai/qml/):

<img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/gpu_to_qpu.png" align="right" width="400px">

* [What is quantum machine learning?](https://pennylane.ai/qml/whatisqml.html)
* [QML tutorials and demos](https://pennylane.ai/qml/demonstrations.html)
* [Frequently asked questions](https://pennylane.ai/faq.html)
* [Key concepts of QML](https://pennylane.ai/qml/glossary.html)
* [QML videos](https://pennylane.ai/qml/videos.html)

You can also check out our [documentation](https://pennylane.readthedocs.io) for [quickstart
guides](https://pennylane.readthedocs.io/en/stable/introduction/pennylane.html) to using PennyLane,
and detailed developer guides on [how to write your
own](https://pennylane.readthedocs.io/en/stable/development/plugins.html) PennyLane-compatible
quantum device.

## Tutorials and demonstrations

Take a deeper dive into quantum machine learning by exploring cutting-edge algorithms on our [demonstrations
page](https://pennylane.ai/qml/demonstrations.html).

<a href="https://pennylane.ai/qml/demonstrations.html">
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/demos.png" width="900px">
</a>

All demonstrations are fully executable, and can be downloaded as Jupyter notebooks and Python
scripts.

If you would like to contribute your own demo, see our [demo submission
guide](https://pennylane.ai/qml/demos_submission.html).

## Videos

Seeing is believing! Check out [our videos](https://pennylane.ai/qml/videos.html) to learn about
PennyLane, quantum computing concepts, and more. 

<a href="https://pennylane.ai/qml/videos.html)">
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/videos.png" width="900px">
</a>

## Contributing to PennyLane

We welcome contributions—simply fork the PennyLane repository, and then make a [pull
request](https://help.github.com/articles/about-pull-requests/) containing your contribution. All
contributors to PennyLane will be listed as authors on the releases. All users who contribute
significantly to the code (new plugins, new functionality, etc.) will be listed on the PennyLane
arXiv paper.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool
projects or applications built on PennyLane.

See our [contributions
page](https://github.com/PennyLaneAI/pennylane/blob/master/.github/CONTRIBUTING.md) and our
[developer hub](https://pennylane.readthedocs.io/en/stable/development/guide.html) for more
details.

## Support

- **Source Code:** https://github.com/PennyLaneAI/pennylane
- **Issue Tracker:** https://github.com/PennyLaneAI/pennylane/issues

If you are having issues, please let us know by posting the issue on our GitHub issue tracker.

We also have a [PennyLane discussion forum](https://discuss.pennylane.ai)—come join the community
and chat with the PennyLane team.

Note that we are committed to providing a friendly, safe, and welcoming environment for all.
Please read and respect the [Code of Conduct](.github/CODE_OF_CONDUCT.md).

## Authors

PennyLane is the work of [many contributors](https://github.com/PennyLaneAI/pennylane/graphs/contributors).

If you are doing research using PennyLane, please cite [our paper](https://arxiv.org/abs/1811.04968):

> Ville Bergholm et al. *PennyLane: Automatic differentiation of hybrid quantum-classical
> computations.* 2018. arXiv:1811.04968

## License

PennyLane is **free** and **open source**, released under the Apache License, Version 2.0.
