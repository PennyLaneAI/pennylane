<img src="https://assets.cloud.pennylane.ai/pennylane_website/spotlights/Spotlight_PLSurvey_2024-03-05.png"
     width="200px"
     align="left" />

Help us shape the future of PennyLane. We’d really appreciate a few minutes of your time to share feedback through our quantum programming survey. [Take the survey!](https://bit.ly/pl-survey-2026-g)

<br clear=left>
<br>

<p align="center">
  <!-- Tests (GitHub actions) -->
  <a href="https://github.com/PennyLaneAI/pennylane/actions?query=workflow%3ATests">
    <img src="https://img.shields.io/github/actions/workflow/status/PennyLaneAI/PennyLane/tests.yml?branch=master&style=flat-square" />
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
  <a href="https://pennylane.ai">PennyLane</a> is a cross-platform Python library for
  <a href="https://pennylane.ai/qml/quantum-computing/">quantum computing</a>,
  <a href="https://pennylane.ai/qml/quantum-machine-learning/">quantum machine learning</a>,
  and
  <a href="https://pennylane.ai/qml/quantum-chemistry/">quantum chemistry</a>.
</p>

<p align="center">
  The definitive open-source framework for quantum programming. Built by researchers, for research.
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/pl-logo-lightmode.png#gh-light-mode-only" width="700px">
    <!--
    Use a relative import for the dark mode image. When loading on PyPI, this
    will fail automatically and show nothing.
    -->
    <img src="./doc/_static/readme/pl-logo-darkmode.png#gh-dark-mode-only" width="700px" onerror="this.style.display='none'" alt=""/>
</p>


## Key Features

<img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/code.png" width="400px" align="right">

- <strong>*Program quantum computers*</strong>. Build quantum circuits with a wide range of state preparations, gates, and measurements. Run on [high-performance simulators](https://pennylane.ai/performance/) or [various hardware devices](https://pennylane.ai/plugins/), with advanced features like mid-circuit measurements and error mitigation.

- <strong>*Master quantum algorithms*</strong>. From NISQ to fault-tolerant quantum computing, unlock algorithms for research and application. Analyze performance, visualize circuits, and access tools for [quantum chemistry](https://docs.pennylane.ai/en/stable/introduction/chemistry.html) and [algorithm development](https://pennylane.ai/search/?contentType=DEMO&categories=algorithms&sort=publication_date).

- <strong>*Machine learning with quantum hardware and simulators*</strong>. Integrate with **PyTorch**, **TensorFlow**, **JAX**, **Keras**, or **NumPy** to define and train hybrid models using quantum-aware optimizers and hardware-compatible gradients for advanced research tasks. [Quantum machine learning quickstart](https://docs.pennylane.ai/en/stable/introduction/interfaces.html).


- <strong>*Quantum datasets*</strong>. Access high-quality, pre-simulated datasets to decrease time-to-research and accelerate algorithm development. [Browse the datasets](https://pennylane.ai/datasets/) or contribute your own data.


- <strong>*Compilation and performance*</strong>. Experimental support for just-in-time
  compilation. Compile your entire hybrid workflow, with support for 
  advanced features such as adaptive circuits, real-time measurement 
  feedback, and unbounded loops. See
  [Catalyst](https://github.com/pennylaneai/catalyst) for more details.

For more details and additional features, please see the [PennyLane website](https://pennylane.ai/features/).

## Installation

PennyLane requires Python version 3.11 and above. Installation of PennyLane, as well as all
dependencies, can be done using pip:

```console
python -m pip install pennylane
```

## Docker support

Docker images are found on the [PennyLane Docker Hub page](https://hub.docker.com/u/pennylaneai), where there is also a detailed description about PennyLane Docker support. [See description here](https://docs.pennylane.ai/projects/lightning/en/stable/dev/docker.html) for more information.

## Getting started

Get up and running quickly with PennyLane by following our [quickstart guide](https://docs.pennylane.ai/en/stable/introduction/pennylane.html), designed to introduce key features and help you start building quantum circuits right away.

Whether you're exploring quantum machine learning (QML), quantum computing, or quantum chemistry, PennyLane offers a wide range of tools and resources to support your research:

<img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/research.png" align="right" width="350px">

### Key Resources:

* [Research-oriented Demos](https://pennylane.ai/qml/demonstrations)
* [Learn Quantum Programming](https://pennylane.ai/qml/) with the [Codebook](https://pennylane.ai/codebook/) and [Coding Challenges](https://pennylane.ai/challenges/)
* [Frequently Asked Questions](https://pennylane.ai/faq)
* [Glossary](https://pennylane.ai/qml/glossary)
* [Videos](https://pennylane.ai/qml/videos)


You can also check out our [documentation](https://pennylane.readthedocs.io) for [quickstart
guides](https://pennylane.readthedocs.io/en/stable/introduction/pennylane.html) to using PennyLane,
and detailed developer guides on [how to write your
own](https://pennylane.readthedocs.io/en/stable/development/plugins.html) PennyLane-compatible
quantum device.

## Demos

Take a deeper dive into quantum computing by exploring cutting-edge algorithms using PennyLane and quantum hardware. [Explore PennyLane demos](https://pennylane.ai/qml/demonstrations).

<a href="https://pennylane.ai/qml/demonstrations">
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/readme/demos.png" width="900px">
</a>

If you would like to contribute your own demo, see our [demo submission
guide](https://pennylane.ai/qml/demos_submission).

## Research Applications

PennyLane is at the forefront of research in quantum computing, quantum machine learning, and quantum chemistry. Explore how PennyLane is used for research in the following publications:

- **Quantum Computing**: [Fast quantum circuit cutting with randomized measurements](https://quantum-journal.org/papers/q-2023-03-02-934/)

- **Quantum Machine Learning**: [Better than classical? The subtle art of benchmarking quantum machine learning models](https://arxiv.org/abs/2403.07059)

- **Quantum Chemistry**: [Accelerating Quantum Computations of Chemistry Through Regularized Compressed Double Factorization](https://quantum-journal.org/papers/q-2024-06-13-1371/)

Impactful research drives PennyLane. Let us know what features you need for your research on [GitHub](https://github.com/PennyLaneAI/pennylane/issues/new?assignees=&labels=enhancement+%3Asparkles%3A&projects=&template=feature_request.yml) or on our [website](https://pennylane.ai/research).



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
[Development guide](https://pennylane.readthedocs.io/en/stable/development/guide.html) for more
details.

## Support

- **Source Code:** https://github.com/PennyLaneAI/pennylane
- **Issue Tracker:** https://github.com/PennyLaneAI/pennylane/issues

If you are having issues, please let us know by posting the issue on our GitHub issue tracker.

Join the [PennyLane Discussion Forum](https://discuss.pennylane.ai/) to connect with the quantum community, get support, and engage directly with our team. It’s the perfect place to share ideas, ask questions, and collaborate with fellow researchers and developers!

Note that we are committed to providing a friendly, safe, and welcoming environment for all.
Please read and respect the [Code of Conduct](.github/CODE_OF_CONDUCT.md).

## Authors

PennyLane is the work of [many contributors](https://github.com/PennyLaneAI/pennylane/graphs/contributors).

If you are doing research using PennyLane, please cite [our paper](https://arxiv.org/abs/1811.04968):

> Ville Bergholm et al. *PennyLane: Automatic differentiation of hybrid quantum-classical
> computations.* 2018. arXiv:1811.04968

## License

PennyLane is **free** and **open source**, released under the Apache License, Version 2.0.
