<p align="center">
  <!-- Tests (GitHub actions) -->
  <a href="https://github.com/PennyLaneAI/pennylane/actions?query=workflow%3ATests">
    <img src="https://img.shields.io/github/actions/workflow/status/PennyLaneAI/PennyLane/tests.yml?branch=main&style=flat-square" />
  </a>
  <!-- CodeCov -->
  <a href="https://codecov.io/gh/PennyLaneAI/pennylane">
    <img src="https://img.shields.io/codecov/c/github/PennyLaneAI/pennylane/main.svg?logo=codecov&style=flat-square" />
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
  <a href="https://pennylane.ai">PennyLane</a> is an open-source quantum software platform
  <a href="https://pennylane.ai/qml/quantum-computing/">quantum computing</a>,
  <a href="https://pennylane.ai/topics/quantum-machine-learning">quantum machine learning</a>,
  and
  <a href="https://pennylane.ai/topics/hamiltonian-simulation">quantum chemistry</a>.
</p>

<p align="center">
  Create meaningful quantum algorithms, from inspiration to implementation.
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/main/doc/_static/readme/pl-logo-lightmode.png#gh-light-mode-only" width="700px">
    <!--
    Use a relative import for the dark mode image. When loading on PyPI, this
    will fail automatically and show nothing.
    -->
    <img src="./doc/_static/readme/pl-logo-darkmode.png#gh-dark-mode-only" width="700px" onerror="this.style.display='none'" alt=""/>
</p>


## Key Features

<img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/main/doc/_static/code.png" width="400px" align="right">


- <strong>*Inspiration to implementation, quickly.*</strong>

  Quantum computing can be complex — PennyLane makes it natural. Leverage the world’s largest library of [research demos](https://pennylane.ai/qml/demonstrations), [interactive tutorials](https://pennylane.ai/codebook/), and state-of-the-art components to build algorithms in [quantum chemistry](https://docs.pennylane.ai/en/stable/introduction/chemistry.html), quantum information, optimization, and [quantum machine learning](https://pennylane.ai/topics/quantum-machine-learning).

- <strong>*Fast where it matters. Scalable where it counts.*</strong>

  Whether executing, compiling, or analyzing, PennyLane is fast. Unlock production-grade performance with [industrial resource estimation](https://pennylane.ai/qml/demos/re_how_to_use_pennylane_for_resource_estimation) and the [Catalyst compiler](https://github.com/PennyLaneAI/Catalyst). Scale up your workflows with the [high-performance Lightning simulators](https://pennylane.ai/performance) on GPUs, supercomputers, and the cloud.

- <strong>*Hardware agnostic, hardware ready.*</strong>

  PennyLane integrates with a wide range of [quantum hardware devices](https://pennylane.ai/devices). Whether superconducting qubits, trapped ion systems, neutral atoms, or photonics, PennyLane provides the tools to [estimate resources](https://pennylane.ai/qml/demos/re_how_to_use_pennylane_for_resource_estimation) and [compile circuits](https://staging.pennylane.ai/topics/quantum-compilation) specifically for the hardware devices of today—and tomorrow!

- <strong>*Participate, collaborate, innovate.*</strong>

  PennyLane is the world’s most [active quantum community](https://staging.pennylane.ai/get-involved). You're part of a global network of [researchers](https://pennylane.ai/research), [developers](https://pennylane.ai/features), and [educators](https://pennylane.ai/education) actively defining the frontier of quantum computing. Whether quantum is your day job or you’re getting your first taste at a [hackathon](https://pennylane.ai/challenges), you’re backed by the [most responsive community](https://discuss.pennylane.ai) in the field.

For more details and additional features, please see the [PennyLane website](https://pennylane.ai/features/) and our most recent [release notes](https://docs.pennylane.ai/en/stable/development/release_notes.html).

## Installation

PennyLane requires Python version 3.11 and above. Installation of PennyLane, as well as all
dependencies, can be done using pip:

```console
python -m pip install pennylane
```

## Docker support

Docker images are found on the [PennyLane Docker Hub page](https://hub.docker.com/u/pennylaneai), where there is also a detailed description about PennyLane Docker support. [See description here](https://docs.pennylane.ai/projects/lightning/en/stable/dev/docker.html) for more information.

## Getting started

Get up and running quickly with PennyLane by following our [interactive tutorials](https://pennylane.ai/codebook/pennylane-fundamentals) and [quickstart guide](https://docs.pennylane.ai/en/stable/introduction/pennylane.html), designed to introduce key features and help you start building quantum circuits right away.

Whether you're exploring quantum machine learning, quantum computing, or quantum chemistry, PennyLane offers a wide range of tools and resources to support your research.

<img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/main/doc/_static/readme/research.png" align="right" width="350px">

### Key Resources

* [Library of research demos](https://pennylane.ai/qml/demonstrations)
* [Learn Quantum Programming](https://pennylane.ai/qml/) with the [Codebook](https://pennylane.ai/codebook/) and [Coding Challenges](https://pennylane.ai/challenges/)
* [PennyLane Discussion Forum](https://discuss.pennylane.ai)

You can also check out our [documentation](https://pennylane.readthedocs.io), and detailed [developer guides](https://docs.pennylane.ai/en/stable/development/guide.html).

## Demos

Take a deeper dive into quantum computing by exploring quantum computing research with the [PennyLane Demos](https://pennylane.ai/qml/demonstrations)—covering fundamental quantum concepts alongside the latest quantum algorithm research results.

If you would like to contribute your own demo, see our [demo submission
guide](https://pennylane.ai/qml/demos_submission).

<a href="https://pennylane.ai/qml/demonstrations">
  <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/main/doc/_static/readme/demos.png" width="900px">
</a>


## Contributing to PennyLane

We welcome contributions—simply fork the PennyLane repository, and then make a [pull
request](https://help.github.com/articles/about-pull-requests/) containing your contribution. All
contributors to PennyLane will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool
projects or applications built on PennyLane.

See our [contributions
page](https://github.com/PennyLaneAI/pennylane/blob/main/.github/CONTRIBUTING.md) and our
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
