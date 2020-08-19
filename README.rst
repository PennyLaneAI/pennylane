.. image:: doc/_static/pennylane_thin.png
    :alt: PennyLane

###################################

.. |CI| image:: https://img.shields.io/github/workflow/status/PennyLaneAI/pennylane/Tests/master?logo=github&style=flat-square
    :alt: GitHub Workflow Status (branch)
    :target: https://github.com/PennyLaneAI/pennylane/actions?query=workflow%3ATests

.. |COV| image:: https://img.shields.io/codecov/c/github/xanaduai/pennylane/master.svg?logo=codecov&style=flat-square
    :alt: Codecov coverage
    :target: https://codecov.io/gh/PennyLaneAI/pennylane

.. |PEP| image:: https://img.shields.io/codefactor/grade/github/PennyLaneAI/pennylane/master?logo=codefactor&style=flat-square
    :alt: CodeFactor Grade
    :target: https://www.codefactor.io/repository/github/pennylaneai/pennylane

.. |DOC| image:: https://img.shields.io/readthedocs/pennylane.svg?logo=read-the-docs&style=flat-square
    :alt: Read the Docs
    :target: https://pennylane.readthedocs.io

.. |VERS| image:: https://img.shields.io/pypi/v/PennyLane.svg?style=flat-square
    :alt: PyPI
    :target: https://pypi.org/project/PennyLane

.. |PY| image:: https://img.shields.io/pypi/pyversions/PennyLane.svg?style=flat-square
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/PennyLane

.. |FORUM| image:: https://img.shields.io/discourse/https/discuss.pennylane.ai/posts.svg?logo=discourse&style=flat-square
    :alt: Discourse posts
    :target: https://discuss.pennylane.ai

.. |LIC| image:: https://img.shields.io/pypi/l/PennyLane.svg?logo=apache&style=flat-square
    :alt: PyPI - License
    :target: https://www.apache.org/licenses/LICENSE-2.0

|CI|  |COV| |PEP| |DOC| |VERS| |PY| |FORUM|

`PennyLane <https://pennylane.ai>`_ is a cross-platform Python library for `differentiable programming <https://en.wikipedia.org/wiki/Differentiable_programming>`__ of quantum computers.

.. raw:: html

    <p align="center">
    <b>Train a quantum computer the same way as a neural network.</b>
    <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/jigsaw.png" width="600px"  align="center">
    </p>

PennyLane provides open-source tools for quantum machine learning, quantum computing, quantum chemistry, and hybrid quantum-classical computing. Extensive examples, tutorials, and demos are available at https://pennylane.ai/qml.

Key Features
============

.. raw:: html

    <img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/code.png" width="300px"  align="right">

- *Device independent*.
  Access quantum hardware and simulators from **Xanadu Strawberry Fields**, **IBM Q**, **Google Cirq**, **Rigetti Forest**, and
  **Microsoft QDK**.

- *Best of both worlds*.
  Build hybrid models by connecting quantum hardware to **PyTorch**, **TensorFlow**, **Keras**, and **NumPy**.

- *Follow the gradient*. Hardware-friendly **automatic differentiation** of quantum circuits.

- *Batteries included*. Built-in tools for **quantum machine learning**, **optimization**, and **quantum chemistry**.

Getting started
===============

For an introduction to quantum machine learning, we have several guides and resources available
on PennyLane's `quantum machine learning page <https://pennylane.ai/qml/>`_:

* `What is quantum machine learning? <https://pennylane.ai/qml/whatisqml.html>`_
* `QML tutorials and demonstrations <https://pennylane.ai/qml/demonstrations.html>`_
* `Frequently asked questions <https://pennylane.ai/faq.html>`_
* `Glossary of key concepts <https://pennylane.ai/qml/glossary.html>`_
* `Curated selection of QML videos <https://pennylane.ai/qml/videos.html>`_

You can also check out our `documentation <https://pennylane.readthedocs.io>`_ for
`quickstart guides <https://pennylane.readthedocs.io/en/stable/introduction/pennylane.html>`_
to using PennyLane, and detailed developer guides on
`how to write your own <https://pennylane.readthedocs.io/en/stable/development/plugins.html>`_
PennyLane-compatible quantum device.

Available plugins
=================

* `PennyLane-SF <https://github.com/PennyLaneAI/pennylane-sf>`_: Supports integration with
  `Strawberry Fields <https://github.com/PennyLaneAI/strawberryfields>`__, a full-stack
  Python library for simulating photonic quantum computing.


* `PennyLane-qiskit <https://github.com/PennyLaneAI/pennylane-qiskit>`_: Supports
  integration with `Qiskit <https://qiskit.org>`__, an open-source quantum
  computation framework by IBM. Provides device support for the Qiskit Aer quantum
  simulators, and IBM Q hardware devices.


* `PennyLane-cirq <https://github.com/PennyLaneAI/pennylane-cirq>`_: Supports
  integration with `Cirq <https://github.com/quantumlib/cirq>`__, an open-source quantum
  computation framework by Google.


* `PennyLane-Forest <https://github.com/rigetti/pennylane-forest>`_: Supports integration
  with `PyQuil <https://github.com/rigetti/pyquil>`__, the
  `Rigetti Forest SDK <https://www.rigetti.com/forest>`__, and the
  `Rigetti QCS <https://www.rigetti.com/qcs>`__, an open-source quantum computation
  framework by Rigetti. Provides device support for the the Quantum Virtual Machine
  (QVM) and Quantum Processing Units (QPUs) hardware devices.


* `PennyLane-Qsharp <https://github.com/PennyLaneAI/pennylane-qsharp>`_: Supports integration
  with the `Microsoft Quantum Development Kit <https://www.microsoft.com/en-us/quantum/development-kit>`__,
  a quantum computation framework that uses the Q# quantum programming language.


For a full list of PennyLane plugins, see `the PennyLane website <https://pennylane.ai/plugins.html>`__.

Installation
============

PennyLane requires Python version 3.6 and above. Installation of PennyLane, as well
as all dependencies, can be done using pip:

.. code-block:: bash

    $ python -m pip install pennylane

Contributing to PennyLane
=========================

We welcome contributions — simply fork the PennyLane repository, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.
All contributers to PennyLane will be listed as authors on the releases. All users who contribute
significantly to the code (new plugins, new functionality, etc.) will be listed on the PennyLane arXiv paper.

We also encourage bug reports, suggestions for new features and enhancements, and even links to
cool projects or applications built on PennyLane.

See our `contributions page <https://github.com/PennyLaneAI/pennylane/blob/master/.github/CONTRIBUTING.md>`_
for more details.


Authors
=======

PennyLane is the work of `many contributors <https://github.com/PennyLaneAI/pennylane/graphs/contributors>`_.

If you are doing research using PennyLane, please cite `our paper <https://arxiv.org/abs/1811.04968>`_:

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, M. Sohaib Alam, Shahnawaz Ahmed,
    Juan Miguel Arrazola, Carsten Blank, Alain Delgado, Soran Jahangiri, Keri McKiernan, Johannes Jakob Meyer,
    Zeyue Niu, Antal Száva, and Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018. arXiv:1811.04968


Support
=======

- **Source Code:** https://github.com/PennyLaneAI/pennylane
- **Issue Tracker:** https://github.com/PennyLaneAI/pennylane/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.

We also have a `PennyLane discussion forum <https://discuss.pennylane.ai>`_ - come join
the discussion and chat with our PennyLane team.


License
=======

PennyLane is **free** and **open source**, released under the Apache License, Version 2.0.
