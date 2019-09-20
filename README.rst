.. image:: doc/_static/pennylane_thin.png
    :alt: PennyLane

###################################

.. |CI| image:: https://img.shields.io/travis/com/XanaduAI/pennylane/master.svg?style=popout-square
    :alt: Travis
    :target: https://travis-ci.com/XanaduAI/pennylane/

.. |COV| image:: https://img.shields.io/codecov/c/github/xanaduai/pennylane/master.svg?style=popout-square
    :alt: Codecov coverage
    :target: https://codecov.io/gh/XanaduAI/pennylane

.. |PEP| image:: https://img.shields.io/codacy/grade/83940d926ef5444798a46378e528249d.svg?style=popout-square
    :alt: Codacy grade
    :target: https://app.codacy.com/app/XanaduAI/pennylane?utm_source=github.com&utm_medium=referral&utm_content=XanaduAI/pennylane&utm_campaign=badger

.. |DOC| image:: https://img.shields.io/readthedocs/pennylane.svg?style=popout-square
    :alt: Read the Docs
    :target: https://pennylane.readthedocs.io

.. |VERS| image:: https://img.shields.io/pypi/v/PennyLane.svg?style=popout-square
    :alt: PyPI
    :target: https://pypi.org/project/PennyLane

.. |PY| image:: https://img.shields.io/pypi/pyversions/PennyLane.svg?style=popout-square
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/PennyLane

.. |FORUM| image:: https://img.shields.io/discourse/https/discuss.pennylane.ai/posts.svg?style=popout-square
    :alt: Discourse posts
    :target: https://discuss.pennylane.ai
    
.. |LIC| image:: https://img.shields.io/pypi/l/PennyLane.svg?style=popout-square
    :alt: PyPI - License
    :target: https://www.apache.org/licenses/LICENSE-2.0

|CI|  |COV| |PEP| |DOC| |VERS| |PY| |FORUM|

`PennyLane <https://pennylane.readthedocs.io>`_ is a cross-platform Python library for quantum machine learning,
automatic differentiation, and optimization of hybrid quantum-classical computations.



Features
========


.. raw:: html

    <img src="https://raw.githubusercontent.com/XanaduAI/pennylane/master/doc/_static/code.png" width="300px"  align="right">


- **Follow the gradient**. Built-in **automatic differentiation** of quantum circuits

- **Best of both worlds**. Support for **hybrid quantum & classical** models

- **Batteries included**. Provides **optimization and machine learning** tools

- **Device independent**. The same quantum circuit model can be **run on different backends**

- **Large plugin ecosystem**. Install plugins to run your computational circuits on more devices, including Strawberry Fields, Rigetti Forest, ProjectQ, and Qiskit

- **Compatible with existing machine learning libraries**. Quantum circuits can interface with PyTorch, Tensorflow, or NumPy (via autograd), allowing hybrid CPU-GPU-QPU computations.


Available plugins
=================

* `PennyLane-SF <https://github.com/XanaduAI/pennylane-sf>`_: Supports integration with `Strawberry Fields <https://github.com/XanaduAI/strawberryfields>`__, a full-stack Python library for simulating continuous variable (CV) quantum optical circuits.


* `PennyLane-Forest <https://github.com/rigetti/pennylane-forest>`_: Supports integration with `PyQuil <https://github.com/rigetti/pyquil>`__, the `Rigetti Forest SDK <https://www.rigetti.com/forest>`__, and the `Rigetti QCS <https://www.rigetti.com/qcs>`__, an open-source quantum computation framework by Rigetti. Provides device support for the the Quantum Virtual Machine (QVM) and Quantum Processing Units (QPUs) hardware devices.


* `PennyLane-qiskit <https://github.com/carstenblank/pennylane-qiskit>`_: Supports integration with `Qiskit Terra <https://qiskit.org/terra>`__, an open-source quantum computation framework by IBM. Provides device support for the Qiskit Aer quantum simulators, and IBM QX hardware devices.


* `PennyLane-PQ <https://github.com/XanaduAI/pennylane-pq>`_: Supports integration with `ProjectQ <https://github.com/ProjectQ-Framework/ProjectQ>`__, an open-source quantum computation framework that supports the IBM quantum experience.


* `PennyLane-Qsharp <https://github.com/XanaduAI/pennylane-qsharp>`_: Supports integration with the `Microsoft Quantum Development Kit <https://www.microsoft.com/en-us/quantum/development-kit>`__, a quantum computation framework that uses the Q# quantum programming language.


Installation
============

PennyLane requires Python version 3.5 and above. Installation of PennyLane, as well as all dependencies, can be done using pip:

.. code-block:: bash

    $ python -m pip install pennylane


Getting started
===============

For getting started with PennyLane, check out our `qubit rotation <https://pennylane.readthedocs.io/en/latest/tutorials/qubit_rotation.html>`_, `Gaussian transformation <https://pennylane.readthedocs.io/en/latest/tutorials/gaussian_transformation.html>`_, `hybrid computation <https://pennylane.readthedocs.io/en/latest/tutorials/hybrid_computation.html>`_, and other machine learning tutorials.

Our `documentation <https://pennylane.readthedocs.io>`_ is also a great starting point to familiarize yourself with the hybrid classical-quantum machine learning approach, and explore the available optimization tools provided by PennyLane. Play around with the numerous devices and plugins available for running your hybrid optimizations — these include the `IBM QX4 quantum chip <https://quantumexperience.ng.bluemix.net/qx/experience>`__, provided by the `PennyLane-PQ <https://github.com/XanaduAI/pennylane-pq>`_ and `PennyLane-qiskit <https://github.com/carstenblank/pennylane-qiskit>`_ plugins, as well as the `Rigetti Aspen-1 QPU <https://www.rigetti.com/qpu>`__.

Finally, detailed documentation on the PennyLane API is provided, for full details on available quantum operations and expectations, and detailed guides on `how to write your own <https://pennylane.readthedocs.io/en/latest/API/overview.html>`_ PennyLane-compatible quantum device.


Contributing to PennyLane
=================================

We welcome contributions — simply fork the PennyLane repository, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.  All contributers to PennyLane will be listed as authors on the releases. All users who contribute significantly to the code (new plugins, new functionality, etc.) will be listed on the PennyLane arXiv paper.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects or applications built on PennyLane.

See our `contributions page <https://github.com/XanaduAI/pennylane/blob/master/.github/CONTRIBUTING.md>`_
for more details.


Authors
=======

Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, and Nathan Killoran.

If you are doing research using PennyLane, please cite `our paper <https://arxiv.org/abs/1811.04968>`_:

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, Carsten Blank, Keri McKiernan, and Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018. arXiv:1811.04968


Support
=======

- **Source Code:** https://github.com/XanaduAI/pennylane
- **Issue Tracker:** https://github.com/XanaduAI/pennylane/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.

We also have a `PennyLane discussion forum <https://discuss.pennylane.ai>`_ - come join the discussion and chat with our PennyLane team.


License
=======

PennyLane is **free** and **open source**, released under the Apache License, Version 2.0.
