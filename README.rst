PennyLane
#########

.. image:: https://img.shields.io/travis/XanaduAI/pennylane/master.svg?style=for-the-badge
    :alt: Travis
    :target: https://travis-ci.org/XanaduAI/pennylane

.. image:: https://img.shields.io/codecov/c/github/xanaduai/pennylane/master.svg?style=for-the-badge
    :alt: Codecov coverage
    :target: https://codecov.io/gh/XanaduAI/pennylane

.. image:: https://img.shields.io/codacy/grade/bd14437d17494f16ada064d8026498dd.svg?style=for-the-badge
    :alt: Codacy grade
    :target: https://app.codacy.com/app/XanaduAI/pennylane?utm_source=github.com&utm_medium=referral&utm_content=XanaduAI/pennylane&utm_campaign=badger

.. image:: https://img.shields.io/readthedocs/pennylane.svg?style=for-the-badge
    :alt: Read the Docs
    :target: https://pennylane.readthedocs.io

.. image:: https://img.shields.io/pypi/v/PennyLane.svg?style=for-the-badge
    :alt: PyPI
    :target: https://pypi.org/project/PennyLane

.. image:: https://img.shields.io/pypi/pyversions/PennyLane.svg?style=for-the-badge
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/PennyLane

`PennyLane <https://pennylane.readthedocs.io>`_ is a full-stack Python library for quantum machine
learning and automatic differentiation of hybrid quantum-classical computations.



Features
========


.. raw:: html

    <img src="https://i.imgur.com/SqlTUZ1.png" width="300px"  align="right">


- **Follow the gradient**. Built-in **automatic differentiation** of quantum circuits

- **Best of both worlds**. Support for **hybrid quantum & classical** models

- **Batteries included**. Provides **optimization and machine learning** tools

- **Device independent**. The same quantum circuit model can be **run on different backends**

- **Large plugin ecosystem**. Install plugins to run your computational circuits on more devices, including Strawberry Fields and ProjectQ


Available plugins
=================

* `PennyLane-SF <https://github.com/XanaduAI/pennylane-sf>`_: Supports integration with `Strawberry Fields <https://github.com/XanaduAI/strawberryfields>`__, a full-stack Python library for simulating continuous variable (CV) quantum optical circuits.


* `PennyLane-PQ <https://github.com/XanaduAI/pennylane-pq>`_: Supports integration with `ProjectQ <https://github.com/ProjectQ-Framework/ProjectQ>`__, an open-source quantum computation framework that supports the IBM quantum experience.

Installation
============

PennyLane requires Python version 3.5 and above. Installation of PennyLane, as well as all dependencies, can be done using pip:

.. code-block:: bash

    $ python -m pip install pennylane


Getting started
===============

For getting started with PennyLane, check out our `qubit rotation <https://pennylane.readthedocs.io/en/latest/tutorials/qubit_rotation.html>`_, `Gaussian transformation <https://pennylane.readthedocs.io/en/latest/tutorials/gaussian_transformation.html>`_, `hybrid computation <https://pennylane.readthedocs.io/en/latest/tutorials/hybrid_computation.html>`_, and other machine learning tutorials.

Our `documentation <https://pennylane.readthedocs.io>`_ is also a great starting point to familiarize yourself with the hybrid classical-quantum machine learning approach, and explore the available optimization tools provided by PennyLane. Play around with the numerous devices and plugins available for running your hybrid optimizations — these include the IBM QX4 quantum chip, provided by the `PennyLane-PQ <https://github.com/XanaduAI/pennylane-pq>`_ plugin.

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

If you are doing research using PennyLane, please cite `our paper <https://arxiv.org/abs/XXXX.XXXXX>`_:

  [Placeholder for paper link]


Support
=======

- **Source Code:** https://github.com/XanaduAI/pennylane
- **Issue Tracker:** https://github.com/XanaduAI/pennylane/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.


License
=======

PennyLane is **free** and **open source**, released under the Apache License, Version 2.0.
