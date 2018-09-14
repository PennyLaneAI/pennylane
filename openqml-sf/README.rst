.. image:: doc/_static/strawberry-fields-text.png
    :alt: Strawberry Fields

##################################################

.. image:: https://img.shields.io/travis/XanaduAI/strawberryfields/master.svg?style=for-the-badge
    :alt: Travis
    :target: https://travis-ci.org/XanaduAI/strawberryfields

.. image:: https://img.shields.io/codecov/c/github/xanaduai/strawberryfields/master.svg?style=for-the-badge
    :alt: Codecov coverage
    :target: https://codecov.io/gh/XanaduAI/strawberryfields

.. image:: https://img.shields.io/codacy/grade/bd14437d17494f16ada064d8026498dd.svg?style=for-the-badge
    :alt: Codacy grade
    :target: https://app.codacy.com/app/XanaduAI/strawberryfields?utm_source=github.com&utm_medium=referral&utm_content=XanaduAI/strawberryfields&utm_campaign=badger

.. image:: https://img.shields.io/readthedocs/strawberryfields.svg?style=for-the-badge
    :alt: Read the Docs
    :target: https://strawberryfields.readthedocs.io

.. image:: https://img.shields.io/pypi/v/StrawberryFields.svg?style=for-the-badge
    :alt: PyPI
    :target: https://pypi.org/project/StrawberryFields

.. image:: https://img.shields.io/pypi/pyversions/StrawberryFields.svg?style=for-the-badge
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/StrawberryFields

`Strawberry Fields <https://strawberryfields.readthedocs.io>`_ is a full-stack Python library for designing,
simulating, and optimizing continuous variable quantum
optical circuits.


Features
========

* An open-source software architecture for **photonic quantum computing**

* A **full-stack** quantum software platform, implemented in Python specifically targeted to the CV model

* Quantum circuits are written using the easy-to-use and intuitive **Blackbird quantum programming language**

* Includes a suite of CV **quantum computer simulators** implemented using **NumPy** and **TensorFlow** - these built-in quantum compiler tools convert and optimize Blackbird code for classical simulation

* Future releases will aim to target experimental backends, including **photonic quantum computing chips**


Installation
============

Strawberry Fields requires Python version 3.5 and above. Installation of Strawberry Fields, as well as all dependencies, can be done using pip:

.. code-block:: bash

    $ python -m pip install strawberryfields


If you are using the ``tensorflow-gpu`` module for TensorFlow GPU support, you can install the following package for GPU support in Strawberry Fields:

.. code-block:: bash

    $ python -m pip install strawberryfields-gpu


Getting started
===============

To see Strawberry Fields in action immediately, try out our `Strawberry Fields Interactive <https://strawberryfields.ai>`_ web application. Prepare your initial states, drag and drop gates, and watch your simulation run in real time right in your web browser.

For getting started with writing your own Strawberry Fields code, check out our `quantum teleportation <https://strawberryfields.readthedocs.io/en/latest/tutorials/tutorial_teleportation.html>`_, `boson sampling <https://strawberryfields.readthedocs.io/en/latest/tutorials/tutorial_boson_sampling.html>`_, and `machine learning <https://strawberryfields.readthedocs.io/en/latest/tutorials/tutorial_machine_learning.html>`_ tutorials.

Our documentation is also a great starting point to familiarize yourself with the framework of `continuous-variable quantum computation <https://strawberryfields.readthedocs.io/en/latest/introduction.html>`_, and check out some important and interesting continuous-variable `quantum algorithms <https://strawberryfields.readthedocs.io/en/latest/quantum_algorithms.html>`_.

Finally, detailed documentation on the `Strawberry fields API <https://strawberryfields.readthedocs.io/en/latest/code/code.html>`_ is provided, for full details on available quantum operations, arguments, and backends.


Contributing to Strawberry Fields
=================================

We welcome contributions - simply fork the Strawberry Fields repository, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`_ containing your contribution.  All contributers to Strawberry Fields will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects or applications built on Strawberry Fields. If your contribution becomes part of Strawberry Fields, or is highlighted in our Gallery, we will send you some exclusive Xanadu Swag™ - including t-shirts, stickers, and more.

.. raw:: html

    <img src="https://i.imgur.com/xSFMt3g.jpg" width="300px"  align="left"> <img src="https://i.imgur.com/dC0U1xG.jpg" width="300px"  align="left">

See our `contributions page <https://github.com/XanaduAI/strawberryfields/blob/master/.github/CONTRIBUTING.md>`_
for more details, and then check out some of the Strawberry Fields `challenges <https://github.com/XanaduAI/strawberryfields/blob/master/.github/CHALLENGES.md>`_ for some inspiration.

|

Authors
=======

Nathan Killoran, Josh Izaac, Nicolás Quesada, Ville Bergholm, Matthew Amy, and Christian Weedbrook.

If you are doing research using Strawberry Fields, please cite `our whitepaper <https://arxiv.org/abs/1804.03159>`_:

  Nathan Killoran, Josh Izaac, Nicolás Quesada, Ville Bergholm, Matthew Amy, and Christian Weedbrook. Strawberry Fields: A Software Platform for Photonic Quantum Computing. *arXiv*, 2018. arXiv:1804.03159


Support
=======

- **Source Code:** https://github.com/XanaduAI/strawberryfields
- **Issue Tracker:** https://github.com/XanaduAI/strawberryfields/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.

We also have a `Strawberry Fields Slack channel <https://u.strawberryfields.ai/slack>`_ -
come join the discussion and chat with our Strawberry Fields team.


License
=======

Strawberry Fields is **free** and **open source**, released under the Apache License, Version 2.0.
