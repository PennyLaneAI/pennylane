PennyLane
#########

:Release: |release|
:Date: |today|

PennyLane is a Python library for building and training machine learning models which include quantum computer circuits.

Features
========

- *Follow the gradient*.
  Built-in **automatic differentiation** of quantum circuits

- *Best of both worlds*.
  Support for **hybrid quantum and classical** models

- *Batteries included*.
  Provides **optimization and machine learning** tools

- *Device independent*.
  The same quantum circuit model can be **run on different backends**

- *Large plugin ecosystem*.
  Install plugins to run your computational circuits on more devices, including `Strawberry Fields <https://github.com/XanaduAI/pennylane-sf>`_ and `ProjectQ <https://github.com/XanaduAI/pennylane-pq>`_

Getting started
===============

To get PennyLane installed and running on your system, begin at the :ref:`download and installation guide <installation>`. Then, familiarize yourself with the PennyLane's :ref:`key concepts <introduction>` for machine learning on quantum circuits.

For getting started with PennyLane, check out our basic :ref:`qubit rotation <qubit_rotation>`, :ref:`Gaussian transformation <gaussian_transformation>`, and :ref:`hybrid quantum optimization <plugins_hybrid>` tutorials. More advanced tutorials include supervised learning, quantum classifiers, and quantum generative adversarial networks (QGANs).

Next, play around with the numerous devices and :ref:`plugins <plugins>` available for running your hybrid models — these include Strawberry Fields, provided by the `PennyLane-SF <https://github.com/XanaduAI/pennylane-pq>`_ plugin, and the IBM QX4 quantum chip, provided by the `PennyLane-PQ <https://github.com/XanaduAI/pennylane-pq>`_ plugin.

Finally, detailed documentation on the PennyLane :ref:`interface <library_overview>` and API is provided. Look there for full details on available quantum operations and expectations, and detailed guides on :ref:`how to write your own <developer_overview>` PennyLane compatible quantum device.

How to cite
===========

If you are doing research using PennyLane, please cite

  [Placeholder for PennyLane paper]

Support
=======

- **Source Code:** https://github.com/XanaduAI/PennyLane
- **Issue Tracker:** https://github.com/XanaduAI/PennyLane/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.

For more details on contributing or performing research with PennyLane, please see
:ref:`research`.

License
=======

PennyLane is **free** and **open source**, released under the Apache License, Version 2.0.


.. toctree::
   :maxdepth: 1
   :caption: Getting started
   :hidden:

   installing
   plugins
   research

.. toctree::
   :maxdepth: 1
   :caption: Key concepts
   :hidden:

   concepts/introduction
   concepts/qfuncs
   concepts/autograd_quantum
   concepts/quantum_nodes
   concepts/hybrid_computation
   references


.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :hidden:

   tutorials/qubit_rotation
   tutorials/gaussian_transformation
   tutorials/plugins_hybrid
   tutorials/advanced_features


.. toctree::
   :maxdepth: 1
   :caption: Library details
   :hidden:

   code/init
   code/qnode
   code/decorator
   code/optimize
   code/configuration
   code/utils

.. toctree::
   :maxdepth: 1
   :caption: Supported operations
   :hidden:

   code/ops
   code/expval

.. toctree::
   :maxdepth: 1
   :caption: Developer API
   :hidden:

   API/overview
   API/device
   API/operation
   API/variable

.. toctree::
   :maxdepth: 1
   :caption: Reference plugins
   :hidden:

   plugins/default_qubit
   plugins/default_gaussian


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
