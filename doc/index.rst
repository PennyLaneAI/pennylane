OpenQML
#######

:Release: |release|
:Date: |today|

OpenQML is a Python library for building and training machine learning models which include quantum computer circuits.

Main features of OpenQML:

- *Follow the gradient*: **automatic differentiation** of quantum circuits
- *Best of both worlds*: support for **hybrid quantum & classical** models
- *Batteries included*: built-in **optimization and machine learning** tools
- *Device independent*: the same quantum circuit model can be **run on different backends**

Getting started
===============


How to cite
===========

If you are doing research using OpenQML, please cite

Support
=======

- **Source Code:** https://github.com/XanaduAI/openQML
- **Issue Tracker:** https://github.com/XanaduAI/openQML/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.

For more details on contributing or performing research with OpenQML, please see
:ref:`research`.

License
=======

OpenQML is **free** and **open source**, released under the Apache License, Version 2.0.


.. toctree::
   :maxdepth: 1
   :caption: Getting started
   :hidden:

   installing
   research

.. toctree::
   :titlesonly:
   :caption: Key concepts
   :hidden:

   concepts/introduction
   concepts/qfuncs
   concepts/autograd_quantum
   concepts/quantum_nodes
   concepts/hybrid_computation
   concepts/conventions
   references


.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   code
   API/configuration
   API/optimize
   API/qnode
   API/qfunc.rst
   API/operation
   API/variable

.. toctree::
   :maxdepth: 1
   :caption: Supported operations
   :hidden:

   API/ops
   API/expectation

.. toctree::
   :maxdepth: 1
   :caption: Plugin API
   :hidden:

   plugins/device
   plugins/default


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
