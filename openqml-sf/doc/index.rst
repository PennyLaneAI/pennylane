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

   introduction
   qfuncs
   autograd_quantum
   quantum_nodes
   hybrid_computation
   conventions
   references


.. toctree::
   :maxdepth: 1
   :caption: API
   :hidden:

   core
   circuit


.. toctree::
   :maxdepth: 1
   :caption: Plugins
   :hidden:

   plugins
   plugins/included_plugins


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
