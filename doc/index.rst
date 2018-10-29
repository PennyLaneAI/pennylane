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
   tutorials/photon_redirection

.. 
   tutorials/photon_redirection.ipynb
   tutorials/quantum_neural_net.ipynb
   tutorials/qubit_rotation.ipynb
   tutorials/variational_quantum_eigensolver.ipynb
   tutorials/variational_classifier.ipynb

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
