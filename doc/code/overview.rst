.. _library_overview:

Overview
========

.. currentmodule:: pennylane

The code documentation serves as a reference where all classes and functions of
PennyLane can be looked up. This is useful in order to check the signatures of inputs
outputs, as well as the different options these functions provide.

Functions and classes are organised in **modules**. To import a class or functions, you need
to specify the full module path, i.e.,

.. code::

    from pennylane.templates.layers import StronglyEntanglingLayer

Some modules are relevant to users, while others are providing more lower-level functionality.


User-facing modules
-------------------

The following modules are relevant for a typical **user** of PennyLane:

+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane <init>`       |  Contains the :func:`device` loader, as well as                                      |
|                               |  the :func:`grad` and :func:`jacobian` functions.                                    |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.configuration |  Contains methods of how to save a PennyLane configuration,                          |
| <configuration>`              |  for example the number of shots for a device.                                       |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.decorator     |  Defines the ``qnode()`` decorator function,                                         |
| <decorator>`                  |  which turns quantum functions into QNodes.                                          |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.measure       |  Introduces different results to extract from a measurement,                         |
| <measure>`                    |  i.e. :func:`expval`, :func:`var` and :func:`samples`.                               |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.ops <ops>`    |  Contains specific operations for quantum circuits,                                  |
|                               |  such as gates and observables.                                                      |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.optimize      |  Contains the optimizers that can be used for basic,                                 |
| <optimize>`                   |  NumPy-compatible quantum nodes, such as :func:`GradientDescentOptimizer`.           |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.qnode <qnode>`|  Defines the :class:`QNode` class.                                                   |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.templates     |  Contains the :mod:`templates.layers` and :mod:`templates.embeddings`                |
| <templates>`                  |  submodules, which contain pre-composed pieces of a quantum circuit.                 |
+-------------------------------+--------------------------------------------------------------------------------------+

Other modules
-------------

The following modules are contain lower-level functionality, which is relevant for developers:

+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.about         |  Print basic information about PennyLane.                                            |
| <about>`                      |                                                                                      |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.init          |  Contains parameter initialization functions                                         |
| <par_init>`                   |                                                                                      |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane._device       |  Defines the abstract :class:`Device` class,                                         |
| <device>`                     |  which plugins inherit from.                                                         |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.interfaces    |  Contains the quantum nodes for different interfaces.                                |
| <interfaces>`                 |                                                                                      |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.operation     |  Defines the abstract :class:`Operation` class from                                  |
| <operation>`                  |  which all operations, such as gates and observables, are derived.                   |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.plugins       |  Contains the default plugins ``default.qubit`` and ``default.gaussian``.            |
| <plugins>`                    |                                                                                      |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.utils         |  Contains a collection of utility functions.                                         |
| <utils>`                      |                                                                                      |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.variable      |  Defines the :class:`Variable` class for tunable                                     |
| <variable>`                   |  parameters.                                                                         |
+-------------------------------+--------------------------------------------------------------------------------------+


.. note::

    If you want to write a plugin that connects PennyLane to hardware
    and simulator backends, visit the :ref:`Plugin API <plugin_overview>`.

For a summary of *all* modules, refer to the :ref:`modindex`, and all methods
can be found in the :ref:`genindex`.

