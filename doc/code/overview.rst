.. _library_overview:

Overview
========

.. currentmodule:: pennylane

The User Documentation is a reference for all methods
that are relevant for a **user** of PennyLane. It lists the following modules:

+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane <init>`       |  Contains the :func:`device` loader, as well as                                      |
|                               |  the :func:`grad` and :func:`jacobian` functions.                                    |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane._device       |  Defines the abstract :class:`Device` class,                                         |
| <device>`                     |  which plugins inherit from.                                                         |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.configuration |  Contains methods of how to save a PennyLane configuration,                          |
| <configuration>`              |  for example the number of shots for a device.                                       |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.decorator     |  Defines the quantum node decorator function,                                        |
| <decorator>`                  |  which turns quantum functions into QNodes.                                          |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.measure       |  Introduces different results to extract from a measurement,                         |
| <measure>`                    |  i.e. expectation values, variances and samples.                                     |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.operation     |  Defines the abstract :class:`Operation` class from                                  |
| <operation>`                  |  which all operations, such as gates and observables, are derived.                   |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.ops <ops>`    |  Contains specific operations for quantum circuits,                                  |
|                               |  such as gates and observables.                                                      |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.optimize      |  Contains the optimizers that can be used for basic,                                 |
| <optimize>`                   |  NumPy-compatible quantum nodes, such as Gradient Descent and Adagrad.               |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.plugins       |  Contains the default plugins `default.qubit` and `default.gaussian`.                |
| <plugins>`                    |                                                                                      |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.qnode <qnode>`|  Defines the :class:`QNode` class.                                                   |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.templates     |  Contains subroutines, like layers and input embeddings,                             |
| <templates>`                  |  that PennyLane creates for ease of use.                                             |
+-------------------------------+--------------------------------------------------------------------------------------+
| :ref:`pennylane.utils         |  Collection of utility functions.                                                    |
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

.. toctree::
    :maxdepth: 1
    :hidden:

    init
    device
    configuration
    decorator
    measure
    operation
    ops
    optimize
    plugins
    qnode
    templates
    utils
    variable


