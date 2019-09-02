.. _library_overview:

Overview
========

.. currentmodule:: pennylane

The User Documentation is a reference for all methods
that are relevant for a **user** of PennyLane. It lists the following modules:

+----------------------------------+-----------------------------------------------------------------------------------------+
| :ref:`pennylane <init>`          |  Contains the :func:`device` loader, as well as                                         |
|                                  |  the :func:`grad` and :func:`jacobian` functions.                                       |
+----------------------------------+-----------------------------------------------------------------------------------------+
| :ref:`pennylane.configuration    |  Shows how you can save your PennyLane configuration,                                   |
| <configuration>`                 |  for example the number of shots for a device.                                          |
+----------------------------------+-----------------------------------------------------------------------------------------+
| :ref:`pennylane.measure          |  Introduces the possible results to get from a measurement,                             |
| <measure>`                       |  i.e. expectation values, variances and samples.                                        |
+----------------------------------+-----------------------------------------------------------------------------------------+
| :ref:`pennylane.ops <ops>`       |  Lists all operations allowed in a quantum circuit,                                     |
|                                  |  such as gates and observables.                                                         |
+----------------------------------+-----------------------------------------------------------------------------------------+
| :ref:`pennylane.optimize         |  Lists the optimizers that can be used for basic,                                       |
| <optimize>`                      |  NumPy-compatible quantum nodes, such as Gradient Descent and Adagrad.                  |
+----------------------------------+-----------------------------------------------------------------------------------------+
| :ref:`pennylane.qnode <qnode>`   |  Contains the central :class:`QNode` class.                                             |
+----------------------------------+-----------------------------------------------------------------------------------------+
| :ref:`pennylane.templates        |  Available subroutines, like layers and input embeddings,                               |
| <templates>`                     |  that PennyLane creates for ease of use.                                                |
+----------------------------------+-----------------------------------------------------------------------------------------+

.. note::

    If you want to write a plugin that connects PennyLane to hardware
    and simulator backends, visit the :ref:`Plugin API <plugin_overview>`.

For a summary of *all* modules, refer to the :ref:`modindex`, and all methods
can be found in the :ref:`genindex`.

