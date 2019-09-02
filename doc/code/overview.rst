.. _library_overview:

Overview
========

.. currentmodule:: pennylane

The User Documentation is a reference for all methods
that are relevant for a **user** of PennyLane. It lists the following modules:


* :ref:`pennylane <init>`

  Contains the :func:`device` loader, as well as the :func:`grad` and :func:`jacobian` functions.

* :ref:`pennylane.qnode <qnode>`

  Contains the central :class:`QNode` class.

* :ref:`pennylane.ops <ops>`

  Lists all operations allowed in a quantum circuit, such as gates and observables.

* :ref:`pennylane.measure <measure>`

  Introduces the possible results to get from a measurement, i.e. expectation values,
  variances and samples.

* :ref:`pennylane.templates <templates>`

  Available subroutines, like layers and input embeddings, that PennyLane
  creates for ease of use.

* :ref:`pennylane.optimize <optimize>`

  Lists the optimizers that can be used for basic, NumPy-compatible quantum nodes, such as
  Gradient Descent and Adagrad.

* :ref:`pennylane.configuration <configuration>`

  Shows how you can save your PennyLane configuration, for example the number
  of shots for a device.

.. note::

    If you are interested in developing PennyLane or if you want to know more about the
    code, visit the :ref:`Developer API <_developer_overview>`.
    If you want to write a plugin that connects PennyLane to hardware
    and simulator backends, visit the :ref:`Plugin API <_plugin_overview>`.

For a summary of *all* modules, refer to the :ref:`modindex`, and all methods
can be found in the :ref:`genindex`.

