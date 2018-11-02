.. _plugins:

Plugins and ecosystem
=====================

PennyLane comes with two reference plugins that are directly included in core PennyLane:

.. autosummary::
   pennylane.plugins.default_qubit
   pennylane.plugins.default_gaussian

In addition to that, the following plugins open up the possibility to use the features of PennyLane with other simulator and quantum hardware devices:


PennyLane Strawberry Fields Plugin
----------------------------------

`Strawberry Fields <https://strawberryfields.readthedocs.io>`_ is a full-stack Python library for designing, simulating, and optimizing continuous variable (CV) quantum optical circuits.

The `PennyLane Strawberry Fields Plugin <https://pennylane-sf.readthedocs.io>`_ allows Strawberry Fields simulators to be used as PennyLane devices.

Features
~~~~~~~~

* Provides two devices to be used with PennyLane: ``strawberryfields.fock`` and ``strawberryfields.gaussian``. These provide access to the Strawberry Fields Fock and Gaussian backends, respectively.

* Supports all core PennyLane CV operations and expectation values.

* Combines Strawberry Fields' optimized simulator suite with PennyLane's automatic differentiation and optimization.


PennyLane ProjectQ Plugin
-------------------------

`ProjectQ <https://github.com/ProjectQ-Framework/ProjectQ>`_ is an open-source quantum compilation framework. ProjectQ is capable of targeting various types of quantum hardware by decomposing quantum cirquits in terms of the available gate set (compiling) and has a built-in high-performance quantum computer simulator.

The `PennyLane ProjectQ Plugin <https://pennylane-pq.readthedocs.io>`_ allows both the software and hardware backends of ProjectQ to be used as devices for PennyLane.

Features
~~~~~~~~

* Provides three devices to be used with PennyLane: ``projectq.simulator``, ``projectq.ibm``, and ``projectq.classical``. These provide access to the respective ProjectQ backends.

* Supports a wide range of PennyLane qubit operations and expectation values across the devices.

* Combines ProjectQ's high-performance simulator and hardware backend support with PennyLane's automatic differentiation and optimization.
