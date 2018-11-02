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

The `PennyLane Strawberry Fields Plugin <https://pennylane-sf.readthedocs.io>`_ allows the Strawberry Fields simulators to be used as PennyLane devices.

`Strawberry Fields <https://strawberryfields.readthedocs.io>`_ is a full-stack Python library for designing, simulating, and optimizing continuous variable (CV) quantum optical circuits.


Features
~~~~~~~~

* Provides two devices to be used with PennyLane: ``strawberryfields.fock`` and ``strawberryfields.gaussian``. These provide access to the Strawberry Fields Fock and Gaussian backends respectively.

* Supports all core PennyLane operations and expectation values across the two devices.

* Combine Strawberry Fields optimized simulator suite with PennyLane's automatic differentiation and optimization.


PennyLane ProjectQ Plugin
-------------------------

The `PennyLane ProjectQ Plugin <https://pennylane-pq.readthedocs.io>`_ allows to use both the software and hardware backends of ProjectQ as devices for PennyLane.

`ProjectQ <https://github.com/ProjectQ-Framework/ProjectQ>`_ is ProjectQ is an open-source compilation framework capable of targeting various types of hardware and a high-performance quantum computer simulator with emulation capabilities, and various compiler plug-ins.


Features
~~~~~~~~

* Provides three devices to be used with PennyLane: ``projectq.simulator``, ``projectq.ibm``, and ``projectq.classical``. These provide access to the respective ProjecQ backends.

* Supports a wide range of PennyLane operations and expectation values across the devices.

* Combine ProjectQ high performance simulator and hardware backend support with PennyLane's automatic differentiation and optimization.
