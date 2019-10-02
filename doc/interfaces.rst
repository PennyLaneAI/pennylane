.. _intro_interfaces:

Interfaces
==========

PennyLane integrates quantum nodes made up of variational circuits with other programming
and machine learning frameworks.
Such frameworks are called *interfaces*. The default interface, implicitly used in the
:ref:`Introduction <pl_intro>`, is NumPy.

Currently, there is support for the following interfaces:

.. toctree::
   :maxdepth: 1

   interfaces/numpy
   interfaces/torch
   interfaces/tfe
