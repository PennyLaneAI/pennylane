 .. role:: html(raw)
   :format: html

.. _pl_intro:

Introduction
============

PennyLane can be understood as a framework that integrates classical and quantum computations (or *nodes*) for the
purpose of automatic differentiation.

The classical computations, as well as the overall optimization,
are executed by a **classical interface** .

The quantum computations, or *quantum variational circuits*, are sent to one of a growing number
of devices for execution; a device can be a classical
simulator or a real quantum device. The communication between the device and PennyLane is
defined by a **plugin**.


.. image:: _static/building_blocks.png
    :align: center
    :width: 650px
    :target: javascript:void(0);


Being a mediator between classical and quantum computing frameworks,
the main job of PennyLane is threefold:

1. PennyLane unifies and simplifies the construction of variational quantum circuits
for hybrid optimization tasks (see also :ref:`Variational Circuits <vcircuits>`).

2. PennyLane sends the circuits via plugins to the desired devices
for execution (see also :ref:`Plugins <plugins>`).

3. PennyLane manages the computation or estimation of gradients
of the variational circuits, and makes them accessible for the classical interface
(see also :ref:`Interfaces <intro_interfaces>`).








