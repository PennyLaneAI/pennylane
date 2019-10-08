 .. role:: html(raw)
   :format: html

.. _pl_intro:

Introduction
============

PennyLane can be understood as a mediator that integrates classical and quantum computations for the
purpose of hybrid optimization.

Bridging the classical and quantum world
----------------------------------------

The classical computations, as well as the overall optimization,
are executed by a **classical interface** . PennyLane's standard interface is :ref:`NumPy <https://numpy.org/>`,
but there is also support for powerful machine learning interfaces like :ref:`PyTorch <https://pytorch.org/>`
and :ref:`Tensorflow <https://www.tensorflow.org/>`.

The quantum computations are sent to a **device** for execution. A device can be a classical
simulator or real quantum hardware. PennyLane comes with default devices, but it can also use external
soft- and hardware to run quantum circuits - such as Xanadu's *StrawberryFields*, Rigetti's *Forest*, IBM's *Quiskit*,
*ProjectQ* or Microsoft's *Q#*.
The communication between PennyLane and external devices is coordinated by a **plugin**.


.. image:: _static/building_blocks.png
    :align: center
    :width: 650px
    :target: javascript:void(0);

The main job of PennyLane is to manage the computation or estimation of gradients
of adaptable quantum circuits, so called *variational circuits*, on quantum devices,
and to make them accessible for the classical interface. The classical interface uses the gradient
information to automatically differentiate through the computation - an essential process in optimization
and machine learning.

Learn more
----------

In the following you can learn more about quantum circuits, interfaces and plugins to external
quantum devices in PennyLane:

1. The section on :ref:`Variational Circuits <intro_vcircuits>` shows you how PennyLane unifies and
simplifies the process of programming quantum circuits with trainable parameters.

2. The section on :ref:`Interfaces <intro_interfaces>` introduces you to how PennyLane is used
with the different classical interfaces for hybrid optimization.

3. The section on :ref:`Plugins <intro_plugins>` gives you an overview of PennyLane's plugin ecosystem,
and teaches you how to write a new plugin for a quantum device.






