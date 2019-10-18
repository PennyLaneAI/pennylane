 .. role:: html(raw)
   :format: html

.. _pl_intro:

Introduction
============

PennyLane integrates classical and quantum computations to
optimize variable parameters that the computations depend on. Prominent examples are
*variational quantum eigensolvers* or *quantum machine learning models*.

Bridging the classical and quantum world
----------------------------------------

The classical computations, as well as the overall optimization,
are executed by a **classical interface** . PennyLane's standard interface is `NumPy <https://numpy.org/>`_,
but there is also support for powerful machine learning interfaces like `PyTorch <https://pytorch.org/>`_
and `Tensorflow <https://www.tensorflow.org/>`_.

The quantum computations are sent to a **device** for execution. A device can be a classical
simulator or real quantum hardware. PennyLane comes with default simulator devices, but it can also use external
software and hardware to run quantum circuits - such as Xanadu's *StrawberryFields*,
Rigetti's *Forest*, IBM's *Quiskit*, *ProjectQ* or Microsoft's *Q#*.
The communication between PennyLane and external devices is coordinated by a **plugin**.


.. image:: ../_static/building_blocks.png
    :align: center
    :width: 650px
    :target: javascript:void(0);

The main job of PennyLane is to manage the computation or estimation of gradients
of parametrized quantum circuits (so called *variational circuits*) on quantum devices,
and to make them accessible to the classical interface.
The classical interface uses the gradient information to automatically differentiate
through the computation - an essential process in optimization and machine learning.

Learn more
----------

In the following sections you can learn more about quantum circuits, interfaces and plugins to external
quantum devices in PennyLane, as well as find its custom optimizers:

1. :ref:`Quantum circuits <intro_vcircuits>` shows you how PennyLane unifies and
simplifies the process of programming quantum circuits with trainable parameters.

2. :ref:`Interfaces <intro_interfaces>` introduces you to how PennyLane is used
with the different classical interfaces to optimize quantum circuits or hybrid computations.

3. :ref:`Quick reference <intro_reference>` lists the optimizers available in the NumPy interface, some
of which are specially designed for quantum optimization.




