.. role:: html(raw)
   :format: html

.. _intro_interfaces:

Classical interfaces
====================

PennyLane integrates quantum nodes made up of variational circuits with other programming
and machine learning frameworks.
Such frameworks are called *interfaces*. The default interface, implicitly used in the previous section on
how to program :ref:`quantum circuits <intro_vcircuits>`, is NumPy.

Currently, there is support for the following three interfaces:

:html:`<br>`

.. image:: ../_static/numpy.jpeg
    :width: 250px
    :target: interfaces/numpy.html

.. image:: ../_static/pytorch.png
    :width: 300px
    :target: interfaces/torch.html

.. image:: ../_static/tensorflow.png
    :width: 100px
    :target: interfaces/tfe.html


.. toctree::
    :hidden:

    interfaces/numpy
    interfaces/torch
    interfaces/tfe




