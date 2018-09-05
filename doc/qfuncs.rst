.. role:: html(raw)
   :format: html

.. _qfuncs:

Quantum functions
================

A quantum function (*qfunc*) is any parameterized function :math:`f(x;\bm{\theta})` which can be evaluated on a quantum circuit using the `Born rule <https://en.wikipedia.org/wiki/Born_rule>`_.

.. math:: f(x; \bm{\theta}) = \langle \hat{B} \rangle = \langle 0 | U^\dagger(x;\bm{\theta})\hat{B}U(x;\bm{\theta}) | 0 \rangle.

.. note:: Here, :math:`\hat{B}` is some observable measured at the circuit output and :math:`| 0 \rangle` is a fixed initial state (e.g., the vacuum or a spin-down state). 

:html:`<br>`

.. figure:: ./_static/quantum_function.svg
    :align: center
    :width: 70%
    :target: javascript:void(0);

    A quantum function is a function which is evaluated by measurements of a programmable quantum computer circuit.

:html:`<br>`

Both the input :math:`x` and the parameters :math:`\bm{\theta}` influence the quantum circuit in the same way: as arguments for the gates which are used to build the unitary :math:`U(x;\bm{\theta})`. 
The measurement operator :math:`\hat{B}` has no dependence on the the input :math:`x` nor the parameters :math:`\bm{\theta}`.

.. todo:: add more discussion here: i) how gate arguments are used to input data (give example with displacement gate?), ii) how is the process same/different for inputs vs parameters? iii) continuous and discrete functions (CV and qubit)
