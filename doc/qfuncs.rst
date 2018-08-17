.. role:: html(raw)
   :format: html

.. _qfuncs:

Quantum functions
================


Classical and quantum information
---------------------------------


Quantum functions
-----------------

A quantum function (*qfunc*) is any parameterized function :math:`f(x;\bm{\theta})` which can be evaluated on a quantum circuit via the Born rule:

.. math:: f(x; \bm{\theta}) = \langle \hat{B} \rangle = \langle 0 | U^\dagger(x;\bm{\theta})\hat{B}U(x;\bm{\theta}) | 0 \rangle.

Here, :math:`\hat{B}` is an observable measured at the circuit output and :math:`| 0 \rangle` is a fixed initial state (e.g., the vacuum or a spin-down state). Both the input :math:`x` and the parameters :math:`\bm{\theta}` enter the quantum circuit as arguments used in the gates which are used to build the unitary :math:`U(x;\bm{\theta})`. For convenience, we can also write the unitary conjugation as a transformation :math:`\mathcal{C}_U` acting on the operator :math:`\hat{B}`:

.. math:: U^\dagger(x;\bm{\theta})\hat{B}U(x;\bm{\theta}) = \mathcal{C}_U(\hat{B}).

Note that the measurement operator :math:`\hat{B}` has no dependence on the the input :math:`x` or the parameters :math:`\bm{\theta}`.




