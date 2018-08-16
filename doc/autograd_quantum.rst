.. role:: html(raw)
   :format: html

.. _autograd_quantum:

Automatic differentiation of quantum functions
============

Quantum gradients
-----------------

In many modern machine learning applications, the ability to automatically compute analytic gradients has shown tremendous practical value. Can we have this same built-in functionality for quantum functions? Yes!

Since qfuncs may be intractable to compute on classical computers, we might expect that the gradients of qfs to be similarly complex. Fortunately, for a given qfunc :math:`f(x;\bm{\theta})`, we can often write the gradient :math:`\nabla_{\bm{\theta}}f(x;\bm{\theta})` as a simple sum of qfuncs, but with shifted parameters: 

.. .. math:: \nabla_{\bm{\theta}}f(x; \bm{\theta}) = \sum_k c_k f(x; \bm{\theta}_k)

:html:`<br>`

.. figure:: ./_static/quantum_gradient.svg
    :align: center
    :width: 70%
    :target: javascript:void(0);

    The same computing device can often be used to evaluate both qfuncs and gradients of qfuncs.

:html:`<br>`

In other words, we can use the same quantum computation device to compute quantum functions and also **gradients of quantum functions**. This is accomplished with minor assistance of a classical coprocessor, which performs the summation.

.. note:: In situations where no formula for quantum gradients is known, OpenQML supports approximate gradient estimation using numerical methods.
