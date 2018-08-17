.. role:: html(raw)
   :format: html

.. _quantum_nodes:

Quantum nodes
================

A quantum node is a computational encapsulation of a quantum function :math:`f(x;\bm{\theta})`. It takes in classical information (the input :math:`x` and the parameters :math:`\bm{\theta}`) and outputs classical information (the expectation value :math:`\langle \hat{B} \rangle`, whose value equals :math:`f(x;\bm{\theta}`)). Quantum computing hardware or simulators are used to evaluate the qfunc associated with a quantum node.

From the perspective a classical computer, a quantum node is just a callable function which maps floating point numbers to floating point numbers.





