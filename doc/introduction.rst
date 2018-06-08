.. role:: html(raw)
   :format: html

.. _introduction:

Introduction
============

OpenQML is a cross-platform library for building and training machine learning models which use quantum circuits.

Key features of OpenQML:

- *Follow the gradient*: **automatic differentiation** of quantum circuits
- *Device independent*: the same quantum circuit model can be **run on different hardware**
- *Best of both worlds*: support for **hybrid quantum & classical** models
- *Batteries included*: built-in **optimization and machine learning** tools

Quantum circuits
--------------------

The key building block in OpenQML is the *variational quantum circuit*. 

These quantum circuits are made up of quantum gates, some (or all) of which are parameterizable. The user specifies the circuit, fixing the gates and the order which they appear, but leaves the gate parameters :math:`\theta_i` unfixed. 


:html:`<br>`

.. figure:: ./_static/var_circuit.svg
    :align: center
    :width: 70%
    :target: javascript:void(0);
    
    An simple example variational circuit built with the parameterizable gates :math:`\{A,B,C\}`.

:html:`<br>`


The quantum circuit performs a unitary transformation :math:`U(\bm{\theta}})`.

The gate parameters can be used to input classical data :math:`\bx` into a quantum circuit (by setting the parameters :math:`\theta_i` of some subset of gates based on the components of :math:`\bx`), and also to enact a transformation on this data. The output of the circuit is given by the expectation value of some measurement operator :math:`B`. Altogether, the circuit computes the function

.. math:: f\theta(\bx) = \langle B \rangle_{\bx,\theta} = \mathrm{Tr}(B~U(x, \theta)\ketbra{0}{0}U^\dagger(\bx, \theta)).

Machine learning with variational circuits
--------------------------------------------------

How can we build machine learning models using programmable quantum circuits?


Quantum circuit gradients
-------------------------

What is the *gradient of a quantum circuit*? 

At the highest level, we picture a quantum circuit as a hardware device that can evaluate functions of the form :math:`f_{\theta}(\bx)`. In machine learning, we want to find the parameter values which make the function :math:`f` optimal for some problem of interest. One way to do this is to perform *gradient descent*: we compute the gradients :math:`\nabla_\theta f(\bx)` and update the parameters to new values based on this gradient information, :math:`\theta\mapsto\theta + \eta\nabla_\theta f(\bx)`. For this, we need a method to evaluate the gradients of the function :math:`f` defined by our quantum circuit.

