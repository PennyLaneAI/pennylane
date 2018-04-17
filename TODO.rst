OpenQML TODO list
=================


Basic idea
----------

Given a quantum computing backend (simulator or actual hardware) via a plugin,
the library takes an optimization problem consisting of a quantum circuit template :math:`f(\theta_i)`
where :math:`\theta_i` is a vector of optimization parameters, some inputs :math:`x_i` and possibly some outputs :math:`y_i`,
a cost/error function based on some expectation values of the circuit, e.g.

.. math::
  C(\theta_i) = \sum_i |\trace(f(\theta_i)(x_i) Y_i) -y_i|^2

and possibly a penalty function. It then trains the circuit by optimizing the thetas to minimize C.
The inputs :math:`x_i` could be initial state preparation procedures, or just arbitrary gate parameters.
The inputs can contain noise, and when using a hardware backend the expectation values :math:`Y_i` are estimated
by averaging a fixed number of measurements, so there's statistical noise in the result.

The trained circuit is tested (or used) with another sample of :math:`x_i` and :math:`y_i`.
Both training and testing steps rely on having a quantum circuit black box at hand.
The thetas are continuous variables, typically rotation angles or in the CV case, real numbers.


Gradient
--------

The optimization algorithm is a gradient-based one, and it should probably be able to handle noisy data.

* stochastic gradient descent

The gradient can be computed in different ways:

1. user giving us a f' black box in addition to f
2. automatically using numerical differentiation based on f evaluations only, `<https://pypi.org/project/Numdifftools/>`_
3. automatically using an analytic method, i.e. given a circuit f construct a circuit for f'

For (3) we need to know something about gates, each plugin may have its own set.


Optimization problems supported
-------------------------------

* 
* 
* 


Features
--------

* We should be able to tell a plugin to build the given circuit, composed of gates in its library with given parameters, and then
  estimate the :math:`\expect{Y_i}` expectation values to a given accuracy, or using a given number of repeats.
* How do we propose a circuit template, or is the user responsible for it? Maybe each plugin should come with a few default templates.
* If the backend/plugin is responsible for both the gates and the circuit template, maybe the only reason we need to know about them
  is to build the gradient circuit? Otherwise it could just be a black box :math:`f(\theta, x_i, y_i)` for us.
* Gradient circuit probably requires that the plugins can communicate to us their gate library, in (gate, generator) pairs.
  Alternatively, if the gate derivative can be computed by shifting the parameter, (gate, derivative_par_shift) pairs.
* Should the plugins build and store a circuit graph with explicit parameter dependencies (the Tensorflow approach)
  and evaluate it with different parameter values, or rebuild the circuit anew each time the parameters change?
* Automatic differentiation of classical input/output/parameter transformation functions: use Tensorflow?


Misc. ideas
-----------

* The above approach assumes a fixed circuit/black box with continuous parameters.
  Maybe we could try to optimize the circuit template too, using discrete optimization methods?
* What about using a quantum device to train a classical model, and use/test it in classical hardware?
