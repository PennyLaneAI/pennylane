Optimizers
==========

In openqml, a nuclear optimizer is a procedure that executes one weight update step along (some function of) the negative gradient of the cost. This update depends in general on 

* The function :math:`f(x)`, from which we calculate a gradient :math:`\nabla f(x)`. If :math:`x` is a vector, the gradient is also a vector whose entries are the partial derivatives of :math:`f` with respect to the elements of :math:`x`. 
* the current weights :math:`x`,
* the (initial) step size :math:`\eta`.

The different optimizers can depend on more hyperparameters. 

In the following, recursive definitions assume that :math:`x^{(0)}` is some initial value in the optimization landscape, and all other step-dependent values are initialized to zero at :math:`t=0`.

Gradient Descent
****************

User-defined hyperparameters: :math:`\eta`.

A step of the gradient descent optimizer computes the new weights via the rule

.. math:: 

    x^{(t+1)} = x^{(t)} - \eta \nabla f(x^{(t)}).

<REF TO CODE>


Momentum
*********
REF: Polyak 1964

User-defined hyperparameters: :math:`\eta`, :math:`m`.

The momentum optimizer adds a "momentum" term to gradient descent which considers the past gradients:

.. math:: 

    x^{(t+1)} = x^{(t)} - a^{(t+1)}.

The accumulator term :math:`a` is updates as follows:

.. math:: 

    a^{(t+1)} = m a^{(t)} + \eta \nabla f(x^{(t)}).



<REF TO CODE>

Nesterov Momentum
*****************

REF: Nesterov 1983?

User-defined hyperparameters: :math:`\eta`, :math:`m`.

Nesterov Momentum works like the Momentum optimizer, but shifts the current input by the momentum term when computing the gradient of the cost,

.. math:: 

    a^{(t+1)} = m a^{(t)} + \eta \nabla f(x^{(t)} - m a^{(t)}).



<REF TO CODE>

Adagrad
*******

User-defined hyperparameters: :math:`\eta_{\text{init}}`.

Adagrad adjusts the learning rate for each parameter :math:`x_i` in :math:`x` based on past gradients. We therefore have to consider each parameter update individually,

.. math:: 

    x^{(t+1)}_i = x^{(t)}_i - \eta_i^{(t+1)} \partial_{w_i} f(x^{(t)}),

where the gradient was replaced by a (scalar) partial derivative. The learning rate in step :math:`t` is given by

.. math::

    \eta_i^{(t+1)} = \frac{ \eta_{\mathrm{init}} }{ \sqrt{a_i^{(t+1)} + \epsilon } }

.. math::

    a_i^{(t+1)} = \sum_{k=1}^t (\partial_{x_i} f(x^{(k)}))^2.


The shift :math:`\epsilon` avoids division by zero and is set to :math:`1e-8` in openqml.

<REF TO CODE>

RMSProp
********

User-defined hyperparameters: :math:`\eta_{\text{init}}`, :math:`\gamma`.

Extensions of Adagrad start the sum :math:`a` over past gradients in the denominator of the learning rate at a finite :math:`t'` with :math:`0 < t' < t`, or decay past gradients to avoid an ever-decreasing learning rate. Root Mean Square propagation is such an adaptation, where


.. math:: 

    a_i^{(t+1)} = \gamma a_i^{(t)} + (1-\gamma) (\partial_{x_i} f(x^{(t)}))^2.

<REF TO CODE>

Adam 
*****

User-defined hyperparameters: :math:`\beta_1`, :math:`\beta_2`.

Reference: https://arxiv.org/pdf/1412.6980.pdf, :cite:`kingma2014adam`.

Adaptive Moment Estimation uses a step-dependent learning rate, a first moment :math:`a` and a second moment :math:`b` (reminiscent of the momentum and velocity of a particle).

.. math:: 

    x^{(t+1)} = x^{(t)} - \eta^{(t+1)} \frac{a^{(t+1)}}{\sqrt{b^{(t+1)}} + \epsilon },

where the update rules for the three values are given by

.. math:: 

    a^{(t+1)} = \frac{\beta_1 a^{(t)} + (1-\beta_1)\nabla f(x^{(t)})}{(1- \beta_1)},

.. math:: 

    b^{(t+1)} = \frac{\beta_2 b^{(t)} + (1-\beta_2) ( \nabla f(x^{(t)}))^{\odot 2} }{(1- \beta_2)},
    
.. math:: 

    \eta^{(t+1)} = \eta^{(t)} \frac{\sqrt{(1-\beta_2)}}{(1-\beta_1)}.

Above, :math:`( \nabla f(x^{(t-1)}))^{\odot 2}` denotes the element-wise square operation, which means that each element in the gradient is multiplied by itself. The hyperparameters :math:`\beta_1` and :math:`\beta_2` can also be step-dependent. Initially, the first and second moment are zero.

The shift :math:`\epsilon` avoids division by zero and is set to :math:`1e-8` in openqml. 


<REF TO CODE>

Natural Gradients
*****************

TODO?








