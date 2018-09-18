Optimizers
==========

In openqml, a nuclear optimizer is a procedure that executes one weight update step along (some function of) the negative gradient of the cost. This update depends in general on 

* The cost function :math:`C(w)`, from which we calculate a gradient :math:`\nabla C(w)`. The gradient is a vector whose entries are the partial derivatives of :math:`C` with respect to the elements of :math:`w`. 
* the current weights :math:`w`,
* the learning rate :math:`\eta`,

The different optimizers add more dependencies to this list.

Gradient Descent
****************

User-defined hyperparameters: :math:`\eta`.

A step of the gradient descent optimizer computes the new weights via the rule

.. math:: 

    w^{(t+1)} = w^{(t)} - \eta \nabla C(w^{(t)}).

<REF TO CODE>


Momentum
*********
REF: Polyak 1964

User-defined hyperparameters: :math:`\eta`, :math:`\gamma`.

The momentum optimizer adds a "momentum" term to gradient descent which considers the past gradients:

.. math:: 

    w^{(t+1)} = w^{(t)} - m^{(t)}.

The momentum is updates as follows:

.. math:: 

    m^{(t)} = \gamma m^{(t-1)} + \eta \nabla C(w^{(t-1)}).



<REF TO CODE>

Nesterov Momentum
*****************

REF: Nesterov 1983?

User-defined hyperparameters: :math:`\eta`, :math:`\gamma`.

Nesterov Momentum works like the Momentum optimizer, but shifts the current input by the momentum term when computing the gradient of the cost,

.. math:: 

    m^{(t)} = \gamma m^{(t-1)} + \eta \nabla C(w^{(t-1)} - \gamma m^{(t-1)}).



<REF TO CODE>

Adagrad
*******

User-defined hyperparameters: :math:`\eta_{\text{init}}`.

Adagrad adjusts the learning rate for each parameter :math:`w_i` in :math:`w` based on past gradients. We therefore have to consider each parameter update individually,

.. math:: 

    w^{(t+1)}_i = w^{(t)}_i - \eta_i^{(t)} \partial_{w_i} C(w^{(t)}),

where the gradient was replaced by a (scalar) partial derivative. The learning rate in step :math:`t` is given by

.. math::

    \eta_i^{(t)} = \frac{ \eta_{\mathrm{init}} }{ \sqrt{s_i^{(t)} + \epsilon},

.. math::

    s_i^{(t)} = \sum_{k=1}^t \partial_{w_i} C(w^{(k)}).


The shift :math:`\epsilon` avoids division by zero and is set to :math:`1e-8` in openqml. 

<REF TO CODE>

RMSProp
********

User-defined hyperparameters: :math:`\eta_{\text{init}}`, :math:`\gamma`.

Extensions of Adagrad start the sum :math:`s` over past gradients in the denominator of the learning rate at a finite :math:`t'` with :math:`0 < t' < t`, or decay past gradients to avoid an ever-decreasing learning rate. Root Mean Square propagation is such an adaptation, where


.. math:: 

    s_i^{(t)} = \gamma s_i^{(t-1)} + (1-\gamma) (\partial_{w_i} C(w^{(k)}))^2.

<REF TO CODE>

Adam 
*****

User-defined hyperparameters: :math:`\beta_1`, :math:`\beta_2`.

Reference: https://arxiv.org/pdf/1412.6980.pdf, :cite:`kingma2014adam`.

Adaptive Moment Estimation uses a step-dependent learning rate, a first moment :math:`m` and a second moment :math:`v` (reminiscent of the momentum and velocity of a particle).

.. math:: 

    w^{(t+1)} = w^{(t)} - \eta^{(t)} \frac{m^{(t)}}{\sqrt{v^{(t)}} + \epsilon },

where the update rules for the three values are given by

.. math:: 

    m^{(t)} = \frac{\beta_1 m^{(t-1)} + (1-\beta_1)\nabla C(w^{(t-1)})}{(1- \beta_1)},

.. math:: 

    v^{(t)} = \frac{\beta_2 v^{(t-1)} + (1-\beta_2) ( \nabla C(w^{(t-1)}))^{\odot 2} }{(1- \beta_2)},
    
.. math:: 

    \eta^{(t)} = \eta^{(t-1)} \frac{\sqrt{(1-\beta_2)}}{(1-\beta_1)}.

Above, :math:`( \nabla C(w^{(t-1)}))^{\odot 2}` denotes the element-wise square operation, which means that each element in the gradient is multiplied by itself. The hyperparameters :math:`\beta_1` and :math:`\beta_2` can also be step-dependent. Initially, the first and second moment are zero.

The shift :math:`\epsilon` avoids division by zero and is set to :math:`1e-8` in openqml. 


<REF TO CODE>

Natural Gradients
*****************

TODO?








