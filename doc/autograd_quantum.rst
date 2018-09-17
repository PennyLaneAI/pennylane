.. role:: html(raw)
   :format: html

.. _autograd_quantum:

Quantum gradients
=================

Automatic differentiation
-------------------------

Derivatives are ubiquitous throughout science and engineering. In recent years, automatic differentiation has become a key feature in many numerical software libraries, in particular for machine learning (e.g., Theano_, Autograd_, Tensorflow_, or Pytorch_). 

Generally speaking, automatic differentiation is the ability for a software library to compute the derivatives of arbitrary numerical code. If you write an algorithm to compute some function :math:`g(x)` (which may include mathematical expressions, but also control flow statements like :code:`if`, :code:`for`, etc.), then automatic differentiation provides an algorithm for :math:`\frac{d}{dx}g(x)` with the same degree of complexity as the original function.

*Automatic* differentiation should be distinguished from other forms of differentiation. *Manual differentiation*, where an expression is differentiated by hand (often on paper), is extremely time-consuming and error-prone. In *numerical differentiation*, such as the finite-difference method familiar from high-school calculus, the derivative of a function is approximated by numerically evaluating the function at two infinitesimaly separated points. However, this method can sometimes be imprecise to do the constraints of floating-point arithmetic.

Computing gradients of quantum functions
----------------------------------------

:ref:`qfuncs` are parameterized functions :math:`f(x;\bm{\theta})` which can be evaluated by measuring a quantum circuit. If we can compute gradients of a quantum function, we could use this information in an optimization or machine learning algorithm, tuning the quantum circuit to produce a desired output. While numerical differentiation is an option, OpenQML is the first software library to support **automatic differentiation of quantum circuits** [#]_.

How is this accomplished? It turns out that the gradient of a quantum function :math:`f(x;\bm{\theta})` can in many cases be expressed as a linear combination of other quantum functions. In fact, these other quantum functions typically use the same circuit, differing only in a shift of the argument. 

:html:`<br>`

.. figure:: ./_static/quantum_gradient.svg
    :align: center
    :width: 70%
    :target: javascript:void(0);

    Decomposing the gradient of a qfunc as a linear combination of qfuncs.

:html:`<br>`

Making a rough analogy to classically computable functions, this is similar to how the derivative of the function :math:`f(x)=\sin(x)` is identical to :math:`\frac{1}{2}\sin(x+\frac{\pi}{2}) - \frac{1}{2}\sin(x-\frac{\pi}{2})`. So the same underlying algorithm can be reused to compute both :math:`\sin(x)` and its derivative (by evaluating at :math:`x\pm\frac{\pi}{2}`). This intuition holds for many quantum functions of interest: the same circuit can be used to compute both the qfunc and gradients of the qfunc [#]_.

A more technical explanation
----------------------------

.. todo:: Need to introduce clean unified definitions up front, then derive the formulas in a way which keeps CV and qubit formalisms on same page as long as possible.

For convenience, let us rewrite the unitary conjugation performed by a quantum circuit as a transformation :math:`\mathcal{C}_U` acting on the operator :math:`\hat{B}`:

.. math:: U^\dagger(x;\bm{\theta})\hat{B}U(x;\bm{\theta}) = \mathcal{C}_U(\hat{B}).

With this notation, a qfunc is simply the matrix element 

.. math:: f(x; \bm{\theta}) = \langle 0 | U^\dagger(x;\bm{\theta})\hat{B}U(x;\bm{\theta}) | 0 \rangle = \langle 0 | \mathcal{C}_U(\hat{B}) | 0 \rangle.

If the transformation :math:`\mathcal{C}_U` depends smoothly on a parameter :math:`\theta_i`, then the associated quantum function will have a well-defined gradient:

.. math:: \nabla_{\theta_i}f(x; \bm{\theta}) = \langle 0 | \nabla_{\theta_i}\mathcal{C}_U(\hat{B}) | 0 \rangle \in \mathbb{R}.

What does this gradient look like? To answer this, we will have to specify how the full circuit unitary :math:`U(x;\bm{\theta})` depends on the specific parameter :math:`\theta_i`. Like any quantum computation, we can decompose a unitary into an ordered sequence of unitary gates from an elementary gate set, each which takes (at most) one argument: 

.. math:: U(x; \bm{\theta}) = U_N(\theta_{N}) U_{N-1}(\theta_{N-1}) \cdots U_i(\theta_i) \cdots U_1(\theta_1) U_0(x).

.. note:: For convenience, we have used the input :math:`x` as the argument for gate :math:`U_0` and the parameters :math:`\bm{\theta}` for the remaining gates. This is not required. Inputs and parameters can be arbitrarily assigned to different gates.

Each of these gates is unitary, and therefore must have the form :math:`U_{j}(\gamma_j)=\exp{(i\gamma_j H_j)}` where :math:`H_j` is a Hermitian operator which generates the gate and :math:`\gamma_j` is the gate parameter. We have also suppressed the subsystems that these gates have been applied to, since it doesn't affect the gradient formula.

Acting on a single one-parameter gate, the gradient formula is straightforward:

.. math:: \nabla_{\gamma} U(\gamma) = \nabla_\gamma\exp{(i\gamma H)} = H\exp{(i\gamma H)} = HU(\gamma).


Since the equations governing quantum circuits are linear, we can pass the gradient through all the unitaries which don't use the parameter :math:`\theta_i`:

.. math:: \nabla_{\theta_i}U(x;\bm{\theta}) = U_N(\theta_{N}) U_{N-1}(\theta_{N-1}) \cdots \left[ \nabla_{\theta_i} U_i(\theta_i) \right] \cdots U_1(\theta_1) U_0(x).

For convenience, let us absorb any gates applied before gate :math:`i` (with indices lower than :math:`i`) into the initial state: :math:`|\psi_{i-1}\rangle = U_{i-1}(\theta_{i-1}) \cdots U_{1}(\theta_{1})|0\rangle`. 
Similarly, any gates applied after gate :math:`i` are combined with the observable :math:`\hat{B}`:
:math:`\hat{B}_{i+1} = U_{N}(\theta_{N}) \cdots U_{i+1}(\theta_{i+1}) \hat{B} U_{i+1}^\dagger(\theta_{i+1}) \cdots U_{N}^\dagger(\theta_{N})`. 

With this simplification, the qfunc becomes

.. math:: f(x; \bm{\theta}) = \langle \psi_{i-1} | U_i(\theta_i) \hat{B}_{i+1} U_i^\dagger(\theta_i) | \psi_{i-1} \rangle

and its gradient has the form

.. math:: \nabla_{\theta_i}f(x; \bm{\theta}) = i\langle \psi_{i-1} | \left[H_i, \hat{B}_{i} \right] | \psi_{i-1} \rangle,

where :math:`\left[H_i, \hat{B}_{i} \right] = H_i \hat{B}_{i} - \hat{B}_{i} H_i` is the commutator.

.. _Theano: https://github.com/Theano/Theano
.. _Autograd: https://github.com/HIPS/autograd
.. _Tensorflow: http://tensorflow.org/
.. _Pytorch: https://pytorch.org/


.. rubric:: Footnotes

.. [#] This should be contrasted with software which can perform automatic differentiation on classical simulations of quantum circuits, such as `Strawberry Fields <https://strawberryfields.readthedocs.io/en/latest/>`_. 

.. [#] In situations where no formula for automatic quantum gradients is known, OpenQML falls back to approximate gradient estimation using numerical methods.

