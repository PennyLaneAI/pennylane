.. role:: html(raw)
   :format: html

.. _autograd_quantum:

Quantum gradients
=================

A quantum function (*qfunc*) is any parameterized function :math:`f(x;\bm{\theta})` which can be evaluated on a quantum circuit via the Born rule:

.. math:: f(x; \bm{\theta}) = \langle \hat{B} \rangle = \langle 0 | U^\dagger(x;\bm{\theta})\hat{B}U(x;\bm{\theta}) | 0 \rangle.

Here, :math:`\hat{B}` is an observable measured at the circuit output and :math:`| 0 \rangle` is a fixed initial state (e.g., the vacuum or a spin-down state). Both the input :math:`x` and the parameters :math:`\bm{\theta}` enter the quantum circuit as arguments used in the gates which are used to build the unitary :math:`U(x;\bm{\theta})`. For convenience, we can also write the unitary conjugation as a transformation :math:`\mathcal{C}_U` acting on the operator :math:`\hat{B}`:

.. math:: U^\dagger(x;\bm{\theta})\hat{B}U(x;\bm{\theta}) = \mathcal{C}_U(\hat{B}).

Note that the measurement operator :math:`\hat{B}` has no dependence on the the input :math:`x` or the parameters :math:`\bm{\theta}`.

Gradients of quantum functions
------------------------------

If the transformation :math:`\mathcal{C}_U` depends smoothly on a parameter :math:`\theta_i`, then the associated quantum function will have a well-defined gradient:

.. math:: \nabla_{\theta_i}f(x; \bm{\theta}) = \langle 0 | \nabla_{\theta_i}\mathcal{C}_U(\hat{B}) | 0 \rangle \in \mathbb{R}.

What does this gradient look like? To answer this, we will have to specify how the full circuit unitary :math:`U(x;\bm{\theta})` depends on the specific parameter :math:`\theta_i`. Like any quantum computation, we can decompose a unitary into an ordered sequence of unitary gates from an elementary gate set, each which takes (at most) one argument: 

.. math:: U(x; \bm{\theta}) = U_N(\theta_{N}) U_{N-1}(\theta_{N-1}) \cdots U_i(\theta_i) \cdots U_1(\theta_1) U_0(x).

.. note:: For convenience, we have used the input :math:`x` as the argument for gate :math:`U_0` and the parameters :math:`\bm{\theta}` for the remaining gates. This is not required. Inputs and parameters can be arbitrarily assigned to different gates.

Each of these gates takes the form :math:`U_{j}(\gamma_j)=\exp{(i\gamma_j H_j)}` where :math:`H_j` is a Hermitian operator which generates the gate and :math:`\gamma_j` is the gate parameter. We have also suppressed the subsystems that these gates have been applied to, since it doesn't affect the gradient formula.

Acting on a single one-parameter gate, the gradient formula is straightforward:

.. math:: \nabla_{\gamma} U(\gamma) = \nabla_\gamma\exp{(i\gamma H)} = H\exp{(i\gamma H)} = HU(\gamma).


Since the equations governing quantum circuits are linear, we can pass the gradient through all the unitaries which don't use the parameter :math:`\theta_i`:

.. math:: \nabla_{\theta_i}U(x;\bm{\theta}) = U_N(\theta_{N}) U_{N-1}(\theta_{N-1}) \cdots \left[ \nabla_{\theta_i} U_i(\theta_i) \right] \cdots U_1(\theta_1) U_0(x).

Note: It might also be useful to have a quantum device that can evaluate standalone gradients, e.g., for calculating forces in quantum chemistry




Backpropagation through hybrid computations
-------------------------------------------

- how does a gradient computation work in a hybrid quantum-classical computation?


.. note:: In situations where no formula for quantum gradients is known, OpenQML supports approximate gradient estimation using numerical methods.

