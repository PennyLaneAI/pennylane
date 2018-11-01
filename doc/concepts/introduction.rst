.. role:: html(raw)
   :format: html

.. _introduction:

Introduction
============

The main principle underlying PennyLane is to make the interface between the quantum and classical worlds seamless. A quantum computing device should not be viewed as a competitor to a classical computer, but rather as an *accelerator*. PennyLane employs a model where both classical and quantum computers are used in the same basic way: as computational **devices** which we program to evaluate mathematical functions.

The core of PennyLane is designed around four main concepts:

1. **Quantum functions**: a class of functions that are naturally evaluated using quantum computer circuits

2. **Quantum gradients**: the gradients of quantum functions, these are themselves built from quantum functions

3. **Quantum nodes**: an abstract representation of quantum circuits which input and output classical information

4. **Hybrid computation**: a computing model which seamlessly integrates both classical and quantum nodes

:html:`<h3>Quantum functions</h3>`

.. rst-class:: admonition see

    See the main :ref:`qfuncs` page for more details.

:html:`<br>`

.. figure:: ../_static/quantum_function.svg
    :align: center
    :width: 70%
    :target: javascript:void(0);

    A quantum function is a function which is evaluated by measurements of a programmable quantum computer circuit.

:html:`<br>`

The primary motivation for building quantum computers is that they should be able to perform computations which are inefficient to run on classical computers. To this end, a parameterized function :math:`f(x;\bm{\theta})` is called a **quantum function** (or **qfunc**) if it can be evaluated using a quantum circuit. 

.. note:: For a function :math:`f(x; \bm{\theta})`, :math:`x` is considered to be the function's input and :math:`\bm{\theta}` are parameters which determine the exact form of :math:`f`.

.. 
    .. seealso:: See the main :ref:`qfuncs` page for more details.

:html:`<h3>Quantum gradients</h3>`

.. rst-class:: admonition see

    See the main :ref:`autograd_quantum` page for more details.

A core element of modern machine learning libraries is the automatic computation of analytic gradients. PennyLane extends this key feature to quantum functions.

Evaluating qfuncs is inefficient on classical computers, so we might expect the gradients of qfuncs to be similarly intractable. Fortunately, we can often compute the gradient of a qfunc :math:`\nabla_{\bm{\theta}}f(x;\bm{\theta})` exactly using a linear combination of closely related qfuncs:

:html:`<br>`

.. figure:: ../_static/quantum_gradient.svg
    :align: center
    :width: 70%
    :target: javascript:void(0);

    Decomposing the gradient of a qfunc as a linear combination of qfuncs.

:html:`<br>`

We can thus **use the same quantum device** to compute both quantum functions and also gradients of quantum functions. This is accomplished with minor assistance of a classical coprocessor, which combines the terms. 


:html:`<h3>Quantum nodes</h3>`

.. rst-class:: admonition see

    See the main :ref:`quantum_nodes` page for more details.

Quantum information is fragile — especially in near-term devices. How can we integrate quantum devices seamlessly and scalably with classical computations?

This leads to the notion of a **quantum node**: a basic computational unit — programmed on a quantum circuit — which evaluates a qfunc. Only classical data can enter or exit a quantum node.

:html:`<br>`

.. figure:: ../_static/quantum_node.svg
    :align: center
    :width: 70%
    :target: javascript:void(0);

    A quantum node encapsulates a quantum circuit. Quantum information cannot exist outside a quantum node.

:html:`<br>`

To a classical device, a quantum node is a black box which can evaluate functions. A quantum device, however, resolves the finer details of the circuit.


:html:`<h3>Hybrid computation</h3>`

.. rst-class:: admonition see

    See the main :ref:`hybrid_computation` page for more details.

In many proposed hybrid algorithms, quantum circuits are used to evaluate quantum functions, and a classical co-processor is used primarily to post-process circuit outputs. But why should the division of labour be so regimented? 

:html:`<br>`

.. figure:: ../_static/hybrid_graph.svg
    :align: center
    :width: 70%
    :target: javascript:void(0);

    An 'true hybrid' quantum-classical computational graph.

:html:`<br>`

In a **true hybrid** computational model, both the classical and the quantum devices are responsible for arbitrary parts of an overall computation, subject to the rules of quantum nodes. This allows quantum and classical devices to be used jointly, each forming an integral and inseparable part of a larger computation.
