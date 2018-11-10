.. role:: html(raw)
   :format: html

.. _introduction:

Introduction
============

The main principle underlying PennyLane is to make the interface between the quantum and classical worlds seamless. A quantum computing device should not be viewed as a competitor to a classical computer, but rather as an *accelerator*. Integrating both types of information processing gives rise to **hybrid computation**.

:html:`<br>`

.. figure:: ../_static/concepts.png
    :align: center
    :width: 70%
    :target: javascript:void(0);

:html:`<br>`

In PennyLane both classical and quantum computers are used in the same basic way: as computational devices which we program to evaluate mathematical functions. We call such functions *nodes*, since they feed information into each other like nodes in a directed graph. **Quantum nodes** are abstract representations of quantum circuits which take classical information as their input and produce classical information as their output.

Each quantum node executes a **variational circuit** — a parametrized quantum computation — on a quantum device.

In optimization and machine learning, models learn by computing gradients of trainable variables. A central feature of PennyLane is the ability to compute the gradients of quantum nodes, or **quantum gradients**. This enables the end-to-end differentiation of hybrid computations.


These four concepts — **hybrid computation**, **quantum nodes**, **variational circuits** — and **quantum gradients**, are central to PennyLane.


:html:`<h3>Hybrid computation</h3>`

.. rst-class:: admonition see

    See the main :ref:`hybrid_computation` page for more details.

:html:`<br>`

.. figure:: ../_static/hybrid_graph.png
    :align: center
    :width: 70%
    :target: javascript:void(0);

:html:`<br>`

*Hybrid quantum algorithms* are  algorithms that **integrate both classical and quantum processing**. In many proposed hybrid algorithms, quantum devices are used to evaluate quantum subroutines, and a classical co-processor is used primarily to post-process circuit outputs. But in principle, hybrid computation can be expanded to much more complex procedures.

In a **true hybrid** computational model, both the classical and the quantum devices are responsible for arbitrary parts of an overall computation, subject to the rules of quantum nodes. This allows quantum and classical devices to be used jointly, each forming an integral and inseparable part of a larger computation.


:html:`<h3>Quantum nodes</h3>`

.. rst-class:: admonition see

    See the main :ref:`quantum_nodes` page for more details.

:html:`<br>`

.. figure:: ../_static/quantumnode.png
    :align: center
    :width: 50%
    :target: javascript:void(0);

:html:`<br>`

Quantum information is fragile — especially in near-term devices. How can we integrate quantum devices seamlessly and scalably with classical computations?

This question leads to the notion of a **quantum node** or **QNode**: a basic computational unit, programmed on a quantum circuit, which carries out a subroutine of quantum information processing. Only classical data can enter or exit a quantum node.

To a classical device, a quantum node is a black box which can evaluate functions. A quantum device, however, resolves the finer details of the circuit.


:html:`<h3>Variational circuits</h3>`

.. rst-class:: admonition see

    See the main :ref:`varcirc` page for more details.

:html:`<br>`

.. figure:: ../_static/varcirc.png
    :align: center
    :width: 50%
    :target: javascript:void(0);

:html:`<br>`

Variational circuits are quantum algorithms that depend on tunable variables, and can therefore be **optimized**. In PennyLane, a variational circuit consists of three ingredients:

1. Preparation of a fixed **initial state** (e.g., the vacuum state or the zero state).

2. A quantum circuit, **parameterized** by both the input :math:`x` and the function parameters :math:`\boldsymbol\theta`.

3. **Measurement** of an observable :math:`\hat{B}` at the output. This observable may be made up from local observables for each wire in the circuit, or just a subset of wires.

Variational circuits provide the internal workings of a QNode, and can be evaluated by running a quantum hardware or simulator device.

:html:`<h3>Quantum gradients</h3>`

.. rst-class:: admonition see

    See the main :ref:`autograd_quantum` page for more details.

:html:`<br>`

.. figure:: ../_static/grad.png
    :align: center
    :width: 60%
    :target: javascript:void(0);

:html:`<br>`

**Automatic computation of gradients and the backpropagation algorithm** are core elements of modern deep learning software. PennyLane extends this key functionality to quantum and hybrid computations.

Evaluating quantum nodes is inefficient on classical computers, so we might expect the gradients of quantum nodes to be similarly intractable. Fortunately, we can often compute the gradient of a quantum node :math:`\nabla f(x;\bm{\theta})` exactly using a linear combination of two quantum nodes, where one variable is shifted.

We can thus **use the same quantum device** to compute both quantum nodes and also gradients of quantum nodes. This is accomplished with minor assistance of a classical coprocessor, which combines the terms.


