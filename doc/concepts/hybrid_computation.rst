.. role:: html(raw)
   :format: html

.. _hybrid_computation:

Hybrid computation
==================

In the previous section, we introduced the notion of :mod:`quantum nodes <pennylane.qnode>`. This abstraction lets us combine quantum functions with classical functions as part of a larger hybrid quantum-classical computation. 

Hybrid computations have been considered in many existing proposals. However, the division of labour between the quantum and classical components is often very rigid. Typically, quantum devices are used to evaluate some circuit(s), and the resulting expectation values are combined in a single classical cost function. 

:html:`<br>`

.. figure:: ../_static/simple_hybrid_graph.svg
    :align: center
    :width: 30%
    :target: javascript:void(0);

    The structure of hybrid quantum-classical computations has historically been limited to quantum circuits whose output is combined in a single classical cost function, e.g., the variational quantum eigensolver [peruzzo2014variational]_. 
    
:html:`<br>`
    
While this approach has shown some success as a first step, it is still too limited. It is easy to imagine many more interesting ways we could combine quantum and classical ingredients into a larger and more complex hybrid computation.


Directed acyclic graphs
-----------------------

:html:`<br>`

.. figure:: ../_static/hybrid_graph.svg
    :align: center
    :width: 70%
    :target: javascript:void(0);

    A *true hybrid* quantum-classical computation. The quantum and classical nodes are arranged in a **directed acyclic graph**.

:html:`<br>`

PennyLane was designed with a much more expressive notion of hybrid computation in mind. Quantum and classical nodes can be combined into an arbitrary `directed acyclic graph <https://en.wikipedia.org/wiki/Directed_acyclic_graph>`_ (DAG). This means that information flows from each node to its successors (which may be QNodes or nodes implementing classical functions), and no cycles (loops) are created. Other than these basic rules, any configuration is supported. Every node in the graph can be either classical or quantum, and quantum nodes running on different devices (e.g., a qubit and a CV device) can be combined in the same computation.

This DAG structure is similar to that appearing in modern deep learning models. In fact, PennyLane supports any machine learning model that can be coded using NumPy. Of course, PennyLane has the added benefit that it also supports quantum circuits seamlessly in the computational graph.

Backpropagation through hybrid computations
-------------------------------------------

Because PennyLane provides a method for evaluating gradients of quantum functions, it is compatible with automatic differentiation techniques like the famous `backpropagation <https://en.wikipedia.org/wiki/Backpropagation>`_ algorithm (a.k.a., reverse mode automatic differentiation), the workhorse algorithm for training deep learning models. 

This means that **PennyLane can differentiate end-to-end through hybrid quantum-classical computations**. Quantum machine learning models can thus be trained in basically the same way that clasical deep learning models are trained.

.. note:: 
    PennyLane leverages the Python library `autograd <https://github.com/HIPS/autograd>`_,  
    which wraps the regular NumPy mathematical library, providing automatic differentiation features.
    PennyLane can support any classical machine learning model which is supported by autograd, as
    well as any hybrid machine learning model supported by the available quantum devices. 

    When building a quantum-classical hybrid model, make sure to import the wrapped version of NumPy 
    which is provided by PennyLane, i.e., :code:`from pennylane import numpy as np`. This will allow
    PennyLane to compute gradients of functions built with NumPy alongside the gradients of quantum 
    circuits.


