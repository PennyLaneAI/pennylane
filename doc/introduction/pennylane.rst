 .. role:: html(raw)
   :format: html

.. _pl_intro:

In a nutshell
=============

PennyLane is a quantum computing software library that emphasises the ability to
automatically *train* or optimise quantum circuits.

.. image:: ../_static/header.png
    :align: right
    :width: 300px
    :target: javascript:void(0);

PennyLane connects quantum computing with powerful machine learning frameworks
like `NumPy <https://numpy.org/>`_'s `autograd <https://github.com/HIPS/autograd>`__,
`JAX <https://github.com/google/jax>`__,
`PyTorch <https://pytorch.org/>`_, and `TensorFlow <https://www.tensorflow.org/>`_,
making them quantum-aware. Its central job is to manage the execution of quantum computations, including
the evaluation of circuits and the computation of their gradients. This information is forwarded to the classical
framework, creating seamless quantum-classical pipelines for quantum applications.

.. image:: ../_static/code.png
    :align: right
    :width: 300px
    :target: javascript:void(0);

PennyLane's design principle is that
circuits can be run on various kinds of simulators or hardware devices without making any changes --
the complex job of optimising communication with the devices, compiling circuits to suit the backend,
and choosing the best gradient strategies is taken care of by the library.
PennyLane comes with default simulator devices, but is well-integrated with external software and hardware to run quantum
circuits---such as IBM's Qiskit, or Google's Cirq, Rigetti's Forest, or Xanadu's Strawberry Fields.

.. image:: ../_static/jigsaw.png
    :align: right
    :width: 300px
    :target: javascript:void(0);