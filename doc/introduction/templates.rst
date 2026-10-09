.. role:: html(raw)
   :format: html

.. _intro_ref_temp:

Templates
=========

PennyLane provides a growing library of pre-coded templates of common variational circuit architectures
that can be used to easily build, evaluate, and train more complex models. In the
literature, such architectures are commonly known as an *ansatz*. Templates can be used to
:ref:`prepare quantum states <intro_ref_temp_stateprep>` as the first operation in a circuit,
or simply as general :ref:`subroutines <intro_ref_temp_subroutines>` that a circuit is built from.

The following are the built-in templates provided by PennyLane.

.. _intro_ref_temp_stateprep:

State Preparations
------------------

State preparation templates transform the zero state :math:`|0\dots 0 \rangle` to another initial
state, and are typically used as the first operation in a circuit.

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.MPSPrep
    ~pennylane.MultiplexerStatePreparation
    ~pennylane.SumOfSlatersPrep
    ~pennylane.PartialUnaryStatePreparation
    ~pennylane.UniformPrep
    ~pennylane.AliasSampling
    ~pennylane.PhaseGradientStatePrep

:html:`</div>`

.. _intro_ref_temp_subroutines:

Arithmetic templates
--------------------

Quantum arithmetic templates enable in-place and out-place modular operations such
as addition, multiplication and exponentiation.

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.SemiAdder
    ~pennylane.Incrementer
    ~pennylane.OutMultiplier
    ~pennylane.SignedOutMultiplier
    ~pennylane.OutSquare
    ~pennylane.SignedOutSquare
    ~pennylane.LeftClassicalComparator
    ~pennylane.LeftQuantumComparator

:html:`</div>`

.. _intro_ref_temp_qchem:

Other subroutines
-----------------

Other useful templates which do not belong to the previous categories can be found here.

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.IQP
    ~pennylane.IQPEmbedding
    ~pennylane.TrotterCDF
    ~pennylane.TrotterCGF
    ~pennylane.TrotterVibronic
    ~pennylane.QuantumPhaseEstimation
    ~pennylane.QFT
    ~pennylane.AQFT
    ~pennylane.FlipSign
    ~pennylane.QSVT
    ~pennylane.GQSP
    ~pennylane.Select
    ~pennylane.OneBodyBlockEncoding
    ~pennylane.QROM
    ~pennylane.SelectPauliRot
    ~pennylane.TemporaryAND
    ~pennylane.BasisRotation

:html:`</div>`

Custom templates
----------------

Creating a custom template can be as simple as defining a function that creates operations and does not have a return
statement:

.. code-block:: python

    import pennylane as qp

    def MyTemplate(a, b, wires):
        c = qp.math.sin(a) + b
        qp.RX(c, wires=wires[0])

    n_wires = 3
    dev = qp.device("lightning.qubit", wires=n_wires)

    @qp.qjit(capture=True)
    @qp.qnode(dev)
    def circuit(a, b):
        MyTemplate(a, b, wires=range(n_wires))
        return qp.expval(qp.PauliZ(0))

>>> circuit(2, 3)
Array(-0.71950657, dtype=float64)

.. note::

    Classical processing inside a template must be compatible with JIT compilation. PennyLane's
    :mod:`math <pennylane.math>` library provides framework-agnostic functions, such as the
    ``qp.math.sin`` used above, that can be used for this purpose.

To turn a quantum function into a template class like the built-in templates above, decorate it with
:func:`~pennylane.subcircuit`. The quantum function must not return anything, and its resources must be
registered with :func:`~pennylane.register_resources`. Its arguments are classified via keyword arguments
such as ``dynamic_argnames``; arguments named ``wires`` are treated as wires by default:

.. code-block:: python

    import numpy as np

    @qp.subcircuit(dynamic_argnames=("weights",))
    @qp.register_resources({qp.RY: 2, qp.CNOT: 1})
    def EntangledRotations(weights, wires):
        qp.RY(weights[0], wires=wires[0])
        qp.RY(weights[1], wires=wires[1])
        qp.CNOT(wires=wires)

    dev = qp.device("lightning.qubit", wires=2)

    @qp.qjit(capture=True)
    @qp.decompose(gate_set={qp.RY, qp.CNOT})
    @qp.qnode(dev)
    def circuit(weights):
        EntangledRotations(weights, wires=[0, 1])
        return qp.expval(qp.PauliZ(1))

    weights = np.array([0.1, 0.2])

``EntangledRotations`` is now an :class:`~.Operator2` subclass, and appears as a single operation in
the circuit. Its quantum function body is registered as its decomposition, which
:func:`~pennylane.decompose` uses to lower it into gates supported by the device:

>>> print(qp.draw(circuit, level="top")(weights))
0: ─╭EntangledRotations(M0)─┤
1: ─╰EntangledRotations(M0)─┤  <Z>
<BLANKLINE>
M0 =
[0.1 0.2]
>>> print(qp.draw(circuit)(weights))
0: ──RY(0.10)─╭●─┤
1: ──RY(0.20)─╰X─┤  <Z>
>>> circuit(weights)
Array(0.97517033, dtype=float64)

As suggested by the camel-case naming, built-in templates in PennyLane are classes. Classes are more complex
data structures than functions, since they can define properties and methods of templates (such as gradient
recipes or matrix representations). Consult the :ref:`Contributing operators <contributing_operators>`
page to learn how to code up your own template class, and how to add it to the PennyLane template library.

Layering Function
-----------------

The layer function creates a new template by repeatedly applying a sequence of quantum
gates to a set of wires. You can import this function both via
``qp.layer`` and ``qp.templates.layer``.

.. autosummary::

    pennylane.layer
