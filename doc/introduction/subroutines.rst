.. role:: html(raw)
   :format: html

.. _intro_ref_temp:

Built-in Subroutines
====================

PennyLane provides a growing library of pre-coded subroutines of common variational circuit architectures
that can be used to easily build, evaluate, and train more complex models. In the
literature, such architectures are commonly known as an *ansatz*. Subroutines can be used to
:ref:`prepare quantum states <intro_ref_temp_stateprep>` as the first operation in a circuit,
or simply as general :ref:`building blocks <intro_ref_temp_subroutines>` that a circuit is built from.

The following are the built-in subroutines provided by PennyLane.

.. _intro_ref_temp_stateprep:

State preparation
-----------------

State preparation subroutines transform the zero state :math:`|0\dots 0 \rangle` to another initial
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

Arithmetic subroutines
----------------------

Quantum arithmetic subroutines enable in-place and out-place modular operations such
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

Other useful subroutines which do not belong to the previous categories can be found here.

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
