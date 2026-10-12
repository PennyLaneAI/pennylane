.. role:: html(raw)
   :format: html

.. _intro_ref_temp:

Built-in subroutines
====================

PennyLane2 provides a growing library of subroutines for common tasks, including
state preparation, data reading and loading, quantum arithmetic, and more.

.. _intro_ref_temp_stateprep:

State preparation
-----------------

State preparation subroutines transform the zero state :math:`|0\dots 0 \rangle`
to another initial state, and are typically used as the first operation in a
circuit.

.. subroutine-table::
    :nosignatures:

    ~pennylane.MPSPrep                        /_static/templates/tensornetworks/MPS_template.png
    ~pennylane.MultiplexerStatePreparation    /_static/templates/state_preparations/multiplexerSP_template.png
    ~pennylane.SumOfSlatersPrep               /_static/templates/state_preparations/sumofslatersprep_template.png
    ~pennylane.PartialUnaryStatePreparation   /_static/templates/state_preparations/partialunarystatepreparation_template.png
    ~pennylane.UniformPrep                    /_static/templates/state_preparations/uniformprep.png
    ~pennylane.AliasSampling                  /_static/templates/state_preparations/aliassampling.png
    ~pennylane.PhaseGradientStatePrep         /_static/templates/state_preparations/phasegradientstateprep_template.png

.. _intro_ref_temp_subroutines:

Arithmetic subroutines
----------------------

Quantum arithmetic subroutines enable in-place and out-place modular operations
such as addition, multiplication, and exponentiation.

.. subroutine-table::
    :nosignatures:

    ~pennylane.SemiAdder                 /_static/templates/arithmetic/semiadder.png
    ~pennylane.Incrementer               /_static/templates/arithmetic/adder.png
    ~pennylane.OutMultiplier             /_static/templates/arithmetic/outmultiplier.png
    ~pennylane.SignedOutMultiplier       /_static/templates/arithmetic/signedoutmultiplier.png
    ~pennylane.OutSquare                 /_static/templates/arithmetic/outsquare.png
    ~pennylane.SignedOutSquare           /_static/templates/arithmetic/signedoutsquare.png
    ~pennylane.LeftClassicalComparator   /_static/templates/arithmetic/leftclassicalcomparator.png
    ~pennylane.LeftQuantumComparator     /_static/templates/arithmetic/leftquantumcomparator.png

.. _intro_ref_temp_qchem:

Other subroutines
-----------------

.. subroutine-table::
    :nosignatures:

    ~pennylane.IQP                      /_static/templates/subroutines/iqp.png
    ~pennylane.IQPEmbedding             /_static/templates/embeddings/iqp.png
    ~pennylane.TrotterCDF               /_static/templates/subroutines/trotter_product.png
    ~pennylane.TrotterCGF               /_static/templates/subroutines/trotter_product.png
    ~pennylane.TrotterVibronic          /_static/templates/subroutines/trotter_product.png
    ~pennylane.QuantumPhaseEstimation   /_static/templates/subroutines/qpe.svg
    ~pennylane.iterative_qpe            /_static/templates/subroutines/iterative_qpe.svg
    ~pennylane.QFT                      /_static/templates/subroutines/qft.svg
    ~pennylane.AQFT                     /_static/templates/subroutines/aqft.png
    ~pennylane.FlipSign                 /_static/templates/subroutines/flip_sign.png
    ~pennylane.QSVT                     /_static/templates/subroutines/qsvt.png
    ~pennylane.GQSP                     /_static/templates/subroutines/gqsp.png
    ~pennylane.Select                   /_static/templates/subroutines/select_cropped.png
    ~pennylane.OneBodyBlockEncoding     /_static/templates/subroutines/onebodyblockencoding.png
    ~pennylane.QROM                     /_static/templates/qrom/qrom_thumbnail.png
    ~pennylane.SelectPauliRot           /_static/templates/subroutines/select_pauli_rot_cropped.png
    ~pennylane.TemporaryAND             /_static/templates/subroutines/temporary_and.png
    ~pennylane.BasisRotation            /_static/templates/subroutines/basis_rotation.jpeg
