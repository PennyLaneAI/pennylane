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

    ~pennylane.MPSPrep                        /_static/subroutine_thumbnails/mps_prep.svg
    ~pennylane.MultiplexerStatePreparation    /_static/subroutine_thumbnails/multiplexer_state_prep.svg
    ~pennylane.SumOfSlatersPrep               /_static/subroutine_thumbnails/sum_of_slaters_prep.svg
    ~pennylane.PartialUnaryStatePreparation   /_static/subroutine_thumbnails/partial_unary_state_prep.svg
    ~pennylane.UniformPrep                    /_static/subroutine_thumbnails/uniform_prep.svg
    ~pennylane.AliasSampling                  /_static/subroutine_thumbnails/alias_sampling.svg
    ~pennylane.PhaseGradientStatePrep         /_static/subroutine_thumbnails/phase_gradient_prep.svg

.. _intro_ref_temp_subroutines:

Arithmetic subroutines
----------------------

Quantum arithmetic subroutines enable in-place and out-place modular operations
such as addition, multiplication, and exponentiation.

.. subroutine-table::
    :nosignatures:

    ~pennylane.SemiAdder                 /_static/subroutine_thumbnails/semi_adder.svg
    ~pennylane.Incrementer               /_static/subroutine_thumbnails/incrementer.svg
    ~pennylane.OutMultiplier             /_static/subroutine_thumbnails/out_multiplier.svg
    ~pennylane.SignedOutMultiplier       /_static/subroutine_thumbnails/signed_out_multiplier.svg
    ~pennylane.OutSquare                 /_static/subroutine_thumbnails/out_square.svg
    ~pennylane.SignedOutSquare           /_static/subroutine_thumbnails/signed_out_square.svg
    ~pennylane.LeftClassicalComparator   /_static/subroutine_thumbnails/left_classical_comparator.svg
    ~pennylane.LeftQuantumComparator     /_static/subroutine_thumbnails/left_quantum_comparator.svg

.. _intro_ref_temp_qchem:

Other subroutines
-----------------

.. subroutine-table::
    :nosignatures:

    ~pennylane.IQP                      /_static/subroutine_thumbnails/iqp.svg
    ~pennylane.IQPEmbedding             /_static/subroutine_thumbnails/iqp_embedding.svg
    ~pennylane.TrotterCDF               /_static/subroutine_thumbnails/trotter_product.svg
    ~pennylane.TrotterCGF               /_static/subroutine_thumbnails/trotter_product.svg
    ~pennylane.TrotterVibronic          /_static/subroutine_thumbnails/trotter_product.svg
    ~pennylane.QuantumPhaseEstimation   /_static/subroutine_thumbnails/qpe.svg
    ~pennylane.iterative_qpe            /_static/subroutine_thumbnails/iterative_qpe.svg
    ~pennylane.QFT                      /_static/subroutine_thumbnails/qft.svg
    ~pennylane.AQFT                     /_static/subroutine_thumbnails/aqft.svg
    ~pennylane.FlipSign                 /_static/subroutine_thumbnails/flip_sign.svg
    ~pennylane.QSVT                     /_static/subroutine_thumbnails/qsvt.svg
    ~pennylane.GQSP                     /_static/subroutine_thumbnails/gqsp.svg
    ~pennylane.Select                   /_static/subroutine_thumbnails/select.svg
    ~pennylane.OneBodyBlockEncoding     /_static/subroutine_thumbnails/one_body_block_encoding.svg
    ~pennylane.QROM                     /_static/subroutine_thumbnails/qrom.svg
    ~pennylane.SelectPauliRot           /_static/subroutine_thumbnails/select_pauli_rot.svg
    ~pennylane.TemporaryAND             /_static/subroutine_thumbnails/temporary_and.svg
    ~pennylane.BasisRotation            /_static/subroutine_thumbnails/basis_rotation.svg
