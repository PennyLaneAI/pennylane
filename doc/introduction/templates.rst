.. role:: html(raw)
   :format: html

.. _intro_ref_temp:

Templates
=========

PennyLane provides a growing library of pre-coded templates of common variational circuit architectures
that can be used to easily build, evaluate, and train more complex models. In the
literature, such architectures are commonly known as an *ansatz*. Templates can be used to
:ref:`embed data <intro_ref_temp_emb>` into quantum states, to define trainable :ref:`layers <intro_ref_temp_lay>`
of quantum gates, to :ref:`prepare quantum states <intro_ref_temp_stateprep>` as the first operation in a circuit,
or simply as general :ref:`subroutines <intro_ref_temp_subroutines>` that a circuit is built from.

The following is a gallery of built-in templates provided by PennyLane.

.. _intro_ref_temp_emb:

Embedding templates
-------------------

Embeddings encode input features into the quantum state of the circuit.
Hence, they usually take a data sample such as a feature vector as an argument. Embeddings can also depend on
trainable parameters, and they may be constructed from repeated layers.

.. gallery-item::
    :description: :doc:`AmplitudeEmbedding <../code/api/pennylane.AmplitudeEmbedding>`
    :figure: _static/templates/embeddings/amplitude.png

.. gallery-item::
    :description: :doc:`AngleEmbedding <../code/api/pennylane.AngleEmbedding>`
    :figure: _static/templates/embeddings/angle.png

.. gallery-item::
    :description: :doc:`BasisEmbedding <../code/api/pennylane.BasisEmbedding>`
    :figure: _static/templates/embeddings/basis.png

.. gallery-item::
    :description: :doc:`DisplacementEmbedding <../code/api/pennylane.DisplacementEmbedding>`
    :figure: _static/templates/embeddings/displacement.png

.. gallery-item::
    :description: :doc:`IQPEmbedding <../code/api/pennylane.IQPEmbedding>`
    :figure: _static/templates/embeddings/iqp.png

.. gallery-item::
    :description: :doc:`QAOAEmbedding <../code/api/pennylane.QAOAEmbedding>`
    :figure: _static/templates/embeddings/qaoa.png

.. gallery-item::
    :description: :doc:`SqueezingEmbedding <../code/api/pennylane.SqueezingEmbedding>`
    :figure: _static/templates/embeddings/squeezing.png

.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_lay:

Layer templates
---------------

Layer architectures define sequences of trainable gates that are repeated like the layers in a
neural network. Note that arbitrary templates or operations can also be repeated using the
:func:`~pennylane.layer` function.

.. gallery-item::
    :description: :doc:`CVNeuralNetLayers <../code/api/pennylane.CVNeuralNetLayers>`
    :figure: _static/templates/layers/cvqnn.png

.. gallery-item::
    :description: :doc:`RandomLayers <../code/api/pennylane.RandomLayers>`
    :figure: _static/templates/layers/random.png

.. gallery-item::
    :description: :doc:`StronglyEntanglingLayers <../code/api/pennylane.StronglyEntanglingLayers>`
    :figure: _static/templates/layers/strongly_entangling.png

.. gallery-item::
    :description: :doc:`SimplifiedTwoDesign <../code/api/pennylane.SimplifiedTwoDesign>`
    :figure: _static/templates/layers/simplified_two_design.png

.. gallery-item::
    :description: :doc:`BasicEntanglerLayers <../code/api/pennylane.BasicEntanglerLayers>`
    :figure: _static/templates/layers/basic_entangler.png

.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_stateprep:

State Preparations
------------------

State preparation templates transform the zero state :math:`|0\dots 0 \rangle` to another initial
state. In contrast to embeddings that can in principle be used anywhere in a circuit,
state preparation is typically used as the first operation.

.. gallery-item::
    :description: :doc:`QutritBasisStatePreparation <../code/api/pennylane.QutritBasisStatePreparation>`
    :figure: _static/templates/state_preparations/basis_qutrit.png

.. gallery-item::
    :description: :doc:`MottonenStatePreparation <../code/api/pennylane.MottonenStatePreparation>`
    :figure: _static/templates/state_preparations/mottonen.png

.. gallery-item::
    :description: :doc:`ArbitraryStatePreparation <../code/api/pennylane.ArbitraryStatePreparation>`
    :figure: _static/templates/subroutines/arbitrarystateprep.png

.. gallery-item::
    :description: :doc:`CosineWindow <../code/api/pennylane.CosineWindow>`
    :figure: _static/templates/state_preparations/thumbnail_cosine_window.png

.. gallery-item::
    :description: :doc:`QROMStatePreparation <../code/api/pennylane.QROMStatePreparation>`
    :figure: _static/templates/state_preparations/thumbnail_qrom.png

.. gallery-item::
    :description: :doc:`MPSPrep <../code/api/pennylane.MPSPrep>`
    :figure: _static/templates/tensornetworks/MPS_template.png

.. gallery-item::
    :description: :doc:`MultiplexerStatePreparation <../code/api/pennylane.MultiplexerStatePreparation>`
    :figure: _static/templates/state_preparations/multiplexerSP_template.png

.. gallery-item::
    :description: :doc:`SumOfSlatersPrep <../code/api/pennylane.SumOfSlatersPrep>`
    :figure: _static/templates/state_preparations/sumofslatersprep_template.png


.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_subroutines:

Arithmetic templates
--------------------

Quantum arithmetic templates enable in-place and out-place modular operations such 
as addition, multiplication and exponentiation.

.. gallery-item::
    :description: :doc:`PhaseAdder <../code/api/pennylane.PhaseAdder>`
    :figure: _static/templates/arithmetic/phaseadder.png

.. gallery-item::
    :description: :doc:`Adder <../code/api/pennylane.Adder>`
    :figure: _static/templates/arithmetic/adder.png

.. gallery-item::
    :description: :doc:`SemiAdder <../code/api/pennylane.SemiAdder>`
    :figure: _static/templates/arithmetic/semiadder.png

.. gallery-item::
    :description: :doc:`OutAdder <../code/api/pennylane.OutAdder>`
    :figure: _static/templates/arithmetic/outadder.png

.. gallery-item::
    :description: :doc:`Multiplier <../code/api/pennylane.Multiplier>`
    :figure: _static/templates/arithmetic/multiplier.png

.. gallery-item::
    :description: :doc:`OutMultiplier <../code/api/pennylane.OutMultiplier>`
    :figure: _static/templates/arithmetic/outmultiplier.png

.. gallery-item::
    :description: :doc:`ModExp <../code/api/pennylane.ModExp>`
    :figure: _static/templates/arithmetic/modexp.png

.. gallery-item::
    :description: :doc:`IntegerComparator <../code/api/pennylane.IntegerComparator>`
    :figure: _static/templates/arithmetic/integercomparator.png

.. gallery-item::
    :description: :doc:`OutPoly <../code/api/pennylane.OutPoly>`
    :figure: _static/templates/arithmetic/outpoly.png


.. raw:: html

        <div style='clear:both'></div>

Quantum Chemistry templates
---------------------------

Quantum chemistry templates define various quantum circuits used in variational algorithms
like VQE to perform quantum chemistry simulations.

.. gallery-item::
    :description: :doc:`AllSinglesDoubles <../code/api/pennylane.AllSinglesDoubles>`
    :figure: _static/templates/subroutines/all_singles_doubles.png

.. gallery-item::
    :description: :doc:`Basis Rotation <../code/api/pennylane.BasisRotation>`
    :figure: _static/templates/subroutines/basis_rotation.jpeg

.. gallery-item::
    :description: :doc:`GateFabric <../code/api/pennylane.GateFabric>`
    :figure: _static/templates/layers/gate_fabric_layer.png

.. gallery-item::
    :description: :doc:`UCCSD <../code/api/pennylane.UCCSD>`
    :figure: _static/templates/subroutines/uccsd.png

.. gallery-item::
    :description: :doc:`k-UpCCGSD <../code/api/pennylane.kUpCCGSD>`
    :figure: _static/templates/subroutines/kupccgsd.png

.. gallery-item::
    :description: :doc:`ParticleConservingU1 <../code/api/pennylane.ParticleConservingU1>`
    :figure: _static/templates/layers/particle_conserving_u1_thumbnail.png

.. gallery-item::
    :description: :doc:`ParticleConservingU2 <../code/api/pennylane.ParticleConservingU2>`
    :figure: _static/templates/layers/particle_conserving_u2.png

.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_tn:

Tensor networks
---------------

Tensor-network templates create quantum circuit architectures where circuit blocks
can be broadcast with the shape and connectivity of tensor networks.

.. gallery-item::
    :description: :doc:`Matrix Product State <../code/api/pennylane.MPS>`
    :figure: _static/templates/tensornetworks/MPS_template.png

.. gallery-item::
    :description: :doc:`Tree Tensor Network <../code/api/pennylane.TTN>`
    :figure: _static/templates/tensornetworks/TTN_template.png

.. gallery-item::
    :description: :doc:`Multi-scale Entanglement Renormalization Ansatz <../code/api/pennylane.MERA>`
    :figure: _static/templates/tensornetworks/MERA_template.png

.. raw:: html

        <div style='clear:both'></div>

Swap networks
--------------

Swap network templates perform qubit routing with linear cost, providing a quadratic advantage in
circuit depth for carrying out all pair-wise interactions between qubits.

.. gallery-item::
    :description: :doc:`Canonical 2-Complete Linear Swap Network <../code/api/pennylane.TwoLocalSwapNetwork>`
    :figure: _static/templates/swap_networks/ccl2.jpeg

.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_qchem:

Other subroutines
-----------------

Other useful templates which do not belong to the previous categories can be found here.

.. gallery-item::
    :description: :doc:`Instantaneous Quantum Polynomial Circuit <../code/api/pennylane.IQP>`
    :figure: _static/templates/subroutines/iqp.png

.. gallery-item::
    :description: :doc:`Grover Diffusion Operator <../code/api/pennylane.GroverOperator>`
    :figure: _static/templates/subroutines/grover.svg

.. gallery-item::
    :description: :doc:`Reflection Operator <../code/api/pennylane.Reflection>`
    :figure: _static/templates/subroutines/reflection.png

.. gallery-item::
    :description: :doc:`Amplitude Amplification <../code/api/pennylane.AmplitudeAmplification>`
    :figure: _static/templates/subroutines/ampamp.png

.. gallery-item::
    :description: :doc:`Interferometer <../code/api/pennylane.Interferometer>`
    :figure: _static/templates/subroutines/interferometer.png

.. gallery-item::
    :description: :doc:`FermionicSingleExcitation <../code/api/pennylane.FermionicSingleExcitation>`
    :figure: _static/templates/subroutines/single_excitation_unitary.png

.. gallery-item::
    :description: :doc:`FermionicDoubleExcitation <../code/api/pennylane.FermionicDoubleExcitation>`
    :figure: _static/templates/subroutines/double_excitation_unitary.png

.. gallery-item::
    :description: :doc:`ArbitraryUnitary <../code/api/pennylane.ArbitraryUnitary>`
    :figure: _static/templates/subroutines/arbitraryunitary.png

.. gallery-item::
  :description: :doc:`ApproxTimeEvolution <../code/api/pennylane.ApproxTimeEvolution>`
  :figure: _static/templates/subroutines/approx_time_evolution.png

.. gallery-item::
  :description: :doc:`QDrift <../code/api/pennylane.QDrift>`
  :figure: _static/templates/subroutines/qdrift.png

.. gallery-item::
  :description: :doc:`TrotterProduct <../code/api/pennylane.TrotterProduct>`
  :figure: _static/templates/subroutines/trotter_product.png

.. gallery-item::
  :description: :doc:`Permute <../code/api/pennylane.Permute>`
  :figure: _static/templates/subroutines/permute.png

.. gallery-item::
  :description: :doc:`QuantumPhaseEstimation <../code/api/pennylane.QuantumPhaseEstimation>`
  :figure: _static/templates/subroutines/qpe.svg

.. gallery-item::
  :description: :doc:`QuantumMonteCarlo <../code/api/pennylane.QuantumMonteCarlo>`
  :figure: _static/templates/subroutines/qmc.svg

.. gallery-item::
    :description: :doc:`QuantumFourierTransform <../code/api/pennylane.QFT>`
    :figure: _static/templates/subroutines/qft.svg

.. gallery-item::
    :description: :doc:`Approximate QFT<../code/api/pennylane.AQFT>`
    :figure: _static/templates/subroutines/aqft.png

.. gallery-item::
    :description: :doc:`CommutingEvolution <../code/api/pennylane.CommutingEvolution>`
    :figure: _static/templates/subroutines/commuting_evolution.png

.. gallery-item::
    :description: :doc:`HilbertSchmidt <../code/api/pennylane.HilbertSchmidt>`
    :figure: _static/templates/subroutines/hst.png

.. gallery-item::
    :description: :doc:`LocalHilbertSchmidt <../code/api/pennylane.LocalHilbertSchmidt>`
    :figure: _static/templates/subroutines/lhst.png

.. gallery-item::
    :description: :doc:`FlipSign operator<../code/api/pennylane.FlipSign>`
    :figure: _static/templates/subroutines/flip_sign.png

.. gallery-item::
    :description: :doc:`QSVT<../code/api/pennylane.QSVT>`
    :figure: _static/templates/subroutines/qsvt.png

.. gallery-item::
    :description: :doc:`GQSP<../code/api/pennylane.GQSP>`
    :figure: _static/templates/subroutines/gqsp.png

.. gallery-item::
    :description: :doc:`Select<../code/api/pennylane.Select>`
    :figure: _static/templates/subroutines/select_cropped.png

.. gallery-item::
    :description: :doc:`ControlledSequence<../code/api/pennylane.ControlledSequence>`
    :figure: _static/templates/subroutines/small_ctrl.png

.. gallery-item::
    :description: :doc:`FABLE <../code/api/pennylane.FABLE>`
    :figure: _static/templates/subroutines/fable.png

.. gallery-item::
    :description: :doc:`Qubitization <../code/api/pennylane.Qubitization>`
    :figure: _static/templates/qubitization/thumbnail_qubitization.png

.. gallery-item::
    :description: :doc:`QROM <../code/api/pennylane.QROM>`
    :figure: _static/templates/qrom/qrom_thumbnail.png

.. gallery-item::
    :description: :doc:`Bucket Brigade QRAM <../code/api/pennylane.BBQRAM>`
    :figure: _static/templates/qram/bbqram_thumbnail.png

.. gallery-item::
    :description: :doc:`Select Only QRAM <../code/api/pennylane.SelectOnlyQRAM>`
    :figure: _static/templates/qram/select_qram_thumbnail.png

.. gallery-item::
    :description: :doc:`Hybrid QRAM <../code/api/pennylane.HybridQRAM>`
    :figure: _static/templates/qram/hybrid_qram_thumbnail.png

.. gallery-item::
    :description: :doc:`PrepSelPrep <../code/api/pennylane.PrepSelPrep>`
    :figure: _static/templates/prepselprep/prepselprep.png

.. gallery-item::
    :description: :doc:`SelectPauliRot <../code/api/pennylane.SelectPauliRot>`
    :figure: _static/templates/subroutines/select_pauli_rot_cropped.png

.. gallery-item::
    :description: :doc:`TemporaryAND <../code/api/pennylane.TemporaryAND>`
    :figure: _static/templates/subroutines/temporary_and.png

.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_constr:


Parameter initializations
-------------------------

Templates that take a weight parameter tensor usually provide methods that return the shape of this tensor.
The shape can for example be used to construct random weights at the beginning of training.

.. code-block:: python

    import pennylane as qml
    from pennylane.templates import BasicEntanglerLayers
    from pennylane import numpy as np

    n_wires = 3
    dev = qml.device('default.qubit', wires=n_wires)

    @qml.qnode(dev)
    def circuit(weights):
        BasicEntanglerLayers(weights=weights, wires=range(n_wires))
        return qml.expval(qml.PauliZ(0))

    shape = BasicEntanglerLayers.shape(n_layers=2, n_wires=n_wires)
    np.random.seed(42)  # to make the result reproducible
    weights = np.random.random(size=shape)

>>> circuit(weights)
tensor(0.72588592, requires_grad=True)

If a template takes more than one weight tensor, the ``shape`` method returns a list of shape tuples.

Custom templates
----------------

Creating a custom template can be as simple as defining a function that creates operations and does not have a return
statement:

.. code-block:: python

    from pennylane import numpy as np

    def MyTemplate(a, b, wires):
        c = np.sin(a) + b
        qml.RX(c, wires=wires[0])

    n_wires = 3
    dev = qml.device('default.qubit', wires=n_wires)

    @qml.qnode(dev)
    def circuit(a, b):
        MyTemplate(a, b, wires=range(n_wires))
        return qml.expval(qml.PauliZ(0))

>>> circuit(2, 3)
-0.7195065654396784

.. note::

    Make sure that classical processing is compatible with the autodifferentiation library you are using. For example,
    if ``MyTemplate`` is to be used with the torch framework, we would have to change ``np.sin`` to ``torch.sin``.
    PennyLane's :mod:`math <pennylane.math>` library contains some advanced functionality for
    framework-agnostic processing.

As suggested by the camel-case naming, built-in templates in PennyLane are classes. Classes are more complex
data structures than functions, since they can define properties and methods of templates (such as gradient
recipes or matrix representations). Consult the :ref:`Contributing operators <contributing_operators>`
page to learn how to code up your own template class, and how to add it to the PennyLane template library.

Layering Function
-----------------

The layer function creates a new template by repeatedly applying a sequence of quantum
gates to a set of wires. You can import this function both via
``qml.layer`` and ``qml.templates.layer``.

.. autosummary::

    pennylane.layer
