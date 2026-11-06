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
    :figure: _static/templates/gallery/AmplitudeEmbedding.svg

.. gallery-item::
    :description: :doc:`AngleEmbedding <../code/api/pennylane.AngleEmbedding>`
    :figure: _static/templates/gallery/AngleEmbedding.svg

.. gallery-item::
    :description: :doc:`BasisEmbedding <../code/api/pennylane.BasisEmbedding>`
    :figure: _static/templates/gallery/BasisEmbedding.svg


.. gallery-item::
    :description: :doc:`IQPEmbedding <../code/api/pennylane.IQPEmbedding>`
    :figure: _static/templates/gallery/IQPEmbedding.svg

.. gallery-item::
    :description: :doc:`QAOAEmbedding <../code/api/pennylane.QAOAEmbedding>`
    :figure: _static/templates/gallery/QAOAEmbedding.svg

.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_lay:

Layer templates
---------------

Layer architectures define sequences of trainable gates that are repeated like the layers in a
neural network. Note that arbitrary templates or operations can also be repeated using the
:func:`~pennylane.layer` function.

.. gallery-item::
    :description: :doc:`RandomLayers <../code/api/pennylane.RandomLayers>`
    :figure: _static/templates/gallery/RandomLayers.svg

.. gallery-item::
    :description: :doc:`StronglyEntanglingLayers <../code/api/pennylane.StronglyEntanglingLayers>`
    :figure: _static/templates/gallery/StronglyEntanglingLayers.svg

.. gallery-item::
    :description: :doc:`SimplifiedTwoDesign <../code/api/pennylane.SimplifiedTwoDesign>`
    :figure: _static/templates/gallery/SimplifiedTwoDesign.svg

.. gallery-item::
    :description: :doc:`BasicEntanglerLayers <../code/api/pennylane.BasicEntanglerLayers>`
    :figure: _static/templates/gallery/BasicEntanglerLayers.svg

.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_stateprep:

State Preparations
------------------

State preparation templates transform the zero state :math:`|0\dots 0 \rangle` to another initial
state. In contrast to embeddings that can in principle be used anywhere in a circuit,
state preparation is typically used as the first operation.

.. gallery-item::
    :description: :doc:`MottonenStatePreparation <../code/api/pennylane.MottonenStatePreparation>`
    :figure: _static/templates/gallery/MottonenStatePreparation.svg

.. gallery-item::
    :description: :doc:`ArbitraryStatePreparation <../code/api/pennylane.ArbitraryStatePreparation>`
    :figure: _static/templates/gallery/ArbitraryStatePreparation.svg

.. gallery-item::
    :description: :doc:`CosineWindow <../code/api/pennylane.CosineWindow>`
    :figure: _static/templates/gallery/CosineWindow.svg

.. gallery-item::
    :description: :doc:`QROMStatePreparation <../code/api/pennylane.QROMStatePreparation>`
    :figure: _static/templates/gallery/QROMStatePreparation.svg

.. gallery-item::
    :description: :doc:`MPSPrep <../code/api/pennylane.MPSPrep>`
    :figure: _static/templates/gallery/MPSPrep.svg

.. gallery-item::
    :description: :doc:`MultiplexerStatePreparation <../code/api/pennylane.MultiplexerStatePreparation>`
    :figure: _static/templates/gallery/MultiplexerStatePreparation.svg

.. gallery-item::
    :description: :doc:`SumOfSlatersPrep <../code/api/pennylane.SumOfSlatersPrep>`
    :figure: _static/templates/gallery/SumOfSlatersPrep.svg

.. gallery-item::
    :description: :doc:`PartialUnaryStatePreparation <../code/api/pennylane.PartialUnaryStatePreparation>`
    :figure: _static/templates/gallery/PartialUnaryStatePreparation.svg

.. gallery-item::
    :description: :doc:`UniformPrep <../code/api/pennylane.UniformPrep>`
    :figure: _static/templates/gallery/UniformPrep.svg

.. gallery-item::
    :description: :doc:`AliasSampling <../code/api/pennylane.AliasSampling>`
    :figure: _static/templates/gallery/AliasSampling.svg

.. gallery-item::
    :description: :doc:`PhaseGradientStatePrep <../code/api/pennylane.PhaseGradientStatePrep>`
    :figure: _static/templates/gallery/PhaseGradientStatePrep.svg


.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_subroutines:

Arithmetic templates
--------------------

Quantum arithmetic templates enable in-place and out-place modular operations such
as addition, multiplication and exponentiation.

.. gallery-item::
    :description: :doc:`PhaseAdder <../code/api/pennylane.PhaseAdder>`
    :figure: _static/templates/gallery/PhaseAdder.svg

.. gallery-item::
    :description: :doc:`Adder <../code/api/pennylane.Adder>`
    :figure: _static/templates/gallery/Adder.svg

.. gallery-item::
    :description: :doc:`SemiAdder <../code/api/pennylane.SemiAdder>`
    :figure: _static/templates/gallery/SemiAdder.svg

.. gallery-item::
    :description: :doc:`Incrementer <../code/api/pennylane.Incrementer>`
    :figure: _static/templates/gallery/Incrementer.svg

.. gallery-item::
    :description: :doc:`OutAdder <../code/api/pennylane.OutAdder>`
    :figure: _static/templates/gallery/OutAdder.svg

.. gallery-item::
    :description: :doc:`Multiplier <../code/api/pennylane.Multiplier>`
    :figure: _static/templates/gallery/Multiplier.svg

.. gallery-item::
    :description: :doc:`OutMultiplier <../code/api/pennylane.OutMultiplier>`
    :figure: _static/templates/gallery/OutMultiplier.svg

.. gallery-item::
    :description: :doc:`SignedOutMultiplier <../code/api/pennylane.SignedOutMultiplier>`
    :figure: _static/templates/gallery/SignedOutMultiplier.svg

.. gallery-item::
    :description: :doc:`OutSquare <../code/api/pennylane.OutSquare>`
    :figure: _static/templates/gallery/OutSquare.svg

.. gallery-item::
    :description: :doc:`SignedOutSquare <../code/api/pennylane.SignedOutSquare>`
    :figure: _static/templates/gallery/SignedOutSquare.svg

.. gallery-item::
    :description: :doc:`ModExp <../code/api/pennylane.ModExp>`
    :figure: _static/templates/gallery/ModExp.svg

.. gallery-item::
    :description: :doc:`IntegerComparator <../code/api/pennylane.IntegerComparator>`
    :figure: _static/templates/gallery/IntegerComparator.svg

.. gallery-item::
    :description: :doc:`LeftClassicalComparator <../code/api/pennylane.LeftClassicalComparator>`
    :figure: _static/templates/gallery/LeftClassicalComparator.svg

.. gallery-item::
    :description: :doc:`LeftQuantumComparator <../code/api/pennylane.LeftQuantumComparator>`
    :figure: _static/templates/gallery/LeftQuantumComparator.svg

.. gallery-item::
    :description: :doc:`OutPoly <../code/api/pennylane.OutPoly>`
    :figure: _static/templates/gallery/OutPoly.svg


.. raw:: html

        <div style='clear:both'></div>

Quantum Chemistry templates
---------------------------

Quantum chemistry templates define various quantum circuits used in variational algorithms
like VQE to perform quantum chemistry simulations.

.. gallery-item::
    :description: :doc:`AllSinglesDoubles <../code/api/pennylane.AllSinglesDoubles>`
    :figure: _static/templates/gallery/AllSinglesDoubles.svg

.. gallery-item::
    :description: :doc:`Basis Rotation <../code/api/pennylane.BasisRotation>`
    :figure: _static/templates/gallery/BasisRotation.svg

.. gallery-item::
    :description: :doc:`GateFabric <../code/api/pennylane.GateFabric>`
    :figure: _static/templates/gallery/GateFabric.svg

.. gallery-item::
    :description: :doc:`UCCSD <../code/api/pennylane.UCCSD>`
    :figure: _static/templates/gallery/UCCSD.svg

.. gallery-item::
    :description: :doc:`k-UpCCGSD <../code/api/pennylane.kUpCCGSD>`
    :figure: _static/templates/gallery/kUpCCGSD.svg

.. gallery-item::
    :description: :doc:`ParticleConservingU1 <../code/api/pennylane.ParticleConservingU1>`
    :figure: _static/templates/gallery/ParticleConservingU1.svg

.. gallery-item::
    :description: :doc:`ParticleConservingU2 <../code/api/pennylane.ParticleConservingU2>`
    :figure: _static/templates/gallery/ParticleConservingU2.svg

.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_tn:

Tensor networks
---------------

Tensor-network templates create quantum circuit architectures where circuit blocks
can be broadcast with the shape and connectivity of tensor networks.

.. gallery-item::
    :description: :doc:`Matrix Product State <../code/api/pennylane.MPS>`
    :figure: _static/templates/gallery/MPS.svg

.. gallery-item::
    :description: :doc:`Tree Tensor Network <../code/api/pennylane.TTN>`
    :figure: _static/templates/gallery/TTN.svg

.. gallery-item::
    :description: :doc:`Multi-scale Entanglement Renormalization Ansatz <../code/api/pennylane.MERA>`
    :figure: _static/templates/gallery/MERA.svg

.. raw:: html

        <div style='clear:both'></div>

Swap networks
--------------

Swap network templates perform qubit routing with linear cost, providing a quadratic advantage in
circuit depth for carrying out all pair-wise interactions between qubits.

.. gallery-item::
    :description: :doc:`Canonical 2-Complete Linear Swap Network <../code/api/pennylane.TwoLocalSwapNetwork>`
    :figure: _static/templates/gallery/TwoLocalSwapNetwork.svg

.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_qchem:

Other subroutines
-----------------

Other useful templates which do not belong to the previous categories can be found here.

.. gallery-item::
    :description: :doc:`Instantaneous Quantum Polynomial Circuit (IQP) <../code/api/pennylane.IQP>`
    :figure: _static/templates/gallery/IQP.svg

.. gallery-item::
    :description: :doc:`Grover Diffusion Operator <../code/api/pennylane.GroverOperator>`
    :figure: _static/templates/gallery/GroverOperator.svg

.. gallery-item::
    :description: :doc:`Reflection Operator <../code/api/pennylane.Reflection>`
    :figure: _static/templates/gallery/Reflection.svg

.. gallery-item::
    :description: :doc:`Amplitude Amplification <../code/api/pennylane.AmplitudeAmplification>`
    :figure: _static/templates/gallery/AmplitudeAmplification.svg

.. gallery-item::
    :description: :doc:`FermionicSingleExcitation <../code/api/pennylane.FermionicSingleExcitation>`
    :figure: _static/templates/gallery/FermionicSingleExcitation.svg

.. gallery-item::
    :description: :doc:`FermionicDoubleExcitation <../code/api/pennylane.FermionicDoubleExcitation>`
    :figure: _static/templates/gallery/FermionicDoubleExcitation.svg

.. gallery-item::
    :description: :doc:`ArbitraryUnitary <../code/api/pennylane.ArbitraryUnitary>`
    :figure: _static/templates/gallery/ArbitraryUnitary.svg

.. gallery-item::
  :description: :doc:`ApproxTimeEvolution <../code/api/pennylane.ApproxTimeEvolution>`
  :figure: _static/templates/gallery/ApproxTimeEvolution.svg

.. gallery-item::
  :description: :doc:`QDrift <../code/api/pennylane.QDrift>`
  :figure: _static/templates/gallery/QDrift.svg

.. gallery-item::
  :description: :doc:`TrotterProduct <../code/api/pennylane.TrotterProduct>`
  :figure: _static/templates/gallery/TrotterProduct.svg

.. gallery-item::
  :description: :doc:`TrotterCDF <../code/api/pennylane.TrotterCDF>`
  :figure: _static/templates/gallery/TrotterCDF.svg

.. gallery-item::
  :description: :doc:`TrotterCGF <../code/api/pennylane.TrotterCGF>`
  :figure: _static/templates/gallery/TrotterCGF.svg

.. gallery-item::
  :description: :doc:`TrotterVibronic <../code/api/pennylane.TrotterVibronic>`
  :figure: _static/templates/gallery/TrotterVibronic.svg

.. gallery-item::
  :description: :doc:`Permute <../code/api/pennylane.Permute>`
  :figure: _static/templates/gallery/Permute.svg

.. gallery-item::
  :description: :doc:`QuantumPhaseEstimation <../code/api/pennylane.QuantumPhaseEstimation>`
  :figure: _static/templates/gallery/QuantumPhaseEstimation.svg

.. gallery-item::
  :description: :doc:`QuantumMonteCarlo <../code/api/pennylane.QuantumMonteCarlo>`
  :figure: _static/templates/gallery/QuantumMonteCarlo.svg

.. gallery-item::
    :description: :doc:`QuantumFourierTransform <../code/api/pennylane.QFT>`
    :figure: _static/templates/gallery/QFT.svg

.. gallery-item::
    :description: :doc:`Approximate QFT<../code/api/pennylane.AQFT>`
    :figure: _static/templates/gallery/AQFT.svg

.. gallery-item::
    :description: :doc:`CommutingEvolution <../code/api/pennylane.CommutingEvolution>`
    :figure: _static/templates/gallery/CommutingEvolution.svg

.. gallery-item::
    :description: :doc:`HilbertSchmidt <../code/api/pennylane.HilbertSchmidt>`
    :figure: _static/templates/gallery/HilbertSchmidt.svg

.. gallery-item::
    :description: :doc:`LocalHilbertSchmidt <../code/api/pennylane.LocalHilbertSchmidt>`
    :figure: _static/templates/gallery/LocalHilbertSchmidt.svg

.. gallery-item::
    :description: :doc:`FlipSign operator<../code/api/pennylane.FlipSign>`
    :figure: _static/templates/gallery/FlipSign.svg

.. gallery-item::
    :description: :doc:`QSVT<../code/api/pennylane.QSVT>`
    :figure: _static/templates/gallery/QSVT.svg

.. gallery-item::
    :description: :doc:`GQSP<../code/api/pennylane.GQSP>`
    :figure: _static/templates/gallery/GQSP.svg

.. gallery-item::
    :description: :doc:`Select<../code/api/pennylane.Select>`
    :figure: _static/templates/gallery/Select.svg

.. gallery-item::
    :description: :doc:`ControlledSequence<../code/api/pennylane.ControlledSequence>`
    :figure: _static/templates/gallery/ControlledSequence.svg

.. gallery-item::
    :description: :doc:`FABLE <../code/api/pennylane.FABLE>`
    :figure: _static/templates/gallery/FABLE.svg

.. gallery-item::
    :description: :doc:`Qubitization <../code/api/pennylane.Qubitization>`
    :figure: _static/templates/gallery/Qubitization.svg

.. gallery-item::
    :description: :doc:`OneBodyBlockEncoding <../code/api/pennylane.OneBodyBlockEncoding>`
    :figure: _static/templates/gallery/OneBodyBlockEncoding.svg

.. gallery-item::
    :description: :doc:`QROM <../code/api/pennylane.QROM>`
    :figure: _static/templates/gallery/QROM.svg

.. gallery-item::
    :description: :doc:`Bucket Brigade QRAM <../code/api/pennylane.BBQRAM>`
    :figure: _static/templates/gallery/BBQRAM.svg

.. gallery-item::
    :description: :doc:`Select Only QRAM <../code/api/pennylane.SelectOnlyQRAM>`
    :figure: _static/templates/gallery/SelectOnlyQRAM.svg

.. gallery-item::
    :description: :doc:`Hybrid QRAM <../code/api/pennylane.HybridQRAM>`
    :figure: _static/templates/gallery/HybridQRAM.svg

.. gallery-item::
    :description: :doc:`Flip Flop QRAM <../code/api/pennylane.FFQRAM>`
    :figure: _static/templates/gallery/FFQRAM.svg

.. gallery-item::
    :description: :doc:`PrepSelPrep <../code/api/pennylane.PrepSelPrep>`
    :figure: _static/templates/gallery/PrepSelPrep.svg

.. gallery-item::
    :description: :doc:`SelectPauliRot <../code/api/pennylane.SelectPauliRot>`
    :figure: _static/templates/gallery/SelectPauliRot.svg

.. gallery-item::
    :description: :doc:`TemporaryAND <../code/api/pennylane.TemporaryAND>`
    :figure: _static/templates/gallery/TemporaryAND.svg

.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_constr:


Parameter initializations
-------------------------

Templates that take a weight parameter tensor usually provide methods that return the shape of this tensor.
The shape can for example be used to construct random weights at the beginning of training.

.. code-block:: python

    import pennylane as qp
    from pennylane.templates import BasicEntanglerLayers
    from pennylane import numpy as np

    n_wires = 3
    dev = qp.device('default.qubit', wires=n_wires)

    @qp.qnode(dev)
    def circuit(weights):
        BasicEntanglerLayers(weights=weights, wires=range(n_wires))
        return qp.expval(qp.PauliZ(0))

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
        qp.RX(c, wires=wires[0])

    n_wires = 3
    dev = qp.device('default.qubit', wires=n_wires)

    @qp.qnode(dev)
    def circuit(a, b):
        MyTemplate(a, b, wires=range(n_wires))
        return qp.expval(qp.PauliZ(0))

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
``qp.layer`` and ``qp.templates.layer``.

.. autosummary::

    pennylane.layer
