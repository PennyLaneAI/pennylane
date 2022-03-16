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

.. customgalleryitem::
    :link: ../code/api/pennylane.AmplitudeEmbedding.html
    :description: AmplitudeEmbedding
    :figure: ../_static/templates/embeddings/amplitude.png

.. customgalleryitem::
    :link: ../code/api/pennylane.AngleEmbedding.html
    :description: AngleEmbedding
    :figure: ../_static/templates/embeddings/angle.png

.. customgalleryitem::
    :link: ../code/api/pennylane.BasisEmbedding.html
    :description: BasisEmbedding
    :figure: ../_static/templates/embeddings/basis.png

.. customgalleryitem::
    :link: ../code/api/pennylane.DisplacementEmbedding.html
    :description: DisplacementEmbedding
    :figure: ../_static/templates/embeddings/displacement.png

.. customgalleryitem::
    :link: ../code/api/pennylane.IQPEmbedding.html
    :description: IQPEmbedding
    :figure: ../_static/templates/embeddings/iqp.png

.. customgalleryitem::
    :link: ../code/api/pennylane.QAOAEmbedding.html
    :description: QAOAEmbedding
    :figure: ../_static/templates/embeddings/qaoa.png

.. customgalleryitem::
    :link: ../code/api/pennylane.SqueezingEmbedding.html
    :description: SqueezingEmbedding
    :figure: ../_static/templates/embeddings/squeezing.png

.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_lay:

Layer templates
---------------

Layer architectures define sequences of trainable gates that are repeated like the layers in a
neural network. Note that arbitrary templates or operations can also be repeated using the
:func:`~pennylane.layer` function.

.. customgalleryitem::
    :link: ../code/api/pennylane.CVNeuralNetLayers.html
    :description: CVNeuralNetLayers
    :figure: ../_static/templates/layers/cvqnn.png

.. customgalleryitem::
    :link: ../code/api/pennylane.RandomLayers.html
    :description: RandomLayers
    :figure: ../_static/templates/layers/random.png

.. customgalleryitem::
    :link: ../code/api/pennylane.StronglyEntanglingLayers.html
    :description: StronglyEntanglingLayers
    :figure: ../_static/templates/layers/strongly_entangling.png

.. customgalleryitem::
    :link: ../code/api/pennylane.SimplifiedTwoDesign.html
    :description: SimplifiedTwoDesign
    :figure: ../_static/templates/layers/simplified_two_design.png

.. customgalleryitem::
    :link: ../code/api/pennylane.BasicEntanglerLayers.html
    :description: BasicEntanglerLayers
    :figure: ../_static/templates/layers/basic_entangler.png

.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_stateprep:

State Preparations
------------------

State preparation templates transform the zero state :math:`|0\dots 0 \rangle` to another initial
state. In contrast to embeddings that can in principle be used anywhere in a circuit,
state preparation is typically used as the first operation.

.. customgalleryitem::
    :link: ../code/api/pennylane.BasisStatePreparation.html
    :description: BasisStatePreparation
    :figure: ../_static/templates/state_preparations/basis.png

.. customgalleryitem::
    :link: ../code/api/pennylane.MottonenStatePreparation.html
    :description: MottonnenStatePrep
    :figure: ../_static/templates/state_preparations/mottonen.png

.. customgalleryitem::
    :link: ../code/api/pennylane.ArbitraryStatePreparation.html
    :description: ArbitraryStatePreparation
    :figure: ../_static/templates/subroutines/arbitrarystateprep.png

.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_subroutines:

Quantum Chemistry templates
---------------------------

Quantum chemistry templates define various quantum circuits used in variational algorithms
like VQE to perform quantum chemistry simulations.

.. customgalleryitem::
    :link: ../code/api/pennylane.AllSinglesDoubles.html
    :description: AllSinglesDoubles
    :figure: ../_static/templates/subroutines/all_singles_doubles.png

.. customgalleryitem::
    :link: ../code/api/pennylane.GateFabric.html
    :description: GateFabric
    :figure: ../_static/templates/layers/gate_fabric_layer.png

.. customgalleryitem::
    :link: ../code/api/pennylane.UCCSD.html
    :description: UCCSD
    :figure: ../_static/templates/subroutines/uccsd.png

.. customgalleryitem::
    :link: ../code/api/pennylane.kUpCCGSD.html
    :description: k-UpCCGSD
    :figure: ../_static/templates/subroutines/kupccgsd.png

.. customgalleryitem::
    :link: ../code/api/pennylane.ParticleConservingU1.html
    :description: ParticleConservingU1
    :figure: ../_static/templates/layers/particle_conserving_u1_thumbnail.png

.. customgalleryitem::
    :link: ../code/api/pennylane.ParticleConservingU2.html
    :description: ParticleConservingU2
    :figure: ../_static/templates/layers/particle_conserving_u2.png

.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_tn:

Tensor networks
-------------------------

Tensor-network templates create quantum circuit architectures where circuit blocks
can be broadcast with the shape and connectivity of tensor networks.

.. customgalleryitem::
    :link: ../code/api/pennylane.MPS.html
    :description: Matrix Product State
    :figure: ../_static/templates/tensornetworks/MPS_template.png

.. customgalleryitem::
    :link: ../code/api/pennylane.TTN.html
    :description: Tree Tensor Network
    :figure: ../_static/templates/tensornetworks/TTN_template.png

.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_qchem:

Other subroutines
-----------------

Other useful templates which do not belong to the previous categories can be found here.

.. customgalleryitem::
    :link: ../code/api/pennylane.GroverOperator.html
    :description: Grover Diffusion Operator
    :figure: ../_static/templates/subroutines/grover.svg

.. customgalleryitem::
    :link: ../code/api/pennylane.Interferometer.html
    :description: Interferometer
    :figure: ../_static/templates/subroutines/interferometer.png

.. customgalleryitem::
    :link: ../code/api/pennylane.FermionicSingleExcitation.html
    :description: FermionicSingleExcitation
    :figure: ../_static/templates/subroutines/single_excitation_unitary.png

.. customgalleryitem::
    :link: ../code/api/pennylane.FermionicDoubleExcitation.html
    :description: FermionicDoubleExcitation
    :figure: ../_static/templates/subroutines/double_excitation_unitary.png

.. customgalleryitem::
    :link: ../code/api/pennylane.ArbitraryUnitary.html
    :description: ArbitraryUnitary
    :figure: ../_static/templates/subroutines/arbitraryunitary.png

.. customgalleryitem::
  :link: ../code/api/pennylane.ApproxTimeEvolution.html
  :description: ApproxTimeEvolution
  :figure: ../_static/templates/subroutines/approx_time_evolution.png

.. customgalleryitem::
  :link: ../code/api/pennylane.Permute.html
  :description: Permute
  :figure: ../_static/templates/subroutines/permute.png

.. customgalleryitem::
  :link: ../code/api/pennylane.QuantumPhaseEstimation.html
  :description: QuantumPhaseEstimation
  :figure: ../_static/templates/subroutines/qpe.svg

.. customgalleryitem::
  :link: ../code/api/pennylane.QuantumMonteCarlo.html
  :description: QuantumMonteCarlo
  :figure: ../_static/templates/subroutines/qmc.svg

.. customgalleryitem::
    :link: ../code/api/pennylane.QFT.html
    :description: QuantumFourierTransform
    :figure: ../_static/templates/subroutines/qft.svg

.. customgalleryitem::
    :link: ../code/api/pennylane.CommutingEvolution.html
    :description: CommutingEvolution
    :figure: ../_static/templates/subroutines/commuting_evolution.png

.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_constr:

Broadcasting function
---------------------

PennyLane offers a broadcasting function to easily construct templates: :func:`~.broadcast`
takes either quantum gates or templates and applies them to wires in a specific pattern.

.. warning::

    While the broadcasting function can make template construction very convenient, it
    adds an overhead and is therefore not recommended when speed is a major concern.

.. customgalleryitem::
    :link: ../code/api/pennylane.broadcast.html
    :description: broadcast (single)
    :figure: ../_static/templates/broadcast_single.png

.. customgalleryitem::
    :link: ../code/api/pennylane.broadcast.html
    :description: broadcast (double)
    :figure: ../_static/templates/broadcast_double.png

.. customgalleryitem::
    :link: ../code/api/pennylane.broadcast.html
    :description: broadcast (double_odd)
    :figure: ../_static/templates/broadcast_double_odd.png

.. customgalleryitem::
    :link: ../code/api/pennylane.broadcast.html
    :description: broadcast (chain)
    :figure: ../_static/templates/broadcast_chain.png

.. customgalleryitem::
    :link: ../code/api/pennylane.broadcast.html
    :description: broadcast (ring)
    :figure: ../_static/templates/broadcast_ring.png

.. customgalleryitem::
    :link: ../code/api/pennylane.broadcast.html
    :description: broadcast (pyramid)
    :figure: ../_static/templates/broadcast_pyramid.png

.. customgalleryitem::
    :link: ../code/api/pennylane.broadcast.html
    :description: broadcast (all-to-all)
    :figure: ../_static/templates/broadcast_alltoall.png

.. customgalleryitem::
    :link: ../code/api/pennylane.broadcast.html
    :description: broadcast (custom)
    :figure: ../_static/templates/broadcast_custom.png

.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_init:

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
    np.random.seed(42)  # to make the result reproducable
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
