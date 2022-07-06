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
    :description: :doc:`BasisStatePreparation <../code/api/pennylane.BasisStatePreparation>`
    :figure: _static/templates/state_preparations/basis.png

.. gallery-item::
    :description: :doc:`MottonenStatePreparation <../code/api/pennylane.MottonenStatePreparation>`
    :figure: _static/templates/state_preparations/mottonen.png

.. gallery-item::
    :description: :doc:`ArbitraryStatePreparation <../code/api/pennylane.ArbitraryStatePreparation>`
    :figure: _static/templates/subroutines/arbitrarystateprep.png

.. raw:: html

        <div style='clear:both'></div>

.. _intro_ref_temp_subroutines:

Quantum Chemistry templates
---------------------------

Quantum chemistry templates define various quantum circuits used in variational algorithms
like VQE to perform quantum chemistry simulations.

.. gallery-item::
    :description: :doc:`AllSinglesDoubles <../code/api/pennylane.AllSinglesDoubles>`
    :figure: _static/templates/subroutines/all_singles_doubles.png

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

.. _intro_ref_temp_qchem:

Other subroutines
-----------------

Other useful templates which do not belong to the previous categories can be found here.

.. gallery-item::
    :description: :doc:`Grover Diffusion Operator <../code/api/pennylane.GroverOperator>`
    :figure: _static/templates/subroutines/grover.svg

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

.. gallery-item::
    :description: :doc:`Broadcast (Single) <../code/api/pennylane.broadcast>`
    :figure: _static/templates/broadcast_single.png

.. gallery-item::
    :description: :doc:`Broadcast (Double) <../code/api/pennylane.broadcast>`
    :figure: _static/templates/broadcast_double.png

.. gallery-item::
    :description: :doc:`Broadcast (Double Odd) <../code/api/pennylane.broadcast>`
    :figure: _static/templates/broadcast_double_odd.png

.. gallery-item::
    :description: :doc:`Broadcast (Chain) <../code/api/pennylane.broadcast>`
    :figure: _static/templates/broadcast_chain.png

.. gallery-item::
    :description: :doc:`Broadcast (Ring) <../code/api/pennylane.broadcast>`
    :figure: _static/templates/broadcast_ring.png

.. gallery-item::
    :description: :doc:`Broadcast (Pyramid) <../code/api/pennylane.broadcast>`
    :figure: _static/templates/broadcast_pyramid.png

.. gallery-item::
    :description: :doc:`Broadcast (All-to-All) <../code/api/pennylane.broadcast>`
    :figure: _static/templates/broadcast_alltoall.png

.. gallery-item::
    :description: :doc:`Broadcast (Custom) <../code/api/pennylane.broadcast>`
    :figure: _static/templates/broadcast_custom.png

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
