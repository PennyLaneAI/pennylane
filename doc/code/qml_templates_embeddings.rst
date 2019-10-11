.. _api_qml_temp_emb:

qml.templates.embeddings
========================

.. currentmodule:: pennylane.templates.embeddings

This module provides quantum circuit architectures that can embed classical data into a quantum state.

CV embeddings
-------------

.. autosummary::
    SqueezingEmbedding
    DisplacementEmbedding

Qubit embeddings
----------------

.. autosummary::
    AmplitudeEmbedding
    AngleEmbedding
    BasisEmbedding




.. note::

    To make the signature of templates resemble other quantum operations used in
    quantum circuits, we treat them as classes here, even though technically
    they are functions.

.. toctree::
    :hidden:

    pennylane.templates.embeddings.AmplitudeEmbedding.rst
    pennylane.templates.embeddings.AngleEmbedding.rst
    pennylane.templates.embeddings.BasisEmbedding.rst
    pennylane.templates.embeddings.DisplacementEmbedding.rst
    pennylane.templates.embeddings.SqueezingEmbedding.rst
