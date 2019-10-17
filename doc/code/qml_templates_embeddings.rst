.. _api_qml_temp_emb:

qml.templates.embeddings
========================

.. currentmodule:: pennylane.templates.embeddings

This module provides quantum circuit architectures that can embed classical data into a quantum state.

CV embeddings
-------------

.. autosummary::
    :toctree: api

    SqueezingEmbedding
    DisplacementEmbedding

Qubit embeddings
----------------

.. autosummary::
    :toctree: api

    AmplitudeEmbedding
    AngleEmbedding
    BasisEmbedding


.. note::

    To make the signature of templates resemble other quantum operations used in
    quantum circuits, we treat them as classes here, even though technically
    they are functions.
