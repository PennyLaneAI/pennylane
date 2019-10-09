.. _docs_templates_embeddings:

qml.templates.embeddings
========================

.. currentmodule:: pennylane.templates.embeddings

This module provides quantum circuit architectures that can embed classical data into a quantum state.

Classes
-------

.. autosummary::
    AmplitudeEmbedding
    AngleEmbedding
    BasisEmbedding
    SqueezingEmbedding
    DisplacementEmbedding


.. note::

    To make the signature of templates resemble other quantum operations used in
    quantum circuits, we treat them as classes here, even though technically
    they are functions.