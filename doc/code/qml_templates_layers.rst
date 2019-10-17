.. _api_qml_temp_lay:

qml.templates.layers
====================

.. currentmodule:: pennylane.templates.layers

This module contains templates for trainable 'layers' of quantum gates.

CV layers
---------

Single layer

.. autosummary::
    :toctree: api

    CVNeuralNetLayer
    Interferometer

Multiple layers

.. autosummary::
    :toctree: api

    CVNeuralNetLayers

Qubit layers
------------

Single layer

.. autosummary::
    :toctree: api

    StronglyEntanglingLayer
    RandomLayer

Multiple layers

.. autosummary::
    :toctree: api

    StronglyEntanglingLayers
    RandomLayers

.. note::

    To make the signature of templates resemble other quantum operations used in
    quantum circuits, we treat them as classes here, even though technically
    they are functions.
