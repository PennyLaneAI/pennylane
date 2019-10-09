.. _docs_templates_layers:

qml.templates.layers
====================

.. currentmodule:: pennylane.templates.layers

This module contains templates for trainable 'layers' of quantum gates.

Classes
-------

Single layer

.. autosummary::
    StronglyEntanglingLayer
    RandomLayer
    CVNeuralNetLayer
    Interferometer

Multiple layers

.. autosummary::
    StronglyEntanglingLayers
    RandomLayers
    CVNeuralNetLayers

.. note::

    To make the signature of templates resemble other quantum operations used in
    quantum circuits, we treat them as classes here, even though technically
    they are functions.
