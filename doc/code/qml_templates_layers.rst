.. _api_qml_temp_lay:

qml.templates.layers
====================

.. currentmodule:: pennylane.templates.layers

This module contains templates for trainable 'layers' of quantum gates.

CV layers
---------

Single layer

.. autosummary::
    CVNeuralNetLayer
    Interferometer

Multiple layers

.. autosummary::
    CVNeuralNetLayers

Qubit layers
------------

Single layer

.. autosummary::
    StronglyEntanglingLayer
    RandomLayer

Multiple layers

.. autosummary::
    StronglyEntanglingLayers
    RandomLayers


.. toctree::
    :hidden:

    pennylane.templates.layers.CVNeuralNetLayer.rst
    pennylane.templates.layers.CVNeuralNetLayers.rst
    pennylane.templates.layers.Interferometer.rst
    pennylane.templates.layers.RandomLayer.rst
    pennylane.templates.layers.RandomLayers.rst
    pennylane.templates.layers.StronglyEntanglingLayer.rst
    pennylane.templates.layers.StronglyEntanglingLayers.rst


.. note::

    To make the signature of templates resemble other quantum operations used in
    quantum circuits, we treat them as classes here, even though technically
    they are functions.
