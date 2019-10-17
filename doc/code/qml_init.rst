.. _api_qml_init:

qml.init
========

.. currentmodule:: pennylane.init

This module contains helper functions that generate initial parameters, for example
to use in templates.


Strongly Entangling Circuit
---------------------------

.. autosummary::
    strong_ent_layers_uniform
    strong_ent_layers_normal
    strong_ent_layer_uniform
    strong_ent_layer_normal

Random Circuit
--------------

.. autosummary::
    random_layers_uniform
    random_layers_normal
    random_layer_uniform
    random_layer_normal

CV Neural Network
-----------------

.. autosummary::
    cvqnn_layers_uniform
    cvqnn_layers_normal
    cvqnn_layer_uniform
    cvqnn_layer_normal

Interferometer
--------------

.. autosummary::
    interferometer_uniform
    interferometer_normal

.. toctree::
    :hidden:

    pennylane.init.cvqnn_layer_normal.rst
    pennylane.init.cvqnn_layers_normal.rst
    pennylane.init.cvqnn_layers_uniform.rst
    pennylane.init.cvqnn_layer_uniform.rst
    pennylane.init.interferometer_normal.rst
    pennylane.init.interferometer_uniform.rst
    pennylane.init.random_layer_normal.rst
    pennylane.init.random_layers_normal.rst
    pennylane.init.random_layers_uniform.rst
    pennylane.init.random_layer_uniform.rst
    pennylane.init.strong_ent_layer_normal.rst
    pennylane.init.strong_ent_layers_normal.rst
    pennylane.init.strong_ent_layers_uniform.rst
    pennylane.init.strong_ent_layer_uniform.rst