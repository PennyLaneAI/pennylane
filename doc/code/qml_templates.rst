qml.templates
=============

This module provides a growing library of templates of common variational
circuit architectures that can be used to easily build,
evaluate, and train quantum nodes.

Embeddings
----------

.. automodapi:: pennylane.templates.embeddings
    :no-heading:
    :include-all-objects:

Layers
------

.. automodapi:: pennylane.templates.layers
    :no-heading:
    :include-all-objects:

Subroutines
-----------

.. automodapi:: pennylane.templates.subroutines
    :no-heading:
    :include-all-objects:

State preperations
------------------

.. automodapi:: pennylane.templates.state_preparations
    :no-heading:
    :include-all-objects:

Custom templates
----------------

Custom templates can be constructed using the template decorator.

.. autosummary::
    :toctree:

    pennylane.templates.template

Utility functions for input checks
----------------------------------

A number of utility functions are provided, which are useful for standard input checks
when constructing custom templates, for example to make sure that arguments have the right shape and types.

.. autosummary::
    :toctree:

    pennylane.templates.utils