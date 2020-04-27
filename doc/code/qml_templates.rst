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

The template decorator can used to register a quantum function as a template.

.. autosummary::
    :toctree:

    pennylane.templates.template

Broadcasting function
---------------------

The broadcast function creates a new template by broadcasting templates (or normal gates) over wires in a
predefined pattern.

.. autosummary::
    :toctree:

    pennylane.templates.broadcast

Utility functions for input checks
----------------------------------

.. automodapi:: pennylane.templates.utils
    :no-heading:
    :include-all-objects:
