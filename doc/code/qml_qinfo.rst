qml.qinfo
=========

Overview
--------

.. warning::

    The ``qinfo`` module is deprecated and scheduled to be removed in v0.40. Most quantum information transforms
    are available as measurement processes (see the :mod:`~pennylane.measurements` module for more details).
    Additionally, the transforms are also available as standalone functions in the :mod:`~pennylane.math` and
    :mod:`~pennylane.gradients` modules.

This module provides a collection of methods to return quantum information quantities from :class:`~.QNode`
returning :func:`~pennylane.state`.

.. currentmodule:: pennylane.qinfo

Transforms
----------

.. automodapi:: pennylane.qinfo.transforms
    :no-heading:
    :no-inherited-members:
    :skip: transform
