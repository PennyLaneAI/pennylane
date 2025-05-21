qml.bose
=========

Overview
--------

This module contains functions and classes for creating and manipulating bosonic operators.

.. currentmodule:: pennylane.bose

BoseWord and BoseSentence
---------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Class
     - Description
   * - :class:`~.BoseWord`
     - Dictionary used to represent a Bose word, a product of bosonic creation and
       annihilation operators, that can be constructed from a standard dictionary.
       
       The keys of the dictionary are tuples of two integers. The first integer represents the
       position of the creation/annihilation operator in the Bose word and the second integer
       represents the mode it acts on. The values of the dictionary are one of ``'+'`` or ``'-'``
       symbols that denote creation and annihilation operators, respectively.
   * - :class:`~.BoseSentence`
     - Dictionary used to represent a Bose sentence, a linear combination of Bose words,
       with the keys as BoseWord instances and the values correspond to coefficients.

Mapping to qubit operators
--------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Function
     - Description
   * - :func:`~.binary_mapping`
     - Convert a bosonic operator to a qubit operator using the standard-binary mapping.
       
       The mapping procedure is described in equations :math:`27-29` in `arXiv:1507.03271 <https://arxiv.org/pdf/1507.03271>`_.
   * - :func:`~.unary_mapping`
     - Convert a bosonic operator to a qubit operator using the unary mapping.
       
       The mapping procedure is described in `arXiv.1909.12847 <https://arxiv.org/abs/1909.12847>`_.
   * - :func:`~.christiansen_mapping`
     - Convert a bosonic operator to a qubit operator using the Christiansen mapping.
       
       This mapping assumes that the maximum number of allowed bosonic states is 2 and works only for
       Christiansen bosons defined in `J. Chem. Phys. 120, 2140 (2004) <https://pubs.aip.org/aip/jcp/article-abstract/120/5/2140/534128/A-second-quantization-formulation-of-multimode?redirectedFrom=fulltext>`_.