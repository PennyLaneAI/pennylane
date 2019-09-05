# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
templates
=========

**Module name**: pennylane.templates

.. currentmodule:: pennylane.templates

This module provides a growing library of templates of common quantum
machine learning circuit architectures that can be used to easily build,
evaluate, and train more complex quantum machine learning models. In the
quantum machine learning literature, such architectures are commonly known as an
**ansatz**.

.. note::

    Templates are constructed out of **structured combinations** of the :mod:`quantum operations <pennylane.ops>`
    provided by PennyLane. This means that **template functions can only be used within a
    valid** :mod:`pennylane.qnode`.

PennyLane conceptually distinguishes two types of templates, **layer architectures** and **input embeddings**.
Most templates are complemented by functions that provide an array of random **initial parameters**.

Layer templates
---------------

Layer architectures, found in :mod:`pennylane.templates.layers`, define sequences of gates that are
repeated like the layers in a neural network. They usually contain only trainable parameters.

The following layer templates are available:

.. currentmodule:: pennylane.templates.layers

+------------------------------------+-------------------------------------------------------------------+
| :ref:`Strongly Entangling Circuit  | Consists of a block of single qubit rotations applied to every    |
| <StronglyEntanglingLayer>`         | qubit, followed by a block of 2-qubit entangling gates            |
+------------------------------------+-------------------------------------------------------------------+
| :ref:`Random Circuit               | Consists of randomly chosen single- and two-qubit gates applied   |
| <RandomLayer>`                     | to randomly chosen qubits.                                        |
+------------------------------------+-------------------------------------------------------------------+


Embedding templates
-------------------

Embeddings, found in :mod:`pennylane.templates.embeddings`, encode input features into the quantum state
of the circuit. Hence, they take a feature vector as an argument. These embeddings can also depend of
trainable parameters, in which case the embedding is learnable.

The following embedding templates are available:

.. currentmodule:: pennylane.templates.embeddings

+------------------------------------+-------------------------------------------------------------------+
| :ref:`Amplitude Embedding          | Prepares a quantum state whose amplitude vector corresponds to    |
| <AmplitudeEmbedding>`              | the input feature vector.                                         |
+------------------------------------+-------------------------------------------------------------------+
| :ref:`Basis Embedding              | Prepares a computational basis state that corresponds to the      |
| <BasisEmbedding>`                  | binary input sequence.                                            |
+------------------------------------+-------------------------------------------------------------------+
| :ref:`Squeezing Embedding          | Encodes an input into the squeezing paparameters of quantum       |
| <SqueezingEmbedding>`              | modes.                                                            |
+------------------------------------+-------------------------------------------------------------------+
| :ref:`Displacement Embedding       | Encodes an input into the squeezing paparameters of quantum       |
| <DisplacementEmbedding>`           | modes.                                                            |
+------------------------------------+-------------------------------------------------------------------+

Parameter initializations
-------------------------

Each trainable template has a dedicated function in :mod:`pennylane.init` which generates a list of
**randomly initialized** arrays for the trainable **parameters**:

"""