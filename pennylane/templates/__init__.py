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
.. _templates:

Circuit Templates
=================

**Module name:** :mod:`pennylane.templates`

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

PennyLane conceptually distinguishes two types of templates, **layer architectures** and **input embeddings**:

* Layer architectures, found in :mod:`pennylane.templates.layers`, define sequences of
  gates that are repeated like the layers in a neural network. They usually contain only trainable parameters.
* Embeddings, found in :mod:`pennylane.templates.embeddings`, encode input features into the quantum state
  of the circuit. Hence, they take a feature vector as an argument. These embeddings can also depend on
  trainable parameters, in which case the embedding is learnable.


The following templates are available:

.. rst-class:: contents local topic

.. toctree::
    :maxdepth: 3

    templates/layers
    templates/embeddings

Each trainable template has a dedicated function in :mod:`pennylane.init` which generates a list of
**randomly initialized** arrays for the trainable **parameters**:

.. rst-class:: contents local topic

.. toctree::
    :maxdepth: 3

    templates/init_parameters

"""