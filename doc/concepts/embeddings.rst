.. _embeddings:

Quantum Embedding
==================

Quantum Embedding is the process of representing classical data as quantum states in a Hilbert space, i.e., *a quantum feature map*. This process is a crucial part of designing quantum algorithms and hence their complexity. For details, see [1][2]. PennyLane provides built-in embedding templates that can do the embedding for the user; see :ref: `<pennylane.templates.embeddings>`. 

Lets consider classical input data consisting of M examples, N dimensional each: :math:`\mathcal{D}=\{x^{1}, \ldots, x^{M}\}`. To embed this onto an n-bit quantum system, we can use various embedding techniques some of whcih are explained breifly below. 


Discrete Variable Embedding
----------------------------


Basis Embedding
^^^^^^^^^^^^^^^^^^^^

For basis embedding, classical data has to be in the form of binary strings; each data point is an N-bit binary string :math:`x^{m}=(b_1,\ldots,b_N)` with :math:`b_i \in \{0,1\}` for :math:`i=1,\ldots,N`. Assuming all features are repesented with unit binary precision (one bit), each input example :math:`x^{m}` can be directly mapped to :math:`\mid x^{m}\rangle` quantum state. **This means that the number of qubits in the quantum system, n, must be at least equal to N; N features are represented by N qubits.** The input data can be reprented in this computational basis as:

.. math:: \mid \mathcal{D} \rangle = \frac{1}{\sqrt{M}} \sum_{m=1}^{M} \mid x^{m} \rangle

For example, lets say :math:`x^{1}=01` and :math:`x^{2}=11`. So, example1 has feature1 in the off state and feature2 in the on state, while example2 has both feature1 and feature2 in the on state. The corresponding basis encoding uses 2 qubits to represent :math:`\mid x^{1} \rangle=\mid 01 \rangle` and :math:`\mid x^{2} \rangle=\mid 11 \rangle`: 

.. math:: \mid \mathcal{D} \rangle = (0) \mid 00 \rangle+\frac{1}{\sqrt{2}}\mid 01 \rangle+(0) \mid 10 \rangle+ \frac{1}{\sqrt{2}} \mid 11 \rangle

.. note:: There will be total of :math:`2^N` amplitudes. Given :math:`M \ll 2^N`, amplitude vectors will be sparse [1]. 

Amplitude Embedding
^^^^^^^^^^^^^^^^^^^^
This is a popular embedding technique where data is encoded into the amplitudes of a quantum state. This can be easily understood if we concatenate all the examples together into one vector, i.e., 
.. :math:: \alpha = [x^1_1,\ldots,x^1_N,x^2_1,\ldots,x^2_N,\ldots,x^M_1,\ldots,x^M_N]
 
This vector has to be normalized; :math:`\mid\alpha\mid^2`.The input data can be reprented in this computational basis as:

.. math:: \mid \mathcal{D} \rangle = \frac{1}{\sqrt{M}} \sum_{m=1}^{M} \mid x^{m} \rangle


Angle Embedding
^^^^^^^^^^^^^^^^^^^^




Continuous Variable Embedding
-------------------------------


Squeezing Embedding
^^^^^^^^^^^^^^^^^^^^


Displacement Embedding
^^^^^^^^^^^^^^^^^^^^^^^^^









.. rubric:: Footnotes
.. [1] Supervised Learning with Quantum Computers by Maria Schuld, Francesco Petruccione. Springer Nature Switzerland AG 2018 
.. [2]  Schuld & Killoran 2019 :cite:`schuld2018quantum`
