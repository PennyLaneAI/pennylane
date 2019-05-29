.. _embeddings:

Quantum Embedding
==================

Quantum Embedding is the process of representing classical data as quantum states in a Hilbert space, i.e., *a quantum feature map*. This process is a crucial part of designing quantum algorithms and hence their complexity. For details, see [#]_ and [#]_. PennyLane provides built-in embedding templates that can do the embedding for the user; see :ref: `<pennylane.templates.embeddings>`. 

Lets consider classical input data consisting of M examples, N dimensional each: :math:`\mathcal{D}=\{x^{1}, \ldots, x^{M}\}`. To embed this onto an n-bit quantum system (n qubits *or* n qumodes for discrete and continuous variable quantum computing, respectively), we can use various embedding techniques some of whcih are explained breifly below. 


Basis Embedding
^^^^^^^^^^^^^^^^^^^^

For basis embedding, classical data has to be in the form of binary strings; each data point is an N-bit binary string :math:`x^{m}=(b_1,\ldots,b_N)` with :math:`b_i \in \{0,1\}` for :math:`i=1,\ldots,N`. Assuming all features are repesented with unit binary precision (one bit), each input example :math:`x^{m}` can be directly mapped to :math:`\mid x^{m}\rangle` quantum state. **This means that the number of quantum subsystems, n, must be at least equal to N; N features are represented by N qubits.** The input data can be reprented in the computational basis as:

.. math:: \mid \mathcal{D} \rangle = \frac{1}{\sqrt{M}} \sum_{m=1}^{M} \mid x^{m} \rangle

For example, lets say :math:`x^{1}=01` and :math:`x^{2}=11`. So, example1 has feature1 in the off state and feature2 in the on state, while example2 has both feature1 and feature2 in the on state. The corresponding basis encoding uses 2 qubits to represent :math:`\mid x^{1} \rangle=\mid 01 \rangle` and :math:`\mid x^{2} \rangle=\mid 11 \rangle`: 

.. math:: \mid \mathcal{D} \rangle = (0) \mid 00 \rangle+\frac{1}{\sqrt{2}}\mid 01 \rangle+(0) \mid 10 \rangle+ \frac{1}{\sqrt{2}} \mid 11 \rangle

.. note:: There will be total of :math:`2^N` amplitudes. Given :math:`M \ll 2^N`, amplitude vectors will be sparse [1]_. 


Amplitude Embedding
^^^^^^^^^^^^^^^^^^^^
This is a popular embedding technique where data is encoded into the amplitudes of a quantum state. This can be easily understood if we concatenate all the input examples :math:`x^m` together into one vector, i.e., 

.. math:: \alpha = N \{ x^1_1, \ldots, x^1_N, x^2_1, \ldots, x^2_N, \ldots, x^M_1, \ldots, x^M_N \}
 
where :math:`N` is the normalization constant; this vector has to be normalized; :math:`\mid\alpha\mid^2`. The input data can now be reprented in the computational basis as:

.. math:: \mid \mathcal{D} \rangle = \sum_{i=1}^{2^n} \alpha_i \mid i \rangle

where :math:`\alpha_i` are the elements of amplitude vector :math:`\alpha` and :math:`\mid i \rangle` are the computational basis vectors corresponding to the number of qubits that are used. The number of amplitudes to be encoded are :math:`N \times M`. As the number of quantum subsystems, n, provide :math:`2^n` amplitudes, amplitude embedding requires :math:`n \sim \log_2({NM})` quantum subsystems.  


.. note:: If the total number of amplitudes to embed :math:`N \times M` are less than :math:`2^n`, *non-informative* constants can be *padded* to :math:`\alpha` [1]_. For example, if we have three example with two features each, we have six amplitudes to embed. However, as we have to use at least three qubits, we have to concatenate two constants at the end of :math:`\alpha`. 


.. rubric:: References
.. [#] Supervised Learning with Quantum Computers by Maria Schuld, Francesco Petruccione. Springer Nature Switzerland AG 2018 
.. [#]  Schuld & Killoran 2019 :cite:`schuld2018quantum`
