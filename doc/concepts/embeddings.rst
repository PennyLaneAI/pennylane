.. role:: html(raw)
   :format: html

.. _embeddings:

Quantum embeddings
===================

A *Quantum Embedding* represents classical data as quantum states in a Hilbert space, i.e., *a quantum feature map*. An embedding takes a classical datapoint :math:`x` and translates it into a set of gate parameters in a quantum circuit. This ends up creating a quantum state :math:`\mid \psi_x \rangle`. This process is a crucial part of designing quantum algorithms and affects their computational power. For more details, see :cite:`Mariabook2018` and :cite:`schuld2018quantum`. PennyLane provides built-in embedding templates that can do the embedding for the user; see :mod:`pennylane.templates.embeddings`. 


Let's consider classical input data consisting of :math:`M` examples, with :math:`N` features each: 

.. math:: \mathcal{D}=\{x^{1}, \ldots, x^{M}\}

where :math:`x^{i}` is a :math:`N` dimensional vector for :math:`i=1,\ldots,M`. To embed this data into :math:`n` quantum subsystems (:math:`n` qubits *or* :math:`n` qumodes for discrete and continuous variable quantum computing, respectively), we can use various embedding techniques, some of which are explained briefly below. 


Basis Embedding
^^^^^^^^^^^^^^^^^^^^

<<<<<<< HEAD
For the basis embedding, classical data has to be in the form of binary strings. The embedded quantum state is the bit-wise translation of a binary string to the corresponding states of the quantum subsystems. For example, :math:`x=1001` is represented by the 4-qubit quantum state :math:`\mid 1001 \rangle`. Hence, one bit of classical information is represented by one quantum subsystem.

Let's consider the classical dataset :math:`\mathcal{D}` mentioned above. For basis embedding, each example has to be a N-bit binary string; :math:`x^{m}=(b_1,\ldots,b_N)` with :math:`b_i \in \{0,1\}` for :math:`i=1,\ldots,N`. Assuming all features are repesented with unit binary precision (one bit), each input example :math:`x^{m}` can be directly mapped to the quantum state :math:`\mid x^{m}\rangle`. **This means that the number of quantum subsystems,** :math:`\bm{n}` **, must be at least equal to** :math:`\bm{N}`. The input data can be represented in the computational basis as

.. math:: \mid \mathcal{D} \rangle = \frac{1}{\sqrt{M}} \sum_{m=1}^{M} \mid x^{m} \rangle.

For example, let's say we have a classical data set containing two examples :math:`x^{1}=01` and :math:`x^{2}=11`. The corresponding basis encoding uses two qubits to represent :math:`\mid x^{1} \rangle=\mid 01 \rangle` and :math:`\mid x^{2} \rangle=\mid 11 \rangle`: 

.. math:: \mid \mathcal{D} \rangle = (0) \mid 00 \rangle+\frac{1}{\sqrt{2}}\mid 01 \rangle+(0) \mid 10 \rangle+ \frac{1}{\sqrt{2}} \mid 11 \rangle

.. note:: For :math:`N` bits, there are :math:`2^N` possible basis states. Given :math:`M \ll 2^N`, the basis embedding of :math:`\mathcal{D}` will be sparse :cite:`Mariabook2018`. 


Amplitude Embedding
^^^^^^^^^^^^^^^^^^^^

In amplitude embedding technique, data is encoded into the amplitudes of a quantum state. This can be easily understood if we concatenate all the input examples :math:`x^m` together into one vector, i.e., 

.. math:: \alpha = C_n \{ x^1_1, \ldots, x^1_N, x^2_1, \ldots, x^2_N, \ldots, x^M_1, \ldots, x^M_N \},
 
where :math:`C_n` is the normalization constant; this vector must be normalized :math:`\mid\alpha\mid^2=1`. The input data can now be represented in the computational basis as

.. math:: \mid \mathcal{D} \rangle = \sum_{i=1}^{2^n} \alpha_i \mid i \rangle,

where :math:`\alpha_i` are the elements of amplitude vector :math:`\alpha` and :math:`\mid i \rangle` are the computational basis vectors corresponding to the number of quantum subsystems that are used. The number of amplitudes to be encoded are :math:`N \times M`. As the number of quantum subsystems, :math:`n`, provide :math:`2^n` amplitudes, **amplitude embedding requires** :math:`\bm{n \geq \log_2({NM})}`  **quantum subsystems.**  


.. note:: If the total number of amplitudes to embed, i.e., :math:`N \times M`, is less less than :math:`2^n`, *non-informative* constants can be *padded* to :math:`\alpha `:cite:`Mariabook2018`. For example, if we have three example with two features each, we have six amplitudes to embed. However, as we have to use at least three qubits, we have to concatenate two constants at the end of :math:`\alpha`. 

.. important:: There are many other embedding techniques with similar embedding protocols. For example, *Squeezing embedding* and *Displacement embedding* are used with continuous variable quantum computing models, where classical information is encoded in the squeezing and displacement operator parameters. *Hamiltonian embedding* uses an implicit technique by encoding information in the evolution of a quantum system :cite:`Mariabook2018`.  


  
