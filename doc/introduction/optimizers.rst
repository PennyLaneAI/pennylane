.. role:: html(raw)
   :format: html

.. _intro_ref_opt:

Optimizers
==========

Optimizers are objects which can be used to automatically update the parameters of a quantum 
or hybrid machine learning model. The optimizers you should use are dependent on your choice
of classical interface (NumPy, PyTorch, and TensorFlow), and are available from different access
points. 

Regardless of their origin, all optimizers provide the same core functionality, 
and PennyLane is fully compatible with all of them. 

NumPy Interface
^^^^^^^^^^^^^^^

When using the standard NumPy interface, PennyLane offers some built-in optimizers.
Some of these are specific to quantum optimization, such as the :mod:`~.QNGOptimizer`.

:html:`<div class="summary-table">`

.. autosummary::
    :nosignatures:

    ~pennylane.AdagradOptimizer
    ~pennylane.AdamOptimizer
    ~pennylane.GradientDescentOptimizer
    ~pennylane.MomentumOptimizer
    ~pennylane.NesterovMomentumOptimizer
    ~pennylane.QNGOptimizer
    ~pennylane.RMSPropOptimizer
    ~pennylane.RotosolveOptimizer
    ~pennylane.RotoselectOptimizer

:html:`</div>`

PyTorch Interface
^^^^^^^^^^^^^^^^^

If you are using the :ref:`PennyLane PyTorch interface <torch_interf>`, you should import one of the native
`PyTorch optimizers <https://pytorch.org/docs/stable/optim.html>`_ (found in ``torch.optim``).

TensorFlow Interface
^^^^^^^^^^^^^^^^^^^^

When using the :ref:`PennyLane TensorFlow interface <tf_interf>`, you will need to leverage one of 
the `TensorFlow optimizers <https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer>`_ 
(found in ``tf.keras.optimizers``).

