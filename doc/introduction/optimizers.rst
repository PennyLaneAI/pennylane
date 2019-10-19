.. role:: html(raw)
   :format: html

.. _intro_ref_opt:

Optimizers
==========

When using the standard NumPy interface, PennyLane offers some custom-made optimizers.
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

:html:`</div>`


.. warning::

  If using the :ref:`PennyLane PyTorch <torch_interf>`
  or the :ref:`PennyLane TensorFlow <tf_interf>` interfaces,
  `PyTorch optimizers <https://pytorch.org/docs/stable/optim.html>`_ and
  TensorFlow optimizers (found in the module ``tf.train``) should be used respectively.

