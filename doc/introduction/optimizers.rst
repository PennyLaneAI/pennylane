.. role:: html(raw)
   :format: html

.. _intro_optimizers:

.. currentmodule:: pennylane.optimize

Optimizers
==========

When using the standard NumPy interface, PennyLane offers some custom-made optimizers.
Some of these are specific to quantum optimization, such as the :mod:`QNGOptimizer`.

.. warning::

  If using the :ref:`PennyLane PyTorch <torch_interf>`
  or the :ref:`PennyLane TensorFlow <tf_interf>` interfaces,
  `PyTorch optimizers <https://pytorch.org/docs/stable/optim.html>`_ and
  TensorFlow optimizers (found in the module ``tf.train``) should be used respectively.


.. autosummary::
   AdagradOptimizer
   AdamOptimizer
   GradientDescentOptimizer
   MomentumOptimizer
   NesterovMomentumOptimizer
   RMSPropOptimizer
   QNGOptimizer



