.. role:: html(raw)
   :format: html

.. _intro_optimizers:

.. currentmodule:: pennylane.optimize

Optimizers
==========

When using the standard NumPy interface, PennyLane offers some custom-made optimizers.
Some of these are specific to quantum optimization, such as the :mod:`QNGOptimizer`.

.. toctree::
    :maxdepth: 2

   ../code/generated/pennylane.optimize.AdagradOptimizer
   ../code/generated/pennylane.optimize.AdamOptimizer
   ../code/generated/pennylane.optimize.GradientDescentOptimizer
   ../code/generated/pennylane.optimize.MomentumOptimizer
   ../code/generated/pennylane.optimize.NesterovMomentumOptimizer
   ../code/generated/pennylane.optimize.RMSPropOptimizer
   ../code/generated/pennylane.optimize.QNGOptimizer


.. warning::

  If using the :ref:`PennyLane PyTorch <torch_interf>`
  or the :ref:`PennyLane TensorFlow <tf_interf>` interfaces,
  `PyTorch optimizers <https://pytorch.org/docs/stable/optim.html>`_ and
  TensorFlow optimizers (found in the module ``tf.train``) should be used respectively.