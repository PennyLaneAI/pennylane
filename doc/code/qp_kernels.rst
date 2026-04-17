qp.kernels
===========

Overview
--------

This subpackage defines functions that relate to quantum kernel methods.
On one hand this includes functions to call a quantum kernel systematically
on training and test datasets to obtain the *kernel matrix*.
On the other hand it provides postprocessing methods for those kernel
matrices which can be used to mitigate device noise and sampling errors.

.. currentmodule:: pennylane.kernels

.. automodapi:: pennylane.kernels
    :no-heading:
    :no-main-docstr:
    :include-all-objects:


Description
-----------

Given a kernel

.. math ::

    k: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}, \quad
    (x_1, x_2)\mapsto k(x_1, x_2)

the kernel matrix of :math:`k` on a training dataset
:math:`\{(x_1, y_1),\cdots (x_n, y_n)\}` with :math:`x_i\in\mathbb{R}^d`
and :math:`y_i\in\{-1, 1\}` is defined as

.. math ::

    K_{ij} = k(x_i, x_j).

For valid kernels, this is a real symmetric positive semi-definite matrix.
We also define the *ideal kernel matrix* for the training dataset which
perfectly predicts whether two points have identical labels or not:

.. math ::

    K^\ast_{ij} = y_i y_j

We can measure the similarity between :math:`K` and :math:`K^\ast`,
through the *kernel polarity* which can be expressed as the Frobenius inner
product between the two matrices:

.. math ::

    \operatorname{P}(k) = \langle K^\ast, K \rangle_F = \sum_{i,j=1}^n y_i y_j k(x_i, x_j)

Additionally, there is the *kernel-target alignment*, which is the normalized
counterpart to the kernel polarity:

.. math ::

    \operatorname{TA}(k) &= \frac{P(k)}{\lVert K^\ast \rVert_F\;\lVert K \rVert_F}\\
        \lVert K\rVert_F &= \sqrt{\sum_{i,j=1}^n k(x_i, x_j)^2}\\
   \lVert K^\ast\rVert_F &= \sqrt{\sum_{i,j=1}^n (y_iy_j)^2}

For datasets with different numbers of training points per class the labels are rescaled
by the number of datapoints in the respective class to avoid that kernel polarity and
kernel-target alignment are dominated by the properties of the kernel for just a single class.

Given a callable kernel function, all these quantities can readily be computed
using the methods in this module.