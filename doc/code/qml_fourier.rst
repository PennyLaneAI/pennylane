qml.fourier
===========

This module contains methods for computing Fourier series representations of
quantum circuits.

Fourier series of quantum circuits
----------------------------------

Consider a quantum circuit that depends on a parameter vector :math:`x` with
length :math:`N`. The circuit involves application of some unitary operations
:math:`U(x)`, and then measurement of a particular expectation value
:math:`P`. Analytically, the expectation value can be computed as

.. math::

   \langle P \rangle = \langle 0 | U^\dagger (x) P U(x) |0\rangle = \langle
   \psi(x) | P | \psi (x)\rangle.

This output is simply a function :math:`f(x) = \langle \psi(x) | P | \psi
(x)\rangle`. Notably, it is a periodic function of the parameters, and
it can thus be expressed as a multidimensional Fourier series: 

.. math::

    f(x) = \sum \limits_{n_1\in \Omega_1} \dots \sum \limits_{n_N \in \Omega_N}
    c_{n_1,\dots, n_N} e^{-i x_1 n_1} \dots e^{-i x_N n_N},

where :math:`n_i` are integer-valued frequencies, :math:`\Omega_i` are the set
of available values for the integer frequencies, and the
:math:`c_{n_1,\ldots,n_N}` are Fourier coefficients.

As a simple example, consider ``simple_circuit`` below, which is a function of a
single parameter.

.. code::

    import pennylane as qml
    from pennylane import numpy as np

    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def simple_circuit(x):
        qml.RX(x[0], wires=0)
        qml.RY(x[0], wires=1)
        qml.CNOT(wires=[1, 0])
        return qml.expval(qml.PauliZ(0))

We can mathematically evaluate the expectation value of this function to be
:math:`\langle Z \rangle = 0.5 + 0.5 \cos(2x)`. Thus, the Fourier coefficients
of this function are :math:`c_0 = 0.5`, :math:`c_1 = c^*_{-1} = 0`, and \
:math:`c_2 = c^*_{-2} = 0.25`.

The PennyLane ``fourier`` module enables calculation of two important aspects of
the Fourier series: the *spectrum*, i.e., the accessible frequencies where the
Fourier coefficients may be non-zero; and the values of the *coefficients*
themselves. Knowledge of the coefficients and accessible spectra is important
for the study of expressivity of quantum circuits, as described in `Schuld,
Sweke and Meyer (2020) <https://arxiv.org/abs/2008.08605>`__ and `Vidal and
Theis, 2019 <https://arxiv.org/abs/1901.11434>`__ -- the more coefficients
available to a quantum model, the larger the class of functions that model can
represent, potentially leading to greater utility for quantum machine learning
applications.

Calculating the Fourier spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The frequency spectra can be calculated using the :func:`~.pennylane.fourier.spectrum`
function. As one may be interested only in the spectra of a subset of the input
parameters, **only the spectrum of differentiable parameters will be calculated**.

.. code::

   >>> from pennylane.fourier import spectrum
   >>> x = np.array([0.5], requires_grad=True)
   >>> spectrum(simple_circuit)
   {tensor(0.5, requires_grad=True): [-2.0, -1.0, 0.0, 1.0, 2.0]}

The set of available frequencies above matches the result we obtained by hand of
:math:`-2, 0, 2`. Note that the :func:`~.pennylane.fourier.spectrum` function
returns the *maximum* possible spectra with respect to the given inputs. It is
possible that certain Fourier coefficients will nevertheless be zero.


Calculating the Fourier coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Knowledge of the frequency spectra enables us to compute the Fourier
coefficients themselves. This can be done using the
:func:`~.pennylane.fourier.fourier_coefficients` function:

.. code::

   >>> from pennylane.fourier import fourier_coefficients
   >>> coeffs = fourier_coefficients(simple_circuit, len(x), 2)
   >>> print(np.round(coeffs, decimals=4))
   [0.5 +0.j 0.  -0.j 0.25+0.j 0.25+0.j 0.  -0.j]

The input to the :func:`~.pennylane.fourier.fourier_coefficients` function are
the function in question, the length of the input vector, and the maximum
frequency for which to calculate the coefficients (also known as the *degree*). (For a quantum function of
multiple inputs with varying order, it may be necessary to use a wrapper
function to ensure the Fourier coefficients are calculated with respect to the
correct input values.)

Internally, the coefficients are computed using numpy's `discrete Fourier
transform <https://numpy.org/doc/stable/reference/generated/numpy.fft.fftn.html>`__
function. The order of the coefficients in the output thus follows the standard
output ordering, i.e., :math:`[c_0, c_1, c_2, c_{-2}, c_{-1}]`, and similarly
for multiple dimensions.

.. note::

   If a frequency lower than the true maximum frequency is used to calculate the
   coefficients, it is possible that `aliasing
   <https://en.wikipedia.org/wiki/Aliasing>`__ will be present in the
   output. Thus, it is good practice to first estimate the maximum frequency of
   a quantum circuit using the :func:`~.pennylane.fourier.spectrum` function. In
   addition, the coefficient calculator also contains a simple anti-aliasing
   filter that will cut off frequencies higher than a given threshold. This can
   be configured by setting the ``lowpass_filter`` option to ``True``, and optionally
   specifying the ``frequency_threshold`` argument (if none is specified, 2 times
   the specified degree will be used as the threshold).

.. automodapi:: pennylane.fourier
    :include-all-objects:
    :no-inheritance-diagram:

