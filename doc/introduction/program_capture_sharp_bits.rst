.. role:: html(raw)
   :format: html

.. _intro_ref_program_capture_sharp_bits:

Program-capture sharp bits and debugging tips
=============================================

Program-capture is PennyLane's new internal representation for hyrbid quantum-classical 
programs that maintains the same familiar feeling frontend that you're used to, 
while providing better performance, harmonious integration with just-in-time compilation 
frameworks like `Catalyst <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__ 
(:func:`~.qjit`) and JAX-jit, and compact representations of programs that preserve 
their structure.

Program-capture in PennyLane is propped up by JAX's internal representation of programs 
called `jaxpr <https://docs.jax.dev/en/latest/jaxpr.html>`__, which works by leveraging 
the Python interpreter to extract the core elements of programs and capture them 
into a light-weight language. 

In part because of PennyLane's NumPy-like syntax and functionality, quantum programs 
written with PennyLane translate nicely into the language of jaxpr, letting your 
quantum-classical programs burn the same fuel that powers JAX and just-in-time compilation.

Our goal with program-capture is to support *more* than just the core features of the 
PennyLane you have come to know and love, but there are some **quirks and restrictions 
to be aware of while we strive towards that ideal**. In this document, we provide 
an overview of said constraints.
