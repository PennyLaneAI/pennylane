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

.. note::
    #. Our short name for PennyLane code that is captured as jaxpr is *plxpr* (PennyLane-jaxpr).
    Program-capture and plxpr can be considered synonymous and interchangeable. 

    #. From here onwards, ``qml.capture.enable`` is assumed to be present within 
    the scope of code examples, unless otherwise stated. This ensures that program-capture
    is enabled.

.. _device_compatibility:

Device compatibility 
--------------------

Currently, ``default.qubit`` and ``lightning.qubit`` are compatible with program-capture.
In addition, ``default.qubit`` requires that ``wires`` be specified. This is in 
contrast to when program-capture is disabled, where automatic qubit management takes
place internally.

.. code-block:: python

    qml.capture.enable()

    @qml.qnode(qml.device('default.qubit'))
    def circuit():
        qml.Hadamard(0)
        return qml.state()

>>> circuit()
NotImplementedError: devices must specify wires for integration with plxpr capture.

.. _valid_data_types:

Valid JAX data types 
--------------------

.. _name_of_section:

Section title 
-------------

blah blah blah

.. code-block:: python
    # nice code block!!!!!!!!!

>>> print("hello plxpr")
hello plxpr

blah blah blah