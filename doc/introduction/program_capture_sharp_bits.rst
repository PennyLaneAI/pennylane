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
    #. From here onwards, ``qml.capture.enable`` is assumed to be present within 
    the scope of code examples, unless otherwise stated. This ensures that program-capture
    is enabled.

    #. Using program-capture requires that JAX be installed. Please consult the 
    JAX documentation for `installation instructions <https://docs.jax.dev/en/latest/installation.html>`__.
    
    #. Our short name for PennyLane code that is captured as jaxpr is *plxpr* (PennyLane-jaxpr).
    Program-capture and plxpr can be considered synonymous and interchangeable. 

.. _device_compatibility:

Device compatibility 
--------------------

Currently, ``default.qubit`` and ``lightning.qubit`` are compatible with program-capture.
In addition, ``default.qubit`` requires that ``wires`` be specified. This is in 
contrast to when program-capture is disabled, where automatic qubit management takes
place internally.

.. code-block:: python
    @qml.qnode(qml.device('default.qubit'))
    def circuit():
        qml.Hadamard(0)
        return qml.state()

>>> circuit()
NotImplementedError: devices must specify wires for integration with plxpr capture.

.. _valid_data_types:

Valid JAX data types 
--------------------

Because of the nature of creating and executing plxpr, it is *best practice* for 
all data types in arguments to quantum functions and QNodes, and positional arguments 
to gates to be JAX-compatible types: ``jax.numpy`` arrays and standard Python 
``int``\ s and ``float``\ s. For example, a ``list`` is not a valid JAX type for
the positional argument in :class:`~.MultiRZ`:

.. code-block:: python
    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.MultiRZ([0.1, 0.2], wires=[0, 1])
        return qml.expval(qml.X(0))

>>> circuit()
TypeError: Value [0.1, 0.2] with type <class 'list'> is not a valid JAX type

Providing a ``list`` as input to a quantum function or QNode is accepted in cases 
where the ``list`` is being indexed into, thereby retrieving a valid JAX type:

.. code-block:: python
    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(x):
        qml.RZ(x[0], wires=0)
        qml.RX(x[1], wires=1)
        return qml.expval(qml.X(0))

>>> circuit([0.1, 0.2])
Array(0., dtype=float32)

JAX-incompatible types, like Python ``range``\ s, are acceptable as keyword arguments:

.. code-block:: python
    dev = qml.device('default.qubit', wires=2)

    @qml.qnode(dev)
    def circuit(x, range_of_wires=None):
        for w in range_of_wires:
            qml.RZ(x[0], wires=w)
            qml.RX(x[1], wires=w)

        return qml.expval(qml.X(0))

>>> circuit([0.1, 0.2], range_of_wires=range(2))
Array(0., dtype=float32)

But, again, using JAX-compatible types wherever possible is recommended.

.. _name_of_section:

Section title 
-------------

blah blah blah

.. code-block:: python
    # nice code block!!!!!!!!!

>>> print("hello plxpr")
hello plxpr

blah blah blah