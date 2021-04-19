.. _contributing_templates:

Contributing templates
======================

The following steps will help you to add your favourite circuit ansatz to
PennyLane's :mod:`template <pennylane.template>` library.

.. note::

    PennyLane internally loosely distinguishes :doc:`different types </introduction/templates>` of templates, such as
    :mod:`embeddings <pennylane.templates.embeddings>`, :mod:`layers <pennylane.templates.layers>`,
    :mod:`state_preparations <pennylane.templates.state_preparations>` or
    :mod:`subroutines <pennylane.templates.subroutines>`. Below you need to replace ``<templ_type>`` with the
    correct template type.

Templates are just gates
~~~~~~~~~~~~~~~~~~~~~~~~

Conceptually, there is no difference in PennyLane between a template or ansatz and a :mod:`gate <pennylane.ops>`.
Both inherit from the :class:`Operation <pennylane.operation.Operation>` class, which has an `expand()` function
that can be used to define a decomposition into other gates. If a device does not recognise the name of the operation,
it calls the `expand` function which returns a :class:`pennylane.tape.QuantumTape` instance that
queues the decomposing gates.

For example, the following shows a simple template for a layer of of `RX` rotation gates:

.. code-block:: python

    import pennylane as qml
    from pennylane.operation import Operation, AnyWires

    class MyNewTemplate(Operation):

        num_params = 1
        num_wires = AnyWires
        par_domain = "A"  # note: this attribute will be deprecated soon

        def expand(self):

            # extract the weights as the first parameter
            # passed to this operation
            weights = self.parameters[0]

            # extract the length of the weights vector using
            # the interface-agnostic `math` module
            num_weights = qml.math.shape(weights)[0]

            with qml.tape.QuantumTape() as tape:

                for i in range(num_weights):
                    qml.RX(weights[i], wires=self.wires[i])

            return tape


The `num_params` and `num_wires` arguments determine that an instance of this template can be created
by passing a single parameter (of arbitrary shape and type), as well as an arbitrary number of wires:

>>> weights = np.array([0.1, 0.2, 0.3])
>>> MyNewTemplate(weights, wires=['a', 'b', 'd'])

As an `Operation`, templates can define other methods and attributes, such as a matrix representation,
a generator, or a a gradient rule.

.. note::

    In principle, templates could also inherit from the :class:`Observable <pennylane.operation.Observable>`
    class and define a sequence of diagonalising gates as an ansatz.

Classical pre-processing
~~~~~~~~~~~~~~~~~~~~~~~~

Templates often perform extensive pre-processing on the arguments they get.
The template should make as many arguments differentiable as possible.

Differentiable arguments can be expected to always be tensors
of the allowed :doc:`interfaces </introduction/interfaces>`, such as `tf.Variable`, or `pennylane.numpy.array`.
This means that we have to use interface-agnostic pre-processing methods inside the templates. A lot of functionality
is provided by the :mod:`pennylane.math` module - for example, the length of the weights in the code above
was computed with the `qml.math.shape(weights)` function.

Non-trivial pre-processing should be implemented by overwriting the `__init__` function of the `Operator` class.
This also allows us to define templates with more flexible signatures than the `(*params, wires)`
signature expected by the `Operator` class.

As an illustration, let us check that the first parameter in `MyNewTemplate` is one-dimensional,
apply a sine function to each weight, and invert the wires that the operation acts on.

.. code-block:: python

    def MyNewTemplate(Operation):

        num_params = 1
        num_wires = AnyWires
        par_domain = "A"  # note: this attribute will be deprecated soon

        def __init__(weights, raw_wires)

            shp = qml.math.shape(weights)
            if len(shp) != 1:
                raise ValueError("Expected one-dimensional weights tensor.")

            new_weights = qml.math.sin(weights)

            inverted_wires = wires[::-1]

            super().__init__(new_weights, wires=inverted_wires)


        def expand(self):

            weights = self.parameters[0]
            num_weights = qml.math.shape(weights)[0]

            with qml.tape.QuantumTape() as tape:
                for i in range(num_weights):
                    qml.RX(weights[i], wires=self.wires[i])

            return tape

The `parameters` and `wires` attributes used in the `expand` functions are now the `new_weights` and `inverted_wires`.


.. warning::

    Not all tensor types support iteration. Avoid expressions like `for w in weights` and
    rather iterate over ranges like `for i in range(num_weights)`.

.. note::

    Use multi-indexing where possible - expressions like `weights[6][5][2]` are usually a
    lot slower than `weights[6, 5, 2]`. 


Adding the template
~~~~~~~~~~~~~~~~~~~

Add the template by adding a new file ``my_new_template.py`` to the correct ``templates/<templ_type>/``
subdirectory. The file contains your new template class.

Make sure you consider the following:

* *Choose the name carefully.* Good names tell the user what a template is good for,
  or what architecture it implements. The class name (i.e., ``MyNewTemplate``) is written in camel case.

* *Explicit decompositions.* Try to implement the decomposition in the `expand` function
  without the use of convenient methods like the :func:`~.broadcast` function - this avoids
  unnecessary overhead.

* *Write an extensive docstring that explains how to use the template.* Include a sketch of the template (add the
  file to the ``doc/_static/templates/<templ_type>/`` directory). You can also display a small usage example
  at the beginning of the docstring.

  You should include an illustrative example of how to use the template before listing the arguments,
  headed by a bold **Example** title.
  If you want to explain the behaviour in more detail, add a section starting
  with the ``.. UsageDetails::`` directive at the end of the docstring,
  where you demonstrate with code examples how to use the templates with different
  settings, for example varying the number of wires, explaining keyword arguments and special cases.
  Use the docstring of one of the existing templates for inspiration, such as
  :func:`AmplitudeEmbedding <pennylane.templates.embeddings.AmplitudeEmbedding>`.

* **Input checks.** While checking the inputs of the template for consistency introduces an overhead,
  it is still advised to do some basic sanity checks, for example making sure that the shape of the
  parameters is correct.

Importing the new template
~~~~~~~~~~~~~~~~~~~~~~~~~~

Import the new template in ``templates/<templ_type>/__init__.py`` by adding the new line

.. code-block:: python

    from .mynewtemplate import MyNewTemplate

Adding your template to the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add your template to the documentation by adding a ``customgalleryitem`` to the correct layer type section in
``doc/introduction/templates.rst``:

.. code-block::

  .. customgalleryitem::
    :link: ../code/api/pennylane.templates.<templ_type>.MyNewTemplate.html
    :description: MyNewTemplate
    :figure: ../_static/templates/<templ_type>/my_new_template.png

.. note::

  This loads the image of the template added to ``doc/_static/templates/test_<templ_type>/`` in Step 1. Make sure that
  this image has the same dimensions and style as other template icons in the folder.

Adding tests
~~~~~~~~~~~~

Don't forget to add tests for your new template to the test suite. Create a separate file
``tests/templates/<templ_type>/test_my_new_template.py`` with all tests. You can get some inspiration from
 :mod:`existing tests <tests.templates.test_embeddings/test_qaoa_emb>`.

