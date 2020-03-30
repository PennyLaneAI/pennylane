How to add a new template
-------------------------

The following steps will help you to add your favourite circuit ansatz to
PennyLane's :mod:`template <pennylane.template>` library.

PennyLane internally distinguishes :ref:`different types <intro_ref_temp>` of templates, such as
:mod:`embeddings <pennylane.templates.embeddings>`, :mod:`layers <pennylane.templates.layers>`,
:mod:`state_preparations <pennylane.templates.state_preparations>` or
:mod:`soubroutines <pennylane.templates.soubroutines>`.
Here we will use the template ``MyNewTemplate`` as an example, and you need to replace ``<templ_type>`` with the
correct template type.

1. **Add the template** by adding a new file ``my_new_template.py`` to the correct ``templates/<templ_type>/``
   subdirectory, which contains your new template. For example, this is a very basic template applying an ``qml.RX``
   gate to each wire:

   .. code-block:: python

        import pennylane as qml
        from pennylane.templates import template  # import the decorator

        @template
        def MyNewTemplate(weights, wires):

            # Check that the inputs have the correct format
            # ...

            for wire, weight in zip(wires, weights):
                qml.RX(weight, wires=wire)

   A template is a function that defines a sequence of quantum gates (without measurements).
   Since the template is called within a :ref:`quantum function <intro_vcirc_qfunc>`,
   it can only contain information processing that is allowed
   inside a quantum functions.

   Make sure you consider the following:

   * To be consistent with other quantum operations (which are classes), the function name (i.e., ``MyNewTemplate``) is
     written in camel case. Choose the name carefully. Good names tell the user what a template is good for, or what architecture
     it implements.

   * A template is "registered" by using the :mod:`template <pennylane.template>` decorator ``@template``.
     This allows us to record the queue of operations of a template, which is very useful for testing:

     .. code-block:: python

        with qml.utils.OperationRecorder() as rec:
            MyNewTemplate(...)

        list_of_gates = rec.queue

   * Consider using the :func:`broadcasting <pennylane.templates.broadcasting>` function to make your code more concise.

   * Write an extensive docstring that explains how to use the template. Include a sketch of the template (add the
     file to the ``doc/_static/templates/<templ_type>/`` directory). You can also display a small usage example
     at the beginning of the docstring.

     At the end of the docstring, add a section starting with the ``.. UsageDetails::`` directive,
     where you demonstrate with code examples how to use the templates with different
     settings, for example varying the number of wires, explaining keyword arguments and special cases.
     For inspiration, check one of the existing templates, such as
     :func:`AmplitudeEmbedding <pennylane.templates.embeddings.AmplitudeEmbedding>`

   * Check the inputs to the template. You can use the functions provided in :mod:`utils <pennylane.templates.utils>`.
     Don't forget that arguments may be passed by the user to the qnode as positional or keyword arguments, and
     by using different interfaces (i.e., a input could be a ``numpy.ndarray`` or a list of
     :class:`Variable <pennylane.variable.Variable>`, depending on how the user uses the template).

2. **Import the new template** in ``templates/<templ_type>/__init__.py`` by adding the new line

   .. code-block:: python

        from .mynewtemplate import MyNewTemplate

3. **Add your template to the documentation** by adding a ``customgalleryitem`` to the correct layer type section in
   ``doc/introduction/templates.rst``:

   .. code-block::

     .. customgalleryitem::
        :link: ../code/api/pennylane.templates.<templ_type>.MyNewTemplate.html
        :description: MyNewTemplate
        :figure: ../_static/templates/<templ_type>/my_new_template.png

   .. note::

      This loads the image of the template added to ``doc/_static/templates`` in Step 1. Make sure that
      this image has the same dimensions and style than other template icons.

4. **Add tests** for your new template to the test suite.

   * Integration tests which check that your template can be called inside a quantum node, and that PennyLane can
     compute gradients with respect to differentiable parameters, are added to ``tests/test_templates.py``.
     Simply add your template to the fixtures (variables indicated by capital letters) to automatically
     run existing tests on your new template.

   * Add a new test class to ``tests/test_templates_<templ_type>.py`` that contains the unit tests for the template.
     Make sure you test all keyword arguments and edge cases like using a single wire.
