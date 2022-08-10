Documentation
=============

Good documentation is just as important, if not more important, than the code itself.
**No matter how good the code is**, without decent documentation, users are deterred from
using the library, and developers are less likely to contribute.

This document describes the requirements, recommendations, and guides for writing documentation
for PennyLane, as well as details for contributing to and building the PennyLane documentation.


Overview
--------

The PennyLane documentation consists of three main sections:

The introductory 'quickstarts'.
    These are short, referential guides designed to quickly
    guide new users around the core functionality of PennyLane, and show available functions and classes.
    These are summaries, *not* full-form tutorials; all :ref:`code examples are minimal <code_examples>`,
    and readers should be directed to function/class docstrings for more details.

    All introductory quickstarts are in ``doc/introduction``, formatted as reStructuredText documents.

Development guides.
    Similar to the introductory quickstarts, but focused on information
    regarding how to contribute to the PennyLane codebase.

    Development guides are in ``doc/development``, formatted as reStructuredText documents.

The API.
    Automatically generated from docstrings in the PennyLane source code. Code stubs, one per
    module, are placed in ``doc/code``. The Sphinx extension
    `sphinx-automodapi <https://github.com/astropy/sphinx-automodapi>`__ is then used
    to automatically generate API documentation pages for public classes and functions
    in ``doc/code/api``.

    PennyLane displays the API using the **import path**---rather than absolute
    path---of the documented code object.

.. note::

    Long-form tutorials, as well as quantum machine learning theory and background information,
    is not part of the PennyLane documentation. Instead, this content is available over in the
    `XanaduAI/QML <https://github.com/XanaduAI/qml>`_ GitHub repository (viewable at
    https://pennylane.ai/qml).


Punctuation and spelling
------------------------

* Use Canadian spellings. That is, "centre" instead of "center", "realize" instead of "realise",
  "acknowledgement" instead of "acknowledgment", "colour" instead of "color", etc.

* Use complete sentences with capitalization and punctuation. The one exemption
  is when describing arguments, return values, attributes, and exceptions; sentence fragments
  may be used for short descriptions:

  .. code-block:: rest

      wires (List[int]): subsystems the operation is applied to
      diff_method (str or None): the method of differentiation to use in the created QNode

  Multi-sentence argument descriptions, or longer sentence fragments with mid-sentence punctuation marks,
  should use full capitalization and punctuation:

  .. code-block:: rest

      device (Device, Sequence[Device]): Corresponding device(s) where the
          ``QNodeCollection`` should be executed. This can either be a single device, or a list
          of devices of length ``len(observables)``.

* Comments may be written more informally than docstrings, as long as consistency and clarity are maintained.
  Capitalization and punctuation should be used with multi-sentence comments to aid with readability.


.. _docstrings:

Docstrings
----------

Documentation strings (docstrings) describe the use and functionality of modules, functions,
classes, and variables. Docstrings in PennyLane are enclosed within triple-quotes (``"""``), and are
written using `reStructuredText (ReST) <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
markup, and are formatted `Google-style <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#google-vs-numpy>`_.

Docstrings are a part of Python; an objects' docstring is stored under the ``__doc__``
attribute, and accessible to the user via the ``help()`` function. In addition, docstrings
are automatically extracted and rendered by Sphinx, and inserted into the
:doc:`API </code/qml>` section of the documentation.

Guiding principles
~~~~~~~~~~~~~~~~~~

Below, we list some general guidelines, before discussing required docstring styles and formatting
for various PennyLane objects.

* **Be clear and concise, and avoid redundant information.** Favour clarity over completeness; too
  much information can distract from conveying the functionality of the code unit.
  Assume the reader has basic domain knowledge, and provide links to internal/external resources
  where deemed necessary.

* **Be mindful with ReST markup**. Overly complex formatting can make the content difficult
  to read for users viewing the docstring in a terminal/Jupyter environment. Try to also avoid
  complex formatting such as tables, which can be difficult to maintain over time.

* **Do not assume the reader can view the accompanying code.** The reader will most likely
  be viewing the docstring via the Sphinx-rendered website, or in a terminal/Jupyter environment.
  For documenting how the code itself works, favour comments.

* **Cross reference code units where applicable.** Use the roles ``:class:``, ``:func:``, ``:attr:``,
  ``:meth:``, and ``:mod:`` where applicable to have Sphinx automatically include a cross-reference
  to the specified code object.

  For example:

  .. code-block:: rest

      The recommended method of creating a QNodeCollection is via :func:`~.map`.

  Here, ``~`` instructs Sphinx to display just the object name, as opposed to the full absolute path
  of the object (``pennylane.map()``). To use custom text, you may use the syntax
  ``:func:`map function <~.map>```.

* **Cross reference internal links.** Internal links should *not* use the URL of the built documentation,
  this is unmaintainable and URLs might change in the future. If linking to a particular page, always use
  ``:doc:`relative/path/filename``` or ``:doc:`/absolute/path/filename``` instead of ``:ref:``. Only
  use ``:ref:`` if linking to a particular subsection, and only when necessary.


Modules
~~~~~~~

Every module should begin with the license boilerplate, followed by a module docstring.

The module docstring must contain a short, single sentence summary of the module, followed
by an (optional) multi-line description of the contents and usage.

.. code-block:: python

    """
    Single sentence that summarizes the module and its contents.

    Optional multi-line description. Make sure to focus on functionality first and
    foremost, and implementation details or background theory afterwards if required.
    """

.. warning::

    Do not manually list the contents of the module; PennyLane uses
    `sphinx-automodapi <https://github.com/astropy/sphinx-automodapi>`__ to automatically
    create module listings.


Functions and methods
~~~~~~~~~~~~~~~~~~~~~

The docstring should provide enough information for the reader to call and use the
function or method without having access to the code.
Most functions or methods should have the following structure:

.. code-block:: python

    def func(arg1, arg2, arg3="default_value", **kwargs):
        """Single sentence that summarizes the function.

        Multi-line description of the function (optional, only if necessary).

        Args:
            arg1 (type): Description.
                Continuation line is indented if needed.
            arg2 (type): description
            arg3 (str): Description. Do not provide the default
                value, as this is already present in the signature.

        Keyword Args:
            kwarg1 (type): Description. Provide the default value if applicable.

        Returns:
            type: Description.
            Continuation line is not indented.

        Raises:
            ExceptionType: description

        .. seealso:: :func:`~.relevant_func`, :class:`~.RelevantClass` (optional)

        **Example**

        Minimal example with 1 or 2 code blocks (required).

        .. details::
            :title: Usage Details

            More complicated use cases, options, and larger code blocks (optional).

        **Related tutorials**

        Links to any relevant PennyLane demos (optional).
        """


For predominantly developer-facing functions and methods (e.g., private functions and methods),
the ``Example`` section is not required. In addition, if the function is very short and its use
is apparent from the signature, it is sufficient for the docstring to simply consist of the single
line summary sentence.

Some notes on the above structure:

* **Function summary:** The summary should provide a brief, basic description of the function. Do not
  include the function or argument names in the summary. It is important that the initial summary
  be a single sentence; Sphinx will truncate the summary at the first period otherwise.

* **Arguments:** To describe named function arguments in the signature. If the type is a PennyLane class or
  function, use the syntax ``(~.Operation)``; do **not** use a Sphinx role such as ``:class:`` or ``:func:``.
  The level of type description is up to the discretion of the code author and reviewers; acceptable
  examples include ``(array)``, ``(array[float])``, ``(dict)``, ``(dict[str, int])``, ``(Sequence[int] or int)``.
  Note that square brakets are used to define the required types of container elements.

  *Note:* if the argument has a default value, *do not* include it in the description if it is declared
  in the signature. Sphinx will automatically extract the default and render it with the argument
  description.

* **Keyword Arguments:** To describe variable length named keyword arguments provided via
  ``**kwargs``. Provide the default values if relevant.

* **Returns:/Yields:** To describe the return (yielded) value of the function (generator). This may
  be omitted if the function signature/summary already sufficiently describe both the return value
  and type.

* **See also:** Provide optional links to relevant pages, functions, classes, methods, etc.

* **Example:** To provide a minimal working example showing basic usage of the function. The example
  should be *minimal* (reduce line counts where possible), but complete; a reader should be able to
  copy and paste the example and get the same output. See :ref:`code_examples` for guidelines on writing
  useful code examples in docstrings.

* **Usage details:** To provide a more complicated usage details showing different edge cases and
  advanced usage of the function, as well as implementation details. This section of the docstring
  is *collapsed by default* in the rendered Sphinx documentation to avoid overwhelming the reader with
  information, however will be displayed in full to users in the terminal/Jupyter notebooks, so
  do not let this section become too long.


Classes
~~~~~~~

The class docstring is placed directly below the class definition:

.. code-block:: python

    class MyClass:
        """Single sentence that summarizes the class.

        Multi-line description of the class (optional, if required).

        Args:
            arg1 (type): Description.
                Continuation line is indented if needed.
            arg2 (type): description

        Keyword Args:
            kwarg1 (type): description

        Attributes:
            attr1 (type): description

        Raises:
            ExceptionType: description

        .. seealso:: :func:`~.relevant_func`, :class:`~.RelevantClass` (optional)

        **Example**

        Minimal example with 1 or 2 code blocks (required).

        .. details::
            :title: Usage Details

            More complicated use cases, options, and larger code blocks (optional).

        **Related tutorials**

        Links to any relevant PennyLane demos (optional).
        """

Docstrings for classes follow a similar structure as functions, with a few differences:

* **Include initialization arguments.** The constructor ``__init__()`` is documented
  here. Note that as the constructor simply returns an instance of the class, no ``Returns:``
  section is included.

* **Do not list methods and properties.** Class properties and methods will be automatically
  listed alongside the class in the generated documentation.

* **List attributes if relevant.**

Document all methods and properties with docstrings below the method declaration,
as you would for functions above. However, there are two
exemptions, where docstrings should *not* be provided:

* **Magic or special methods.** This includes methods such as ``__init__``, ``__call__``,
  ``__len__``, ``__getitem__``, etc. Only provide docstrings for special methods if
  their behaviour is modified in a non-standard way (e.g., if negative indices are allowed
  as arguments for ``__getitem__``).

* **Overwritten methods.** Overwritten inherited methods automatically inherit the parent methods
  docstring, even when overwritten. This is particularly useful when creating an instance
  of an abstract base class; the abstract method docstring is defined in the parent, and automatically
  inherited by the child class, even when overwritten.

  As with special methods, only provide docstrings for overwritten methods if
  their behaviour is modified in a non-standard way (as defined in the parent class).


Variables
~~~~~~~~~

Module-level variables may also be optionally documented, by providing a triple-quote docstring
**below** the variable definition. For example,

.. code-block:: python

    x = {"John": 23, "James": 54}
    """dict[str, int]: stores the ages of known users"""

The syntax is the same as those used for describing function arguments.


.. _code_examples:

Code examples
-------------

Code examples are very important; they *show* readers how the function or class should be used.
When writing code examples for docstrings, use the following guidelines:

- You may assume that PennyLane is imported as ``qml`` and NumPy is imported as ``np`` in the code examples.
  All other imports must be specified explicitly.

- For single line statements and associated output, use Python console syntax (``pycon``):

  .. code-block:: pycon

      >>> circuit(0.5, 0.1)
      [0.43241234312, -0.543534534]

  Multi-line statements should use ``...`` to indicate continuation lines:

  .. code-block:: pycon

      >>> dev = qml.device("default.qubit", wires=1)
      >>> @qml.qnode(dev)
      >>> def circuit(x):
      ...     qml.RX(x, wires=0)
      ...     return qml.expval(qml.PauliZ(0))
      >>> circuit(0.5)
      0.8775825618903726

  For larger, more complicated code blocks, favour standard Python code-block with
  Python console syntax for displaying output:

  .. code-block:: rest

      .. code-block:: python3

          dev = qml.device("default.qubit", wires=1)
          @qml.qnode(dev)
          def circuit(x):
              qml.RX(x, wires=0)
              return qml.expval(qml.PauliZ(0))

      Executing this circuit:

      >>> circuit(0.5)
      0.8775825618903726


Commenting code
---------------

While docstrings describe *what* a function or class does, or how it is used, **comments**
are used to document the underlying implementation. Use comments where applicable to make
your code easy to follow and understand, keeping to the following guidelines.

* **Use comments to explain the implementation or algorithm, never to describe the code.**
  Assume the reader has basic understanding of common programming principles and syntax.
  Do not assume the reader knows what you're trying to do with it!

  .. note::

      If additional implementation details have been requested by reviewers during code review,
      these must be incorporated as code comments.

* **Code should be self-documented where possible.** Code should be clear and concise,
  with the logical flow easily followed by the reader. This can be done by using common
  programming patterns alongside sensible variable and function names.

  Note that this does not exclude the use of code comments; comments are particularly valuable
  in quantum software, where the underlying algorithm can be highly non-trivial. Rather,
  self-documented code simply excludes *unnecessary* comments.

* **Comments should be as close to the described code as possible.** These can either be
  single- or multi-line comments *above* the described code block,

  .. code-block:: python3

      # Note: in the following template definition, we pass the observable, measurement,
      # and wires as *default arguments* to named parameters. This is to avoid
      # Python's late binding closure behaviour
      # (see https://docs.python-guide.org/writing/gotchas/#late-binding-closures)
      def circuit(params, _obs=obs, _m=m, _wires=wires, **kwargs):
          template(params, wires=_wires, **kwargs)
          return MEASURE_MAP[_m](_obs)

  or, for shorter comments, single line comments at the end of the line.

* **Avoid markup and complex formatting in comments.** Markup, such as text formatting and
  hyperlink markup, are unnecessary in comments as they are not rendered, and will simply
  be read in plain text. Use text formatting sparingly for emphasis, and simply insert URLs
  directly. In addition, avoid complex formatting such as tables---these are difficult
  to maintain and modify.


Contributing documentation
--------------------------

Contributions to the PennyLane documentation are encouraged; to contribute to the introductory
quickstarts or developer guides, simply make a :doc:`pull request on GitHub <pullrequests>`.

If you are making a contribution to the PennyLane source code, **all new and modified
functions and code must be clearly commented and documented**. See below for guidelines
on code docstrings, as well as how to add a new module or package to the API documentation.

In addition, specific additions to the code base must also be reflected in the
introductory quickstarts:

* **Operations**: new operations should be added to the :doc:`/introduction/operations` quickstart
  located at ``doc/introduction/operations.rst``. For more details, see :doc:`../adding_operators`.

* **Templates**: new templates should be added to the :doc:`/introduction/templates` quickstart,
  located at ``doc/introduction/templates.rst``. For more details, see :doc:`../adding_operators`.

* **Optimizers**: new optimizers should be added to the relevant quickstart section
  in :doc:`/introduction/interfaces`, located at ``doc/introduction/interfaces.rst``.

* **Measurement**: new measurement functions should be added to the :doc:`/introduction/measurements` quickstart,
  located at ``doc/introduction/measurements.rst``.

* **Interfaces**: new interfaces should include a quickstart guide in the ``introduction/interfaces``
  directory, with a link and table of contents entry added to the ``introduction/interfaces.rst`` page.

Finally, any underlying logic change, new feature, or UI change to the core PennyLane QNode interface
should be reflected on the :doc:`/introduction/circuits` quickstart, located at
``doc/introduction/circuits.rst``.


Adding a new module to the docs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several steps to adding a new module to the documentation:

1. Make sure your module has a one- to two-line module docstring, that summarizes
   what the module purpose is, and what it contains.

2. Add a file ``doc/code/qml_module_name.rst``, that contains the following:

   .. literalinclude:: example_module_rst.txt
       :language: rest

3. Add ``code/qml_module_name`` to the table of contents at the bottom of ``doc/index.rst``.


Adding a new package to the docs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding a new subpackage to the documentation requires a slightly different process than
a module:

1. Make sure your package ``__init__.py`` file has a one- to two-line module docstring,
   that summarizes what the package purpose is, and what it contains.

2. At the bottom of the ``__init__.py`` docstring, add an autosummary table that contains
   all modules in your package:

   .. literalinclude:: example_module_autosummary.txt
       :language: rest

   All modules should also contain a module docstring that summarizes the module.

3. Add a file ``doc/code/qml_package_name.rst``, that contains the following:

   .. literalinclude:: example_package_rst.txt
       :language: rest

4. Add ``code/qml_package_name`` to the table of contents at the bottom of ``doc/index.rst``.


Building the documentation
--------------------------

To build the documentation, in addition to the standard PennyLane dependencies,
the following additional packages are required:

* `Sphinx <http://sphinx-doc.org/>`_ == 2.2.2
* `sphinx-automodapi <https://github.com/astropy/sphinx-automodapi>`__
* `pygments-github-lexers <https://github.com/liluo/pygments-github-lexers>`_
* `m2r <https://github.com/miyakogi/m2r>`_
* `sphinx-copybutton <https://github.com/ExecutableBookProject/sphinx-copybutton>`_

In addition, some pages in the documentation have additional dependencies:

* The latest version of PyTorch and TensorFlow are required to build the interface documentation,
* The latest version of TensorNetwork is required to build the ``default.tensor`` documentation.

These can all be installed via ``pip``:

.. code-block:: console

    $ pip install -r doc/requirements.txt

To build the HTML documentation, go to the top-level directory and run

.. code-block:: bash

    make docs

The documentation can then be found in the :file:`doc/_build/html/` directory.

.. note::

    To build the interfaces documentation, PyTorch and TensorFlow will need to
    be installed, see :ref:`install_interfaces`.

.. note::

  If you are running Python3.8 on an M1 Mac you need to set the following environment variables
  before installing the requirements to be able to install the grpcio package required by TensorFlow
  (`see thread <https://github.com/grpc/grpc/issues/25082#issuecomment-778392661>`):

  .. code-block:: bash

    export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
    export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1