Software tests
==============

Running the tests
~~~~~~~~~~~~~~~~~

The PennyLane test suite requires the Python ``pytest`` package, as well as:

* ``pytest-cov``: determines test coverage
* ``pytest-mock``: allows replacing components with dummy/mock objects
* ``flaky``: manages tests with non-deterministic behaviour

These requirements can be installed via ``pip``:

.. code-block:: bash

    pip install pytest pytest-cov pytest-mock flaky

The `tests <https://github.com/PennyLaneAI/pennylane/tree/master/tests>`__ folder of the root PennyLane directory contains the PennyLane test suite. Run all tests in this folder via:

.. code-block:: bash

    python -m pytest --ignore=tests/beta tests

Using ``python -m`` ensures that the tests run with the correct Python version if multiple versions are on the system. The ``tests/beta`` folder can contain failing tests, so ``--ignore=tests/beta`` excludes them from execution.

As the entire test suite takes some time, locally running only relevant files speeds the debugging cycle.  For example, if a developer was adding a new non-parametric operation, they could run:

.. code-block:: bash

    python -m pytest tests/ops/qubit/test_non_parametric_ops.py

The slowest tests are marked with ``slow`` and can be deselected by:

.. code-block:: bash

    python -m pytest -m "not slow" tests

Pytest supports many other command-line options, which can be found with the command:

.. code-block:: bash

    pytest --help

Or by visiting the `pytest documentation <https://docs.pytest.org/en/latest/reference/reference.html#id88>`__ . 

PennyLane provides a set of integration tests for all PennyLane plugins and devices. See the documentation on these tests under the section on the `device API <https://pennylane.readthedocs.io/en/latest/code/api/pennylane.devices.tests.html>`__. These tests can be run from the PennyLane root folder by:

.. code-block:: bash

    pytest pennylane/devices/tests --device=default.qubit --shots=1000

All PennyLane tests and the device suite on core devices can be run from the PennyLane root folder via:

.. code-block:: bash

    make test


Testing Matplotlib based code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Matplotlib images can display differently due to various factors outside the standard developer's control, such as image backend and available fonts. Even though matplotlib provides
`functionality for pointwise comparison of images <https://matplotlib.org/stable/api/testing_api.html#module-matplotlib.testing>`__ , they require caching
correct images in a particular location and are sensitive to details we don't need to test. 

Instead of performing per-pixel comparison of saved images, we can instead inspect the  `figure <https://matplotlib.org/stable/api/figure_api.html?highlight=figure#matplotlib.figure.Figure>`__
and `axes <https://matplotlib.org/stable/api/axes_api.html?highlight=axes#module-matplotlib.axes>`__
objects to ascertain whether they contain the correct information. The figure should contain the axis object in its ``fig.axes`` attribute, and the axis object should contain the `Artists <https://matplotlib.org/stable/tutorials/intermediate/artists.html>`__ that get displayed. These artists relevant to us are located in one of three attributes. Each attribute is a list of relevant objects, ordered as they were added:

* ``ax.texts``
* ``ax.lines``
* ``ax.patches``

Instead of testing every relevant piece of information for all objects in the graphic, we can check key pieces of information to make sure everything looks decent.  These key pieces of information can include (but are not limited to):

* number of objects
* type of objects
* location

**Text objects**

`Text objects <https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text>`__
are stored in ``ax.texts``.  While the text object has many methods and attributes for relevant information, the two most commonly used in testing text objects are:

* ``text_obj.get_text()`` : Get the string value for the text object
* ``text_obj.get_position()``: Get the ``(x,y)`` position of the object

**Lines**

`2D lines <https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html?highlight=line2d#matplotlib.lines.Line2D>`__ are stored in ``ax.lines``.  PennyLane's
circuit drawing code uses lines for wires, SWAP gates, and controlled operations. The most important method for checking lines is ``line_obj.get_data()``.  For easier reading, you
can also use ``line_obj.get_xdata()`` and ``line_obj.get_ydata()``.

**Patches**

`Patches <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html?highlight=patch#matplotlib.patches.Patch>`__
can be a wide variety of different objects, like:

* `Rectangle <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html?highlight=rectangle#matplotlib.patches.Rectangle>`__
* `Circle <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Circle.html?highlight=circle#matplotlib.patches.Circle>`__
* `Arc <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Arc.html?highlight=arc#matplotlib.patches.Arc>`__
* `Fancy Arrow <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyArrow.html?highlight=fancyarrow#matplotlib.patches.FancyArrow>`__

Each can have its own getter methods and attributes.  For example, an arc has ``theta1`` and ``theta2``. ``dir(patch_obj)`` can help developers determine which methods and attributes a given object has.

For Rectangles, the most relevant methods are:

* ``rectangle_obj.get_xy()``
* ``rectangle_obj.get_width()``
* ``rectangle_obj.get_height()``