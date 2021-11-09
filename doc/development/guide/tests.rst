Software tests
==============

Running the test suite
~~~~~~~~~~~~~~~~~~~~~~

The PennyLane test suite requires the Python ``pytest`` package, as well as ``pytest-cov``
for test coverage, ``pytest-mock`` for mocking, and ``flaky`` for automatically rerunning flaky tests; these can be installed via ``pip``:

.. code-block:: bash

    pip install pytest pytest-cov pytest-mock flaky

To ensure that PennyLane is working correctly, the test suite can be run by
navigating to the source code folder and running

.. code-block:: bash

    make test

while the test coverage can be checked by running

.. code-block:: bash

    make coverage

The output of the above command will show the coverage percentage of each
file, as well as the line numbers of any lines missing test coverage.



Testing Matplotlib based code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Matplotlib images can be displayed differently based on the system generating them, such as default resolution and available fonts.
Even though matplotlib does provide
`functionality for pointwise comparison of images <https://matplotlib.org/stable/api/testing_api.html#module-matplotlib.testing>`__ , they can be rather fragile and require caching
correct images in a particular location.

Instead of performing per-pixel comparison of saved images, we can instead inspect the  `figure <https://matplotlib.org/stable/api/figure_api.html?highlight=figure#matplotlib.figure.Figure>`__
and `axes <https://matplotlib.org/stable/api/axes_api.html?highlight=axes#module-matplotlib.axes>`__
objects to ascertain whether they contain the correct objects. The figure should contain the axis object in its `fig.axes` attribute, and the axis object should contain the objects that get displayed. These are stored in one of three attributes. Each attribute is a list of relevant objects, ordered corresponding to when they were added:

* ``ax.texts``
* ``ax.lines``
* ``ax.patches``

Instead of testing every relevant piece of information for all objects in the graphic, we can just check key pieces of information to make sure everything looks decent.  These key pieces of information can include (but are not limited to):

* number of objects
* type of objects
* location

**Text objects**

`Text objects <https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text>`__
are stored in ``ax.texts``.  While the text object can contain a great variety of methods for pulling out relevant information, the two most commonly used in testing text objects are:

* ``text_obj.get_text()`` : Get the string value for the text object
* ``text_obj.get_position()``: Get the ``(x,y)`` position of the object

**Lines**

`2D lines <https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html?highlight=line2d#matplotlib.lines.Line2D>`__ are stored in ``ax.lines``.  PennyLane's
circuit drawing code uses lines for wires, SWAP gates, and controlled operations. The most important method for checking lines is ``line_obj.get_data()``.  For easier reading, you
can also use ``line_obj.get_xdata()`` and ``line_obj.get_ydata()``.

**Patches**

`Patches <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html?highlight=patch#matplotlib.patches.Patch>`__
include a wide variety of different objects, like:

* `Rectangle <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html?highlight=rectangle#matplotlib.patches.Rectangle>`__
* `Circle <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Circle.html?highlight=circle#matplotlib.patches.Circle>`__
* `Arc <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Arc.html?highlight=arc#matplotlib.patches.Arc>`__
* `Fancy Arrow <https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyArrow.html?highlight=fancyarrow#matplotlib.patches.FancyArrow>`__

Each can have it's own getter methods and attributes.  For example, an arc has `theta1` and `theta2`. ``dir(patch_obj)`` can help developers determine which methods an attributes a given object has.