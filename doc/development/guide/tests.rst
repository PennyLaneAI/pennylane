Software tests
==============

Requirements
~~~~~~~~~~~~
The PennyLane test suite requires the Python ``pytest`` package, as well as
some extentions thereof, for example:

* ``pytest-cov``: determines test coverage
* ``pytest-mock``: allows replacing components with dummy/mock objects
* ``flaky``: manages tests with non-deterministic behaviour
* ``pytest-benchmark``: benchmarks the performance of functions, and can be used to ensure consistent runtime
* ``pytest-xdist``: currently used to force some tests to run on the same thread to avoid race conditions

If you properly followed the :doc:`installation guide <./installation>`, you should have all of these packages and others installed in your
environment, so you can go ahead and put your code to the test!

Creating a test
~~~~~~~~~~~~~~~
Every test has to be added to the PennyLane `test folder <https://github.com/PennyLaneAI/pennylane/tree/master/tests>`__.
The test folder follows the structure of the PennyLane module folder. Therefore, tests needs to be added to the corresponding subfolder of the functionality they are testing.

Most tests typically will not require the use of an interface or autodifferentiation framework (such as Autograd, PyTorch, and JAX). Tests without an interface will be marked
as a ``core`` test automatically by pytest (the functionality for this is located in ``conftest.py``). For such general tests, you can follow the structure of the example below,
where it is recommended that you follow general `pytest guidelines <https://docs.pytest.org/>`__:

.. code-block:: python

    import pennylane as qp

    def test_circuit_expval(self):
        """ Test that the circuit expectation value for PauliX is 0."""
        dev = qp.device("default.qubit", wires=1)

        @qp.qnode(dev)
        def circuit():
            return qp.expval(qp.PauliX(0))

        expval = circuit()

        assert expval == 0

This test will be marked automatically as a ``core`` test.

On the other hand, some tests require specific interfaces and need to be marked in order to be run on our Github test suite.
Tests involving interfaces have to be marked with their respective marker:

- ``@pytest.mark.autograd``,

- ``@pytest.mark.torch``,

- ``@pytest.mark.tf``, and

- ``@pytest.mark.jax``.

If tests involve multiple interfaces, one should add the marker:

- ``@pytest.mark.all_interfaces``.

.. warning::
    Please do not use ``pytest.importorskip`` inside your tests. Instead, simply import the autodifferentiation package
    as needed inside your marked test. The mark will automatically ensure that the test is skipped if the
    autodifferentiation framework is not installed.

Tests that are not marked but do import an interface will lead to a failure in the GitHub test suite.

Below you can find an example for testing a PennyLane template with Jax:

.. code-block:: python

    def circuit_template(features):
        qp.AngleEmbedding(features, range(3))
        return qp.expval(qp.PauliZ(0))

    def circuit_decomposed(features):
        qp.RX(features[0], wires=0)
        qp.RX(features[1], wires=1)
        qp.RX(features[2], wires=2)
        return qp.expval(qp.PauliZ(0))

    @pytest.mark.jax
    def test_jax(self, tol):
        """Tests the jax interface."""

        import jax
        import jax.numpy as jnp

        features = jnp.array([1.0, 1.0, 1.0])

        dev = qp.device("default.qubit", wires=3)

        circuit = qp.QNode(circuit_template, dev, interface="jax")
        circuit2 = qp.QNode(circuit_decomposed, dev, interface="jax")

        res = circuit(features)
        res2 = circuit2(features)
        assert qp.math.allclose(res, res2, atol=tol, rtol=0)

Another example of a test involving multiple interfaces is shown below:

.. code-block:: python

        def circuit(features):
            qp.AngleEmbedding(features, range(3))
            return qp.expval(qp.PauliZ(0))

        @pytest.mark.parametrize("interface",[
            pytest.param("jax", marks=pytest.mark.jax),
            pytest.param("torch", marks=pytest.mark.torch),
        ])
        def test_all_interfaces_gradient_agree(self):
            """Test the results are similar between torch and jax"""
            import torch
            import jax.numpy as jnp

            dev = qp.device("default.qubit", wires=3)

            features_torch = torch.Tensor([1.0, 1.0, 1.0])
            features_jax = jnp.array([1.0, 1.0, 1.0])

            circuit_torch = qp.QNode(circuit, dev, interface="torch")
            circuit_jax = qp.QNode(circuit, dev, interface="jax")

            res_torch = circuit_torch(features_torch)
            res_jax = circuit_jax(features_torch)

            assert np.allclose(res_torch, res_jax)


Running the tests
~~~~~~~~~~~~~~~~~

The `tests <https://github.com/PennyLaneAI/pennylane/tree/master/tests>`__ folder of the root PennyLane directory contains the PennyLane test suite. Run all tests in this folder via:

.. code-block:: bash

    python -m pytest tests

Using ``python -m`` ensures that the tests run with the correct Python version if multiple versions are on the system.
As the entire test suite takes some time, locally running only relevant files speeds up the debugging cycle. For example,
if a developer was adding a new non-parametric operation, they could run:

.. code-block:: bash

    python -m pytest tests/ops/qubit/test_non_parametric_ops.py

Using ``pytest -m`` offers the possibility to select and run tests with specific markers. For example,
if Jax is installed and a developer wants to run only Jax related tests, they could run:

.. code-block:: bash

    python -m pytest tests -m "jax"

There exists markers for interfaces (``autograd``, ``torch``, ``tf``, ``jax``), for multiple interfaces (``all_interfaces``) and
also for certain PennyLane submodules (``qchem`` and ``qcut``).

For running ``qchem`` tests, one can run the following:

.. code-block:: bash

    python -m pytest tests -m "qchem"

The slowest tests are marked with ``slow`` and can be deselected by:

.. code-block:: bash

    python -m pytest -m "not slow" tests

The ``pytest -m`` option supports Boolean combinations of markers. It is therefore possible to run both JAX and PyTorch
tests by writing:

.. code-block:: bash

    python -m pytest -m "jax and torch" tests

or Jax tests that are not slow:

.. code-block:: bash

    python -m pytest -m "jax and not slow" tests

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