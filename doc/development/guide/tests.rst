Software tests
==============

The PennyLane test suite requires the Python ``pytest`` package, as well as ``pytest-cov``
for test coverage; these can be installed via ``pip``:

.. code-block:: bash

    pip install pytest pytest-cov

To ensure that PennyLane is working correctly, the test suite can then be run by
navigating to the source code folder and running

.. code-block:: bash

    make test

while the test coverage can be checked by running

.. code-block:: bash

    make coverage

The output of the above command will show the coverage percentage of each
file, as well as the line numbers of any lines missing test coverage.
