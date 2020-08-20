Submitting a pull request
=========================

Before submitting a pull request, please make sure the following is done:

* **All new features must include a unit test.** If you've fixed a bug or added
  code that should be tested, add a test to the ``tests`` directory.

  PennyLane uses pytest for testing; common fixtures can be found in the ``tests/conftest.py``
  file.

* **All new functions and code must be clearly commented and documented.**

  Have a look through the source code at some of the existing function docstrings---
  the easiest approach is to simply copy an existing docstring and modify it as appropriate.

  If you do make documentation changes, make sure that the docs build and render correctly by
  running ``make docs``. See our :doc:`documentation guidelines <documentation>` for more details.

* **Ensure that the test suite passes**, by running ``make test``.

* **Make sure the modified code in the pull request conforms to the PEP8 coding standard.**

  The PennyLane source code conforms to `PEP8 standards <https://www.python.org/dev/peps/pep-0008/>`_.
  Before submitting the PR, you can autoformat your code changes using the
  `Black <https://github.com/psf/black>`_ Python autoformatter, with max-line length set to 100:

  .. code-block:: bash

      black -l 100 pennylane/path/to/modified/file.py

  We check all of our code against `Pylint <https://www.pylint.org/>`_ for errors.
  To lint modified files, simply ``pip install pylint``, and then from the source code
  directory, run

  .. code-block:: bash

      pylint pennylane/path/to/modified/file.py

Submitting to the PennyLane repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ready, submit your fork as a `pull request <https://help.github.com/articles/about-pull-requests>`_
to the PennyLane repository, filling out the pull request template. This template is added
automatically to the comment box when you create a new issue.

* When describing the pull request, please include as much detail as possible
  regarding the changes made/new features added/performance improvements. If including any
  bug fixes, mention the issue numbers associated with the bugs.

* Once you have submitted the pull request, three things will automatically occur:

  - The **test suite** will automatically run on `Travis CI <https://travis-ci.org/PennyLaneAI/pennylane>`_
    to ensure that all tests continue to pass.

  - Once the test suite is finished, a **code coverage report** will be generated on
    `Codecov <https://codecov.io/gh/PennyLaneAI/pennylane>`_. This will calculate the percentage
    of PennyLane covered by the test suite, to ensure that all new code additions
    are adequately tested.

  - Finally, the **code quality** is calculated by
    `Codefactor <https://app.codacy.com/app/PennyLaneAI/pennylane/dashboard>`_,
    to ensure all new code additions adhere to our code quality standards.

Based on these reports, we may ask you to make small changes to your branch before
merging the pull request into the master branch. Alternatively, you can also
`grant us permission to make changes to your pull request branch
<https://help.github.com/articles/allowing-changes-to-a-pull-request-branch-created-from-a-fork/>`_.
