### General guidelines

* Do not make a pull request for minor typos/cosmetic code changes - make an issue instead.

### Before submitting

Before submitting a pull request, please make sure the following is done:

* All new features must include a unit test.
  If you've fixed a bug or added code that should be tested, add a test to the test directory!

* All new functions and code must be clearly commented and documented.
  Have a look through the source code at some of the existing function docstrings - the easiest
  approach is to simply copy an existing docstring and modify it as appropriate.
  If you do make documentation changes, make sure that the docs build and render correctly by running `make docs`.

* Ensure that the test suite passes, by running `make test`.

* Make sure the modified code in the pull request conforms to the PEP8 coding standard.
  The PennyLane source code conforms to PEP8 standards (https://www.python.org/dev/peps/pep-0008/).
  We check all of our code against Pylint (https://www.pylint.org/). To lint modified files, simply
  `pip install pylint`, and then from the source code directory, run `pylint pennylane/path/to/file.py`.

### Pull request template

When ready to submit, delete everything above the dashed line and fill in the pull request template.

Please include as much detail as possible regarding the changes made/new features
added/performance improvements. If including any bug fixes, mention the issue numbers associated with the bugs.

------------------------------------------------------------------------------------------------------------

**Description of the Change:**

**Benefits:**

**Possible Drawbacks:**

**Related GitHub Issues:**
