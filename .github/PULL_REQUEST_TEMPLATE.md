### Before submitting

Please complete the following checklist when submitting a PR:

- [ ] All new features must include appropriate unit tests.
      If you've fixed a bug or added code that should be tested, add a test to the
      test directory!

- [ ] All new functions and code must be clearly commented and documented.
      If you do make documentation changes, make sure that the docs build and
      render correctly by running `make docs`.

- [ ] Ensure that the test suite passes, by running `make test`.

- [ ] Add a new entry to the `doc/releases/changelog-dev.md` file, summarizing the
      change, and including a link back to the PR.

- [ ] The PennyLane source code conforms to
      [PEP8 standards](https://www.python.org/dev/peps/pep-0008/).
      We check all of our code against [Pylint](https://www.pylint.org/).
      To lint modified files, simply `pip install pylint`, and then
      run `pylint pennylane/path/to/file.py`.

When all the above are checked, delete everything above the dashed
line and fill in the pull request template.

------------------------------------------------------------------------------------------------------------

**Context:**

**Description of the Change:**

**Benefits:**

**Possible Drawbacks:**

**Related GitHub Issues:**

**Related Shortcut Stories:**
