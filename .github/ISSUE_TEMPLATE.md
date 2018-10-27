#### Before posting an issue

Search existing GitHub issues to make sure the issue does not already exist:
  https://github.com/xanaduai/pennylane/issues

If posting a PennyLane issue, delete everything above the dashed line, and fill in the template.

If making a feature request, delete the following template and describe, in detail, the feature and why it is needed.

For general technical details:
* Check out our documentation: https://pennylane.readthedocs.io

-----------------------------------------------------------------------------------------------------------------------

#### Issue description

Description of the issue - include code snippets and screenshots here if relevant.

* *Expected behavior:* (What you expect to happen)

* *Actual behavior:* (What actually happens)

* *Reproduces how often:* (What percentage of the time does it reproduce?)


#### System information

* **Operating system:**
  Include the operating system version if you can, e.g., Ubuntu Linux 16.04


* **PennyLane version:**
  This can be found by running
  python -c "import pennylane as sf; print(sf.version())"

* **Python version:**
  This can be found by running: python --version


* **NumPy and SciPy versions:**
  These can be found by running
  python -c "import numpy as np; import scipy as sp; print(np.__version__,sp.__version__)"

* **Installation method:**

  Did you install PennyLane via pip, or directly from the GitHub repository source code?

  If installed via GitHub, what branch/commit did you use? You can get this information by opening a terminal in the
  PennyLane source code directory, and pasting the output of:
  git log -n 1 --pretty=format:"%H%n%an%n%ad%n%s"


#### Source code and tracebacks

Please include any additional code snippets and error tracebacks related to the issue here.


#### Additional information

Any additional information, configuration or data that might be necessary to reproduce the issue.
