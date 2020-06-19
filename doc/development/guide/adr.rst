Architecture Design Records
===========================

Any significant design decision in PennyLane must first be discussed and agreed to by the
core development team before implementation begins. This is done by drafting an
`Architectural Design Record (ADR) <https://github.com/joelparkerhenderson/architecture_decision_record>`__,
which details the **context** behind the proposal, potential solutions, and any
consequences should the ADR be accepted.

This document describes the requirements and recommendations for PennyLane ADRs.

When to write an ADR?
---------------------

Not all contributions and modifications of the PennyLane source code requires an ADR.
Bug fixes and minor modifications can usually proceed directly to a pull request. Further,
straightforward modifications, where no major architectural or user interface decisions need
to be made---such as adding a new template or optimizer---can usually proceed directly
to a pull request.

On the other hand, the following changes typically require a ADR proposal:

* Changes to the user interface

* Changes to the device or plugin API

* Changes that affect multiple core components of the library

* New features that require decisions as to source code location and user interface

* Structural changes to the source code, documentation, or tests

If you are unsure if a change requires an ADR proposal or not, please open a
`GitHub issue <https://github.com/XanaduAI/pennylane/issue>`__ or post in the
`discussion forum <https://discuss.pennylane.ai>`__ to discuss it with the PennyLane
development team.

