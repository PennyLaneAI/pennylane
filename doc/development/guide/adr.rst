Architecture Design Records
===========================

Any significant design decision in PennyLane must first be discussed and agreed to by the
core development team before implementation begins. This is done by drafting an
`Architectural Design Record (ADR) <https://github.com/joelparkerhenderson/architecture_decision_record>`__,
which details the **context** behind the proposal, potential solutions, and any
consequences should the ADR be accepted.

This document describes the requirements and recommendations for PennyLane ADRs.

When to write an ADR
--------------------

Not all contributions and modifications of the PennyLane source code requires an ADR.
Bug fixes and minor modifications can usually proceed directly to a pull request. Further,
straightforward modifications, where no major architectural or user interface decisions need
to be made---such as adding a new template or optimizer---can usually proceed directly
to a pull request.

On the other hand, the following changes typically require an ADR proposal:

* Changes to the user interface

* Changes to the device or plugin API

* Changes that affect multiple core components of the library

* New features that require decisions as to source code location and user interface

* Structural changes to the source code, documentation, or tests

If you are unsure if a change requires an ADR proposal or not, please open a
`GitHub issue <https://github.com/PennyLaneAI/pennylane/issues>`__ or post in the
`discussion forum <https://discuss.pennylane.ai>`__ to discuss it with the PennyLane
development team.

What to include in a PennyLane ADR
----------------------------------

The ADR is a Markdown or reStructuredText document that outlines:


* The context behind this design discussion,

* Analysis of potential options, pros/cons of each approach, and any implementation 'gotchas' that
  might arise, and

* A summary of the decision and an outline of the required work package.

The ADR must have the following components:

**Title**
    The title should directly reflect the decision made by the ADR.

    The title should be 50 characters or less, reflect the final decision, and be written in present
    tense imperative form. For example, "Move common plugin utilities to PennyLane core", or "Add a
    high-level ``qnn`` module to PennyLane".

**Status**
    This acts as a *record* of status updates for the ADR.

    Statuses should be one of: 'Proposed', 'Accepted', 'Rejected', 'Deprecated', 'Superseded', or
    'Amended'.  When the status changes, leave the old status and create a new entry for the new
    status, so that we have a log of when changes were made.

**Context**
    Background information, alongside any other general organizational or
    architectural needs, that allows readers to place the ADR in context.

    The context section should provide sufficient information such that readers of the ADR understand
    the problem the ADR is addressing, and *why* the ADR is required. This section can include
    code/pseudo-code snippets, diagrams, and images if relevant.

**Analysis**
    The outcome of any research undertaken to make the decision.

    This should include an overview of all possible alternatives, potential risks and downsides,
    assumptions that have been made, and all implications. As with the context section, this section
    can include code/pseudo-code snippets, diagrams, and images if relevant.

**Decision**
    Outlines the final decision(s), in present-tense imperative form.

    The decision section should also include:

    - A summary of the resulting work package. This is typically a bullet-point list,
      with each item corresponding to a task such as a pull request or follow-on ADR.
      Care should be taken such that the work package is sufficiently broken down; a series
      of smaller, self-contained pull requests should be favoured over fewer, larger pull requests.

    - Additional or further questions. An ADR may be accepted with some open questions, as long as
      their resolution is (a) not a short-term priority, or (b) must be informed by the upcoming
      implementation. These questions, and any missing information, should be recorded here.

In addition, the following *optional* sections may be included:

* A **review** section, where notes can be made *after* the decision has been implemented for a
  record of the actual outcome and effect the decision has had.

* A **further reading** section where links can be posted to relevant external documentation,
  articles, etc.


Further reading
---------------

* `Documenting Architecture Decisions blog post by Fabian Keller
  <https://www.fabian-keller.de/blog/documenting-architecture-decisions/>`__

* `Joel Parker Henderson: Architecture Decision Record
  <https://github.com/joelparkerhenderson/architecture_decision_record>`__

* `ADR links and tooling <https://adr.github.io/>`__
