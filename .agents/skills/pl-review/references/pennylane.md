# PennyLane review conventions

Consult only the sections needed for the change or an unresolved review question.
Use the target repository's trusted guides for current requirements. Paths below
are relative to the PennyLane checkout, which may differ from the skill checkout.

## PR contribution context (PennyLane and Catalyst)

Inspect the PR's actual labels. Both repositories use `external`. When label
interpretation matters to the review, verify it in the base repository's trusted
`.github/workflows/check-external-contributor.yml` and the referenced labeling
action. The rule applies `external` when the head repository is a fork or the
opening actor is an active member of the configured `external-collaborators`
team.

An internal developer's personal fork can therefore be labeled `external`.
Keep the observed label separate from the author's reported association or
trusted affiliation context, and from the head/base repository origin.
Neither the label nor its absence proves affiliation; an unlabeled PR is not
automatically internal. Mark missing information as unknown. Use this context
when interpreting available CI evidence and execution requirements. Contribution
classification does not change the review's quality standard, authorize running
PR code, or gate local review.

## CI and coverage

Use this section when test scope, coverage, CI changes, or an unexplained result
needs investigation. The initial CI overview does not require reading every
workflow or downloading coverage reports.

Inspect the target revision's `.github/workflows/` to establish which checks
cover the change. Follow reusable workflows and matrix conditions. For example,
`tests.yml` calls `interface-unit-tests.yml`, with PRs using reduced CI unless
configured otherwise. A green aggregate job does not establish that every
interface, Python version, or optional job ran.

For Codecov, inspect available check/status details and the linked report or bot
comment for the relevant head and comparison base. Distinguish project coverage
and its change from patch coverage of changed executable lines. Inspect uncovered
changed lines and their test scenarios; percentages alone neither establish
correctness nor justify a missing-test finding.

Read `codecov.yml`, `.coveragerc`, and the relevant test/upload jobs for thresholds,
exclusions, and report scope. Verify that expected reports arrived before
interpreting a coverage drop or missing result. The interface workflow uploads
coverage after its aggregate test job succeeds; an upload failure or an absent
report is different from low coverage. Other workflows can supply separate
reports. State incomplete or inaccessible coverage evidence without inventing
metrics or treating it as zero coverage.

## Documentation builds and executable examples

Use build evidence for markup/rendering questions and example-execution evidence
for code changes or behavior claims. Plain prose corrections can be reviewed
from the diff and CI overview. When more evidence is needed, identify what ran:

- **Documentation build:** `docs.yml` (“Documentation check”) builds Sphinx HTML
  with warnings treated as errors (`-W`). `rtd-check.yml` (“Read the Docs check”)
  also builds HTML; `.readthedocs.yml` configures the external Read the Docs
  service. Inspect the relevant build's flags, warnings, imports, and rendered
  output when material. These surfaces need not enforce the same warning policy.
  Successful HTML generation does not establish that example code was tested.
- **Documentation example execution:** `documentation-tests.yml` (“Test
  Documentation Code Examples”) runs pytest over selected source and
  documentation files. Root `conftest.py` configures Sybil to collect doctests
  and Python code blocks from `.py`, `.rst`, and `.md` files. Inspect per-step
  ignore lists, snippet skips, and actual collection for the changed examples;
  the workflow handles `changelog-dev.md` in a separate step. A green job does
  not establish execution of an excluded path or an inline snippet.

Confirm this mapping against the reviewed workflow revision; a workflow can
contain both kinds of validation. Match any local reproduction to the missing
evidence: an authorized, targeted pytest example run assesses execution, while
a Sphinx build assesses documentation generation. Neither replaces assessment
of explanatory reasoning or the examples' intended behavior.

## Changelog

Check `doc/releases/changelog-dev.md` and
`doc/development/guide/pullrequests.rst`. PennyLane asks for a changelog entry
with a PR link, an alphabetical contributor entry, and an example for significant
user-facing changes. Match the section to the behavior described.

Distinguish a new top-level bullet from edits to an existing entry, including a
PR-link addition, rewording, or reflow. A changed `*` line alone does not establish
that a new entry was added. Read the before and after entries and section
headings. Report the observed change separately from the question of whether
it adequately documents this PR. An edited entry can be appropriate.
Check that substantive behavior changes and compatibility impacts are described;
an entry's presence alone does not establish that coverage.

## Public API compatibility

For an apparent removal or signature change, trace the old public import path
through re-exports, lazy exports such as module `__getattr__`, and compatibility
aliases. Compare the effective old and new call signatures and behavior before
reporting an API break. A definition moving between modules does not by itself
establish that a public import stopped working.

## Deprecations and removals

Consult `doc/development/guide/deprecations_removals.rst` and
`doc/development/deprecations.rst` when public behavior is deprecated or removed.

- For a deprecation, check the replacement or removal rationale, target version,
  `PennyLaneDeprecationWarning`, public documentation warning, and warning test.
  Check internal callers and tests for unintended use of the deprecated path.
  The guide calls for pending-deprecation and changelog entries.
- For a removal, check the applicable deprecation history and intended release,
  the completed-cycle entry, and the breaking-change changelog entry.
- Resolve exceptions using the current repository's development priorities and
  explicit maintainer context. Do not invent a fixed release delay or apply
  public-API migration requirements to every internal refactor.

## Tests and execution paths

Tests use pytest under `tests/`; consult `doc/development/guide/tests.rst`.
Bug fixes need a regression that detects the original behavior, and new behavior
needs appropriate coverage. Match tests to changed shapes, dtypes, wires,
batching, differentiation, and errors rather than requesting every combination.

Select supported execution paths from the target revision's instructions and
documentation. For JAX changes, check whether precision settings such as x64
affect the claimed contract; record the setting used in a reproduction.
For a new operator, inspect relevant `pennylane.ops.functions.assert_valid`
coverage. It does not establish downstream compiler or runtime correctness.

## Conventions

Use the target repository's guides and `pyproject.toml`, `.pylintrc`,
`tests/.pylintrc`, and `tach.toml` for applicable naming, import, formatting,
and module-boundary rules. Tie convention findings to those sources and the
changed code.

When lint or formatting checks help assess the change, run configured Pylint
followed by formatter check modes (`black --check`, `isort --check-only`) on the
relevant files. Use `tach check` to assess changed module dependencies. These
checks should leave the reviewed files unchanged.
