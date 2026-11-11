# Focused deeper review

Use the sections that answer the question which triggered deeper review. Stop
expanding once the evidence supports an assessment; other dimensions need not
be audited.

## Behavior, design, and compatibility

Trace the affected implementation, callers, tests, and documentation. Judge the
change against its stated purpose and intended downstream use. For design
concerns, explain responsibility boundaries, coupling, or complexity through
their concrete maintenance consequences.

For numerical or quantum behavior, consider relevant shapes, dtypes, wires,
batching, differentiation, tracing, and state effects. Follow the target
repository's supported execution paths and current development priorities.
Distinguish intentional behavior changes with adequate migration from
regressions; use the repository's API/deprecation guidance where applicable.

When contracts cross frontend, compiler, or runtime boundaries, inspect their
definitions and consumers. Use the ecosystem's `CONTEXT.md` when available.
Treat submodules as separate repositories, recording each revision/base and
naming the repository in findings. Similarly named concepts need not have
equivalent behavior.

For large changes, prioritize affected contracts and disclose material omissions.
Do not exclude relevant files solely by path or extension. Use Git object reads
for old versions; if the reviewed snapshot changes, recheck affected conclusions.

## Validation and environment

Check whether the tests detect the changed behavior and meaningful failure
cases. For a missing-test concern, name the consumer/execution path,
input/configuration, expected result, and gap in existing coverage. Frontend
capture or JAX JIT checks alone do not establish Catalyst compilation/runtime
correctness.

Choose the smallest authorized test or reproduction that resolves the
uncertainty. Reuse a suitable environment. Use an available environment skill
for necessary setup or repair, following repository instructions. Verify imports
and build artifacts match the intended revisions; use comparable base/head
configurations for regression or performance comparisons. Keep scratch work in
the repository's designated location, outside the reviewed diff.

Record what ran and its scope/configuration. Distinguish missing coverage, unrun
tests, and confirmed failures. Passing checks establish only their tested cases.
Separate performance measurements from estimates. When execution is unavailable,
report the resulting uncertainty rather than implying validation occurred.

## CI failures and evidence gaps

Investigate the relevant job/step logs and annotations. For missing or skipped
evidence, inspect the applicable triggers, path filters, conditions, dependencies,
and workflow-approval state. For CI edits, trace the changed job and its callers
or dependents as needed to establish the effect.

Distinguish change-induced failures from existing failures and infrastructure
problems. Resolve material diagnostics as supported findings, open questions,
or explained non-issues. A missing report is not evidence of zero coverage.
Use the relevant repository's coverage and documentation-test configuration to
establish what a result covers; a build result and an example-execution result
answer different questions.

## Documentation and explanations

Check changed reasoning and the conditions behind stated guarantees. Correct
output does not establish that its explanation is sound. Verify syntax and
call semantics in API examples, including inline snippets outside example-test
collection; distinguish mathematical notation from executable code. Evaluate
changed errors and diagnostics where their meaning affects users.

Consult changelog, deprecation, and convention rules only as they apply to the
change. For PennyLane, use the corresponding sections of
[pennylane.md](pennylane.md).

## Findings and synthesis

Reopen cited context in the reviewed snapshot to verify locations and quotations.
Substantiate general claims such as “no callers” or “not tested” with scoped
searches and inspected matches; absence of a match alone is not proof.

Explain triggering scenarios, evidence, and impact for defects; identify
introduced issues versus relevant pre-existing context. Explain tradeoffs for
improvements and reserve questions for material uncertainty. Consolidate repeated
diagnostics. A design observation can cite several files or a call path without
requiring one inline quote.

Return an overall assessment, including material design/testing conclusions
even when no defect is confirmed. Avoid quotas, generic advice, and unsupported
style preferences. Keep the report proportionate to the change and evidence.
