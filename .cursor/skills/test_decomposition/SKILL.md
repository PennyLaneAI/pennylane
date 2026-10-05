---
name: test-decomposition
description: >-
  Write unit tests for PennyLane graph-based decomposition rules (list_decomps,
  resources, conditions, AnnotatedQueue, capture). Use when adding or updating
  tests for qp.add_decomps / DecompositionRule, or when the user asks to test
  a decomposition.
---

# Test decomposition rules

Each registered decomposition rule should be unit tested for registered resources,
conditions, and rule behaviour. Extract rules via `qp.list_decomps(op_type)` by
integer index or rule name. Test each rule separately with `AnnotatedQueue` and
with capture enabled. Capture tests should use `jax.make_jaxpr` and
`CollectOpsandMeas`. If the jaxpr has structured control flow (`cond`, `while`, `for`),
assert those primitives on the jaxpr directly. Prefer putting setup inline in the
test body over class methods / class variables.

## Fetch and invoke

```python
rule = qp.list_decomps(OpType)["rule_name"]  # or [0]

# AnnotatedQueue
with qp.queuing.AnnotatedQueue() as q:
    # operator 1 syntax
    rule(*op.parameters, wires=op.wires, **op.hyperparameters)
    # operator 2 syntax
    rule(*op.arguments)
tape = qp.tape.QuantumScript.from_queue(q)

# Capture — wires become separate jaxpr args
jaxpr = jax.make_jaxpr(rule)(U, wires=[0, 1])
collector = CollectOpsandMeas()
collector.eval(jaxpr.jaxpr, jaxpr.consts, U, 0, 1)
decomp_ops = collector.state["ops"]
```

Imports:
- `CollectOpsandMeas` ← `pennylane.tape.plxpr_conversion`
- `cond_prim`, `for_loop_prim`, `while_loop_prim` ← `pennylane.capture.primitives`
- Enable capture with `@pytest.mark.capture` (`enable_disable_plxpr`)

## Resources and conditions

For Operator1:
```python
assert rule.is_applicable(**op.resource_params)  # often num_wires=...
assert not rule.is_applicable(num_wires=wrong_n)

assert rule.compute_resources(**op.resource_params) == Resources({...})
```
For Operator2:
```python
assert rule.is_applicable(**op.arguments)  # same as operator call siganture
assert not rule.is_applicable(**op2.arguments)

assert rule.compute_resources(**op.arguments) == Resources({...})
```

Resource dict keys:
- Operator2 (e.g. `RZ`): `abstractify(qp.RZ)` from `pennylane.core.operator`
  or `qp.RZ(qp.typing.Float, qp.typing.Wire)` if the operator does not have a fixed signature.
- Operator1: `qp.resource_rep(Op, **resource_params)`
- Compare with `Resources` from `pennylane.decomposition.resources`

## Capture gotchas

- Under capture, optional branches often still emit `cond` (e.g. `GlobalPhase`).
  Assert `cond_prim` / `for_loop_prim` on `jaxpr.eqns`; do not rely only on
  collected ops to prove control flow.
- `qp.matrix` on tapes can be wrong while capture is
  on. Temporarily `qp.capture.disable()` before the matrix assert, then
  re-enable if needed.
- Prefer structure checks (types, wires, hyperparams) under capture; use matrix
  checks after collecting concrete ops.
