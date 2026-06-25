---
name: deep-thinking
description: When and how to delegate hard problems to the extended-thinking `deep-thinker` subagent. Use this skill before tackling a problem that needs careful multi-step reasoning — especially challenging coding, algorithm/systems design, subtle debugging, proofs, or ambiguous planning.
---

# Deep Thinking

You run with extended thinking OFF by default — fast and cheap. For problems that
genuinely need careful, multi-step reasoning, you can delegate to a separate
`deep-thinker` subagent whose model reasons deeply before answering.

## Availability

The `deep-thinker` subagent only exists when the operator sets `KYNETIC_THINKING=1`.
Check your `task` tool: if `deep-thinker` is listed as a target, delegation is
available. If it is NOT listed, the flag is off — reason inline as usual and do not
mention the subagent.

## When to delegate

Hand a problem to `deep-thinker` when it is genuinely hard:

* **Challenging coding** — tricky algorithms, non-obvious data structures, concurrency
  or performance work, subtle bugs where the cause isn't apparent, or designs with
  many interacting constraints.
* **Systems / architecture design** with real trade-offs to weigh.
* **Proofs, derivations, or careful correctness arguments.**
* **Planning under ambiguity** — multi-step plans where a wrong early choice is costly.
* **Decisions with non-obvious trade-offs** that benefit from reasoning through several
  options before committing.

## When NOT to delegate

Handle these inline — delegation adds latency and cost for no benefit:

* **Simple or routine coding** — boilerplate, small edits, well-known patterns, glue code.
* Lookups, formatting, summarizing, or anything you can answer confidently at a glance.
* Tasks that are mostly mechanical rather than reasoning-bound.

Rule of thumb: if you'd solve it correctly on the first try without much thought, do it
inline. If you'd want to slow down and reason it through, delegate.

## How to delegate

Call the `task` tool targeting `deep-thinker`. Give it:

1. The full problem statement.
2. The relevant context (file paths, constraints, what you've already tried, what
   "done" looks like).
3. What you want back (a decision, a design, a fix, a proof).

It returns a clear, reasoned answer — not its full scratch work. Review the answer,
then act on it yourself. You remain responsible for the final result.
