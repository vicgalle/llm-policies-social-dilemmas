# Proposer log

This is the proposer's append-only narrative of the meta-harness search.
Each entry is one decision: **what** the proposer did, **why** (informed by
which trace findings or prior harness outcomes), and **what was learned**
(to inform later branches).

Format:

```
## YYYY-MM-DD HH:MM — <action>

**Parent**: h0042
**New harness**: h0043 (mode=M3)
**Hypothesis**: ...
**Diagnosis from h0042 traces**: ...
**Result**: efficiency=... (n=5 seeds), Pareto=...
**Followup**: ...
```

Entries are written by the proposer via `tools.py log <text>`.
