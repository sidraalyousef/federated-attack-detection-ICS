# Progress Log

## Timeline

- **Design phase (Semester 1):** 9 weeks, weeks 6–14, starting 2026-11-02
- **Implementation phase (Semester 2):** 14 weeks

### Department deadlines

| Milestone | Date |
|---|---|
| Draft Proposal | 2026-11-01 |
| Proposal | 2026-11-29 |
| Progress-I | 2027-01-03 |

## Decisions log

- **2026-10-xx** — Topic finalized: attack detection on critical infrastructure. Team: Sid, Salma, Fowzi.
- **2026-10-xx** — Models: LSTM and convolutional autoencoders (chosen over GDN/MTAD-GAT for
  tractability within the project timeline).
- **2026-10-31** — Dataset plan: SWaT + WADI as primary train/eval sets, HAI added for
  generalization testing (previously SWaT-only).
- Final demo requirement: a **live running system**, not just offline results — keep this in
  mind when scoping the federated client/server implementation for Semester 2.

## Current status

Design phase — proposal drafting, literature review (see `docs/literature-review.md`), data
exploration.
