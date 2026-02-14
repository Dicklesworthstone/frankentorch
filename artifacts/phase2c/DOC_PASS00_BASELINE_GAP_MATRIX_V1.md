# DOC_PASS00_BASELINE_GAP_MATRIX_V1.md

Date: 2026-02-14  
Schema: `ft-doc-gap-matrix-v1`  
Bead: `bd-3v0.23.1`

## 1. Scope and Reproducibility

Target docs:
- `EXHAUSTIVE_LEGACY_ANALYSIS.md`
- `EXISTING_PYTORCH_STRUCTURE.md`

Baseline commands used:
```bash
wc -l -w EXHAUSTIVE_LEGACY_ANALYSIS.md EXISTING_PYTORCH_STRUCTURE.md
rg -n "^##|^###" EXHAUSTIVE_LEGACY_ANALYSIS.md EXISTING_PYTORCH_STRUCTURE.md
rg -io "unit|property|e2e|logging|differential|metamorphic|adversarial" EXHAUSTIVE_LEGACY_ANALYSIS.md EXISTING_PYTORCH_STRUCTURE.md | wc -l
awk 'BEGIN{sec="";start=0} /^## /{if(sec!=""){printf "%s\t%d\n",sec, NR-start}; sec=$0; start=NR} END{if(sec!=""){printf "%s\t%d\n",sec, NR-start+1}}' EXHAUSTIVE_LEGACY_ANALYSIS.md
awk 'BEGIN{sec="";start=0} /^## /{if(sec!=""){printf "%s\t%d\n",sec, NR-start}; sec=$0; start=NR} END{if(sec!=""){printf "%s\t%d\n",sec, NR-start+1}}' EXISTING_PYTORCH_STRUCTURE.md
```

## 2. Baseline Metrics (Current)

| Doc | Lines | Words | `##/###` headings | test/log cross-ref terms |
|---|---:|---:|---:|---:|
| `EXHAUSTIVE_LEGACY_ANALYSIS.md` | 276 | 1438 | 20 | 4 |
| `EXISTING_PYTORCH_STRUCTURE.md` | 75 | 326 | 11 | 0 |
| Total | 351 | 1764 | 31 | 4 |

## 3. Document-Level Quantitative Targets

| Metric | Baseline | Target | Gate |
|---|---:|---:|---|
| Combined lines across both docs | 351 | >= 4200 | ~12x overall expansion (minimum) |
| Combined words across both docs | 1764 | >= 22000 | >= 12.5x content depth |
| Explicit source-anchor references (file + symbol + rationale) | sparse | >= 450 | no section ships without anchors |
| Unit/property cross-references | 0-4 range | >= 120 | explicit test evidence mapping |
| Differential/metamorphic/adversarial cross-references | sparse | >= 110 | parity + robustness traceability |
| E2E/logging/replay cross-references | sparse | >= 95 | forensics-first operability |

Density requirements (applied per 100 lines in final docs):
- unit/property refs >= 2.5
- differential/metamorphic/adversarial refs >= 2.0
- e2e/logging/replay refs >= 1.8

## 4. Section Gap Matrix — `EXHAUSTIVE_LEGACY_ANALYSIS.md`

Columns:
- `current_lines`: measured via `awk` section spans.
- `factor`: required expansion multiplier for this section.
- `target_lines`: `ceil(current_lines * factor)`.
- `anchor_quota`: minimum count of explicit legacy/source anchors required.
- `u/p`, `diff/adv`, `e2e/log`: minimum section-local cross-reference quotas.

| Section | current_lines | factor | target_lines | anchor_quota | u/p | diff/adv | e2e/log | Primary omission to close | Primary follow-on bead |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `0. Mission and Completion Criteria` | 9 | 10 | 90 | 6 | 4 | 3 | 3 | lacks measurable parity-closure thresholds by subsystem | `bd-3v0.23.14` |
| `1. Source-of-Truth Crosswalk` | 13 | 10 | 130 | 18 | 3 | 3 | 2 | weak trace map to concrete source symbols | `bd-3v0.23.2` |
| `2. Quantitative Legacy Inventory` | 15 | 12 | 180 | 20 | 3 | 3 | 2 | missing subsystem-level inventory slices and growth trends | `bd-3v0.23.2` |
| `3. Subsystem Extraction Matrix` | 13 | 16 | 208 | 32 | 6 | 6 | 4 | insufficient coverage of nested subsystem boundaries | `bd-3v0.23.2` |
| `4. Alien-Artifact Invariant Ledger` | 14 | 18 | 252 | 30 | 7 | 7 | 5 | needs formalized invariants + violation witness plans | `bd-3v0.23.4` |
| `5. Native/C++/CUDA Boundary Register` | 10 | 14 | 140 | 22 | 4 | 5 | 4 | missing explicit hostile-edge cases by boundary | `bd-3v0.23.9` |
| `6. Compatibility and Security Doctrine` | 13 | 18 | 234 | 28 | 5 | 7 | 4 | missing explicit fail-closed drift decision tables | `bd-3v0.23.9` |
| `7. Conformance Program (parent)` | 23 | 20 | 460 | 40 | 10 | 12 | 10 | fixture families too shallow for full parity | `bd-3v0.23.10` |
| `7.1 Fixture families` | (nested) | 20 | 220 | 24 | 8 | 8 | 6 | missing operator-family breadth and adversarial corpus quotas | `bd-3v0.23.10` |
| `7.2 Differential harness outputs` | (nested) | 20 | 220 | 24 | 7 | 9 | 7 | missing full drift taxonomy and replay contract density | `bd-3v0.23.10` |
| `8. Extreme Optimization Program` | 24 | 18 | 432 | 26 | 5 | 6 | 5 | lacks per-hotspot profile protocol and proof obligations | `bd-3v0.23.6` |
| `9. RaptorQ-Everywhere Artifact Contract` | 13 | 16 | 208 | 20 | 3 | 4 | 4 | no end-to-end recovery drill matrix per artifact class | `bd-3v0.23.8` |
| `10. Phase-2 Execution Backlog` | 20 | 16 | 320 | 24 | 6 | 6 | 5 | backlog items not decomposed into full evidence chains | `bd-3v0.23.5` |
| `11. Residual Gaps and Risks` | 6 | 14 | 84 | 14 | 3 | 4 | 3 | lacks formal risk register with owners + closure gates | `bd-3v0.23.9` |
| `12. Deep-Pass Hotspot Inventory` | 19 | 16 | 304 | 22 | 4 | 5 | 3 | hotspot prioritization lacks pass-level target metrics | `bd-3v0.23.6` |
| `13. Phase-2C Extraction Payload Contract` | 21 | 20 | 420 | 30 | 8 | 8 | 6 | payload contract not yet exhaustive over deferred surfaces | `bd-3v0.23.4` |
| `14. Strict/Hardened Compatibility Drift Budgets` | 15 | 18 | 270 | 24 | 5 | 7 | 4 | lacks packet-family drift histograms and escalation tiers | `bd-3v0.23.9` |
| `15. Extreme-Software-Optimization Law` | 18 | 16 | 288 | 20 | 4 | 5 | 3 | lacks profile artifacts crosswalk by packet | `bd-3v0.23.6` |
| `16. RaptorQ Evidence Topology and Recovery Drills` | 16 | 16 | 256 | 22 | 3 | 4 | 5 | no corruption scenario matrix tied to decode proofs | `bd-3v0.23.8` |
| `17. Phase-2C Exit Checklist` | 9 | 14 | 126 | 12 | 4 | 4 | 4 | checklist lacks mandatory evidence cardinality thresholds | `bd-3v0.23.14` |

Projected doc size after targets: >= 4600 lines (including nested depth expansion details).

## 5. Section Gap Matrix — `EXISTING_PYTORCH_STRUCTURE.md`

| Section | current_lines | factor | target_lines | anchor_quota | u/p | diff/adv | e2e/log | Primary omission to close | Primary follow-on bead |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `1. Legacy Oracle` | 5 | 10 | 50 | 10 | 2 | 2 | 1 | no precise branch/version/source drift tracking | `bd-3v0.23.2` |
| `2. Subsystem Map` | 8 | 16 | 128 | 24 | 4 | 4 | 2 | map lacks deep ownership boundaries and coupling map | `bd-3v0.23.2` |
| `3. Semantic Hotspots` | 11 | 18 | 198 | 26 | 6 | 6 | 4 | no formal hotspot invariants and failure signatures | `bd-3v0.23.4` |
| `4. Compatibility-Critical Behaviors` | 7 | 18 | 126 | 20 | 5 | 6 | 3 | missing full compatibility matrix and mode-split deltas | `bd-3v0.23.9` |
| `5. Security and Stability Risk Areas` | 7 | 18 | 126 | 20 | 4 | 6 | 3 | threat classes too coarse, no explicit exploit pathways | `bd-3v0.23.9` |
| `6. Extraction Sequencing Boundary` | 16 | 16 | 256 | 24 | 4 | 5 | 3 | sequencing not decomposed into closure criteria by wave | `bd-3v0.23.5` |
| `6.1 Immediate wave` | (nested) | 16 | 120 | 12 | 3 | 3 | 2 | lacks packet-level entry/exit conditions | `bd-3v0.23.5` |
| `6.2 Mandatory closure waves` | (nested) | 16 | 120 | 12 | 3 | 3 | 2 | lacks full parity closure risk ledger | `bd-3v0.23.5` |
| `7. High-Value Conformance Fixture Families` | 7 | 20 | 140 | 22 | 6 | 7 | 6 | no quantified fixture depth targets per family | `bd-3v0.23.10` |
| `8. First Packet Anchors Already Extracted` | 7 | 14 | 98 | 16 | 3 | 3 | 2 | no cross-packet anchor normalization conventions | `bd-3v0.23.3` |
| `9. Extraction Notes for Rust Spec` | 5 | 16 | 80 | 14 | 3 | 4 | 3 | missing normative drafting rules and review rubric | `bd-3v0.23.14` |

Projected doc size after targets: >= 1500 lines.

## 6. High-Risk Omission Queue (Priority-Ordered)

| Priority | Omission class | Impact | Beads that must close it |
|---|---|---|---|
| P0 | Missing complete source-anchor cartography (module/package ownership + boundaries) | blocks trustworthy expansion of every downstream section | `bd-3v0.23.2` |
| P0 | Missing symbol/API census with observable-behavior tags | prevents complete parity-surface accounting | `bd-3v0.23.3` |
| P0 | Missing formal state/invariant mapping for tensor/dispatch/autograd/serialization | high risk of hidden semantic drift | `bd-3v0.23.4` |
| P1 | Missing execution-path control-flow narratives for critical flows | weakens e2e and failure forensics mapping | `bd-3v0.23.5` |
| P1 | Missing complexity/perf/memory characterization by hotspot | optimization work lacks hard budgets | `bd-3v0.23.6` |
| P1 | Missing concurrency/lifecycle ordering semantics | race/deadlock and replay nondeterminism risk | `bd-3v0.23.7` |
| P1 | Missing failure taxonomy and recovery semantics | poor operator/developer incident response | `bd-3v0.23.8` |
| P1 | Missing security/compatibility undefined-zone inventory | fail-open drift risk | `bd-3v0.23.9` |
| P0 | Missing unit/property + differential + e2e/logging crosswalk | cannot validate parity claims end-to-end | `bd-3v0.23.10` |

## 7. Cross-Cutting Validation Gate Note

This is a docs/planning bead. Execution evidence is carried by implementation/conformance beads.
N/A execution evidence mapping:
- Unit/property execution evidence: `bd-3v0.12.5`, `bd-3v0.13.5`, `bd-3v0.14.5`, `bd-3v0.15.5`, `bd-3v0.17.5`.
- Differential/metamorphic/adversarial execution evidence: `bd-3v0.12.6`, `bd-3v0.13.6`, `bd-3v0.14.6`, `bd-3v0.15.6`, `bd-3v0.17.6`.
- E2E/logging execution evidence: `bd-3v0.12.7`, `bd-3v0.13.7`, `bd-3v0.14.7`, `bd-3v0.15.7`, `bd-3v0.17.7`.

## 8. Done Check for `bd-3v0.23.1`

- [x] Gap matrix covers all top-level and nested sections in both docs.
- [x] Targets include explicit unit/property, differential/adversarial, and e2e/logging quotas.
- [x] High-risk omissions are prioritized and mapped to specific follow-on doc-pass beads.
- [x] Matrix is reproducible from explicit baseline command outputs.
