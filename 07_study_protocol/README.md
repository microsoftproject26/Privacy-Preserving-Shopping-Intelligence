# 07_study_protocol - pre-registration, locks, registers and delivery verification

## Purpose

The governance record of the study: the pre-registration and its amendments, the locks and
gates that fixed every rule before the value it governed existed, the hash-chained TEST
exposure register, the handover documents, the deployment manifests of the frozen server code
trees, and the delivery-verification evidence of the closure phase. **Everything in this folder
is a record and is never edited.** Corrections go into new files, errata or new register
entries.

## What the record contains, in order

1. **Pre-registration lineage.** The registered deployment study and its amendments (01 to 07,
   06A to 06P) and the separately registered follow-up analysis, kept as hashed files in
   `preregistration/`; the contrast files are in `locks_and_gates/` and the federated
   improvement ladder in `registers/`.
2. **Locks, freezes and the gate chain.** The data protocol lock, the architecture lock after
   the backbone contest, the federated selections, the contrast registry (`ANA_CONTRASTS_v4`,
   hash prefix `f1769f0b`), the analysis freeze (final one `59c83c86`), and the lock of all
   methods (`04a3cd19`: 131 arms, 56 sealed federated checkpoints, the TEST plan). Two
   check-only refusals before the validation read are part of the record: both were
   tool-definition defects, and no value was read.
3. **The single validation read (3 October 2026).** MAIN validation was read exactly once
   (decision-log row 400); everything after it is labelled exploratory.
4. **The confirmatory test evaluation (5-7 October 2026, 21:03 UTC on 5 October to 10:09 UTC
   on 7 October).** The main analysis and the follow-up analysis shared one sealed scoring of
   the REES46 TEST week (26,254 users, 229,742 decisions): register entries 10 and 11. The
   frozen test-scoring tree that ran it is kept as a record and must not be edited.
5. **The TEST exposure register.** A hash-chained file with 19 entries (0 errors) plus the
   closure-phase extension (entries 20 to 23) in new mirrors that leave the original untouched:
   entries 1-9 are pre-evaluation (counts-only censuses, the recorded historical exposure of
   the pilot studies, manifest and integrity checks); entries 10-11 are the confirmatory looks;
   entries 12-23 are exploratory looks, each a pre-scoring registration entry and a
   post-scoring disclosure entry pair.
6. **Technical closure and delivery (12-17 October 2026).** Verification of every saved result
   (final closure counts: 1,087 tests passing and 328 expected negative controls, 0 unresolved),
   the merged, verified delivery folder of 17 October 2026 (338,858 files, 338,857 verified),
   and the delivery entry documents.

## Folder map

| Folder | Contents |
|---|---|
| `handover/` | Handover documents 01-07, 11-14, 18 and 19 (research design, data, governance, validation results, follow-up analysis, benchmarks, server operations, file map, glossary, open issues, experiment register, incident and fix log) |
| `handover/reference/` | The team handover of 30 September 2026, the validation claim register, headline template and per-role metric tables, and the three Phase-2 hand-offs that the code cites |
| `deployment_manifests/` | Sync and tool manifests of the frozen server code trees (`r4` to `r9`, `r7b`, `p2r1`, `bench5`, `bench5b`) with the A40 environment lock (`ENV_LOCK_A40.txt`, `ENV_PKGS_A40.txt`) and `SHA256SUMS.txt` |
| `delivery_verification/` | Read-only closure verifiers: delivery and technical-closure verification, protected-tree inventories, update-norm summaries, compatibility and locked-benchmark retests, local suite runners |
| `tools/` | Protocol tools as run at the time: the test-exposure register tool, the follow-up lock and no-op checks, the data-protocol lock writers, code sync and transfer helpers |

## How to run

This stage has no `run.py` and nothing here is re-run as an experiment: re-running a study
means the original flat code tree whose manifests are recorded in `deployment_manifests/`
(see the root [`README.md`](../README.md) on hash-pinned code). The delivery-verification
scripts are read-only checks meant to run against a copy of the original workspace, never
against the originals:

```bash
uv run python 07_study_protocol/delivery_verification/verify_project_delivery.py --help
uv run python 07_study_protocol/delivery_verification/summarize_update_norms.py --help
```

## results/ and figures/

None by design; this stage keeps records, not results. Curated result tables live in
[`../06_evaluation_and_benchmarks/results/`](../06_evaluation_and_benchmarks/README.md) and the
other stage folders.

## Known limitations

Working material that a reader does not need is not published: review briefs and gate packets, worker hand-offs, team-internal guides, drafts, backup copies of operations scripts and the old supervisor decks. Some records, including the decision log, still name those files; the decisions and verdicts they led to are kept in the decision log and the files that remain.

The registrations are team-internal records - hashed files, a decision log and a hash-chained
register - and no third-party timestamps are claimed. The process made mistakes, which the
records keep (an amendment overwritten by a second session and restored byte-identically; a
duplicated decision-log row number; jobs stopped by an organization spending limit); none
changed a reported result, and the incident log (`handover/19_INCIDENT_AND_FIX_LOG.md`)
lists them. The independent reviews recorded here were AI-assisted reviews, not reviews by
human domain experts. The register shows that TEST was sealed and never used for selection; it
also shows, in entries 1-4, that it was not never opened.
