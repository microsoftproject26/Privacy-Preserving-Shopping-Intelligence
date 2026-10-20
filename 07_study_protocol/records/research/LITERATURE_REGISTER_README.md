# Literature register — how to read `LITERATURE_REGISTER.json`

**What this is.** A thesis-grade citation map over all 34 sources in the V4 package's own research
material. It answers, per source: which project question it can support, exactly what it says (with
its package card ID), what protocol detail it gives (if any), how it differs from this project's
protocol, and what it does/does not license us to claim.

**What this is not.** No new web research was done. Nothing was invented: every title, URL, year,
source classification, finding, decision and limit traces to the V4 package's own research files. An
unknown field (author names, venue/journal names — the package never records either, for any of the
34 sources) is `null` with a note explaining the gap.

## Source files read (extraction only, all under
`PPSI_SINGLE_BACKBONE_V4_2026_09_26\PPSI_SINGLE_BACKBONE_V4_2026_09_26\`)
- `research/REFERENCES.json` — the 34 source cards (id, title, url, source_kind, year, reviewed_on,
  review_depth, takeaway, project_decision, does_not_establish, implementation_url,
  reproduced_by_this_package, and for R28/R30 a bibliographic_note).
- `research/RESEARCH_REVIEW.md` — the same 34 cards in prose form (cross-checked; identical content).
- `research/BENCHMARK_COMPARABILITY.md` — the one place with concrete comparator numbers (used for R05's
  `protocol_context.metric`).
- `research/SEARCH_LOG.md` — search scope, what was excluded from required implementation, and reading-
  depth caveats (used for the review_status wording and general framing).
- `research/references.bib` — bibtex keys (`ppsi_r01`…`ppsi_r34`), reused here as `citation_key`.
- `14_RESEARCH_TO_IMPLEMENTATION.md` — the same 34 cards again in a different layout plus one summary
  table of evidence families (cross-checked; no new facts found beyond REFERENCES.json).

All four card-bearing files (REFERENCES.json, RESEARCH_REVIEW.md, 14_RESEARCH_TO_IMPLEMENTATION.md,
references.bib) agree with each other for every one of the 34 sources; no conflicts were found.

## Field-by-field mapping (task packet -> register -> package origin)

| Register field | Package origin | Notes |
|---|---|---|
| `card_id` | REFERENCES.json `id` (R01…R34) | Used in-text as "(R12)" style citations per the task's instruction. |
| `citation_key` | references.bib key (`ppsi_r01`…) | |
| `title`, `year`, `doi_or_arxiv_or_url` (= `url`) | REFERENCES.json, verbatim | |
| `authors`, `venue` | **Not present anywhere in the package**, for any of the 34 sources | `null` + `authors_note`/`venue_note` explaining the gap and flagging it for verification before thesis submission (the package itself flags this for R28/R30 as a `bibliographic_note`; the same gap in fact applies to all 34). |
| `source_type` | Derived from REFERENCES.json `source_kind` + the URL host | This is an editorial re-classification into the task's fixed enum (`peer_reviewed / preprint / official_repo / official_docs / dataset_page / other`), which the package does not use itself. Basis is disclosed per entry in `source_type_basis` (e.g. arxiv.org host -> `preprint`; proceedings.mlr.press or a doi.org/ACM link -> `peer_reviewed`; github.com author/project repos -> `official_repo`; vendor docs sites -> `official_docs`; research.google listing pages -> `other`, since they are a company's own publication listing, not a preprint server or proceedings site). |
| `project_questions_supported` | Derived from each card's `takeaway`/`project_decision` wording | Short slugs (e.g. `central_baseline_gru`, `evaluation_split_protocol`, `personalization_shared_private_split`) grouping sources by the design question they inform. Not a package-given taxonomy; a classification aid over package-given facts. |
| `exact_useful_finding` | REFERENCES.json `takeaway`, quoted, + card ID | |
| `protocol_context` (dataset/preprocessing/split/candidates/metric) | Filled only where a card (or BENCHMARK_COMPARABILITY.md, for R05) actually states something in that category; otherwise all five sub-fields are `null` with a `note` explaining why (most federated-learning/tooling cards state no REES46-specific dataset/split/metric detail at all). | |
| `differences_from_our_project` | REFERENCES.json `does_not_establish`, quoted, + card ID | The package has no separately labeled "differences" field; this is disclosed in `differences_from_our_project_note` on every entry. |
| `claim_allowed` | REFERENCES.json `project_decision`, quoted, + card ID | |
| `claim_not_allowed` | REFERENCES.json `does_not_establish`, quoted, + card ID | |
| `review_status` | REFERENCES.json `review_depth` | Identical verbatim string on every one of the 34 cards: "Primary landing page/abstract or official repository/documentation sections; not a full independent reproduction." Recorded as-is, per card. |
| `not_directly_comparable` | Judgment call from each card's `does_not_establish` text (`reproduced_by_this_package` is `false` for all 34) | `true` for every paper/repo describing an algorithm, benchmark or protocol not run under this project's exact protocol (29 of 34). `false` for the 6 sources that are not benchmark/protocol comparisons at all: R01 (this is the publisher card for our own dataset), R25/R26/R27/R31 (framework/runtime documentation), R32 (agent-tooling documentation). Reasoning is disclosed per entry in `not_directly_comparable_note`. |
| `package_raw` | The full REFERENCES.json record for that card, embedded verbatim | Included on every entry so every derived field above is independently auditable against its exact source without leaving this file. |

## Coverage and validation
- 34 of 34 REFERENCES.json entries appear exactly once (`R01`…`R34`, no duplicates, no gaps).
- Validated as strict JSON with the project's pinned interpreter:
  `next_item_eda\.venv\Scripts\python.exe` (`json.load` succeeds; `entry_count == len(entries) == 34`;
  `{entry ids} == {REFERENCES.json ids}`; every entry carries all task-required fields; `source_type` is
  one of the six allowed values; `not_directly_comparable` is boolean).

## Known limits (carried over from the package, not resolved here)
- No independent reproduction of any external benchmark was performed by this package or by this task;
  every card says so (`reproduced_by_this_package: false`, review_depth as above).
- Author and venue metadata is missing for all 34 sources and must be verified against the original
  source before thesis submission (the package explicitly flags this for R28 and R30; the same caveat
  applies uniformly here since the package never carries author/venue fields for any entry).
- The Time to Split DOI (R12) was access-limited at review time; the package used indexed publisher text
  and the author repository instead of the paywalled version.
