# Protocol Trial Vector KB Schema

## Goal

Define the canonical schema for the MedAI `protocol` trial knowledge base built from the TREC Clinical Trials corpus under `/home/yuanzy/MadAI/数据/totrials`.

This document does **not** define storage technology yet. It defines:

- what the online trial knowledge base record is
- which raw XML fields should be retained
- which fields should be embedded
- which fields should stay as metadata only
- how one trial should be chunked for retrieval

This order is intentional: schema first, vectorization second, persistence third.

## Source Scope

The `totrials` directory contains three different data roles:

1. `topics*.xml`
   These are query-side benchmark cases. They are used to validate retrieval and to understand what kinds of patient facts the retriever must match.
2. `qrels*.txt`
   These are relevance labels for evaluation. They are not part of the online trial KB.
3. `corpus_2021_2022/` and `corpus_2023/`
   These archives contain the actual ClinicalTrials.gov XML documents. These XML records are the source of truth for the trial KB.

The protocol trial KB should be built from the ClinicalTrials.gov XML corpus, not from `topics` or `qrels`.

## Design Principles

1. Keep the KB trial-centric.
   The online entity is a clinical trial identified by `nct_id`.
2. Separate source facts from retrieval text.
   Raw XML fields should be preserved in normalized form before assembling any embedding text.
3. Separate vector fields from filter fields.
   Not every field should be embedded. Some fields are better used for filtering, ranking, or audit only.
4. Support both long-text and structured-case queries.
   The 2021 and 2022 topics are narrative cases; the 2023 topics are template-like fielded cases. The schema must support both retrieval styles.
5. Make chunking explicit.
   Persistence format depends on whether we embed whole trials or chunks. Chunk schema must be defined before the index format.

## Non-Goals

- Do not store `qrels` inside the online KB.
- Do not treat TREC topic fields as trial metadata.
- Do not decide FAISS, Qdrant, Chroma, or any other backend in this document.
- Do not prematurely compress the raw XML into only the current MedAI treatment payload fields.

## Online Objects

Two online objects are needed.

1. `trial_record`
   The canonical normalized representation of one ClinicalTrials.gov trial.
2. `trial_chunk`
   A retrievable unit derived from a `trial_record`.

The retriever may search chunks, but the protocol node should consume trial-level candidates with evidence attached.

## Trial-Level Schema

### A. Identity and provenance

These fields uniquely identify the trial and allow audit back to the corpus snapshot.

| Field | Type | Required | Source | Notes |
| --- | --- | --- | --- | --- |
| `nct_id` | `str` | yes | XML `id_info/nct_id` | Primary key |
| `source_url` | `str` | no | XML `required_header/url` | Link back to ClinicalTrials.gov |
| `source_corpus` | `str` | yes | build-time metadata | `totrials_2021_2022` or `totrials_2023` snapshot source |
| `source_archive` | `str` | yes | build-time metadata | Zip shard file name |
| `source_member_path` | `str` | yes | zip member path | XML path inside archive |
| `source_snapshot_date` | `str` | no | XML `required_header/download_date` | Corpus processing date |
| `xml_sha256` | `str` | yes | derived | Used for rebuild detection |

### B. Core trial descriptors

These are the top-level descriptors that almost always matter for retrieval.

| Field | Type | Required | Source |
| --- | --- | --- | --- |
| `brief_title` | `str` | no | XML `brief_title` |
| `official_title` | `str` | no | XML `official_title` |
| `acronym` | `str` | no | XML `acronym` |
| `brief_summary` | `str` | no | XML `brief_summary/textblock` |
| `detailed_description` | `str` | no | XML `detailed_description/textblock` |
| `overall_status` | `str` | no | XML `overall_status` |
| `study_type` | `str` | no | XML `study_type` |
| `phase` | `str` | no | XML `phase` |
| `primary_purpose` | `str` | no | XML `study_design_info/primary_purpose` |
| `intervention_model` | `str` | no | XML `study_design_info/intervention_model` |
| `allocation` | `str` | no | XML `study_design_info/allocation` |
| `masking` | `str` | no | XML `study_design_info/masking` |
| `enrollment` | `str` | no | XML `enrollment` |
| `enrollment_type` | `str` | no | XML `enrollment/@type` |

### C. Condition and intervention descriptors

These fields are high-value retrieval anchors and should be preserved as lists.

| Field | Type | Required | Source |
| --- | --- | --- | --- |
| `conditions` | `list[str]` | no | XML `condition` |
| `condition_mesh_terms` | `list[str]` | no | XML `condition_browse/mesh_term` |
| `keywords` | `list[str]` | no | XML `keyword` |
| `interventions` | `list[str]` | no | XML `intervention/intervention_name` |
| `intervention_types` | `list[str]` | no | XML `intervention/intervention_type` |
| `intervention_descriptions` | `list[str]` | no | XML `intervention/description` |
| `intervention_mesh_terms` | `list[str]` | no | XML `intervention_browse/mesh_term` |
| `arm_group_labels` | `list[str]` | no | XML `arm_group/arm_group_label` |
| `arm_group_types` | `list[str]` | no | XML `arm_group/arm_group_type` |
| `arm_group_descriptions` | `list[str]` | no | XML `arm_group/description` |

### D. Eligibility and population descriptors

These are important both for semantic matching and later filtering.

| Field | Type | Required | Source |
| --- | --- | --- | --- |
| `eligibility_text` | `str` | no | XML `eligibility/criteria/textblock` |
| `gender` | `str` | no | XML `eligibility/gender` |
| `minimum_age` | `str` | no | XML `eligibility/minimum_age` |
| `maximum_age` | `str` | no | XML `eligibility/maximum_age` |
| `healthy_volunteers` | `str` | no | XML `eligibility/healthy_volunteers` |

### E. Outcome and evidence descriptors

These fields are especially useful for protocol-stage evidence grounding.

| Field | Type | Required | Source |
| --- | --- | --- | --- |
| `primary_outcome_measures` | `list[str]` | no | XML `primary_outcome/measure` |
| `primary_outcome_descriptions` | `list[str]` | no | XML `primary_outcome/description` |
| `primary_outcome_time_frames` | `list[str]` | no | XML `primary_outcome/time_frame` |
| `secondary_outcome_measures` | `list[str]` | no | XML `secondary_outcome/measure` |
| `secondary_outcome_descriptions` | `list[str]` | no | XML `secondary_outcome/description` |
| `secondary_outcome_time_frames` | `list[str]` | no | XML `secondary_outcome/time_frame` |
| `reference_pmids` | `list[str]` | no | XML `reference/PMID` |
| `reference_citations` | `list[str]` | no | XML `reference/citation` |

### F. Time and geography descriptors

These are mainly metadata and ranking aids.

| Field | Type | Required | Source |
| --- | --- | --- | --- |
| `start_date` | `str` | no | XML `start_date` |
| `completion_date` | `str` | no | XML `completion_date` |
| `primary_completion_date` | `str` | no | XML `primary_completion_date` |
| `study_first_submitted` | `str` | no | XML `study_first_submitted` |
| `study_first_posted` | `str` | no | XML `study_first_posted` |
| `last_update_submitted` | `str` | no | XML `last_update_submitted` |
| `last_update_posted` | `str` | no | XML `last_update_posted` |
| `countries` | `list[str]` | no | XML `location_countries/country` |
| `facility_cities` | `list[str]` | no | XML `location/facility/address/city` |
| `facility_states` | `list[str]` | no | XML `location/facility/address/state` |
| `facility_countries` | `list[str]` | no | XML `location/facility/address/country` |

## Derived Fields

These are normalized fields produced at build time. They are part of the KB contract even though they are not direct XML fields.

| Field | Type | Why |
| --- | --- | --- |
| `display_title` | `str` | `brief_title -> official_title -> nct_id` fallback |
| `normalized_status` | `str` | Stable normalization of `overall_status` for filtering/ranking |
| `condition_terms` | `list[str]` | Union of `conditions`, `condition_mesh_terms`, and `keywords` after dedupe |
| `intervention_terms` | `list[str]` | Union of intervention names and intervention mesh terms |
| `title_text` | `str` | Compact title string for retrieval assembly |
| `overview_text` | `str` | Concise high-signal summary built from title, summary, phase, purpose, conditions, interventions |
| `eligibility_inclusion_text` | `str` | Inclusion subsection when parsable |
| `eligibility_exclusion_text` | `str` | Exclusion subsection when parsable |
| `has_results_references` | `bool` | Whether references exist |
| `age_floor_years` | `float | null` | Parsed lower age for optional filtering |
| `age_ceiling_years` | `float | null` | Parsed upper age for optional filtering |

## Vectorization Policy

The KB should not embed every field the same way.

### Fields that should participate in semantic retrieval

These should be assembled into embedding text.

- `display_title`
- `acronym`
- `brief_summary`
- `detailed_description`
- `conditions`
- `condition_mesh_terms`
- `keywords`
- `interventions`
- `intervention_descriptions`
- `intervention_mesh_terms`
- `arm_group_descriptions`
- `eligibility_text`
- `primary_outcome_measures`
- `primary_outcome_descriptions`
- `secondary_outcome_measures`
- `secondary_outcome_descriptions`
- `reference_citations`

### Fields that should stay metadata-only

These are better for filtering, ranking, and audit than for direct embedding.

- `nct_id`
- `source_url`
- `source_corpus`
- `source_archive`
- `source_member_path`
- `xml_sha256`
- `overall_status`
- `normalized_status`
- `study_type`
- `phase`
- `primary_purpose`
- `allocation`
- `masking`
- `enrollment`
- `gender`
- `minimum_age`
- `maximum_age`
- `healthy_volunteers`
- `start_date`
- `completion_date`
- `primary_completion_date`
- `study_first_submitted`
- `study_first_posted`
- `last_update_submitted`
- `last_update_posted`
- `countries`
- `facility_cities`
- `facility_states`
- `facility_countries`
- `reference_pmids`

### Fields that should contribute to ranking boosts, not direct embedding weight

- `overall_status`
- `normalized_status`
- `phase`
- `study_type`
- `primary_purpose`
- country and site availability fields
- age and sex compatibility

## Chunk-Level Schema

The first retrievable unit should be a `trial_chunk`.

### Required fields

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `chunk_id` | `str` | yes | Stable ID, e.g. `NCT...::overview::0` |
| `nct_id` | `str` | yes | Parent trial |
| `chunk_type` | `str` | yes | One of the defined chunk types below |
| `sequence` | `int` | yes | Order within chunk type |
| `text` | `str` | yes | Human-readable chunk text |
| `embedding_text` | `str` | yes | Exact text passed to the embedding model |
| `source_fields` | `list[str]` | yes | Which trial fields produced this chunk |
| `token_estimate` | `int` | no | Optional build-time estimate |
| `rank_weight` | `float` | yes | Weight used during trial aggregation |

### Chunk types for v1

1. `overview`
   High-signal summary of title, condition, intervention, phase, study type, purpose, and brief summary.
2. `description`
   Detailed description chunks.
3. `eligibility_inclusion`
   Inclusion criteria chunks.
4. `eligibility_exclusion`
   Exclusion criteria chunks.
5. `outcomes`
   Primary and secondary outcome chunks.
6. `arms_interventions`
   Arm and intervention description chunks.

### Trial aggregation contract

Retrieval may return chunks, but protocol should consume trial candidates with at least:

- `nct_id`
- `display_title`
- `score`
- `matched_chunks`
- `best_evidence_text`
- `matched_fields`
- `overall_status`
- `phase`
- `study_type`
- `conditions`
- `interventions`

This is necessary so the protocol node can explain why a trial was surfaced instead of only exposing a title and status.

## Recommended Chunk Assembly

### 1. Overview chunk

Purpose: strong first-pass semantic match for both free-text and structured-case queries.

Suggested assembly:

- title
- acronym
- study type
- phase
- primary purpose
- overall status
- conditions
- condition mesh terms
- interventions
- brief summary

### 2. Description chunks

Purpose: capture trial rationale, disease context, and treatment framing from long text.

Suggested source:

- `detailed_description`

Chunking:

- paragraph-preserving
- target moderate chunk size
- allow overlap only if needed later during implementation

### 3. Eligibility chunks

Purpose: support direct matching against patient inclusion and exclusion details.

Suggested source:

- `eligibility_text`

Preferred split:

- split into inclusion vs exclusion when the source text clearly contains both
- otherwise keep paragraph chunks under a generic eligibility chunk type

### 4. Outcomes chunks

Purpose: surface what the study is actually trying to measure, which matters at protocol stage.

Suggested source:

- primary outcome measures and descriptions
- secondary outcome measures and descriptions

### 5. Arms and interventions chunks

Purpose: surface concrete treatment/control structure when the query implies a therapy choice.

Suggested source:

- `arm_group_*`
- intervention names and descriptions

## What Not To Put Into The KB

The following belong to evaluation or query processing, not to the trial corpus object itself.

- TREC topic number
- TREC template name as trial metadata
- qrels relevance labels
- topic-side field names such as `HAM-A`, `GOLD stage`, `oxygen saturation`, `HER2`

Those query-side fields matter, but they should be matched against trial chunks, not written back as if they were native trial fields.

## Why Persistence Is Not First

Persistence depends on these decisions:

1. Are we storing one vector per trial or many vectors per trial chunk?
2. Which normalized fields are retained in the online object?
3. Which fields are assembled into `embedding_text`?
4. Which chunk types exist?
5. Which metadata fields must be available at retrieval time for filtering and ranking?

Without those answers, "persistent storage" is underspecified. We might otherwise build:

- the wrong index granularity
- the wrong metadata payload
- the wrong rebuild key
- the wrong retrieval output contract

So the correct order for this project is:

1. define trial schema
2. define chunk schema
3. define vectorization assembly
4. define retrieval output contract
5. then choose persistence format and build pipeline

## Recommended Next Step

After this schema is accepted, implement a build script that:

1. reads ClinicalTrials.gov XML from `数据/totrials/corpus_*`
2. emits normalized `trial_record` JSONL
3. emits `trial_chunk` JSONL
4. leaves actual vector index construction as the next step

That will let MedAI validate field coverage before committing to a vector backend.
