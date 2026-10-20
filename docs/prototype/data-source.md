# REES46 Data Source & Dataset Manifest — October 2019

## 1. Purpose

This document is the single source record for the primary raw dataset used in Phase 1 of the project **Privacy-Preserving Shopping Intelligence via Federated & On-Device Learning**.

It intentionally consolidates:
- data source and provenance,
- raw-file identity,
- SHA-256 checksum,
- Dataset Manifest v1,
- schema verification,
- row-count and date-range verification,
- event vocabulary,
- storage and reproducibility notes.

No separate checksum or manifest report is maintained for this task.

---

## 2. Dataset Selection

- **Dataset:** REES46 Multi-category eCommerce Behavior Data
- **Dataset type:** Behavior events
- **Selected partition:** October 2019
- **Raw filename:** `2019-Oct.csv`
- **Role:** Primary Phase 1 dataset
- **Intended use:** T1, T2, T3 and the centralized / federated / on-device experimental pipeline

---

## 3. Source & Provenance

### Official REES46 dataset page

`https://rees46.com/en/datasets`

Selected dataset entry:

- **Category:** Multi-category
- **Type:** Behavior events
- **Published period:** October 2019 – April 2020

### Download source used

`https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store?select=2019-Oct.csv`

The October 2019 partition, `2019-Oct.csv`, was downloaded from this release.

### Acquisition date

`2026-08-15`

---

## 4. Raw File Identity

| Field | Value |
|---|---|
| Filename | `2019-Oct.csv` |
| Size | `5,668,612,855 bytes` |
| SHA-256 | `FEDD938409B5F836EC89B39C861B13DAD99FC7CD9BEB1FDDD97A2D50488B5B80` |
| Raw rows | `42,448,764` |
| Minimum timestamp | `2019-10-01 00:00:00 UTC` |
| Maximum timestamp | `2019-10-31 23:59:59 UTC` |

### Checksum record

```text
FEDD938409B5F836EC89B39C861B13DAD99FC7CD9BEB1FDDD97A2D50488B5B80  2019-Oct.csv
```

The checksum was calculated directly from the downloaded raw CSV before preprocessing.

---

## 5. Dataset Manifest v1

```yaml
manifest_version: 1

dataset:
  name: REES46 Multi-category eCommerce Behavior Data
  partition: 2019-10
  role: primary
  intended_use:
    - T1
    - T2
    - T3
    - centralized_training
    - federated_training
    - on_device_evaluation

source:
  official_page: https://rees46.com/en/datasets
  download_page: https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store?select=2019-Oct.csv
  acquisition_date: 2026-08-15

raw_file:
  filename: 2019-Oct.csv
  size_bytes: 5668612855
  sha256: FEDD938409B5F836EC89B39C861B13DAD99FC7CD9BEB1FDDD97A2D50488B5B80
  row_count: 42448764

time_range:
  min: "2019-10-01 00:00:00 UTC"
  max: "2019-10-31 23:59:59 UTC"

event_types:
  - view
  - cart
  - purchase

columns:
  - event_time
  - event_type
  - product_id
  - category_id
  - category_code
  - brand
  - price
  - user_id
  - user_session
```

---

## 6. Read-Only Verification

The raw CSV was inspected before preprocessing.

### Expected columns

All 9 expected source fields were verified:

1. `event_time`
2. `event_type`
3. `product_id`
4. `category_id`
5. `category_code`
6. `brand`
7. `price`
8. `user_id`
9. `user_session`

### Observed event vocabulary

The full-file validation returned exactly:

- `view`
- `cart`
- `purchase`

No additional event type was observed in the October 2019 partition.

### Date-range verification

The observed event window is:

`2019-10-01 00:00:00 UTC` → `2019-10-31 23:59:59 UTC`

### Row-count verification

Observed raw rows:

`42,448,764`

---

## 7. Initial Source Schema Observations

Read-only inspection showed:

- `event_time`: source timestamp text in UTC
- `event_type`: text
- `product_id`: numeric identifier
- `category_id`: numeric identifier
- `category_code`: text; may contain missing values
- `brand`: text; may contain missing values
- `price`: numeric
- `user_id`: numeric identifier
- `user_session`: text identifier

These are source-level observations only. Canonical dtypes, nullability rules, normalization, and downstream contracts are defined later in the data pipeline.

---

## 8. Raw Data Storage Policy

The raw multi-GB CSV is stored locally outside the Git repository.

The repository should contain code, configuration, documentation, processing logic, reproducibility metadata, and aggregate experiment outputs.

The raw `2019-Oct.csv` file should not be committed to normal Git history.

---

## 9. Attribution & Use Notes

The project uses the dataset for academic research and experimentation.

Source attribution should preserve reference to:
- the Kaggle dataset release,
- REES46.

Before any future external redistribution of raw data, the current source/hosting terms should be reviewed again.

---

## 10. Verification Status

- [x] Raw file downloaded
- [x] Filename recorded
- [x] File size recorded
- [x] SHA-256 calculated
- [x] CSV readability verified
- [x] Expected 9 columns verified
- [x] Full row count verified
- [x] October 2019 time window verified
- [x] Event vocabulary verified
- [x] Raw data stored outside Git
- [x] Source and provenance recorded
- [x] Dataset Manifest v1 included in this document
- [x] Checksum record included in this document

---

## 11. Handoff

This verified raw dataset is ready to be used as the input for the next data-pipeline task:

**S1-DS-02 — CSV → Parquet**

All downstream data artifacts should remain traceable to the raw-file identity recorded in this document.
