"""v4data.reference_counts (owner: Data Lead, DATA-04) - the census and receipt values the build must reproduce.

Each value is QUOTED from the named file (paths relative to .local_ai/ppsi_v4/); nothing here is computed.
`code/tests_data/test_reference_counts.py` re-reads those files locally and asserts every value, so the copy cannot
drift. The server has no g0/ census files, which is why the values travel with the code.
No RETRO_TEST number appears here (EVALUATION_CONTRACT_V4 section 10.4).
"""
from __future__ import annotations

# key -> (value, source)
FULL_POPULATION = {
    "TRAIN.decisions": (52760314, "g0/data_census/SPLIT_COUNTS.json per_old_split.TRAIN.decisions"),
    "TRAIN.loss_mask_true": (40180608, "SPLIT_COUNTS.json per_old_split.TRAIN.loss_mask_true"),
    "TRAIN.target_oov_true": (56560, "SPLIT_COUNTS.json per_old_split.TRAIN.target_oov_true"),
    "TRAIN.distinct_users": (4204379, "SPLIT_COUNTS.json per_old_split.TRAIN.distinct_users"),
    "TRAIN.status.OBSERVED_IN_PARTITION": (40237168, "SPLIT_COUNTS.json per_old_split.TRAIN.target_status_counts"),
    "TRAIN.status.NO_OBSERVED_NEXT": (12512673, "SPLIT_COUNTS.json per_old_split.TRAIN.target_status_counts"),
    "TRAIN.status.AMBIGUOUS_BOUNDARY": (10472, "SPLIT_COUNTS.json per_old_split.TRAIN.target_status_counts"),
    "TRAIN.status.BOUNDARY_UNRESOLVED": (1, "SPLIT_COUNTS.json per_old_split.TRAIN.target_status_counts"),
    "VALIDATION_SELECT.decisions": (10930090, "SPLIT_COUNTS.json per_old_split.VALIDATION_SELECT.decisions"),
    "VALIDATION_SELECT.eval_mask_true": (8826488, "SPLIT_COUNTS.json per_old_split.VALIDATION_SELECT.eval_mask_true"),
    "VALIDATION_SELECT.distinct_users": (1139657, "SPLIT_COUNTS.json per_old_split.VALIDATION_SELECT.distinct_users"),
    "VALIDATION_CONFIRM.decisions": (3427555, "SPLIT_COUNTS.json per_old_split.VALIDATION_CONFIRM.decisions"),
    "VALIDATION_CONFIRM.eval_mask_true": (2461568, "SPLIT_COUNTS.json per_old_split.VALIDATION_CONFIRM.eval_mask_true"),
    "VALIDATION_CONFIRM.distinct_users": (700093, "SPLIT_COUNTS.json per_old_split.VALIDATION_CONFIRM.distinct_users"),
    "VALIDATION.decisions": (14357645, "SPLIT_COUNTS.json v4_validation_union.decisions"),
    "VALIDATION.eval_mask_true": (11288056, "SPLIT_COUNTS.json v4_validation_union.eval_mask_true"),
    "VALIDATION.target_oov_true": (96871, "SPLIT_COUNTS.json v4_validation_union.target_oov_true"),
    "VALIDATION.target_class_ge0": (11191185, "SPLIT_COUNTS.json v4_validation_union.target_class_ge0"),
    "VALIDATION.distinct_users": (1578912, "SPLIT_COUNTS.json v4_validation_union.distinct_users"),
    "VALIDATION.status.OBSERVED_IN_PARTITION": (11288056, "SPLIT_COUNTS.json v4_validation_union.target_status_counts"),
    "VALIDATION.status.NO_OBSERVED_NEXT": (3064778, "SPLIT_COUNTS.json v4_validation_union.target_status_counts"),
    "VALIDATION.status.AMBIGUOUS_BOUNDARY": (4489, "SPLIT_COUNTS.json v4_validation_union.target_status_counts"),
    "VALIDATION.status.BOUNDARY_UNRESOLVED": (322, "SPLIT_COUNTS.json v4_validation_union.target_status_counts"),
    "VALIDATION.censor.CEN_LEGACY_BARRIER": (170, "company/reviews/G0B_PROTOCOL_FABLE_POSITION.md:32; g0/BARRIER_COUNT.json L21"),
    "VALIDATION.censor.CEN_AMBIGUOUS_NEXT_BARRIER": (4489, "G0B_PROTOCOL_FABLE_POSITION.md:32; SPLIT_COUNTS.json L83"),
    "VALIDATION.censor.CEN_RIGHT_EDGE": (152, "G0B_PROTOCOL_FABLE_POSITION.md:32; g0/BARRIER_COUNT.json L16"),
    "VALIDATION.censor.CEN_NO_NEXT": (3064778, "G0B_PROTOCOL_FABLE_POSITION.md:32; SPLIT_COUNTS.json L82"),
    "label_time.train_loss_target_ts_ge_train_end": (0, "g0/data_census/LABEL_TIME_AUDIT.json"),
    "label_time.train_loss_target_ts_null": (0, "LABEL_TIME_AUDIT.json"),
    "label_time.val_eval_target_ts_ge_val_end": (0, "LABEL_TIME_AUDIT.json"),
    "label_time.decisions_ts_ge_2019_11_30_non_test": (0, "LABEL_TIME_AUDIT.json decisions_with_decision_ts_ge_2019_11_30"),
    "catalogue.n_rows": (158486, "g0/data_census/CATALOGUE_SUPPORT.json verification.n_catalogue_rows"),
    "catalogue.support_mismatch": (0, "CATALOGUE_SUPPORT.json verification.n_mismatch"),
    "catalogue.K1_products_with_train_support": (191890, "CATALOGUE_SUPPORT.json verification.distinct_products_with_any_train_support_K1"),
    "VALIDATION.oov_absent_from_train_s1_excluded": (50124, "CATALOGUE_SUPPORT.json thresholds.1.validation_union_label_coverage.excluded_count"),
}

POPULATION = {
    "universe_size": (2694998, "g0/data_census/COHORT_PROJECTIONS.json universe_size; g0/receipts_qa01/COHORT_RECEIPTS.json"),
    "users_in_train_zero_eligible": (1509381, "g0/data_census/CLIENT_CENSUS.json s5.users_in_train_with_zero_eligible_examples"),
    "val_users_with_ge1_eval": (1023523, "CLIENT_CENSUS.json s5.validation_union_users.total_with_ge1_eval_row"),
    "val_users_warm": (642228, "CLIENT_CENSUS.json s5.validation_union_users.warm_ge1_eligible_train_example"),
    "val_users_cold": (381295, "CLIENT_CENSUS.json s5.validation_union_users.cold_zero_eligible_train_examples"),
    "s5_total_examples": (40180608, "CLIENT_CENSUS.json s5.total_examples"),
    "cohort16384.population_sha256": ("3b845f51260fab7d10590827eafd0c9d8b58a3813f482d88b5ca4826f77a8c92", "COHORT_PROJECTIONS.json cohorts.16384"),
    "cohort65536.population_sha256": ("3d680aa7222fb5c9246d03fe6088464c78095d90a344e3453acb68e98d53a6f4", "COHORT_PROJECTIONS.json cohorts.65536"),
    "cohort262144.population_sha256": ("54e1d3c2539fd4cd7fe22d2b8ef91bed1a786d69be27867406bb19141ac48184", "COHORT_PROJECTIONS.json cohorts.262144"),
    "cohortALL.population_sha256": ("db46b81d6ceb7d0c766e76ca2d0f5dede905cedf4b91881e168be4476b9bee14", "COHORT_PROJECTIONS.json cohorts.ALL"),
    "cohort262144.eligible_examples": (3906929, "COHORT_PROJECTIONS.json cohorts.262144.eligible_train_examples_total"),
    "cohort262144.val_eval_rows": (808610, "COHORT_PROJECTIONS.json cohorts.262144.validation_union_eval_rows_total"),
    "cohort262144.users_ge1_eval": (62368, "COHORT_PROJECTIONS.json cohorts.262144.users_with_ge1_validation_eval_row"),
    "cohort65536.eligible_examples": (990362, "COHORT_PROJECTIONS.json cohorts.65536.eligible_train_examples_total"),
    "cohort65536.val_eval_rows": (200985, "COHORT_PROJECTIONS.json cohorts.65536.validation_union_eval_rows_total"),
    "cohort65536.users_ge1_eval": (15487, "COHORT_PROJECTIONS.json cohorts.65536.users_with_ge1_validation_eval_row"),
    "cohort262144.rankable": (801762, "g0/receipts_qa01/COHORT_RECEIPTS.json cohorts.262144"),
    "cohort262144.oov": (6848, "COHORT_RECEIPTS.json cohorts.262144"),
    "cohort262144.classes_ge1_positive": (133154, "COHORT_RECEIPTS.json cohorts.262144"),
    "cohort262144.classes_ge5_positive": (72682, "COHORT_RECEIPTS.json cohorts.262144"),
    "cohort262144.val_classes_zero_positive": (5158, "COHORT_RECEIPTS.json cohorts.262144"),
    "cohort65536.rankable": (199276, "COHORT_RECEIPTS.json cohorts.65536"),
    "cohort65536.oov": (1709, "COHORT_RECEIPTS.json cohorts.65536"),
    "cohort65536.classes_ge1_positive": (90545, "COHORT_RECEIPTS.json cohorts.65536"),
    "cohort65536.classes_ge5_positive": (31425, "COHORT_RECEIPTS.json cohorts.65536"),
    "cohort65536.val_classes_zero_positive": (7619, "COHORT_RECEIPTS.json cohorts.65536"),
    "cold.COLD_NO_TRAIN_EVENTS.users": (299035, "g0/receipts_qa01/COLD_STRATA.json VALIDATION_union"),
    "cold.COLD_NO_TRAIN_EVENTS.eval_rows": (2270516, "COLD_STRATA.json VALIDATION_union"),
    "cold.COLD_TRAIN_EVENTS_NO_LABEL.users": (82260, "COLD_STRATA.json VALIDATION_union"),
    "cold.COLD_TRAIN_EVENTS_NO_LABEL.eval_rows": (587149, "COLD_STRATA.json VALIDATION_union"),
    "cohort262144.per_client_min": (1, "COHORT_PROJECTIONS.json cohorts.262144.examples_quantiles.min"),
    "cohort262144.per_client_p50": (6, "COHORT_PROJECTIONS.json cohorts.262144.examples_quantiles.p50"),
    "cohort262144.per_client_max": (3014, "COHORT_PROJECTIONS.json cohorts.262144.examples_quantiles.max"),
}


def value(key: str):
    if key in FULL_POPULATION:
        return FULL_POPULATION[key][0]
    return POPULATION[key][0]
