"""Public sequential-recommendation benchmarks in one leave-one-out layout.

Converters for the Amazon Reviews 2023 5-core release (`amazon2023`) and the MBHT Taobao / Tmall session data
(`mbht`) write the ml-1m style layout used by public SASRec benchmark releases (leave_one_out/train.csv +
holdout.csv + statistics.csv; `layout`, `check_layout`). `sampled_metrics` computes sampled ranking metrics for
comparison with papers that report them. CPU only; nothing here reads the project's private data.
"""
