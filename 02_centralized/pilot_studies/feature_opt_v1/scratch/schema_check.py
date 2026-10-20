import json, re, pyarrow.parquet as pq
names = set(pq.read_schema("/home/jovyan/shared_data/rees46_nextitem_v1/model_ready/user_bucket=0/features.parquet").names)
src = open("code/stream_fit.py").read() + open("code/feature_view.py").read()
fg = json.load(open("/home/jovyan/shared_data/rees46_nextitem_v1/model_ready/FEATURE_GROUPS.json"))
need = set(["split", "local_weekday_idx", "daypart_idx"]) | set(fg["event_numeric_block_order"]) | set(fg["event_quality_flags_order"]) | set(fg["user_context_order"]) | set(fg["user_context_masks_order"])
print("n_columns", len(names), "missing:", sorted(need - names))
