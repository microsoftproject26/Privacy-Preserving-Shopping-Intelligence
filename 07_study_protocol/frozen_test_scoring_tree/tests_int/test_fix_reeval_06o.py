"""FIX-REEVAL-06O (PREG1_AMENDMENT_06O O1-O2) - v4int.reeval admits the registered central sources C_FULL (seeds 2026 /
2027: best.pt = PRACTICAL_BEST, endpoint_6.0.pt = CONTROLLED_BUDGET) and PRE (theta_pre = endpoint_6.0.pt) on role
UNSEEN_VAL as DESCRIPTIVE preview references (CPU only; the real C_FULL s2026 / PRE RUN_START fixtures of FIX-REEVAL,
the s2027 record derived from the real s2026 one exactly as ANA-05 does).

  * the dry run of C_FULL s2026 / s2027 PB / CB and PRE (CB and the PB label) on UNSEEN_VAL passes EVERY source check
    (classification, FIX-REEVAL binding = the real registry hashes, checkpoint file / kind / mark / meta, PRE init
    record) and plans the ROLE key; it reports preview_reference NON_MAIN_UNSEEN_VAL and the DESCRIPTIVE row labels,
    and no no_ft_reference;
  * every existing source refusal still refuses on UNSEEN_VAL (binding, config hash, checkpoint sha, meta, file rule,
    kind, mark, PRE init record, seed, pair / claims, a MAIN-pair run under a source label);
  * rows: comparison_status DESCRIPTIVE + preview_reference (none a comparison-key field); REEVAL_RECORD carries
    preview_reference; the C_SMALL / FA no-FT rows, records, dry runs and refusal text are the pre-06O ones (pinned);
    C_FULL / PRE on MAIN and the panels are unchanged;
  * tools/ana_block_index.py lists the reeval-written preview dirs as EXCLUDED (never an entry) beside an admitted
    C_SMALL no-FT dir; an unlabelled C_FULL UNSEEN_VAL record keeps the old refusal; a preview label elsewhere refuses.
Negative controls (test_nc_*, strict xfail on AssertionError).
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from intkit import nc
from test_ana05_reeval import derived_start, rehash, s2027_run
from test_dev_ftpre import _central, _fl_run, _rv
from test_gate_b_reeval import REG, _sha, argv, real_start, source_run
from v4int import reeval as R

V4 = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "reeval_records"
TAG = {"C_FULL": "CFULL_s2026", "PRE": "PRE"}
BASE_SELF = json.loads((REG / "REGISTRY_V2.json").read_text(encoding="utf-8"))["self_sha256"]
PB, CB = "PRACTICAL_BEST", "CONTROLLED_BUDGET"
UV = "UNSEEN_VAL"
UV_EVAL = "UNEW_VAL_VALIDATION_32768"
PREVIEW = "NON_MAIN_UNSEEN_VAL"
# the pre-06O refusal text of check_no_ft_args (reeval 615f1c60), written out here
OLD_NO_FT_MSG = ("role UNSEEN_VAL is the device selection panel: only the no-FT references C_SMALL (--ckpt-kind central, "
                 "SASREC_D64_B2), FA (--ckpt-kind fl, SASREC_D64_B2) are evaluated there, not {arm} / --ckpt-kind {kind} "
                 "/ {variant} (06L-C §C16)")
OLD_NO_FT_RULE = ("06L-C §C16: an evaluation-only 'no-FT' reference of C_SMALL / FA on UNSEEN_VAL (the device selection "
                  "panel); DESCRIPTIVE, never a headline and never a selection input")


def rec(name: str, kind: str) -> dict:
    return json.loads((FIXTURES / f"{TAG[name]}_{kind}.json").read_text(encoding="utf-8"))


def preview_run(t: Path, name: str, seed: int, ckpt: str) -> Path:
    return s2027_run(t, ckpt=ckpt) if seed == 2027 else source_run(t, name, ckpt=ckpt)


# ------------------------------------------------------------------------------------------------ the dry run
CASES = (("C_FULL", 2026, "best.pt", PB), ("C_FULL", 2026, "endpoint_6.0.pt", CB),
         ("C_FULL", 2027, "best.pt", PB), ("C_FULL", 2027, "endpoint_6.0.pt", CB),
         ("PRE", 2026, "endpoint_6.0.pt", CB), ("PRE", 2026, "endpoint_6.0.pt", PB))


def check_preview_dry_run(tmp_path, run=R.run):
    for i, (name, seed, ckpt, rule) in enumerate(CASES):
        t = tmp_path / f"pv{i}"
        cp = preview_run(t, name, seed, ckpt)
        try:
            out = run(argv(cp, t, name, role=UV, selection_rule=rule, seed=seed))
        except R.ReevalRefused as e:
            raise AssertionError(f"{name} s{seed} {rule} on UNSEEN_VAL refused: {e}")
        assert out["dry_run"] and out["role"] == UV and out["arm_label"] == name, out
        assert out.get("preview_reference") == PREVIEW and "no_ft_reference" not in out, out
        cs = out["central_source"]
        assert cs["source"] == name and cs["checks"]["all_ok"], cs
        st, la, sh = rec(name, "RUN_START"), rec(name, "LAUNCH_0"), rec(name, "SHIM_0")
        b = cs["binding"]                                             # the REAL registry binding (FIX-REEVAL)
        assert b["comparison_key_registry_self_sha256"] == la["comparison_key"]["manifest_registry_self_sha256"]
        assert b["trainer_view_self_sha256"] == st["key_base"]["manifest_registry_self_sha256"] == \
            sh["trainer_view"]["self_sha256"]
        assert b["view_checks"]["all_ok"] and b["ext_file_sha256"] == sh["ext_registry"]["file_sha256"]
        assert out["planned_key"].startswith("ROLE_KEY"), out["planned_key"]
        assert out["checkpoint_file"]["file"] == ckpt and out["checkpoint_meta"]["seed"] == seed
        lab = out["row_labels"]
        assert lab["central_source"] == name and lab["comparison_status"] == "DESCRIPTIVE", lab
        assert lab["preview_reference"] == PREVIEW and lab["preview_reference_rule"] == R.PREVIEW_RULE
        assert not any(k.startswith("no_ft") for k in lab), lab
        assert (cs.get("seed_status") == "DESCRIPTIVE") == (seed == 2027), cs
        if name == "PRE":
            assert out["pre_init_record"]["fl_init_sha256"] == _sha(cp)
        assert not (t / "out").exists()


def test_c_full_and_pre_preview_dry_run_on_unseen_val(tmp_path):
    check_preview_dry_run(tmp_path)


def test_other_roles_unchanged_for_the_sources(tmp_path):
    """C_FULL / PRE on MAIN and a panel: no preview key in the dry-run record or its row labels."""
    for name, role, ckpt, rule in (("C_FULL", "MAIN", "best.pt", PB), ("PRE", "UNSEEN_HOLDOUT", "endpoint_6.0.pt", CB)):
        t = tmp_path / f"{name}_{role}"
        out = R.run(argv(source_run(t, name, ckpt=ckpt), t, name, role=role, selection_rule=rule))
        assert "preview_reference" not in out and "no_ft_reference" not in out
        assert not any(k.startswith("preview") for k in out["row_labels"]) and "comparison_status" not in \
            out["row_labels"]


# ------------------------------------------------------------------------------------------------ refusals
def _edited(name, fn):
    st = copy.deepcopy(real_start(name))
    fn(st)
    return st


def _base_key(s):
    s["config"]["key"]["manifest_registry_self_sha256"] = BASE_SELF
    rehash(s)


def refusal_cases() -> list:
    """(tag, name, run kwargs for source_run, argv kwargs, message) - every source check on UNSEEN_VAL."""
    return [
        ("cf_base_key", "C_FULL", dict(start=_edited("C_FULL", _base_key)), {}, "FIX-REEVAL"),
        ("pre_base_key", "PRE", dict(start=_edited("PRE", _base_key)), dict(selection_rule=CB), "FIX-REEVAL"),
        ("cf_not_rehashed", "C_FULL", dict(start=_edited("C_FULL", lambda s: s["config"].update(peak_lr=1.0))), {},
         "config_sha256 = RunConfig.sha256"),
        ("cf_kb_ext", "C_FULL", dict(kb_over={"manifest_registry_self_sha256": "f" * 64}), {},
         "key_base.manifest_registry_self_sha256"),
        ("pre_view", "PRE", dict(start=_edited("PRE", lambda s: s["manifest_registry"].update(self_sha256="0" * 64))),
         dict(selection_rule=CB), "RUN_START.manifest_registry.self_sha256"),
        ("cf_sha", "C_FULL", {}, dict(expect_sha256="0" * 64), "--expect-sha256"),
        ("cf_meta", "C_FULL", dict(meta_over={"config_sha256": "d" * 64}), {}, "config_sha256"),
        ("cf_rule", "C_FULL", dict(ckpt="best.pt"), dict(selection_rule=CB), "--selection-rule"),
        ("cf_e8", "C_FULL", dict(ckpt="endpoint_8.0.pt"), dict(selection_rule=CB), "checkpoint file"),
        ("cf_kind", "C_FULL", dict(ckpt="best.pt", kind="endpoint_6.0"), {}, "carries kind"),
        ("cf_mark", "C_FULL", dict(ckpt="endpoint_6.0.pt", meta_over={"mark": 5.5}), dict(selection_rule=CB),
         "meta mark"),
        ("pre_best", "PRE", dict(ckpt="best.pt"), {}, "checkpoint file"),
        ("pre_init_sha", "PRE", dict(init_over={"sha256": "0" * 64}), dict(selection_rule=CB), "fl_init.sha256"),
        ("pre_no_init", "PRE", dict(init=False), dict(selection_rule=CB), "PRETRAIN_FOR_FL_INIT.json"),
        ("pre_s2027", "PRE", {}, dict(seed=2027), "seed"),
        ("cf_s2028", "C_FULL", dict(start=rehash(_edited("C_FULL", lambda s: s["config"].update(seed=2028))),
                                     meta_over={"seed": 2028}), dict(seed=2028), "seed"),
        ("cf_pair", "C_FULL", dict(start_over={"eval_provider": {"manifest_id": UV_EVAL}}), {}, "eval_provider"),
        ("cf_run_id", "C_FULL", dict(start_over={"run_id": "C_BIG_02B_SASREC_s2026"}), {}, "run id prefix"),
        ("pre_as_cfull", "PRE", {}, dict(arm_label="C_FULL", variant="SASREC_D256_B3", recipe_id="02B_SASREC"),
         "model_variant"),                                             # the PRE run under the C_FULL label
        ("pre_label_on_cfull", "C_FULL", {}, dict(arm_label="PRE"), "only the no-FT references"),   # PRE != D256
    ]


def check_preview_refusals(tmp_path, run=R.run):
    for tag, name, run_kw, arg_kw, msg in refusal_cases():
        t = tmp_path / tag
        cp = source_run(t, name, **run_kw)
        refused = None
        try:
            run(argv(cp, t, name, role=UV, **arg_kw))
        except R.ReevalRefused as e:
            refused = str(e)
        assert refused is not None and msg in refused, (tag, msg, refused)
        assert not (t / "out").exists()


def test_every_source_check_still_refuses_on_unseen_val(tmp_path):
    check_preview_refusals(tmp_path)


def test_s2027_binding_still_refuses_on_unseen_val(tmp_path):
    st = derived_start()
    st["config"]["key"]["manifest_registry_self_sha256"] = BASE_SELF
    cp = s2027_run(tmp_path, start=rehash(st))
    with pytest.raises(R.ReevalRefused, match="FIX-REEVAL"):
        R.run(argv(cp, tmp_path, "C_FULL", role=UV, seed=2027))


def test_source_labels_on_non_source_runs_refuse(tmp_path):
    """A MAIN-pair run under a source label on UNSEEN_VAL: the source classification refuses (C_BIG-shaped as C_FULL,
    C_SMALL-shaped as PRE); the preview source guard itself refuses a non-source / other-source run."""
    run = tmp_path / "cbig" / "run"
    cp = run / "sealed" / "checkpoints" / "best.pt"
    cp.parent.mkdir(parents=True)
    cp.write_bytes(b"x")
    (run / "RUN_START.json").write_text(json.dumps({
        "run_id": "02B_SASREC_MAIN_s2026", "config": {"family": "SASREC", "model_variant": "SASREC_D256_B3",
                                                       "recipe_id": "02B_SASREC", "seed": 2026},
        "key_base": {"split": "VALIDATION", "run_label": "PRODUCTION"},
        "train_provider": {"manifest_id": "MAIN_TRAIN_5_100_131072"},
        "eval_provider": {"manifest_id": "MAIN_VALIDATION_5_100_131072"}}), encoding="utf-8")
    with pytest.raises(R.ReevalRefused, match="not the registered C_FULL"):
        R.run(argv(cp, tmp_path / "cbig", "C_FULL", role=UV))
    with pytest.raises(R.ReevalRefused, match="not the registered PRE"):
        R.run(_rv(_central(tmp_path / "c"), tmp_path / "c", arm="PRE"))
    check_preview_source_guard()


def check_preview_source_guard():
    """check_no_ft_source (the re-assertion after classify_central_source): a C_FULL / PRE preview whose run is not that
    registered source refuses (a MAIN-pair run, the other source, an FL run); the matching source passes."""
    for arm, src in (("C_FULL", {"kind": "central", "central_source": None}),
                     ("PRE", {"kind": "central", "central_source": "C_FULL"}),
                     ("C_FULL", {"kind": "central", "central_source": "PRE"}),
                     ("PRE", {"kind": "fl", "arm": "FA", "central_source": "PRE"})):
        refused = None
        try:
            R.check_no_ft_source(SimpleNamespace(role=UV, arm_label=arm), src)
        except R.ReevalRefused as e:
            refused = str(e)
        assert refused is not None and "preview reference" in refused and "06O" in refused, (arm, src, refused)
    for arm in ("C_FULL", "PRE"):
        R.check_no_ft_source(SimpleNamespace(role=UV, arm_label=arm), {"kind": "central", "central_source": arm})


def test_pre_06o_refusal_text_is_unchanged(tmp_path):
    """Every other arm / kind / variant on UNSEEN_VAL refuses with the byte-identical pre-06O text; a source label on
    an FL checkpoint keeps its check_args refusal."""
    cp = _central(tmp_path / "c")
    for arm, variant, recipe in (("C_BIG", "SASREC_D64_B2", "SMALL_SASREC_D64_B2"),
                                 ("C_FULL", "SASREC_D64_B2", "SMALL_SASREC_D64_B2"),       # a source, wrong variant
                                 ("PRE", "SASREC_D256_B3", "02B_SASREC"),
                                 ("FT_C", "SASREC_D64_B2", "SMALL_SASREC_D64_B2")):
        with pytest.raises(R.ReevalRefused) as ei:
            R.run(_rv(cp, tmp_path / "c", arm=arm, variant=variant, recipe=recipe))
        assert str(ei.value) == OLD_NO_FT_MSG.format(arm=arm, kind="central", variant=variant), str(ei.value)
    with pytest.raises(R.ReevalRefused, match="registered CENTRAL source"):
        R.run(_rv(_fl_run(tmp_path / "f"), tmp_path / "f", kind="fl", arm="C_FULL"))


# ------------------------------------------------------------------------------------------------ labels / records
def _args(role=UV, arm="C_FULL", rule=PB):
    return SimpleNamespace(role=role, arm_label=arm, selection_rule=rule, checkpoint_id=f"{arm}:ckpt",
                           ckpt_kind="central")


def check_preview_labels():
    """run()'s labelling (label_rows) and record (add_no_ft_record) for the preview references."""
    src_reg = {"ext_self_sha256": "e" * 64}
    for arm, seed in (("C_FULL", 2026), ("C_FULL", 2027), ("PRE", 2026)):
        extra = dict(R.source_row_labels(arm, src_reg), **R.source_seed_labels(arm, seed))
        row = R.label_rows([{"metric": "mrr", "k": 20}], _args(arm=arm), checkpoint_sha256="c" * 64,
                           source_run_id=f"{arm}_run", efe=6.0, extra=extra)[0]
        assert row["comparison_status"] == "DESCRIPTIVE" and row.get("preview_reference") == PREVIEW, row
        assert row["preview_reference_rule"] == R.PREVIEW_RULE and row["row_status"] == "REEVAL"
        assert row["central_source"] == arm and not any(k.startswith("no_ft") for k in row)
        assert R.add_no_ft_record({"role": UV}, _args(arm=arm)) == {
            "role": UV, "preview_reference": PREVIEW, "preview_reference_rule": R.PREVIEW_RULE}


def test_preview_rows_and_record_are_labelled():
    check_preview_labels()


def test_preview_labels_are_not_comparison_key_fields():
    from v4eval.rows import COMPARISON_KEY
    assert not set(R.preview_labels(UV, "C_FULL")) & set(COMPARISON_KEY)
    assert R.preview_labels("MAIN", "C_FULL") == {} and R.preview_labels(UV, "C_SMALL") == {}
    assert R.PREVIEW_SOURCES == ("C_FULL", "PRE") and set(R.PREVIEW_SOURCES) <= set(R.CENTRAL_SOURCES)


def test_no_ft_and_main_labels_are_the_pre_06o_ones():
    """C_SMALL / FA on UNSEEN_VAL: exactly the pre-06O no-FT labels / record (pinned); C_FULL on MAIN: REEVAL, no new key."""
    old = {"comparison_status": "DESCRIPTIVE", "no_ft_reference": True, "no_ft_reference_role": UV,
           "no_ft_reference_arm": None, "no_ft_reference_rule": OLD_NO_FT_RULE}
    for arm in ("C_SMALL", "FA"):
        want = dict(old, no_ft_reference_arm=arm)
        row = R.label_rows([{"m": 1}], _args(arm=arm), checkpoint_sha256="k", source_run_id="x", efe=1.0, extra={})[0]
        assert {k: row[k] for k in want} == want and not any(k.startswith("preview") for k in row)
        assert R.add_no_ft_record({"role": UV}, _args(arm=arm)) == {"role": UV, "no_ft_reference": want}
    row = R.label_rows([{"m": 1}], _args(role="MAIN"), checkpoint_sha256="k", source_run_id="x", efe=1.0, extra={})[0]
    assert row["comparison_status"] == "REEVAL" and not any(k.startswith(("preview", "no_ft")) for k in row)
    assert R.add_no_ft_record({"role": "MAIN"}, _args(role="MAIN")) == {"role": "MAIN"}


def test_no_ft_dry_runs_are_the_pre_06o_records(tmp_path):
    cp = _central(tmp_path / "c")
    got = R.run(_rv(cp, tmp_path / "c"))
    assert got == {"dry_run": True, "checkpoint_sha256": _sha(cp), "checkpoint_kind": "central",
                   "source_run": str(cp.parent.parent.parent), "source_run_id": got["source_run_id"],
                   "no_ft_reference": {"comparison_status": "DESCRIPTIVE", "no_ft_reference": True,
                                       "no_ft_reference_role": UV, "no_ft_reference_arm": "C_SMALL",
                                       "no_ft_reference_rule": OLD_NO_FT_RULE, "planned": True}}
    fcp = _fl_run(tmp_path / "f")
    got = R.run(_rv(fcp, tmp_path / "f", kind="fl", arm="FA"))
    assert set(got) == {"dry_run", "checkpoint_sha256", "checkpoint_kind", "source_run", "source_run_id", "fl_round",
                        "fl_efe", "no_ft_reference"} and got["no_ft_reference"]["no_ft_reference_arm"] == "FA"


# ------------------------------------------------------------------------------------------------ the ANA index
def _load_index_tool():
    tools = V4 / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location("_fix06o_ana_block_index", tools / "ana_block_index.py")
    BI = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = BI
    spec.loader.exec_module(BI)
    return BI


def _write(root: Path, name: str, arm: str, seed: int, rule: str, role: str = UV, record_extra=None, label=True):
    """One REEVAL_A40 dir through reeval's own label_rows / add_no_ft_record / write_outputs (synthetic rows)."""
    a = _args(role=role, arm=arm, rule=rule)
    extra = {}
    record = {"format": R.RECORD_FORMAT, "arm_label": arm, "seed": seed, "role": role, "selection_rule": rule,
              "checkpoint_id": f"{arm}_s{seed}:{rule}", "checkpoint_sha256": "k" * 64, "device_class": "CPU-SYN",
              "eval_manifest_id": UV_EVAL if role == UV else "MAIN_VALIDATION_5_100_131072"}
    if arm in R.CENTRAL_SOURCES:
        d = R.CENTRAL_SOURCES[arm]
        cs = {"source": arm, "arm_label": arm, "ext_role": d["ext_role"], "pair": {"train": d["train"], "eval": d["eval"]},
              "ext_self_sha256": "e" * 64}
        record["central_source"] = R.add_source_seed_record(cs, arm, seed)
        extra.update(R.source_row_labels(arm, cs), **R.source_seed_labels(arm, seed))
    rows = R.label_rows([{"metric": "mrr", "k": 20, "value": 0.1}], a, checkpoint_sha256="k" * 64, source_run_id="x",
                        efe=6.0, extra=extra)
    if label:
        R.add_no_ft_record(record, a)
    record.update(record_extra or {})
    return R.write_outputs(root / "REEVAL_A40" / name, tag=role, run_id=f"REEVAL_{arm}_{role}_s{seed}_{rule}", role=role,
                           rows=rows, result={"n": 1}, ranks=np.zeros(3, dtype=np.int32), record=record)


PREVIEW_DIRS = (("C_FULL_UNSEEN_VAL_s2026_PB", "C_FULL", 2026, PB), ("C_FULL_UNSEEN_VAL_s2026_CB", "C_FULL", 2026, CB),
                ("C_FULL_UNSEEN_VAL_s2027_PB", "C_FULL", 2027, PB), ("C_FULL_UNSEEN_VAL_s2027_CB", "C_FULL", 2027, CB),
                ("PRE_UNSEEN_VAL_s2026_CB", "PRE", 2026, CB))


def check_index_excludes_previews(tmp_path, BI=None):
    BI = BI or _load_index_tool()
    root = tmp_path / "R"
    _write(root, "C_SMALL_UNSEEN_VAL_s2026_PB", "C_SMALL", 2026, PB)
    for name, arm, seed, rule in PREVIEW_DIRS:
        rec_ = _write(root, name, arm, seed, rule)
        assert rec_["preview_reference"] == PREVIEW and "no_ft_reference" not in rec_
    lines = []
    rc = BI.main(["--split", "VALIDATION", "--device-class", "CPU-SYN", "--reeval", str(root / "REEVAL_A40"),
                  "--out", str(tmp_path / "idx")], say=lines.append)
    assert rc == 0, lines
    idx = json.loads((tmp_path / "idx" / "BLOCK_INDEX.json").read_text(encoding="utf-8"))
    assert [(e["arm"], e["role"], e["rule"]) for e in idx["entries"]] == [("C_SMALL", UV, PB)], idx["entries"]
    exc = {Path(x["dir"]).name: x["reason"] for x in idx["excluded"]}
    assert set(exc) == {n for n, *_ in PREVIEW_DIRS} and set(exc.values()) == {BI.RR.PREVIEW_REFERENCE_LABEL}, exc
    assert "never a block entry, a contrast" in BI.RR.PREVIEW_REFERENCE_LABEL
    assert not (tmp_path / "idx" / "REEVAL" / "C_FULL_UNSEEN_VAL_s2026_PB").exists()     # nothing copied


def test_index_excludes_the_preview_dirs(tmp_path):
    check_index_excludes_previews(tmp_path)


def test_index_refuses_unlabelled_or_misplaced_preview_records(tmp_path):
    BI = _load_index_tool()
    root = tmp_path / "a"                                             # C_FULL on UNSEEN_VAL without the label
    _write(root, "C_FULL_UNSEEN_VAL_s2026_PB", "C_FULL", 2026, PB, label=False)
    lines = []
    assert BI.main(["--split", "VALIDATION", "--device-class", "CPU-SYN", "--reeval", str(root / "REEVAL_A40"),
                    "--out", str(tmp_path / "ia")], say=lines.append) == BI.REFUSED
    assert "admits only the no-FT references" in lines[0], lines
    root = tmp_path / "b"                                             # the preview label on a MAIN record
    _write(root, "C_FULL_s2026_PB", "C_FULL", 2026, PB, role="MAIN", record_extra={"preview_reference": PREVIEW})
    lines = []
    assert BI.main(["--split", "VALIDATION", "--device-class", "CPU-SYN", "--reeval", str(root / "REEVAL_A40"),
                    "--out", str(tmp_path / "ib")], say=lines.append) == BI.REFUSED
    assert "preview_reference" in lines[0] and "06O" in lines[0], lines


# ------------------------------------------------------------------------------------------------ negative controls
@nc("the pre-06O UNSEEN_VAL restriction (no-FT references only) must not pass the C_FULL / PRE preview dry run")
def test_nc_preview_not_admitted(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "PREVIEW_SOURCES", ())                                      # MUTANT
    check_preview_dry_run(tmp_path)


@nc("preview rows without the DESCRIPTIVE / preview_reference labels must not pass the labelling check")
def test_nc_preview_rows_unlabelled(monkeypatch):
    monkeypatch.setattr(R, "preview_labels", lambda role, arm: {})                     # MUTANT
    check_preview_labels()


@nc("a preview path that skips the FIX-REEVAL source binding must not pass the UNSEEN_VAL refusals")
def test_nc_preview_binding_unchecked(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "check_source_binding", lambda name, start, ext, ext_path: {"view_checks": {"all_ok": True}})
    check_preview_refusals(tmp_path)


@nc("a preview path whose source guard is skipped accepts a C_FULL / PRE label on a run that is not that source")
def test_nc_preview_source_guard_skipped(monkeypatch):
    orig = R.check_no_ft_source
    monkeypatch.setattr(R, "check_no_ft_source",                                      # MUTANT: guard skipped
                        lambda args, src: None if R.is_preview_reference(args.role, args.arm_label) else orig(args, src))
    check_preview_source_guard()


@nc("an index without the preview exclusion refuses the whole REEVAL_A40 root (the freeze index cannot be built)")
def test_nc_index_exclusion_removed(tmp_path, monkeypatch):
    BI = _load_index_tool()
    monkeypatch.setattr(BI.RR, "PREVIEW_REFERENCE_ARMS", ())                           # MUTANT
    check_index_excludes_previews(tmp_path, BI)
