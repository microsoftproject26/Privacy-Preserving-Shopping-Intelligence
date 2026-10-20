"""CLI refusals: --final without a LOCK, unimplemented arms, unknown datasets."""
from __future__ import annotations

from v4bench import run as R


def test_final_refuses_without_lock(release, tmp_path, capsys):
    rc = R.main(["final", "--dataset", release["name"], "--data-root", str(release["root"]),
                 "--runs-root", str(tmp_path / "runs"), "--threads", "1"])
    assert rc == 2 and "LOCK" in capsys.readouterr().err
    assert not (tmp_path / "runs" / release["name"] / "TEST_LOOK.json").exists()


def test_unimplemented_and_unknown_arms_refused(release, tmp_path, capsys):
    base = ["--dataset", release["name"], "--data-root", str(release["root"]), "--runs-root", str(tmp_path / "r"),
            "--threads", "1", "--seed", "2026"]
    assert R.main(["run", "--arm", "DP8", *base]) == 2
    assert R.main(["run", "--arm", "NOT_AN_ARM", *base]) == 2
    assert R.main(["run", "--arm", "FT_FA", *base]) == 2          # FT needs --base-run


def test_describe_runs_on_train_only(release, capsys):
    (release["root"] / release["name"] / "leave_one_out" / "holdout.csv").unlink()
    assert R.main(["describe", "--dataset", release["name"], "--data-root", str(release["root"]), "--threads", "1"]) == 0
    assert '"K"' in capsys.readouterr().out
