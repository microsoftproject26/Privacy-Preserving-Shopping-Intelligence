"""Both disk thresholds (company/plans/DISK_RETENTION_PLAN.md S5; package 10 S3):
  REFUSE: free - planned_peak < 8 GiB floor.
  PAUSE:  free < 10 GiB target (checked only once REFUSE does not apply).
  OK:     otherwise.
Disk usage is mocked throughout (SYS-03 requirement); no test depends on the real free space of
the machine it runs on.
"""
from __future__ import annotations

from suptestkit import fake_disk_usage_fn

from v4sup.disk import disk_check


def test_ample_room_is_ok(workspace):
    dd = disk_check(workspace, planned_peak_gib=1.0, disk_usage_fn=fake_disk_usage_fn(20.0))
    assert dd.state == "OK"


def test_below_target_but_above_floor_pauses(workspace):
    # free=9.5, peak=0 -> headroom 9.5 >= floor 8, but free < target 10
    dd = disk_check(workspace, planned_peak_gib=0.0, disk_usage_fn=fake_disk_usage_fn(9.5))
    assert dd.state == "PAUSE"


def test_headroom_below_floor_refuses(workspace):
    # free=8.5, peak=1 -> headroom 7.5 < floor 8
    dd = disk_check(workspace, planned_peak_gib=1.0, disk_usage_fn=fake_disk_usage_fn(8.5))
    assert dd.state == "REFUSE"


def test_deep_in_the_red_refuses_not_just_pauses(workspace):
    # free=7, peak=0 -> headroom 7 < floor 8: REFUSE takes precedence over PAUSE
    dd = disk_check(workspace, planned_peak_gib=0.0, disk_usage_fn=fake_disk_usage_fn(7.0))
    assert dd.state == "REFUSE"


def test_matches_g0b_concurrent_contest_example():
    """DISK_RETENTION_PLAN.md: 12.81 GiB free, 3.4 GiB planned peak -> 9.4 GiB headroom, which
    fits the 8 GiB floor (this is exactly the case the old "refuse if free - peak < 10" rule
    would have wrongly blocked)."""
    dd = disk_check("/does/not/matter", planned_peak_gib=3.4, disk_usage_fn=fake_disk_usage_fn(12.81))
    assert dd.state == "OK"


def test_boundary_exactly_at_floor_is_ok(workspace):
    # free=18, peak=10 -> headroom exactly 8.0: not "< floor", so OK, not REFUSE
    dd = disk_check(workspace, planned_peak_gib=10.0, disk_usage_fn=fake_disk_usage_fn(18.0))
    assert dd.state == "OK"


def test_boundary_exactly_at_target_is_ok(workspace):
    # free=10 exactly, peak=0 -> not "< target", so OK, not PAUSE
    dd = disk_check(workspace, planned_peak_gib=0.0, disk_usage_fn=fake_disk_usage_fn(10.0))
    assert dd.state == "OK"
