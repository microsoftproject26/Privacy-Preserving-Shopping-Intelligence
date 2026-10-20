import json
from pathlib import Path

from tests.experiments.test_schemas import (
    _make_common_initialization,
    _make_experiment_config,
    _make_experiment_result,
    _make_metric_record,
    _make_model_config,
    _make_sys_measurement,
)


def main() -> None:
    root = Path(__file__).parent.parent.parent
    fixtures_dir = root / "fixtures" / "experiments" / "contracts"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    fixtures = {
        "model_config.json": _make_model_config(),
        "experiment_config_r1.json": _make_experiment_config(),
        "metric_record.json": _make_metric_record(),
        "sys_measurement.json": _make_sys_measurement(),
        "experiment_result.json": _make_experiment_result(),
        "common_initialization.json": _make_common_initialization(),
    }

    for filename, data in fixtures.items():
        path = fixtures_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    print(f"Generated {len(fixtures)} contracts fixtures.")

    # S1-PR-03 Common Initialization Seed Proofs
    import hashlib

    from scripts.experiments.initialization import (
        generate_fixture_linear_v1_state,
        serialize_state_dict,
    )

    init_dir = root / "fixtures" / "experiments" / "common_initialization"
    init_dir.mkdir(parents=True, exist_ok=True)

    mcfg = _make_model_config()
    mcfg_path = init_dir / "fixture_model_config.v1.json"
    with open(mcfg_path, "w", encoding="utf-8") as f:
        json.dump(mcfg, f, indent=2)

    seeds = [13, 42, 2026]
    for s in seeds:
        state = generate_fixture_linear_v1_state(mcfg, s)
        bin_data = serialize_state_dict(state)
        bin_hash = hashlib.sha256(bin_data).hexdigest()

        bin_path = init_dir / f"seed-{s}.state.v1.bin"
        bin_path.write_bytes(bin_data)

        # Build the CommonInitialization record
        ci_meta = _make_common_initialization()
        ci_meta["seed"] = s
        ci_meta["artifact_kind"] = "FIXTURE_PROOF"
        ci_meta["state_sha256"] = bin_hash
        ci_meta["state_size_bytes"] = len(bin_data)

        json_path = init_dir / f"seed-{s}.common_init.v1.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(ci_meta, f, indent=2)

        print(f"Generated seed {s} proof: {bin_hash} ({len(bin_data)} bytes)")


if __name__ == "__main__":
    main()
