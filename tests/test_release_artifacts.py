import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_are_valid_and_output_free() -> None:
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 4
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("outputs", []) == []
                assert cell.get("execution_count") is None


def test_final_untruncated_plugin_rerun_is_frozen() -> None:
    root = ROOT / "reference_results" / "paper_untruncated_20260909"
    expected_rows = {
        "D_selection_paper/D_validation_raw.csv": 4000,
        "main_paper/repetition_results.csv": 400,
        "scenario_b_paper/repetition_results.csv": 200,
    }
    for relative, expected in expected_rows.items():
        with (root / relative).open(newline="", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == expected

    for manifest_path in root.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["config"]["profile"] == "paper"
        assert manifest["config"]["repetitions"] == 50
        assert manifest["config"]["clip_estimator"] is False
        assert manifest["software_release"] == "1.0.0"
        assert "git_revision" not in manifest
        assert "git_worktree_dirty" not in manifest


def test_single_finalist_empirical_measure_nn_rerun_is_frozen() -> None:
    root = ROOT / "reference_results" / "nn_empirical_measure_single_finalist_20260911"
    expected_rows = {
        "nn_A_paper/repetition_results.csv": 400,
        "nn_B_paper/repetition_results.csv": 200,
    }
    for relative, expected in expected_rows.items():
        with (root / relative).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        keys = {
            (row["regime"], row["M"], row["N_train_per_class"], row["repetition"])
            for row in rows
        }
        assert len(rows) == expected
        assert len(keys) == expected
        assert all(row["N_train_per_class"] == row["N_validation_per_class"] for row in rows)
        assert all(row["N_test_per_class"] == "1000" for row in rows)
        assert all(row["finalists_evaluated"] == "1" for row in rows)
        assert all(row["class_collapse"] in {"0", "1"} for row in rows)
        assert any(row["class_collapse"] == "1" for row in rows)

    for manifest_path in root.glob("*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["config"]["profile"] == "paper"
        assert manifest["config"]["repetitions"] == 50
        assert manifest["config"]["validation_protocol"] == "empirical_measure_system"
        assert manifest["software_release"] == "1.0.0"
        assert "git_revision" not in manifest
        assert "git_worktree_dirty" not in manifest


def test_tracked_artifacts_do_not_contain_local_absolute_paths() -> None:
    suffixes = {".py", ".md", ".json", ".ipynb", ".toml", ".yml", ".yaml", ".cff", ".sh"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "results" in path.parts:
            continue
        if path.suffix in suffixes or path.name in {"Makefile"}:
            text = path.read_text(encoding="utf-8")
            assert "/" + "Users/" not in text
            assert "file" + "://" not in text
