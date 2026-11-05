import pytest

from catbond.manifest import relative_command, repo_root, write_manifest
from catbond.pipeline import TIV_USD, reference_layer


def test_reference_layer_rounds_to_five_basis_points_of_tiv():
    a, e = reference_layer({50: 0.01224 * TIV_USD, 100: 0.01814 * TIV_USD})
    assert a == pytest.approx(0.0120 * TIV_USD) and e == pytest.approx(0.0180 * TIV_USD)
    a, e = reference_layer({50: 0.01226 * TIV_USD, 100: 0.0183 * TIV_USD})
    assert a == pytest.approx(0.0125 * TIV_USD) and e == pytest.approx(0.0185 * TIV_USD)


def test_manifest_has_no_absolute_repository_path(tmp_path, monkeypatch):
    script = repo_root() / "scripts" / "run_hazard.py"
    monkeypatch.setattr("sys.argv", [str(script), "--seed", "7"])
    assert relative_command([str(script), "--seed", "7"]) == (
        "python3 scripts/run_hazard.py --seed 7"
    )
    write_manifest(tmp_path, 7, {"note": "x"}, name="M.txt")
    text = (tmp_path / "M.txt").read_text()
    assert str(repo_root()) not in text
    assert "seed:    7" in text and "note: x" in text
