from __future__ import annotations

import json

import pytest

from scripts.train_model import _source_tree_sha256, _write_report


def test_source_tree_hash_is_stable_and_tracks_path_and_content(tmp_path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "b.py").write_text("B = 2\n", encoding="utf-8")
    (source / "a.py").write_text("A = 1\n", encoding="utf-8")

    first = _source_tree_sha256(source)
    assert _source_tree_sha256(source) == first

    (source / "a.py").write_text("A = 3\n", encoding="utf-8")
    assert _source_tree_sha256(source) != first


def test_source_tree_hash_rejects_an_empty_tree(tmp_path) -> None:
    with pytest.raises(ValueError, match="no Python source files"):
        _source_tree_sha256(tmp_path)


def test_write_report_adds_top_level_provenance_stamp(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    source_hash = "a" * 64

    _write_report(report_path, {"metric": 0.5}, source_hash)

    assert json.loads(report_path.read_text(encoding="utf-8")) == {
        "metric": 0.5,
        "source_provenance": source_hash,
        "source_tree_sha256": source_hash,
    }
