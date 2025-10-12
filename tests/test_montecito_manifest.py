from pathlib import Path

from tools.montecito_manifest import iter_files, write_manifest


def test_iter_files_orders_results(tmp_path: Path) -> None:
    (tmp_path / "a").write_text("alpha", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b").write_bytes(b"beta")

    results = list(iter_files(tmp_path))

    assert [entry[0] for entry in results] == [Path("a"), Path("nested/b")]
    assert results[0][1] == 5
    assert len(results[0][2]) == 32


def test_write_manifest_creates_csv(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("content", encoding="utf-8")
    destination = tmp_path / "manifest.csv"

    write_manifest(tmp_path, destination)

    lines = destination.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "filename,bytes,md5"
    assert len(lines) == 2
    assert lines[1].startswith("file.txt,7,")
