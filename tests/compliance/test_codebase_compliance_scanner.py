from pathlib import Path

from tests.codebase_compliance import scan_source_size_violations


def _write_module(root: Path, body: str) -> Path:
    source = root / "src" / "sample.py"
    source.parent.mkdir()
    source.write_text(body, encoding="utf-8")
    return source


def test_callable_boundary_is_exact(tmp_path: Path) -> None:
    lines = ["def exact():"] + ["    value = 1"] * 48 + ["    return value"]
    _write_module(tmp_path, "\n".join(lines) + "\n")
    result = scan_source_size_violations(tmp_path / "src")
    assert result.callable_violations == ()


def test_callable_over_boundary_is_reported(tmp_path: Path) -> None:
    lines = ["async def too_long():"] + ["    value = 1"] * 49 + ["    return value"]
    _write_module(tmp_path, "\n".join(lines) + "\n")
    result = scan_source_size_violations(tmp_path / "src")
    assert [(item.name, item.lines) for item in result.callable_violations] == [
        ("too_long", 51)
    ]


def test_file_boundary_is_exact(tmp_path: Path) -> None:
    _write_module(tmp_path, "\n".join(["value = 1"] * 500) + "\n")
    result = scan_source_size_violations(tmp_path / "src")
    assert result.file_violations == ()


def test_file_over_boundary_is_reported(tmp_path: Path) -> None:
    _write_module(tmp_path, "\n".join(["value = 1"] * 501) + "\n")
    result = scan_source_size_violations(tmp_path / "src")
    assert [(item.name, item.lines) for item in result.file_violations] == [
        ("<file>", 501)
    ]


def test_nested_and_method_callables_are_scanned(tmp_path: Path) -> None:
    body = (
        "class Model:\n    def method(self):\n"
        + "\n".join("        value = 1" for _ in range(49))
        + "\n        return value\n"
    )
    _write_module(tmp_path, body)
    result = scan_source_size_violations(tmp_path / "src")
    assert [item.name for item in result.callable_violations] == ["Model.method"]
