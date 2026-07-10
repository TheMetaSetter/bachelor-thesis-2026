"""Small AST-based checks for the repository readability contract."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SizeViolation:
    """One source or callable that exceeds its configured hard limit."""

    path: Path
    name: str
    lines: int
    limit: int
    line: int


@dataclass(frozen=True)
class AuditResult:
    """Immutable result containing file and callable violations."""

    file_violations: tuple[SizeViolation, ...]
    callable_violations: tuple[SizeViolation, ...]

    @property
    def violations(self) -> tuple[SizeViolation, ...]:
        """Return all violations in deterministic path/name order."""
        return tuple(
            sorted(
                self.file_violations + self.callable_violations,
                key=lambda item: (str(item.path), item.line, item.name),
            )
        )


def _callable_name(node: ast.AST, parents: tuple[str, ...]) -> str:
    name = getattr(node, "name", "<callable>")
    return ".".join((*parents, name))


def _iter_callables(tree: ast.AST):
    """Yield callable nodes, including nested and asynchronous definitions."""

    def walk(current: ast.AST, parents: tuple[str, ...]):
        for child in ast.iter_child_nodes(current):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield child, _callable_name(child, parents)
                yield from walk(child, (*parents, child.name))
            elif isinstance(child, ast.ClassDef):
                yield from walk(child, (*parents, child.name))
            else:
                yield from walk(child, parents)

    yield from walk(tree, ())


def scan_source_size_violations(source_root: Path) -> AuditResult:
    """Scan ``src/**/*.py`` for files over 500 or callables over 50 lines."""
    file_violations: list[SizeViolation] = []
    callable_violations: list[SizeViolation] = []
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root.parent)
        lines = len(path.read_text(encoding="utf-8").splitlines())
        if lines > 500:
            file_violations.append(SizeViolation(relative, "<file>", lines, 500, 1))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node, name in _iter_callables(tree):
            end = getattr(node, "end_lineno", node.lineno)
            length = end - node.lineno + 1
            if length > 50:
                callable_violations.append(
                    SizeViolation(relative, name, length, 50, node.lineno)
                )
    return AuditResult(tuple(file_violations), tuple(callable_violations))
