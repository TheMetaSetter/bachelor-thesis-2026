#!/usr/bin/env python3
"""Create a read-only map of project documentation for Codex sessions."""
from __future__ import annotations
import hashlib, json, re, subprocess, sys, tempfile
from collections import defaultdict
from pathlib import Path

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DOC_DIR_NAMES = ("documents", "docs", "documentation", "doc")
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "site-packages"}
IMPORTANT_WORDS = ("summary", "decision", "plan", "report", "audit", "phase", "proposal", "lessons", "spec", "result", "review")


def read_config(root: Path) -> dict[str, str]:
    values = {}
    config = root / ".codex/cleandocs.toml"
    if not config.is_file(): return values
    for line in config.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        match = re.fullmatch(r"([a-z_]+)\s*=\s*['\"]([^'\"]*)['\"]", line)
        if match: values[match.group(1)] = match.group(2)
    return values


def find_documents_dir(root: Path, configured: str | None = None) -> Path | None:
    if configured:
        path = root / configured
        return path if path.is_dir() and path != root else None
    for name in DOC_DIR_NAMES:
        path = root / name
        if path.is_dir(): return path
    for path in root.rglob("*"):
        if path.is_dir() and path.name in DOC_DIR_NAMES and not any(p in SKIP_DIRS for p in path.relative_to(root).parts): return path
    return None


def _context(path: Path) -> tuple[str, str, str]:
    try: lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError: return path.stem.replace("-", " "), "", "Could not read this file."
    title = next((x.lstrip("# ").strip() for x in lines if x.startswith("#")), path.stem.replace("-", " "))
    status_match = next((re.match(r"\s*(?:status|state):\s*(.*)", x, re.I) for x in lines[:30] if re.match(r"\s*(status|state):", x, re.I)), None)
    status = status_match.group(1).strip() if status_match else ""
    paragraph = []
    for line in lines:
        text = line.strip()
        if not text:
            if paragraph: break
            continue
        if text.startswith(("#", "```", "|", "- [")) or re.match(r"(?:status|state|date|scope):", text, re.I): continue
        paragraph.append(text)
    intro = re.sub(r"\s+", " ", " ".join(paragraph) or "No opening summary found.")
    if len(intro) > 280: intro = intro[:277].rsplit(" ", 1)[0] + "..."
    return title, status, intro


def _score(path: Path) -> int:
    name = path.stem.lower()
    score = sum(2 for word in IMPORTANT_WORDS if word in name)
    try: body = path.read_text(encoding="utf-8", errors="ignore")[:12000].lower()
    except OSError: body = ""
    return score + sum(1 for word in ("decision", "scope", "status:", "conclusion", "result") if word in body)


def _dates(docs: Path):
    grouped = defaultdict(lambda: defaultdict(list))
    for p in docs.rglob("*.md"):
        rel = p.relative_to(docs)
        if any(part in SKIP_DIRS for part in rel.parts): continue
        date = next((part for part in rel.parts if DATE_RE.fullmatch(part)), None)
        if date:
            pos = rel.parts.index(date)
            category = rel.parts[pos + 1] if pos + 1 < len(rel.parts) - 1 else "(date folder root)"
            grouped[date][category].append(p)
    return grouped


def build_report(root: Path) -> str | None:
    config = read_config(root)
    if config.get("cleandocs", "off").lower() != "on": return None
    docs = find_documents_dir(root, config.get("documents_dir"))
    if docs is None: return "Cleandocs is on, but no documentation folder was found. Set documents_dir in .codex/cleandocs.toml."
    tree = subprocess.run(["tree", "-a", "-I", ".DS_Store", str(docs)], capture_output=True, text=True, check=False)
    tree_text = tree.stdout.rstrip() or f"tree command failed: {tree.stderr.strip()}"
    lines = ["# Documentation map", "", f"Repository: `{root}`", f"Documentation folder: `{docs.relative_to(root)}`", "", "## File tree", "", "```text", tree_text, "```", "", "## Key files by date", ""]
    grouped = _dates(docs)
    latest = max(grouped) if grouped else None
    if not grouped: lines.append("No dated folders were found.")
    for date in sorted(grouped):
        for category, paths in sorted(grouped[date].items()):
            label = f"{date} — {category}" + (" (LATEST DATE)" if date == latest else "")
            lines.extend([f"### {label}", ""])
            chosen = sorted(paths, key=lambda p: (-_score(p), p.name.lower()))[:5 if date == latest else 3]
            for p in chosen:
                title, status, intro = _context(p)
                lines.append(f"- `{p.relative_to(docs)}` — **{title}**")
                if status: lines.append(f"  - Status: {status}")
                lines.append(f"  - Background: {intro}")
            lines.append("")
    lines.extend(["## Key files outside dated folders", ""])
    dated = {p for cats in grouped.values() for paths in cats.values() for p in paths}
    other = [p for p in docs.rglob("*.md") if p not in dated and not any(x in SKIP_DIRS for x in p.relative_to(docs).parts)]
    for folder in sorted({p.parent for p in other}):
        chosen = sorted((p for p in other if p.parent == folder), key=lambda p: (-_score(p), p.name.lower()))[:2]
        lines.extend([f"### {folder.relative_to(docs) if folder != docs else '.'}", ""])
        for p in chosen:
            title, status, intro = _context(p)
            lines.append(f"- `{p.relative_to(docs)}` — **{title}**")
            if status: lines.append(f"  - Status: {status}")
            lines.append(f"  - Background: {intro}")
        lines.append("")
    lines.append("Importance is estimated from names and contents. Review the full tree before deciding a document is redundant.")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    try: event = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError): return 0
    cwd = Path(event.get("cwd") or Path.cwd()).resolve()
    result = subprocess.run(["git", "-C", str(cwd), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)
    if result.returncode: return 0
    root = Path(result.stdout.strip()).resolve()
    report = build_report(root)
    if report is None: return 0
    repo_key = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:10]
    out = Path(tempfile.gettempdir()) / f"cleandocs-{root.name}-{repo_key}.md"
    out.write_text(report, encoding="utf-8")
    docs = find_documents_dir(root, read_config(root).get("documents_dir"))
    dates = _dates(docs) if docs else {}
    print(f"Cleandocs report: `{out}`. It begins with the full `tree` file listing, then key files grouped by dated folder. Latest date: {max(dates) if dates else 'none found'}. Read it and explain the document stories in simple English. This hook changed no source documents.")
    return 0

if __name__ == "__main__": raise SystemExit(main())
