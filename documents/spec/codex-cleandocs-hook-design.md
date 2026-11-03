# Codex document review hook design

Status: approved design; implemented in `bachelor-thesis-2026` and `tsad-lib`.

## Goal

At Codex session start, show a useful map of the current project's documentation and help the user find important files and understand how dated work connects.

## Selected approach

Use a report-only Codex `SessionStart` command hook. Install it for `bachelor-thesis-2026` and `tsad-lib` first. Do not use `PostToolUse`; checking after every edit would add repeated noise. Do not delete, move, or edit documents automatically.

## Runtime flow

1. Read the Codex hook input and find the Git repository root.
2. Check `.codex/cleandocs.toml`; run only when `cleandocs = "on"`.
3. Discover the documentation directory instead of assuming its name. Prefer a top-level directory named `documents`; allow a configured alternate path for other repositories. If several documentation directories exist, use the configured path; otherwise choose the first conventional folder found.
4. Run the `tree` command over that directory and include the full file listing in the report.
5. After the tree, group dated folders chronologically and summarize likely key documents in each folder. Give the newest date special attention.
6. For each selected document, use its title, status, and opening summary to explain its background and connection to nearby work. Write the report in simple English.
7. Keep the hook output short enough for Codex context. Save the full tree and report to a uniquely named file in the system temporary folder and return a short notice and report path to Codex.

## Future opt-in

Add `.codex/` to each repository’s `.gitignore` before creating local hook files. Each repository opts in through `.codex/cleandocs.toml` with `cleandocs = "on"`. The hook is installed at repository level; a future repository can add the shared hook script/config and turn this setting on. The script accepts an optional configured documentation path, so repository layouts may differ.

## Limits

The hook can identify likely important files from folder/date structure, document titles, status, and summaries. It cannot decide scientific importance as reliably as a person. It must describe its selection as a suggestion and retain the full tree so the user can inspect omitted files.

## Verification

Test both repositories, including each discovered document path, tree output, date grouping, plain-English summaries, disabled behavior, and a third temporary repository with a different documentation-folder name. Confirm that the hook never changes or deletes documentation.
