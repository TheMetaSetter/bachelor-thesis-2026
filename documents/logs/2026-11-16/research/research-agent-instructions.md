---
date: 2026-11-16T13:31:24+07:00
topic: "Preference sources for AGENTS.md"
status: complete
revision: 463d203c2dd1cc08b4e0b99fc7786e5920154d85
branch: simplifying
---

# Research: Agent instructions

## Finding

Both preference files exist in this repository.
The previous `AGENTS.md` did not name them.
The new file must direct agents to both sources.
These findings describe documents and paths, not tested runtime behavior.

## Sources

- [codebase-specific-preferences.md:1](../../../../codebase-specific-preferences.md#L1) contains project preferences.
- [codebase-agnostic-preferences.md:13](../../../../codebase-agnostic-preferences.md#L13) contains general preferences.
- [codebase-agnostic-preferences.md:31](../../../../codebase-agnostic-preferences.md#L31) requires ASD-STE100 for English technical writing.
- [codebase-agnostic-preferences.md:67](../../../../codebase-agnostic-preferences.md#L67) requires research before implementation.
- [codebase-specific-preferences.md:20](../../../../codebase-specific-preferences.md#L20) defines the log date format.
- [codebase-specific-preferences.md:73](../../../../codebase-specific-preferences.md#L73) defines model ownership and permitted helpers.

## Differences and missing evidence

- The project preferences require one public model entrypoint per model at line 75.
- They permit small helper files at line 77.
- The previous `AGENTS.md` required all model training code in one file.
- That instruction was stricter than the preference source.
- The prompt paths at lines 21–24 are missing.
- The supplied research prompt exists under `prompts/primitive_prompts/`.

## Research method

The supplied [research prompt](</Users/conquerormikrokosmos/Downloads/LAPTOP MAC/MYUNIVERSITY/ĐẠI HỌC QUỐC GIA TPHCM/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/prompts/primitive_prompts/1_research_prompt.md>) requires source evidence and clear limits.
It separates implemented behavior from documented requirements.
It requires a dated Markdown research report.
This review checks preference documents and their referenced paths.
It does not establish that the code follows these preferences.

## Use for initialization

Read both preference files before work.
Keep detailed rules in their owning files.
Report missing references and conflicting requirements before affected changes.
Keep `AGENTS.md` short.
