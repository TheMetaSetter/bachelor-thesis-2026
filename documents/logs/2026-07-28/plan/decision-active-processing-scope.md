# Active processing scope decision

## Decision

The active benchmark runtime keeps only the SMD processing flow required by
`full-spec-v2.md` and `full-spec-v3.md`:

- THESIS two-stage offline training;
- RedLamp comparative baseline;
- THESIS online adaptation and its benchmark variants;
- shared SMD loading, windowing, scoring, checkpoint, and reporting code.

AnomalyArchive is archival scope. Its parser, analysis helpers, and direct
forensic scripts remain available for historical analysis, but the active
runtime registry does not expose it. Archival scripts must register it
explicitly when they need it. Its archival config support stays separate
until those historical reports are migrated.

Operational scripts under `scripts/ops/` remain available because they inspect,
validate, retain, or report experiment artifacts. They are not processing
flows for the main experiments and are not imported by the active benchmark
entrypoints.

## Plain-language rule

If a new benchmark config is for the main experiments, it must use
`dataset_name: smd`. AnomalyArchive work must be an explicit archival tool
invocation, not an accidental active benchmark path.
