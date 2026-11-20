# Experiment terminology for the next version

Status: terminology authority for the next version of `bachelor-thesis-2026`.

## Core definitions

- An **experiment** is a set of multiple runs.
- A **run** is one execution of a configured method on a dataset/entity/seed and protocol.
- A **matrix** describes combinations of run parameters.
- A **cell** identifies one unit of execution in that matrix.
- An **experiment directory** contains the runs and experiment-level reports for one experiment.

## Related terms

- **Offline phase** and **online phase** name execution stages.
- **Online test-time adaptation** names method behavior that updates learned parameters during the test stream. An online phase may contain methods that do not adapt.
- `O0/O1/O2` and `A0/A1/A2` name THESIS/MUSTARD ablations only. Baselines use method-specific profiles.
- MOMENT is a method, not an adapter.

## Naming rules

Use **experiment** or **run** in prose according to the definitions above. Do not use “benchmark” in new prose, filenames, directory names, or identifiers. Rename project-owned paths and identifiers containing that term when migrating their callers. Keep historical output paths as artifact identities.

## Scope

This note governs terminology for the proposed next version. It does not rename current runtime objects, configs, source files, or historical artifacts by itself. Implementation needs a caller and artifact migration plan.
