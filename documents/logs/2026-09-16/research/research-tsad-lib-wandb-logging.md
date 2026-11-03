---
date: 2026-09-16 Asia/Ho_Chi_Minh
topic: "W&B logging for tsad-lib"
status: complete
revision: 9aa4c35
repository: "/Users/conquerormikrokosmos/Downloads/LAPTOP MAC/IT/AI&DS/tsad-lib"
---

# Research: W&B logging for tsad-lib

## Question

Find the smallest safe boundary for W&B experiment logging. Do not change code.

## Confirmed current flow

`tsad.api.run()` creates `ExperimentRunner`. `ExperimentRunner.run()` calls `run_request()` once per `RunRequest`. That method creates the local run directory, writes `run_manifest.json`, `resolved_config.yaml`, metrics, provenance, optional checkpoints, and `completion.json`. It then returns `Result`.

The current project has no `wandb` import, call, or dependency. `codebase_preferences.md` requires W&B for experiment artifacts and history, but this is a documented requirement, not implementation.

## Smallest useful boundary

Log one W&B run only after a local run has produced its final metrics and required artifacts. The logger receives the existing `RunRequest`, resolved configuration, metric dictionary, and completed local run directory. It must not enter model training, data preparation, metric calculation, or `ArtifactStore`.

This scope records one experiment history entry, final scalar metrics, configuration, and one versioned artifact directory. It deliberately does not add epoch-level loss history because current runners expose no common epoch callback.

## Minimal source changes proposed

| Location | Change |
| --- | --- |
| `pyproject.toml` | Add `wandb` as a runtime dependency. |
| Proposed `src/tsad/wandb_logging.py` | One small function that starts a W&B run, logs config and metrics, attaches the existing run directory as an `experiment` artifact, and finishes the run. |
| `src/tsad/experiment.py` | Call that function after local artifact validation and before the local completion marker. |
| Proposed `tests/test_wandb_logging.py` | Add one fake-W&B unit test and one `api.run()` integration test. |

`RunRequest.label()` is already a stable run name source. It contains no characters that W&B disallows in run IDs, but the design uses it as the human-readable run name, not an explicit W&B ID.

## Six alternative designs

| ID | Design | Benefit | Cost | Decision |
| --- | --- | --- | --- | --- |
| A | Put `wandb.init`, `log`, and `log_artifact` directly in `run_request`. | Few changed lines. | External API is coupled to orchestration and tests need deep monkeypatching. | Reject. |
| B | Add one small `wandb_logging.py` function and call it once from `run_request`. | Clear boundary; fake module tests one function; one integration seam. | One new file. | Select. |
| C | Add a logger protocol, concrete W&B class, null logger, and constructor injection. | Maximum substitution. | Adds interfaces and state without a current second backend. | Reject. |
| D | Add W&B calls to each model runner for epoch-level metrics. | Detailed learning curves. | Touches unrelated runners and requires a common callback contract. | Reject. |
| E | Instrument `ArtifactStore` so every local write uploads immediately. | Captures every file. | Changes atomic local semantics; uploads partial artifacts. | Reject. |
| F | Run a separate post-processing command that scans `outputs/` and uploads finished directories. | No runtime change. | History can be skipped or detached from execution failure. | Reject. |

## Decision framework: MECE decision tree

1. Is the required information already available at one completed-run boundary? Yes: `run_request()` has request, config, metrics, and `run_dir`.
2. Does the task require an additional backend now? No. Therefore no protocol or registry.
3. Must tests avoid W&B network and credentials? Yes. Therefore isolate the SDK calls in one function and pass a fake module in its unit test.
4. Must a local run be marked complete if W&B publication fails? No. Therefore publish before `write_completion()` so resume treats the result as incomplete and starts a fresh attempt.

## Three challenges to the selected design

1. A direct inline call is smaller. It loses a clean fake boundary, so its apparent line-count saving makes unit tests less clear.
2. A protocol would make mocks easy. It creates two concepts, a protocol and a concrete implementation, although the project has one logger only.
3. Logging after `write_completion()` would preserve a successful local run when the network fails. It falsely advertises a reproducible completed experiment even when the required W&B record is absent.

Design B remains the smallest design that preserves one owner, one external seam, and separate unit and integration tests.

## Evidence

- `pyproject.toml:10-17` has runtime dependencies but no W&B package.
- `src/tsad/api.py:42-58` enters the experiment runner.
- `src/tsad/experiment.py:395-681` owns one run's local artifacts and completion boundary.
- `src/tsad/types.py:50-63` defines the request identity and label.
- `tests/test_public_contracts.py:77-85` already exercises one public baseline run.
- `codebase_preferences.md:22` requires W&B experiment logging.

## External API verification

W&B documents `wandb.init()` for runs, `run.log_artifact()` for output artifacts, and `Artifact.add_dir()` for a local directory. The selected design uses only these documented calls: [W&B artifact documentation](https://docs.wandb.ai/models/artifacts/construct-an-artifact), [W&B init documentation](https://docs.wandb.ai/models/ref/python/functions/init).
