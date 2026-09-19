---
date: 2026-09-16 Asia/Ho_Chi_Minh
topic: "W&B logging for tsad-lib"
status: ready
revision: 9aa4c35
source_structure: documents/logs/09-16-2026/structure/structure-tsad-lib-wandb-logging.md
related_documents:
  - documents/logs/09-16-2026/research/research-tsad-lib-wandb-logging.md
  - documents/logs/09-16-2026/plan/plan-tsad-lib-wandb-logging.md
---

# Detailed Implementation: W&B logging for tsad-lib

## Desired end state

For each successful `ExperimentRunner.run_request()`, W&B receives one run named by `RunRequest.label()`, the resolved configuration, final scalar metrics, and one `experiment` artifact for the existing `run_dir`. The local completion marker is written only after that publication succeeds.

## File changes

| File | Status | Responsibility |
| --- | --- | --- |
| `pyproject.toml` | Modify | Declare the W&B runtime dependency. |
| `src/tsad/wandb_logging.py` | Proposed new file | Own exactly the W&B SDK calls. |
| `src/tsad/experiment.py` | Modify | Invoke the publisher at the completed-run boundary. |
| `tests/test_wandb_logging.py` | Proposed new file | Hold focused unit and integration tests. |

## Phase 1: Tests first

### Stage 1.1: Unit test

1. Create `tests/test_wandb_logging.py`.
2. Define a fake module with `init()` and `Artifact()`.
3. Define a fake run that records `log()`, `log_artifact()`, and `finish()`.
4. Define a fake artifact that records `add_dir()`.
5. Write `test_publish_completed_run_logs_config_metrics_and_artifact`.
6. Give it a `RunRequest`, a small config dictionary, a temporary `run_dir`, and `{"VUS-PR": 0.7}`.
7. Call the proposed publisher with the fake module.
8. Assert the project is `tsad`, the name is `request.label()`, config equals the supplied dictionary, metrics equal the supplied dictionary, artifact type is `experiment`, artifact directory equals `run_dir`, and finish occurs once.
9. Run `.venv/bin/python -m pytest tests/test_wandb_logging.py::test_publish_completed_run_logs_config_metrics_and_artifact -q`.
10. Confirm failure occurs because the publisher does not exist.

### Stage 1.2: Integration test

1. In the same test file, reuse the existing small SMD layout from `tests/test_public_contracts.py` or copy only its local helper.
2. Write `test_public_run_publishes_after_local_artifacts_are_written`.
3. Replace the publisher imported by `tsad.experiment` with a recording function.
4. Run `tsad.api.run()` for one isolation-forest smoke request.
5. Assert the returned result is completed.
6. Assert the recording function sees an existing `metrics.json`, `provenance.json`, and `run_manifest.json` whose status is `completed`.
7. Assert `completion.json` does not exist during the recording call and exists after `api.run()` returns.
8. Run `.venv/bin/python -m pytest tests/test_wandb_logging.py::test_public_run_publishes_after_local_artifacts_are_written -q`.
9. Confirm failure occurs because `ExperimentRunner` does not call the publisher.

## Phase 2: Minimal production code

### Stage 2.1: Dependency

1. Add `"wandb"` to `[project].dependencies` in `pyproject.toml`.
2. Keep it a runtime dependency because an ordinary `tsad run` invokes the publisher.

### Stage 2.2: Publisher

1. Create `src/tsad/wandb_logging.py`.
2. Define `publish_completed_run(wandb_module, request, config, run_dir, metrics)`.
3. Call `wandb_module.init(project="tsad", name=request.label(), config=config)`.
4. Call `run.log(metrics)` once.
5. Construct `wandb_module.Artifact(name=request.label(), type="experiment")`.
6. Call `artifact.add_dir(str(run_dir))` once.
7. Call `run.log_artifact(artifact)` once.
8. Call `run.finish()` in `finally` so an artifact-upload error still closes the W&B run.
9. Do not catch publication errors in this function.

### Stage 2.3: Experiment boundary

1. Import `wandb` and `publish_completed_run` in `src/tsad/experiment.py`.
2. After `artifact_store.write_manifest(... status="completed")`, call `publish_completed_run(wandb, request, vars(config), run_dir, metrics)`.
3. Keep this call before `artifact_store.write_completion(required=required_artifacts)`.
4. Do not add a CLI flag, environment-variable policy, retry loop, or model callback.
5. Run each new test again. Confirm each passes.
6. Run `.venv/bin/python -m pytest tests/test_public_contracts.py tests/test_release_contracts.py -q`.
7. Confirm all selected tests pass.

## Phase 3: One real smoke validation

### Stage 3.1: Authentication

1. The human authenticates the intended W&B account.
2. Confirm the project name `tsad` is acceptable for that account.

### Stage 3.2: Smoke run

1. Select one existing SMD entity only.
2. Run the existing `tsad run` command with `benchmark_smoke` and the normal output root.
3. Do not run a matrix or full benchmark.

### Stage 3.3: Inspect results

1. Open the resulting W&B run.
2. Confirm its name equals the local request label.
3. Confirm its config contains the resolved values.
4. Confirm its summary/history contains final metrics.
5. Confirm it has one `experiment` artifact containing the local run files.
6. Confirm local `completion.json` exists.

## Error and recovery contract

- A W&B initialization, metric, artifact, or upload error propagates to `run_request()`.
- `run_request()` returns a failed `Result` under its existing exception handling.
- `completion.json` is absent, so `resume=True` follows the existing partial-run path and archives it before a new attempt.
- Existing local run files remain available for diagnosis.

## Final verification

- Focused unit test passes with no network.
- Focused API integration test passes with no network.
- Existing public and release contract tests pass.
- One authenticated smoke run shows the W&B run and artifact.
