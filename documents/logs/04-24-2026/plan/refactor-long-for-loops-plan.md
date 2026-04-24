# Refactor Long For Loops Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the reviewed long `for` loops so the codebase follows the readability rule in `codebase_preferences.md`.

**Architecture:** Keep the current module boundaries. Refactor long loops into small, clearly named helper methods in the same owning class or parser so behavior remains local and readable. Do not introduce new shared abstractions unless two call sites use the exact same behavior.

**Tech Stack:** Python, PyTorch, pytest, existing `console_print`, existing checkpoint/logging helpers.

---

## Scope And File Map

- Modify: `src/engine/trainer.py`
  - Break the epoch loop into epoch preparation, training epoch execution, optional synthetic validation, metric construction, and best-checkpoint update helpers.
  - Break validation-batch result collection into smaller helper methods.
- Modify: `src/engine/online_loop.py`
  - Break the online step loop into single-step execution, metric construction, record construction, and checkpoint helpers.
- Modify: `src/data/datasets/smd.py`
  - Break SMD entity parsing into entity tensor loading, validation split calculation, raw sequence construction, and collection append helpers.
- Modify: `src/models/thesis_multitask.py`
  - Break memory initialization token collection into clean-token and optional synthetic-normal-token helpers.
- Tests to run:
  - `pytest -q tests/test_learning_rate_scheduler.py tests/test_console_instrumentation.py`
  - `pytest -q tests/test_online_entrypoint.py tests/test_online_state_roundtrip.py tests/test_online_adaptation_step.py`
  - `pytest -q tests/test_smd_dataset_shapes.py tests/test_smoke_loader_limits.py`
  - `pytest -q tests/test_multitask_memory_initialization.py tests/test_multitask_memory_bootstrap.py tests/test_multitask_memory_updates.py`

## Refactor Constraints

- Preserve all public method signatures: `Trainer.train`, `OnlineLoop.run`, `SMDDatasetParser.parse`, and `ThesisMultitaskModel._collect_memory_initialization_token_pool_from_loader`.
- Preserve metric keys, checkpoint filenames, checkpoint extra state keys, console log messages, and returned dictionary shapes.
- Keep model-specific logic inside `src/models/thesis_multitask.py` to respect the `1 model - 1 file` rule.
- Prefer explicit helper names over compact abstractions.
- Add or keep tests focused on behavior, not helper implementation details.

---

### Task 1: Add Trainer Characterization Tests

**Files:**
- Modify: `tests/test_learning_rate_scheduler.py`
- Modify: `tests/test_console_instrumentation.py`

- [ ] **Step 1: Add a trainer behavior test for checkpoint monitor preservation**

Add this test near the existing best-checkpoint tests in `tests/test_learning_rate_scheduler.py`:

```python
def test_trainer_refactor_preserves_epoch_metrics_and_best_checkpoint(
    tmp_path: Path,
) -> None:
    model = DummyPlateauModel(
        val_loss_sequence=[0.9, 0.4],
        val_synth_loss_sequence=[0.8, 0.7],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler, scheduler_monitor_metric = build_scheduler_from_experiment_config(
        optimizer,
        _build_scheduler_experiment_config(
            patience=1,
            monitor_metric="val_synth_loss",
        ),
    )
    experiment_logger = ExperimentLogger(tmp_path / "logs")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_monitor_metric=scheduler_monitor_metric,
        checkpoint_manager=CheckpointManager(tmp_path / "checkpoints"),
        experiment_logger=experiment_logger,
        device="cpu",
    )
    batch = _build_batch()

    try:
        outputs = trainer.train(
            train_loader=[batch],
            val_loader=[batch],
            scaler_state={
                "feature_mean": torch.zeros(38),
                "feature_std": torch.ones(38),
            },
            config={"experiment_name": "trainer-refactor-characterization"},
            epochs=2,
        )
    finally:
        experiment_logger.close()

    metric_history = outputs["metric_history"]
    assert len(metric_history) == 2
    assert metric_history[0]["epoch"] == 1
    assert metric_history[1]["epoch"] == 2
    assert metric_history[0]["train_loss"] == 1.0
    assert metric_history[1]["val_loss"] == 0.4
    assert metric_history[1]["val_synth_loss"] == 0.7
    assert metric_history[1]["scheduler_monitor_val_synth_loss"] == 0.7
    assert outputs["best_checkpoint_path"] is not None
```

- [ ] **Step 2: Run the characterization tests before refactoring**

Run:

```bash
pytest -q tests/test_learning_rate_scheduler.py::test_trainer_refactor_preserves_epoch_metrics_and_best_checkpoint tests/test_console_instrumentation.py::test_run_training_experiment_emits_runtime_console_messages
```

Expected: both tests pass before refactoring.

---

### Task 2: Refactor `Trainer._run_validation_epoch`

**Files:**
- Modify: `src/engine/trainer.py:191-241`
- Test: `tests/test_learning_rate_scheduler.py`

- [ ] **Step 1: Add a validation-batch helper**

Add this helper above `_run_validation_epoch`:

```python
    def _run_validation_batch(
        self,
        *,
        val_batch: dict[str, Any],
        val_batch_index: int,
        epoch_index: int,
        stage_name: str,
        step_method: Any,
    ) -> dict[str, Any]:
        batch_on_device = self._move_batch_to_device(val_batch)
        console_print(
            stage_name.upper(),
            "Processing validation batch",
            epoch=epoch_index + 1,
            batch_index=val_batch_index,
        )
        step_output = step_method(batch_on_device)
        console_print(
            stage_name.upper(),
            "Completed validation batch",
            epoch=epoch_index + 1,
            batch_index=val_batch_index,
            step_log=step_output["log"],
        )
        return step_output
```

- [ ] **Step 2: Add classification and runtime collection helpers**

Add these helpers near `_run_validation_batch`:

```python
    @staticmethod
    def _step_output_has_stage_classification_metrics(
        step_output: dict[str, Any],
        stage_name: str,
    ) -> bool:
        return (
            step_output["outputs"].get("logits") is not None
            and "classification_labels" in step_output["batch"]
            and f"{stage_name}_classification_loss" in step_output["log"]
        )

    @staticmethod
    def _append_classification_history_if_available(
        *,
        step_output: dict[str, Any],
        stage_name: str,
        logits_history: list[torch.Tensor],
        label_history: list[torch.Tensor],
    ) -> None:
        if not Trainer._step_output_has_stage_classification_metrics(
            step_output,
            stage_name,
        ):
            return
        logits_history.append(step_output["outputs"]["logits"].detach().cpu())
        label_history.append(
            step_output["batch"]["classification_labels"].detach().cpu()
        )

    @staticmethod
    def _append_forward_pass_seconds_if_available(
        *,
        step_output: dict[str, Any],
        forward_pass_seconds_history: list[float],
    ) -> None:
        output_aux = step_output["outputs"]["aux"]
        if "forward_pass_seconds" in output_aux:
            forward_pass_seconds_history.append(float(output_aux["forward_pass_seconds"]))
```

- [ ] **Step 3: Rewrite `_run_validation_epoch` loop as orchestration**

Replace the body inside the `for val_batch_index...` loop with:

```python
                step_output = self._run_validation_batch(
                    val_batch=val_batch,
                    val_batch_index=val_batch_index,
                    epoch_index=epoch_index,
                    stage_name=stage_name,
                    step_method=step_method,
                )
                stage_logs.append(step_output["log"])
                self._append_classification_history_if_available(
                    step_output=step_output,
                    stage_name=stage_name,
                    logits_history=logits_history,
                    label_history=label_history,
                )
                self._append_forward_pass_seconds_if_available(
                    step_output=step_output,
                    forward_pass_seconds_history=forward_pass_seconds_history,
                )
```

- [ ] **Step 4: Run trainer validation tests**

Run:

```bash
pytest -q tests/test_learning_rate_scheduler.py tests/test_console_instrumentation.py
```

Expected: all selected tests pass.

---

### Task 3: Refactor `Trainer.train` Epoch Loop

**Files:**
- Modify: `src/engine/trainer.py:243-433`
- Test: `tests/test_learning_rate_scheduler.py`, `tests/test_console_instrumentation.py`

- [ ] **Step 1: Add epoch setup helper**

Add above `train`:

```python
    def _prepare_training_epoch(
        self,
        *,
        train_loader: Any,
        epoch_index: int,
        epochs: int,
    ) -> None:
        self.model.train()
        if hasattr(self.model, "set_epoch_context"):
            self.model.set_epoch_context(epoch_index=epoch_index, total_epochs=epochs)
        if hasattr(self.model, "maybe_initialize_memories_from_loader"):
            memory_initialized = self.model.maybe_initialize_memories_from_loader(
                train_loader=train_loader,
                device=self.device,
            )
            console_print(
                "TRAIN",
                "Checked prototype memory initialization hook",
                epoch=epoch_index + 1,
                memory_initialized=memory_initialized,
            )
        console_print("TRAIN", "Starting epoch", epoch=epoch_index + 1)
```

- [ ] **Step 2: Add training-batch helper**

Add:

```python
    def _run_training_batch(
        self,
        *,
        train_batch: dict[str, Any],
        train_batch_index: int,
        epoch_index: int,
    ) -> dict[str, Any]:
        batch_on_device = self._move_batch_to_device(train_batch)
        console_print(
            "TRAIN",
            "Processing training batch",
            epoch=epoch_index + 1,
            batch_index=train_batch_index,
        )
        step_output = self.model.training_step(batch_on_device)
        loss = step_output["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        console_print(
            "TRAIN",
            "Completed optimizer step",
            epoch=epoch_index + 1,
            batch_index=train_batch_index,
            loss=float(loss.detach().cpu()),
            step_log=step_output["log"],
        )
        return step_output
```

- [ ] **Step 3: Add training epoch helper**

Add:

```python
    def _run_training_epoch(
        self,
        *,
        train_loader: Any,
        epoch_index: int,
    ) -> tuple[
        list[dict[str, float]], list[torch.Tensor], list[torch.Tensor], list[float]
    ]:
        train_logs: list[dict[str, float]] = []
        train_logits_history: list[torch.Tensor] = []
        train_label_history: list[torch.Tensor] = []
        train_forward_pass_seconds_history: list[float] = []

        for train_batch_index, train_batch in enumerate(train_loader, start=1):
            step_output = self._run_training_batch(
                train_batch=train_batch,
                train_batch_index=train_batch_index,
                epoch_index=epoch_index,
            )
            train_logs.append(step_output["log"])
            if (
                step_output["outputs"].get("logits") is not None
                and "classification_labels" in step_output["batch"]
            ):
                train_logits_history.append(
                    step_output["outputs"]["logits"].detach().cpu()
                )
                train_label_history.append(
                    step_output["batch"]["classification_labels"].detach().cpu()
                )
            self._append_forward_pass_seconds_if_available(
                step_output=step_output,
                forward_pass_seconds_history=train_forward_pass_seconds_history,
            )

        return (
            train_logs,
            train_logits_history,
            train_label_history,
            train_forward_pass_seconds_history,
        )
```

- [ ] **Step 4: Add optional synthetic validation helper**

Add:

```python
    def _run_optional_synthetic_validation_epoch(
        self,
        *,
        val_loader: Any,
        epoch_index: int,
    ) -> tuple[
        list[dict[str, float]], list[torch.Tensor], list[torch.Tensor], list[float]
    ]:
        if not hasattr(self.model, "synthetic_validation_step"):
            return [], [], [], []

        if hasattr(self.model, "prepare_synthetic_validation_epoch"):
            self.model.prepare_synthetic_validation_epoch()

        return self._run_validation_epoch(
            val_loader=val_loader,
            epoch_index=epoch_index,
            stage_name="val_synth",
            step_method_name="synthetic_validation_step",
        )
```

- [ ] **Step 5: Add epoch metrics helper**

Add:

```python
    def _build_epoch_metrics(
        self,
        *,
        epoch_index: int,
        train_logs: list[dict[str, float]],
        val_logs: list[dict[str, float]],
        val_synth_logs: list[dict[str, float]],
        train_logits_history: list[torch.Tensor],
        train_label_history: list[torch.Tensor],
        train_forward_pass_seconds_history: list[float],
        val_synth_logits_history: list[torch.Tensor],
        val_synth_label_history: list[torch.Tensor],
        val_synth_forward_pass_seconds_history: list[float],
    ) -> dict[str, Any]:
        epoch_metrics: dict[str, Any] = {"epoch": epoch_index + 1}
        epoch_metrics.update(self._aggregate_logs(train_logs))
        epoch_metrics.update(self._aggregate_logs(val_logs))
        epoch_metrics.update(self._aggregate_logs(val_synth_logs))
        epoch_metrics.update(
            self._aggregate_multitask_classification_metrics(
                logits_history=train_logits_history,
                label_history=train_label_history,
                forward_pass_seconds_history=train_forward_pass_seconds_history,
                stage_name="train",
            )
        )
        epoch_metrics.update(
            self._aggregate_multitask_classification_metrics(
                logits_history=val_synth_logits_history,
                label_history=val_synth_label_history,
                forward_pass_seconds_history=val_synth_forward_pass_seconds_history,
                stage_name="val_synth",
            )
        )
        epoch_metrics.update(self._step_learning_rate_scheduler(epoch_metrics))
        return epoch_metrics
```

- [ ] **Step 6: Add metric logging helper**

Add:

```python
    def _record_epoch_metrics(
        self,
        *,
        epoch_index: int,
        epoch_metrics: dict[str, Any],
    ) -> None:
        self.metric_history.append(epoch_metrics)
        self.experiment_logger.log_metrics(epoch_metrics)
        console_print(
            "TRAIN",
            "Completed epoch",
            epoch=epoch_index + 1,
            epoch_metrics=epoch_metrics,
        )
```

- [ ] **Step 7: Add best-checkpoint helper**

Add:

```python
    def _save_best_checkpoint_if_improved(
        self,
        *,
        epoch_metrics: dict[str, Any],
        epoch_index: int,
        best_checkpoint_monitor_metric: str,
        best_checkpoint_monitor_mode: str,
        best_checkpoint_metric_value: float,
        scaler_state: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[float, Any | None]:
        if best_checkpoint_monitor_metric not in epoch_metrics:
            raise KeyError(
                f"Best checkpoint monitor metric '{best_checkpoint_monitor_metric}' is missing from epoch metrics"
            )
        current_checkpoint_metric_value = float(
            epoch_metrics[best_checkpoint_monitor_metric]
        )
        if not self._is_best_checkpoint_metric_improved(
            candidate_metric_value=current_checkpoint_metric_value,
            best_metric_value=best_checkpoint_metric_value,
            monitor_mode=best_checkpoint_monitor_mode,
        ):
            return best_checkpoint_metric_value, None

        best_checkpoint_metric_value = current_checkpoint_metric_value
        console_print(
            "CHECKPOINT",
            "Checkpoint monitor improved; saving best checkpoint",
            epoch=epoch_index + 1,
            checkpoint_monitor_metric=best_checkpoint_monitor_metric,
            checkpoint_monitor_mode=best_checkpoint_monitor_mode,
            checkpoint_monitor_value=current_checkpoint_metric_value,
            best_checkpoint_metric_value=best_checkpoint_metric_value,
        )
        best_checkpoint_path = self.checkpoint_manager.save_checkpoint(
            checkpoint_name="best.pt",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler_state=scaler_state,
            config=config,
            epoch=epoch_index + 1,
            metric_history=self.metric_history,
            extra_state=(
                self.model.get_checkpoint_extra_state()
                if hasattr(self.model, "get_checkpoint_extra_state")
                else None
            ),
        )
        return best_checkpoint_metric_value, best_checkpoint_path
```

- [ ] **Step 8: Replace the long `for epoch_index` body**

Replace the loop body in `train` with orchestration:

```python
            self._prepare_training_epoch(
                train_loader=train_loader,
                epoch_index=epoch_index,
                epochs=epochs,
            )
            (
                train_logs,
                train_logits_history,
                train_label_history,
                train_forward_pass_seconds_history,
            ) = self._run_training_epoch(
                train_loader=train_loader,
                epoch_index=epoch_index,
            )

            self.model.eval()
            val_logs, _, _, _ = self._run_validation_epoch(
                val_loader=val_loader,
                epoch_index=epoch_index,
                stage_name="val",
                step_method_name="validation_step",
            )
            (
                val_synth_logs,
                val_synth_logits_history,
                val_synth_label_history,
                val_synth_forward_pass_seconds_history,
            ) = self._run_optional_synthetic_validation_epoch(
                val_loader=val_loader,
                epoch_index=epoch_index,
            )

            epoch_metrics = self._build_epoch_metrics(
                epoch_index=epoch_index,
                train_logs=train_logs,
                val_logs=val_logs,
                val_synth_logs=val_synth_logs,
                train_logits_history=train_logits_history,
                train_label_history=train_label_history,
                train_forward_pass_seconds_history=train_forward_pass_seconds_history,
                val_synth_logits_history=val_synth_logits_history,
                val_synth_label_history=val_synth_label_history,
                val_synth_forward_pass_seconds_history=val_synth_forward_pass_seconds_history,
            )
            self._record_epoch_metrics(
                epoch_index=epoch_index,
                epoch_metrics=epoch_metrics,
            )
            (
                best_checkpoint_metric_value,
                candidate_best_checkpoint_path,
            ) = self._save_best_checkpoint_if_improved(
                epoch_metrics=epoch_metrics,
                epoch_index=epoch_index,
                best_checkpoint_monitor_metric=best_checkpoint_monitor_metric,
                best_checkpoint_monitor_mode=best_checkpoint_monitor_mode,
                best_checkpoint_metric_value=best_checkpoint_metric_value,
                scaler_state=scaler_state,
                config=config,
            )
            if candidate_best_checkpoint_path is not None:
                best_checkpoint_path = candidate_best_checkpoint_path
```

- [ ] **Step 9: Run trainer tests**

Run:

```bash
pytest -q tests/test_learning_rate_scheduler.py tests/test_console_instrumentation.py tests/test_one_train_step.py tests/test_checkpoint_roundtrip.py
```

Expected: all selected tests pass.

---

### Task 4: Refactor `OnlineLoop.run`

**Files:**
- Modify: `src/engine/online_loop.py:53-220`
- Test: `tests/test_online_entrypoint.py`, `tests/test_online_state_roundtrip.py`, `tests/test_online_adaptation_step.py`

- [ ] **Step 1: Add parameter snapshot helper**

Add above `run`:

```python
    def _clone_target_parameters(self) -> list[torch.Tensor]:
        return [
            parameter.detach().clone()
            for parameter in self.model.get_parameter_group(
                self.model.target_param_group
            )
        ]
```

- [ ] **Step 2: Add online score helper**

Add:

```python
    def _compute_window_score_mean(self, batch_on_device: dict[str, Any]) -> float:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(batch_on_device)
        return float(outputs["window_scores"].mean().detach().cpu())
```

- [ ] **Step 3: Add single optimization-step helper**

Add:

```python
    def _run_online_training_step(
        self,
        batch_on_device: dict[str, Any],
    ) -> None:
        self.model.train()
        step_output = self.model.training_step(batch_on_device)
        loss = step_output["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

- [ ] **Step 4: Add metric builder helper**

Add:

```python
    @staticmethod
    def _build_online_step_metrics(
        *,
        step_index: int,
        pre_window_score_mean: float,
        post_window_score_mean: float,
        update_norm: float,
        post_outputs: dict[str, Any],
    ) -> dict[str, float]:
        return {
            "online/step": step_index,
            "online/pre_window_score_mean": pre_window_score_mean,
            "online/post_window_score_mean": post_window_score_mean,
            "online/update_norm": update_norm,
            "online/alignment_loss": float(
                post_outputs["aux"]["alignment_loss"].detach().cpu()
            ),
            "online/prototype_alignment_loss": float(
                post_outputs["aux"]["prototype_alignment_loss"].detach().cpu()
            ),
            "online/projector_drift": float(
                post_outputs["aux"]["projector_drift"].detach().cpu()
            ),
        }
```

- [ ] **Step 5: Add post-forward helper returning outputs and score**

Add:

```python
    def _run_post_update_forward(
        self,
        batch_on_device: dict[str, Any],
    ) -> tuple[dict[str, Any], float]:
        self.model.eval()
        with torch.no_grad():
            post_outputs = self.model.forward(batch_on_device)
        post_window_score_mean = float(
            post_outputs["window_scores"].mean().detach().cpu()
        )
        return post_outputs, post_window_score_mean
```

- [ ] **Step 6: Add single online-step helper**

Add:

```python
    def _run_online_step(
        self,
        *,
        batch: dict[str, Any],
        step_index: int,
    ) -> dict[str, Any]:
        batch_on_device = self._move_batch_to_device(batch)
        console_print("ONLINE", "Processing online step", step_index=step_index)

        pre_window_score_mean = self._compute_window_score_mean(batch_on_device)
        trainable_parameters = self._clone_target_parameters()
        self._run_online_training_step(batch_on_device)
        post_outputs, post_window_score_mean = self._run_post_update_forward(
            batch_on_device
        )
        updated_parameters = self._clone_target_parameters()
        update_norm = self._measure_update_norm(
            trainable_parameters,
            updated_parameters,
        )
        step_metrics = self._build_online_step_metrics(
            step_index=step_index,
            pre_window_score_mean=pre_window_score_mean,
            post_window_score_mean=post_window_score_mean,
            update_norm=update_norm,
            post_outputs=post_outputs,
        )
        return {
            "step_metrics": step_metrics,
            "pre_window_score_mean": pre_window_score_mean,
            "post_window_score_mean": post_window_score_mean,
        }
```

- [ ] **Step 7: Add record and checkpoint helpers**

Add:

```python
    @staticmethod
    def _build_online_record(
        *,
        batch: dict[str, Any],
        step_index: int,
        step_metrics: dict[str, Any],
        pre_window_score_mean: float,
        post_window_score_mean: float,
    ) -> dict[str, Any]:
        return {
            "step": step_index,
            "entity_ids": [meta["entity_id"] for meta in batch["meta"]],
            "stream_steps": [meta["stream_step"] for meta in batch["meta"]],
            "start_indices": [meta["start_index"] for meta in batch["meta"]],
            "end_indices": [meta["end_index"] for meta in batch["meta"]],
            "pre_window_score_mean": pre_window_score_mean,
            "post_window_score_mean": post_window_score_mean,
            "alignment_loss": step_metrics["online/alignment_loss"],
            "projector_drift": step_metrics["online/projector_drift"],
        }

    def _build_online_checkpoint_extra_state(
        self,
        online_batcher: OnlineWindowBatcher,
    ) -> dict[str, Any]:
        return {
            "stream_state_dict": online_batcher.state_dict(),
            "projector_anchor_state_dict": self.model.get_projector_anchor_state_dict(),
            "target_param_group": self.model.target_param_group,
            "online_metric_history": self.metric_history,
            "reset_policy_state": {
                "reset_policy": self.model.reset_policy,
                "reset_alignment_threshold": self.model.reset_alignment_threshold,
            },
        }
```

- [ ] **Step 8: Add checkpoint save helper**

Add:

```python
    def _save_online_checkpoint(
        self,
        *,
        checkpoint_name: str,
        online_batcher: OnlineWindowBatcher,
        scaler_state: dict[str, Any],
        config: dict[str, Any],
        epoch: int,
    ) -> Any:
        return self.checkpoint_manager.save_checkpoint(
            checkpoint_name=checkpoint_name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=None,
            scaler_state=scaler_state,
            config=config,
            epoch=epoch,
            metric_history=self.metric_history,
            extra_state=self._build_online_checkpoint_extra_state(online_batcher),
        )
```

- [ ] **Step 9: Replace `run` loop body with orchestration**

The loop body should become:

```python
            step_result = self._run_online_step(batch=batch, step_index=step_index)
            step_metrics = step_result["step_metrics"]
            self.metric_history.append(step_metrics)
            if step_index % log_every_n_steps == 0:
                self.experiment_logger.log_metrics(step_metrics)
            console_print(
                "ONLINE",
                "Completed online step",
                step_index=step_index,
                step_metrics=step_metrics,
            )
            records.append(
                self._build_online_record(
                    batch=batch,
                    step_index=step_index,
                    step_metrics=step_metrics,
                    pre_window_score_mean=step_result["pre_window_score_mean"],
                    post_window_score_mean=step_result["post_window_score_mean"],
                )
            )

            if step_index % checkpoint_every_n_steps == 0:
                console_print(
                    "CHECKPOINT",
                    "Saving periodic online checkpoint",
                    step_index=step_index,
                )
                final_checkpoint_path = self._save_online_checkpoint(
                    checkpoint_name=f"online_step_{step_index}.pt",
                    online_batcher=online_batcher,
                    scaler_state=scaler_state,
                    config=config,
                    epoch=step_index,
                )
```

Replace the final checkpoint block with `_save_online_checkpoint(checkpoint_name="online_final.pt", epoch=len(self.metric_history), ...)`.

- [ ] **Step 10: Run online tests**

Run:

```bash
pytest -q tests/test_online_entrypoint.py tests/test_online_state_roundtrip.py tests/test_online_adaptation_step.py tests/test_online_reference_checkpoint.py
```

Expected: all selected tests pass.

---

### Task 5: Refactor `SMDDatasetParser.parse`

**Files:**
- Modify: `src/data/datasets/smd.py:53-180`
- Test: `tests/test_smd_dataset_shapes.py`, `tests/test_smoke_loader_limits.py`

- [ ] **Step 1: Add entity-id selection helper**

Add above `parse`:

```python
    def _select_entity_ids(
        self,
        *,
        train_files_by_entity: dict[str, Path],
        test_files_by_entity: dict[str, Path],
        label_files_by_entity: dict[str, Path],
    ) -> list[str]:
        if self.entity_ids is None:
            return sorted(train_files_by_entity.keys())

        selected_entity_ids = list(self.entity_ids)
        if not selected_entity_ids:
            raise ValueError(
                "SMD parser requires at least one entity_id when filtering is enabled"
            )
        for entity_id in selected_entity_ids:
            if entity_id not in train_files_by_entity:
                raise ValueError(
                    f"Requested SMD entity is missing from train split: {entity_id}"
                )
            if entity_id not in test_files_by_entity:
                raise ValueError(
                    f"Requested SMD entity is missing from test split: {entity_id}"
                )
            if entity_id not in label_files_by_entity:
                raise ValueError(
                    f"Requested SMD entity is missing from test_label split: {entity_id}"
                )
        return selected_entity_ids
```

- [ ] **Step 2: Add tensor loading and validation helpers**

Add:

```python
    def _load_entity_tensors(
        self,
        *,
        entity_id: str,
        train_files_by_entity: dict[str, Path],
        test_files_by_entity: dict[str, Path],
        label_files_by_entity: dict[str, Path],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        train_tensor = self._load_feature_matrix(train_files_by_entity[entity_id])
        test_tensor = self._load_feature_matrix(test_files_by_entity[entity_id])
        test_labels = self._load_label_vector(label_files_by_entity[entity_id])
        console_print(
            "DATA",
            "Loaded SMD entity files",
            entity_id=entity_id,
            train_tensor=summarize_tensor(train_tensor),
            test_tensor=summarize_tensor(test_tensor),
            test_labels=summarize_tensor(test_labels),
        )
        if test_tensor.shape[0] != test_labels.shape[0]:
            raise ValueError(
                f"Test labels do not match test sequence length for {entity_id}"
            )
        return train_tensor, test_tensor, test_labels

    def _compute_train_validation_split(
        self,
        *,
        train_tensor: torch.Tensor,
        entity_id: str,
    ) -> tuple[int, int]:
        validation_length = max(
            1,
            int(train_tensor.shape[0] * self.validation_split_ratio),
        )
        train_cutoff = train_tensor.shape[0] - validation_length
        if train_cutoff < 1:
            raise ValueError(
                f"Validation split ratio leaves no training data for {entity_id}"
            )
        return train_cutoff, validation_length
```

- [ ] **Step 3: Add entity sequence builder helper**

Add:

```python
    def _build_entity_split_sequences(
        self,
        *,
        entity_id: str,
        train_tensor: torch.Tensor,
        test_tensor: torch.Tensor,
        test_labels: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        train_cutoff, validation_length = self._compute_train_validation_split(
            train_tensor=train_tensor,
            entity_id=entity_id,
        )
        train_sequence = self._build_raw_sequence(
            x_tensor=train_tensor[:train_cutoff].clone(),
            entity_id=entity_id,
            split="train",
            point_labels=torch.zeros(train_cutoff, dtype=torch.long),
        )
        val_sequence = self._build_raw_sequence(
            x_tensor=train_tensor[train_cutoff:].clone(),
            entity_id=entity_id,
            split="val",
            point_labels=torch.zeros(validation_length, dtype=torch.long),
        )
        test_sequence = self._build_raw_sequence(
            x_tensor=test_tensor.clone(),
            entity_id=entity_id,
            split="test",
            point_labels=test_labels.clone(),
        )
        return train_sequence, val_sequence, test_sequence
```

- [ ] **Step 4: Replace entity loop body**

Replace the body of `for entity_id in selected_entity_ids:` with:

```python
            train_tensor, test_tensor, test_labels = self._load_entity_tensors(
                entity_id=entity_id,
                train_files_by_entity=train_files_by_entity,
                test_files_by_entity=test_files_by_entity,
                label_files_by_entity=label_files_by_entity,
            )
            train_sequence, val_sequence, test_sequence = (
                self._build_entity_split_sequences(
                    entity_id=entity_id,
                    train_tensor=train_tensor,
                    test_tensor=test_tensor,
                    test_labels=test_labels,
                )
            )
            train_sequences.append(train_sequence)
            val_sequences.append(val_sequence)
            test_sequences.append(test_sequence)
```

Also replace the existing entity selection block with:

```python
        selected_entity_ids = self._select_entity_ids(
            train_files_by_entity=train_files_by_entity,
            test_files_by_entity=test_files_by_entity,
            label_files_by_entity=label_files_by_entity,
        )
```

- [ ] **Step 5: Run SMD tests**

Run:

```bash
pytest -q tests/test_smd_dataset_shapes.py tests/test_smoke_loader_limits.py tests/test_public_data_api.py
```

Expected: all selected tests pass.

---

### Task 6: Refactor Thesis Multitask Memory Initialization Loop

**Files:**
- Modify: `src/models/thesis_multitask.py:690-756`
- Test: `tests/test_multitask_memory_initialization.py`

- [ ] **Step 1: Add clean-token helper**

Add above `_collect_memory_initialization_token_pool_from_loader`:

```python
    def _collect_clean_memory_initialization_tokens(
        self,
        batch_on_device: dict[str, Any],
    ) -> torch.Tensor:
        clean_batch = self._prepare_clean_batch(
            batch_on_device,
            stage_name="memory_init",
        )
        return self.encoder(clean_batch)["hidden"].reshape(-1, self.hidden_dim)
```

- [ ] **Step 2: Add synthetic-normal-token helper**

Add:

```python
    def _collect_synthetic_normal_memory_initialization_tokens(
        self,
        batch_on_device: dict[str, Any],
    ) -> torch.Tensor | None:
        if not (
            self.memory_initialization_with_synthetic_windows
            and self.use_synthetic_augmentation
        ):
            return None

        synthetic_batch = self.synthetic_anomaly_injector.augment_batch(
            self._clone_batch(batch_on_device)
        )
        synthetic_hidden = self.encoder(synthetic_batch)["hidden"]
        normal_time_step_mask = synthetic_batch["synthetic_anomaly_mask"] == 0
        synthetic_normal_hidden = synthetic_hidden[normal_time_step_mask]
        if synthetic_normal_hidden.numel() == 0:
            return None
        return synthetic_normal_hidden
```

- [ ] **Step 3: Replace body of memory initialization loop**

Inside `for batch_index, raw_batch in enumerate(train_loader):`, keep the limit and device movement, then replace clean/synthetic logic with:

```python
                clean_hidden_tokens.append(
                    self._collect_clean_memory_initialization_tokens(batch_on_device)
                )
                synthetic_normal_hidden = (
                    self._collect_synthetic_normal_memory_initialization_tokens(
                        batch_on_device
                    )
                )
                if synthetic_normal_hidden is not None:
                    synthetic_normal_hidden_tokens.append(synthetic_normal_hidden)
```

- [ ] **Step 4: Run memory initialization tests**

Run:

```bash
pytest -q tests/test_multitask_memory_initialization.py tests/test_multitask_memory_bootstrap.py tests/test_multitask_memory_updates.py
```

Expected: all selected tests pass.

---

### Task 7: Final Review And Regression Suite

**Files:**
- Inspect: `src/engine/trainer.py`
- Inspect: `src/engine/online_loop.py`
- Inspect: `src/data/datasets/smd.py`
- Inspect: `src/models/thesis_multitask.py`

- [ ] **Step 1: Re-run AST long-loop scan**

Run:

```bash
python3 -c "import ast, subprocess
paths=subprocess.check_output(['rg','--files','-g','*.py','-g','!bsc-thesis-ref-codebases/**','-g','!data/**','-g','!__pycache__/**'], text=True).splitlines()
rows=[]
for path in paths:
    with open(path, encoding='utf-8') as file:
        tree=ast.parse(file.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)) and hasattr(node, 'end_lineno'):
            rows.append((node.end_lineno-node.lineno+1, path, node.lineno, node.end_lineno))
for lines, path, start, end in sorted(rows, reverse=True)[:20]:
    print(f'{lines:4d} lines | {path}:{start}-{end}')
"
```

Expected:
- `src/engine/trainer.py:271-428` no longer appears as a 158-line loop.
- `src/engine/online_loop.py:78-187` no longer appears as a 110-line loop.
- `src/data/datasets/smd.py:111-164` no longer appears as a 54-line loop.
- `src/models/thesis_multitask.py:705-736` no longer appears as a 32-line mixed-responsibility loop.

- [ ] **Step 2: Run targeted regression tests**

Run:

```bash
pytest -q \
  tests/test_learning_rate_scheduler.py \
  tests/test_console_instrumentation.py \
  tests/test_online_entrypoint.py \
  tests/test_online_state_roundtrip.py \
  tests/test_online_adaptation_step.py \
  tests/test_online_reference_checkpoint.py \
  tests/test_smd_dataset_shapes.py \
  tests/test_smoke_loader_limits.py \
  tests/test_public_data_api.py \
  tests/test_multitask_memory_initialization.py \
  tests/test_multitask_memory_bootstrap.py \
  tests/test_multitask_memory_updates.py
```

Expected: all selected tests pass.

- [ ] **Step 3: Inspect diff for behavior-only refactor**

Run:

```bash
git diff -- src/engine/trainer.py src/engine/online_loop.py src/data/datasets/smd.py src/models/thesis_multitask.py tests/test_learning_rate_scheduler.py
```

Expected:
- Existing public return keys are unchanged.
- Existing checkpoint extra-state keys are unchanged.
- Existing console log messages are unchanged unless a test was updated intentionally.
- Each reviewed long loop now reads as orchestration over helper methods.
