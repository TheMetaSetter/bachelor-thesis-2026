from __future__ import annotations

"""Checkpoint serialization helpers for the thesis multitask model."""

from typing import Any

import torch

from src.core.console import console_print


class ThesisMultitaskStateSerializationMixin:
    def _normalize_verification_metadata_source(self) -> None:
        if not bool(getattr(self, "memory_initialized", False)):
            return
        if not isinstance(self.anomalous_codeword_mask, torch.Tensor):
            return
        if not isinstance(self.anomaly_radii, torch.Tensor):
            return
        if getattr(self, "verification_metadata_source", "") in {
            "",
            "uninitialized",
            "disabled",
        }:
            self.verification_metadata_source = "train_anomaly_tokens_q99"

    def _validate_verification_metadata_state(self) -> None:
        if not bool(getattr(self, "memory_initialized", False)):
            return
        if not isinstance(self.anomalous_codeword_mask, torch.Tensor):
            raise ValueError("verification metadata requires anomalous_codeword_mask")
        if not isinstance(self.anomaly_radii, torch.Tensor):
            raise ValueError("verification metadata requires anomaly_radii")
        if self.anomalous_codeword_mask.ndim != 1:
            raise ValueError("anomalous_codeword_mask must be one-dimensional")
        if self.anomaly_radii.ndim != 1:
            raise ValueError("anomaly_radii must be one-dimensional")
        if self.anomalous_codeword_mask.numel() == 0:
            raise ValueError("anomalous_codeword_mask must be non-empty")
        if self.anomaly_radii.numel() == 0:
            raise ValueError("anomaly_radii must be non-empty")
        if self.anomalous_codeword_mask.shape != self.anomaly_radii.shape:
            raise ValueError(
                "verification metadata mask and radii must have same shape"
            )
        if not torch.isfinite(self.anomaly_radii).all().item():
            raise ValueError("anomaly_radii must contain only finite values")
        if (self.anomaly_radii < 0).any().item():
            raise ValueError("anomaly_radii must be non-negative")
        if getattr(self, "verification_metadata_source", "") in {
            "",
            "uninitialized",
            "disabled",
        }:
            raise ValueError(
                "verification metadata source must be concrete when memory is initialized"
            )
        if isinstance(self.discrete_codebook, torch.Tensor):
            expected_shape = (self.discrete_codebook.shape[0],)
            if self.anomalous_codeword_mask.shape != expected_shape:
                raise ValueError(
                    "verification metadata mask must match discrete codebook size"
                )
            if self.verification_codeword_class_ids is not None and (
                self.verification_codeword_class_ids.shape != expected_shape
            ):
                raise ValueError(
                    "verification_codeword_class_ids checkpoint shape mismatch"
                )
            if self.verification_contributing_token_counts is not None and (
                self.verification_contributing_token_counts.shape != expected_shape
            ):
                raise ValueError(
                    "verification_contributing_token_counts checkpoint shape mismatch"
                )

    def get_checkpoint_extra_state(self) -> dict[str, Any]:
        self._normalize_verification_metadata_source()
        self._validate_verification_metadata_state()
        state = self.get_memory_lifecycle_state()
        if isinstance(self.anomalous_codeword_mask, torch.Tensor):
            state["anomalous_codeword_mask"] = (
                self.anomalous_codeword_mask.detach().cpu().tolist()
            )
        if isinstance(self.anomaly_radii, torch.Tensor):
            state["anomaly_radii"] = self.anomaly_radii.detach().cpu().tolist()
        state["verification_metadata_source"] = self.verification_metadata_source
        state["verification_metadata_schema_version"] = int(
            getattr(self, "verification_metadata_schema_version", 1)
        )
        state["verification_metadata_split"] = str(
            getattr(self, "verification_metadata_split", "synthetic_train")
        )
        state["verification_metadata_initialization_seed"] = int(
            getattr(self, "verification_metadata_initialization_seed", 0)
        )
        if isinstance(self.verification_codeword_class_ids, torch.Tensor):
            state["verification_codeword_class_ids"] = (
                self.verification_codeword_class_ids.detach().cpu().tolist()
            )
        if isinstance(self.verification_contributing_token_counts, torch.Tensor):
            state["verification_contributing_token_counts"] = (
                self.verification_contributing_token_counts.detach().cpu().tolist()
            )
        state["verification_radius_quantile"] = 0.99
        state["verification_metadata_label_source"] = self.discrete_memory_label_source
        state["stochastic_inference"] = self.stochastic_inference
        state["monte_carlo_samples"] = self.monte_carlo_samples
        state["continuous_temperature"] = self.continuous_temperature
        state["discrete_temperature"] = self.discrete_temperature
        state["variance_correction"] = self.variance_correction
        state["return_mc_samples"] = self.return_mc_samples
        state["sample_retention_policy"] = self.sample_retention_policy
        return state

    def get_memory_tensor_state(self) -> dict[str, torch.Tensor | None]:
        return {
            "continuous_prototype_bank": (
                None
                if self.continuous_prototype_bank is None
                else self.continuous_prototype_bank.detach().clone()
            ),
            "discrete_codebook": (
                None
                if self.discrete_codebook is None
                else self.discrete_codebook.detach().clone()
            ),
            "discrete_ema_counts": (
                None
                if self.discrete_ema_counts is None
                else self.discrete_ema_counts.detach().clone()
            ),
            "discrete_ema_sums": (
                None
                if self.discrete_ema_sums is None
                else self.discrete_ema_sums.detach().clone()
            ),
        }

    def mark_memories_initialized(
        self, initialization_epoch: int | None = None
    ) -> None:
        self.memory_initialized = True
        self.memory_training_enabled = not self.freeze_memories_after_initialization
        self.memory_ready_for_initialization = False
        self.memory_initialization_epoch = initialization_epoch
        console_print(
            "MODEL",
            "Marked prototype memories as initialized",
            initialization_epoch=initialization_epoch,
            memory_state=self.get_memory_lifecycle_state(),
        )

    def load_checkpoint_extra_state(self, extra_state: dict[str, Any] | None) -> None:
        if not extra_state:
            return
        if (
            "verification_metadata_schema_version" in extra_state
            and int(extra_state["verification_metadata_schema_version"]) != 1
        ):
            raise ValueError("verification_metadata_schema_version must be 1")
        if (
            "verification_metadata_split" in extra_state
            and extra_state.get("verification_metadata_split") != "synthetic_train"
        ):
            raise ValueError("verification_metadata_split must be synthetic_train")
        if "verification_radius_quantile" in extra_state and not (
            0.0 < float(extra_state["verification_radius_quantile"]) <= 1.0
        ):
            raise ValueError("verification_radius_quantile must be in (0, 1]")
        if (
            "verification_metadata_label_source" in extra_state
            and extra_state.get("verification_metadata_label_source")
            != self.discrete_memory_label_source
        ):
            raise ValueError(
                "verification_metadata_label_source must match discrete_memory_label_source"
            )
        if (
            "verification_metadata_initialization_seed" in extra_state
            and int(extra_state["verification_metadata_initialization_seed"]) < 0
        ):
            raise ValueError(
                "verification_metadata_initialization_seed must be non-negative"
            )
        self.memory_initialized = bool(
            extra_state.get("memory_initialized", self.memory_initialized)
        )
        self.memory_training_enabled = bool(
            extra_state.get("memory_training_enabled", self.memory_training_enabled)
        )
        self.memory_ready_for_initialization = bool(
            extra_state.get(
                "memory_ready_for_initialization",
                self.memory_ready_for_initialization,
            )
        )
        self.memory_initialization_epoch = extra_state.get(
            "memory_initialization_epoch",
            self.memory_initialization_epoch,
        )
        mask = extra_state.get("anomalous_codeword_mask")
        radii = extra_state.get("anomaly_radii")
        if mask is not None and radii is not None:
            if not isinstance(self.discrete_codebook, torch.Tensor):
                raise ValueError("verification metadata requires a discrete codebook")
            mask = torch.as_tensor(mask, dtype=torch.bool)
            radii = torch.as_tensor(radii, dtype=torch.float32)
            if mask.shape != (self.discrete_codebook.shape[0],):
                raise ValueError("anomalous_codeword_mask checkpoint shape mismatch")
            if (
                radii.shape != mask.shape
                or not torch.isfinite(radii).all().item()
                or (radii < 0).any().item()
            ):
                raise ValueError("anomaly_radii checkpoint shape or value mismatch")
            self.anomalous_codeword_mask = mask.detach().bool().clone()
            self.anomaly_radii = radii.detach().float().clone()
            self.verification_metadata_source = str(
                extra_state.get("verification_metadata_source", "checkpoint")
            )
            if "verification_codeword_class_ids" in extra_state:
                codeword_class_ids = torch.as_tensor(
                    extra_state["verification_codeword_class_ids"], dtype=torch.long
                )
                if codeword_class_ids.shape != mask.shape:
                    raise ValueError(
                        "verification_codeword_class_ids checkpoint shape mismatch"
                    )
                self.verification_codeword_class_ids = (
                    codeword_class_ids.detach().clone()
                )
            if "verification_contributing_token_counts" in extra_state:
                contributing_token_counts = torch.as_tensor(
                    extra_state["verification_contributing_token_counts"],
                    dtype=torch.float32,
                )
                if contributing_token_counts.shape != mask.shape:
                    raise ValueError(
                        "verification_contributing_token_counts checkpoint shape mismatch"
                    )
            self.verification_contributing_token_counts = (
                contributing_token_counts.detach().clone()
            )
            self.verification_metadata_schema_version = int(
                extra_state.get("verification_metadata_schema_version", 1)
            )
            self.verification_metadata_split = str(
                extra_state.get("verification_metadata_split", "synthetic_train")
            )
            self.verification_metadata_initialization_seed = int(
                extra_state.get("verification_metadata_initialization_seed", 0)
            )
            self._normalize_verification_metadata_source()
            self._validate_verification_metadata_state()
        if "stochastic_inference" in extra_state:
            if bool(extra_state["stochastic_inference"]) != self.stochastic_inference:
                raise ValueError("checkpoint stochastic_inference does not match model")
        if "monte_carlo_samples" in extra_state:
            if int(extra_state["monte_carlo_samples"]) != self.monte_carlo_samples:
                raise ValueError("checkpoint monte_carlo_samples does not match model")
        if "variance_correction" in extra_state:
            if int(extra_state["variance_correction"]) != self.variance_correction:
                raise ValueError("checkpoint variance_correction does not match model")
        if "sample_retention_policy" in extra_state:
            if (
                str(extra_state["sample_retention_policy"])
                != self.sample_retention_policy
            ):
                console_print(
                    "MODEL",
                    "Allowing compatibility mismatch for sample_retention_policy",
                    checkpoint_value=str(extra_state["sample_retention_policy"]),
                    model_value=self.sample_retention_policy,
                )
