from __future__ import annotations

"""Batch preparation and forward helpers for the thesis multitask routing path."""

import time
from typing import Any

import torch

from src.core.console import (
    console_print,
    debug_print_if,
    summarize_batch,
    summarize_label_distribution,
    summarize_tensor,
)
from src.core.contracts import validate_batch, validate_model_outputs


def _prepare_clean_batch(
    self: Any, batch: dict[str, Any], stage_name: str
) -> dict[str, Any]:
    if (
        "classification_labels" in batch
        and "synthetic_anomaly_mask" in batch
        and "augmentation_metadata" in batch
    ):
        console_print(
            stage_name.upper(),
            "Received pre-augmented multitask batch",
            **summarize_batch(batch),
            classification_label_distribution=summarize_label_distribution(
                batch["classification_labels"]
            ),
        )
        return self._clone_batch(batch)

    if stage_name == "train" and self.use_synthetic_augmentation:
        return self.synthetic_anomaly_injector.augment_batch(batch)

    prepared_batch = self._clone_batch(batch)
    batch_size, window_size, _ = prepared_batch["x"].shape
    prepared_batch["classification_labels"] = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=prepared_batch["x"].device,
    )
    prepared_batch["classification_class_names"] = self._classification_class_names()
    prepared_batch["synthetic_anomaly_mask"] = torch.zeros(
        batch_size,
        window_size,
        dtype=torch.long,
        device=prepared_batch["x"].device,
    )
    prepared_batch["augmentation_metadata"] = [
        {
            "is_synthetic_anomaly": False,
            "anomaly_family": "clean",
            "anomaly_family_index": None,
            "start_index": None,
            "end_index": None,
            "affected_channels": [],
            "family_parameters_by_channel": {},
        }
        for _ in range(batch_size)
    ]
    if prepared_batch["point_labels"] is None:
        prepared_batch["point_labels"] = prepared_batch[
            "synthetic_anomaly_mask"
        ].clone()
    console_print(
        stage_name.upper(),
        "Prepared clean multitask batch",
        **summarize_batch(prepared_batch),
        classification_label_distribution=summarize_label_distribution(
            prepared_batch["classification_labels"]
        ),
    )
    return prepared_batch


def _build_contrastive_token_masks(
    self: Any, batch: dict[str, Any]
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    anomaly_mask = batch.get("synthetic_anomaly_mask")
    if anomaly_mask is None or not self.enable_two_view_contrastive:
        return None, None
    return anomaly_mask == 0, anomaly_mask == 1


def _resolve_active_memory_banks(
    self: Any,
    hidden: torch.Tensor,
    *,
    normal_token_mask: torch.Tensor | None,
    anomaly_token_mask: torch.Tensor | None,
    stage_name: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    active_continuous_memory_bank = None
    if self.continuous_prototype_bank is not None:
        if self._should_update_memory(stage_name):
            active_continuous_memory_bank = self._update_continuous_memory_bank(
                hidden,
                token_mask=normal_token_mask,
            )
        else:
            active_continuous_memory_bank = self._normalize_memory_vectors(
                self.continuous_prototype_bank
            )

    active_discrete_codebook = None
    if self.discrete_codebook is not None:
        if self._should_update_memory(stage_name):
            self._update_discrete_codebook_memory(
                hidden,
                token_mask=anomaly_token_mask,
            )
        active_discrete_codebook = self._normalize_memory_vectors(
            self.discrete_codebook
        )

    return active_continuous_memory_bank, active_discrete_codebook


def forward(
    self: Any, batch: dict[str, Any], stage_name: str = "train"
) -> dict[str, Any]:
    validate_batch(batch)
    console_print("MODEL", "Multitask forward input batch", **summarize_batch(batch))
    forward_start_time = time.perf_counter()
    encoder_outputs = self.encoder(batch)
    hidden = encoder_outputs["hidden"]
    normal_token_mask, anomaly_token_mask = _build_contrastive_token_masks(self, batch)
    monte_carlo_forward_outputs = None
    if self._phase_uses_prototype_path():
        active_continuous_memory_bank, active_discrete_codebook = (
            _resolve_active_memory_banks(
                self,
                hidden,
                normal_token_mask=normal_token_mask,
                anomaly_token_mask=anomaly_token_mask,
                stage_name=stage_name,
            )
        )
        continuous_outputs = self._continuous_prototype_lookup(
            hidden,
            stage_name=stage_name,
            active_memory_bank=active_continuous_memory_bank,
        )
        discrete_outputs = self._discrete_prototype_lookup(
            hidden,
            stage_name=stage_name,
            active_codebook=active_discrete_codebook,
            precomputed_assignment_logits=None,
            precomputed_assignment_probabilities=None,
        )
        fusion_outputs = self._compute_fusion_outputs(
            continuous_hidden=continuous_outputs["prototype_context"],
            discrete_hidden=discrete_outputs["quantized_hidden"],
            base_hidden=hidden,
            paired_hidden=batch.get("paired_hidden_for_fusion"),
        )
        stochastic_query = None
        if self.stochastic_inference:
            query_bundle = self.build_stochastic_queries(
                hidden,
                stage_name=stage_name,
                active_memory_bank=active_continuous_memory_bank,
                active_codebook=active_discrete_codebook,
            )
            continuous_retrieved_samples = self.sample_continuous_retrieval(
                query_bundle,
                self.monte_carlo_samples,
            )
            discrete_retrieved_samples = self.sample_discrete_retrieval(
                query_bundle,
                self.monte_carlo_samples,
            )
            discrete_topk_ids = self.sample_discrete_topk_ids(
                query_bundle,
                self.monte_carlo_samples,
            )
            stochastic_query = {
                "schema_version": 3,
                "enabled": True,
                "num_samples": self.monte_carlo_samples,
                "continuous_temperature": self.continuous_temperature,
                "discrete_temperature": self.discrete_temperature,
                "continuous_retrieved_samples": continuous_retrieved_samples,
                "discrete_retrieved_samples": discrete_retrieved_samples,
                "discrete_topk_ids": discrete_topk_ids,
            }
            if not self.training:
                monte_carlo_forward_outputs = self._build_monte_carlo_forward_outputs(
                    batch,
                    fusion_outputs=fusion_outputs,
                    query_bundle=query_bundle,
                    continuous_samples=continuous_retrieved_samples,
                    discrete_samples=discrete_retrieved_samples,
                    discrete_topk_ids=discrete_topk_ids,
                )
                debug_print_if(
                    "THESIS_DEBUG_UQ_TRACE",
                    "MODEL",
                    "Built Monte Carlo forward outputs",
                    stage_name=stage_name,
                    training=bool(self.training),
                    stochastic_inference=bool(self.stochastic_inference),
                    phase_uses_prototype_path=bool(self._phase_uses_prototype_path()),
                    memory_initialized=bool(self.memory_initialized),
                    memory_bypass_active=bool(query_bundle.memory_bypass_active),
                    has_uncertainty=(
                        monte_carlo_forward_outputs["aux"].get("uncertainty")
                        is not None
                    ),
                    has_stochastic_query=(
                        monte_carlo_forward_outputs["aux"].get("stochastic_query")
                        is not None
                    ),
                    stochastic_query_keys=list(
                        monte_carlo_forward_outputs["aux"]
                        .get("stochastic_query", {})
                        .keys()
                    ),
                )
    else:
        active_continuous_memory_bank = None
        active_discrete_codebook = None
        continuous_outputs, discrete_outputs, fusion_outputs = (
            self._build_phase_passthrough_outputs(hidden)
        )
        stochastic_query = None

    hidden_reconstruction = fusion_outputs["hidden_reconstruction"]
    hidden_classification = fusion_outputs["hidden_classification"]
    recon = self.reconstruction_head(hidden_reconstruction)
    if hidden_classification.shape[1] != self.window_size:
        raise ValueError(
            "hidden_classification must have window dimension "
            f"{self.window_size}, but received {hidden_classification.shape[1]}"
        )
    flattened_classification_hidden = hidden_classification.reshape(
        hidden_classification.shape[0],
        self.window_size * self.hidden_dim,
    )
    logits = None
    class_probabilities = None
    monte_carlo_uncertainty = None
    raw_point_scores = torch.mean((recon - batch["x"]) ** 2, dim=-1)
    point_scores, point_score_calibrated = self.transform_official_point_scores(
        raw_point_scores
    )
    window_scores = raw_point_scores.mean(dim=1)
    if not self.training and monte_carlo_forward_outputs is not None:
        recon = monte_carlo_forward_outputs["outputs"]["recon"]
        logits = monte_carlo_forward_outputs["outputs"]["logits"]
        point_scores = monte_carlo_forward_outputs["outputs"]["point_scores"]
        window_scores = monte_carlo_forward_outputs["outputs"]["window_scores"]
        raw_point_scores = monte_carlo_forward_outputs["aux"]["raw_point_scores"]
        point_score_calibrated = monte_carlo_forward_outputs["aux"][
            "point_score_calibrated"
        ]
        class_probabilities = monte_carlo_forward_outputs["aux"].get(
            "class_probabilities"
        )
        stochastic_query = monte_carlo_forward_outputs["aux"].get("stochastic_query")
        monte_carlo_uncertainty = monte_carlo_forward_outputs["aux"].get("uncertainty")
    elif self.enable_classification_path:
        logits = self.classification_head(flattened_classification_hidden)
        class_probabilities = torch.softmax(logits, dim=-1)
    outputs = {
        "hidden": hidden,
        "pooled": flattened_classification_hidden,
        "recon": recon,
        "logits": logits,
        "point_scores": point_scores,
        "window_scores": window_scores,
        "aux": {
            "encoder": encoder_outputs["aux"],
            "continuous_branch": continuous_outputs,
            "discrete_branch": discrete_outputs,
            "fusion": fusion_outputs["aux"],
            "active_continuous_memory_bank": active_continuous_memory_bank,
            "active_discrete_codebook": active_discrete_codebook,
            "hidden_reconstruction": hidden_reconstruction,
            "hidden_classification": hidden_classification,
            "alpha": fusion_outputs["alpha"],
            "beta": fusion_outputs["beta"],
            "class_probabilities": class_probabilities,
            "raw_point_scores": raw_point_scores,
            "point_score_calibrated": point_score_calibrated,
            "deterministic_geometry": {
                "hidden_reconstruction": hidden_reconstruction,
                "hidden_classification": hidden_classification,
                "alpha": fusion_outputs["alpha"],
                "beta": fusion_outputs["beta"],
                "fusion": fusion_outputs["aux"],
            },
            "classification_class_names": self._classification_class_names(),
            "memory": self.get_memory_lifecycle_state(),
            "stochastic_query": stochastic_query,
            "uncertainty": monte_carlo_uncertainty,
            "retention": {
                "return_mc_samples": self.return_mc_samples,
                "sample_retention_policy": self.sample_retention_policy,
            },
            "forward_pass_seconds": time.perf_counter() - forward_start_time,
        },
    }
    validate_model_outputs(outputs)
    debug_print_if(
        "THESIS_DEBUG_UQ_TRACE",
        "MODEL",
        "Forward output summary",
        stage_name=stage_name,
        training=bool(self.training),
        stochastic_inference=bool(self.stochastic_inference),
        phase_uses_prototype_path=bool(self._phase_uses_prototype_path()),
        memory_initialized=bool(self.memory_initialized),
        has_uncertainty=outputs["aux"].get("uncertainty") is not None,
        has_stochastic_query=outputs["aux"].get("stochastic_query") is not None,
        stochastic_query_keys=list(
            outputs["aux"].get("stochastic_query", {}).keys()
            if isinstance(outputs["aux"].get("stochastic_query"), dict)
            else []
        ),
    )
    console_print(
        "MODEL",
        "Multitask forward outputs",
        hidden=summarize_tensor(outputs["hidden"]),
        pooled=summarize_tensor(outputs["pooled"]),
        recon=summarize_tensor(outputs["recon"]),
        logits=summarize_tensor(outputs["logits"]),
        point_scores=summarize_tensor(outputs["point_scores"]),
        window_scores=summarize_tensor(outputs["window_scores"]),
        continuous_hidden=summarize_tensor(
            outputs["aux"]["continuous_branch"]["prototype_context"]
        ),
        discrete_hidden=summarize_tensor(
            outputs["aux"]["discrete_branch"]["quantized_hidden"]
        ),
        assignment_probabilities=summarize_tensor(
            outputs["aux"]["discrete_branch"]["assignment_probabilities"]
        ),
        hidden_reconstruction=summarize_tensor(outputs["aux"]["hidden_reconstruction"]),
        hidden_classification=summarize_tensor(outputs["aux"]["hidden_classification"]),
        forward_pass_seconds=outputs["aux"]["forward_pass_seconds"],
    )
    return outputs
