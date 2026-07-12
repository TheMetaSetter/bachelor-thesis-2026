from __future__ import annotations

"""Phase passthrough helpers for the thesis multitask model."""

import torch


class ThesisMultitaskStatePassthroughMixin:
    def _build_phase_passthrough_outputs(
        self,
        hidden: torch.Tensor,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        continuous_outputs = {
            "hidden": hidden,
            "prototype_context": hidden,
            "prototype_logits": None,
            "prototype_weights": None,
            "aux": {
                "branch_name": "continuous",
                "enabled": False,
                "num_prototypes": 0,
                "memory_bypass_active": True,
                "memory_initialized": self.memory_initialized,
            },
        }
        discrete_outputs = {
            "hidden": hidden,
            "quantized_hidden": hidden,
            "assignment_logits": None,
            "assignment_probabilities": None,
            "code_indices": None,
            "aux": {
                "branch_name": "discrete",
                "enabled": False,
                "num_codes": 0,
                "memory_bypass_active": True,
                "memory_initialized": self.memory_initialized,
                "query_mode": "phase_passthrough",
                "topk": 0,
                "query_temperature": 0.0,
            },
        }
        fusion_outputs = {
            "hidden_reconstruction": hidden,
            "hidden_classification": hidden,
            "alpha": hidden.new_zeros(hidden.shape[0]),
            "beta": hidden.new_zeros(hidden.shape[0]),
            "aux": {
                "fusion_mode": "phase_direct_passthrough",
                "alpha": 0.0,
                "beta": 0.0,
                "alpha_std": 0.0,
                "beta_std": 0.0,
                "alpha_logit": float(self.alpha_logit.detach().cpu()),
                "beta_logit": float(self.beta_logit.detach().cpu()),
                "cka_reconstruction_mean": 0.0,
                "cka_reconstruction_std": 0.0,
                "cka_classification_mean": 0.0,
                "cka_classification_std": 0.0,
                "warmup_active": self.schedule_state["warmup_active"],
                "temperature": self.gumbel_temperature,
            },
        }
        return continuous_outputs, discrete_outputs, fusion_outputs
