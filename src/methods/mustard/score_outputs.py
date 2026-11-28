"""Calculate reconstruction errors and uncertainty for MUSTARD scores."""

import torch

from ...scoring import reconstruction_scores


def point_errors(window, reconstructions, space, scaler):
    settings = {"score_space": space, "scaler": scaler}
    return torch.stack(
        [
            reconstruction_scores(window, sample, settings)
            for sample in reconstructions.unbind(dim=1)
        ],
        dim=1,
    )


def uncertainty(passes, errors, requested, correction):
    if len(passes) <= correction:
        return {}

    def variance(name):
        values = [item[name] for item in passes]
        if name == "logits":
            values = [value.softmax(dim=-1) for value in values]
        return torch.stack(values, dim=1).var(dim=1, correction=correction)

    result = {
        "point_anomaly_score_variance": errors.var(dim=1, correction=correction),
        "window_anomaly_score_variance": errors.mean(dim=-1).var(
            dim=1, correction=correction
        ),
    }
    for name, prefix in (
        ("continuous", "continuous_retrieval"),
        ("discrete", "discrete_retrieval"),
        ("reconstruction", "reconstruction"),
    ):
        full = variance(name)
        if name == "reconstruction":
            result["reconstruction_variance_full"] = full
        result[f"{prefix}_variance_point"] = full.mean(dim=-1)
        result[f"{prefix}_variance_window"] = full.mean(dim=(-1, -2))
    result["classification_probability_variance"] = variance("logits")
    result["classification_variance_mean"] = result[
        "classification_probability_variance"
    ].mean(dim=-1)
    return {name: result[name].cpu().numpy() for name in requested}
