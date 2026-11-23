"""Calculate MUSTARD training losses for the selected offline variant."""

import torch
from torch.nn import functional as F


def _classification_loss(logits, labels, correction):
    classes = logits.shape[-1]
    alpha, beta = correction["alpha"], correction["beta"]
    one_hot = F.one_hot(labels.long(), classes).to(logits.dtype)
    targets = one_hot * (1.0 - alpha - beta * classes) + beta
    targets[:, 0] += alpha
    return -(targets * logits.log_softmax(dim=-1)).sum(dim=-1).mean()


def _balanced_score_loss(errors, mask):
    normal, anomalous = errors[~mask], errors[mask]
    if not len(normal) or not len(anomalous):
        return None
    center = normal.mean().detach()
    scale = normal.std(unbiased=False).detach().clamp_min(1e-6)
    return (
        F.softplus((normal - center) / scale).mean()
        + F.softplus((center - anomalous) / scale).mean()
    ) / 2


def _contrastive(clean, augmented, normal, temperature):
    anchor = F.normalize(clean[normal], dim=-1)
    positive = F.normalize(augmented[normal], dim=-1)
    if not len(anchor):
        return clean.new_zeros(())
    labels = torch.arange(len(anchor), device=anchor.device)
    return F.cross_entropy(anchor @ positive.T / temperature, labels)


def compute_loss(
    outputs: dict, targets: dict, settings: dict
) -> dict[str, torch.Tensor]:
    """Return the loss and each term that is defined for this batch."""
    mask = targets["synthetic_anomaly_mask"].bool()
    errors = (outputs["reconstruction"] - targets["x"]).square().mean(dim=-1)
    reconstruction = errors[~mask].mean() if (~mask).any() else errors.mean()
    classification = _classification_loss(
        outputs["logits"],
        targets["classification_labels"],
        settings["label_correction"],
    )
    terms = {
        "reconstruction": reconstruction,
        "classification": classification,
        "classification_branch": classification,
    }
    weights = settings["loss_weights"]
    variant, stage = settings["variant"], settings["stage"]
    if variant == "O1" and stage == "stage_a":
        point_score = _balanced_score_loss(errors, mask)
        if point_score is not None:
            terms["point_score"] = point_score
            terms["classification_branch"] = (classification + point_score) / 2
    total = (
        weights["reconstruction"] * reconstruction
        + weights["classification"] * terms["classification_branch"]
    )
    if variant == "O2" and stage == "stage_a":
        terms["contrastive"] = _contrastive(
            outputs["clean_features"],
            outputs["query"],
            ~mask,
            settings["contrastive_temperature"],
        )
        total = total + weights["contrastive"] * terms["contrastive"]
    terms["total"] = total
    return terms
