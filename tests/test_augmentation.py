import torch

from src.methods.augmentation import augment


SETTINGS = {"min_segment_fraction": 0.3, "max_segment_fraction": 0.6,
            "spike_scale": 3.0, "anomaly_visibility_boost": 1.6, "family_intensity": {}}


def sample():
    time = torch.linspace(0, 12, 48)
    features = torch.stack((time.sin(), time.cos(), (time * 1.3).sin()), dim=-1)
    return features.repeat(12, 1, 1)


def test_retained_families_labels_masks_and_provenance():
    source = sample()
    altered, record = augment(source, SETTINGS, generator=torch.Generator().manual_seed(92))
    labels = record["classification_labels"]
    masks = record["synthetic_anomaly_mask"]
    assert sorted(labels.tolist()) == list(range(12))
    assert masks.shape == (12, 48)
    assert masks.dtype == torch.bool
    assert not masks[labels == 0].any()
    assert masks[labels != 0].any(dim=1).all()
    assert len(record["augmentation_metadata"]) == 12
    assert len(record["classification_class_names"]) == 12
    assert torch.equal(source, sample())
    assert torch.equal(altered[labels == 0], source[labels == 0])
    assert torch.equal(altered[~masks], source[~masks])


def test_explicit_generator_and_global_rng_isolation():
    source = sample()
    rng = torch.Generator().manual_seed(92)
    saved_state = rng.get_state()
    global_state = torch.random.get_rng_state()
    one, record = augment(source, SETTINGS, generator=rng)
    two, repeated = augment(source, SETTINGS, generator=torch.Generator().set_state(saved_state))
    assert torch.equal(one, two)
    assert torch.equal(record["classification_labels"], repeated["classification_labels"])
    assert torch.equal(global_state, torch.random.get_rng_state())
    assert not torch.equal(saved_state, rng.get_state())
