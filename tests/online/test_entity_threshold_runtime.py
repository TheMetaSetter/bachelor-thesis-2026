from src.engine.online_tta.online_engine import _run_online_execution_sequences


def _artifact(entity_id, value):
    return {
        "entity_id": entity_id,
        "thresholds": {
            "online_ewma_point": {"value": value},
            "input_window": {"value": value + 1},
            "latent_window_low": {"value": value + 2},
            "latent_window_high": {"value": value + 3},
        },
    }


def test_execution_selects_a_distinct_artifact_for_each_entity(monkeypatch) -> None:
    seen = []

    def run_sequence(**kwargs):
        seen.append(
            (kwargs["sequence"]["meta"]["entity_id"], kwargs["threshold_value"])
        )
        return [], []

    monkeypatch.setattr(
        "src.engine.online_tta.online_engine._run_online_sequence", run_sequence
    )
    context = {
        "data_bundle": {
            "scaled_sequences": {
                "test": [{"meta": {"entity_id": "m1"}}, {"meta": {"entity_id": "m2"}}]
            }
        },
        "threshold_artifacts": {"m1": _artifact("m1", 1.0), "m2": _artifact("m2", 9.0)},
        "model": object(),
        "optimizer": object(),
        "online_variant": "A2",
        "batch_size": 1,
        "view_noise_std": 0.0,
        "view_dropout_probability": 0.0,
        "device": "cpu",
        "verification_buffer": object(),
        "hard_old_guard": object(),
        "signature_history": [],
        "max_online_steps": 0,
    }
    _run_online_execution_sequences(context=context, protocol_config={})
    assert seen == [("m1", 1.0), ("m2", 9.0)]
