from demo.app import run_live_demo


def test_live_demo_waits_for_window_and_hides_labels() -> None:
    received = []

    def score(payload):
        received.append(payload)
        return sum(payload["window"])

    values = [
        {"x": value, "point_labels": 1, "meta": {"entity_id": "m1"}}
        for value in range(4)
    ]
    outputs = run_live_demo(values=values, window_size=3, score_callback=score)
    assert outputs == [{"end_index": 3, "score": 3}, {"end_index": 4, "score": 6}]
    assert all("point_labels" not in payload for payload in received)
    assert all(set(payload) == {"x", "meta", "window", "end_index"} for payload in received)
