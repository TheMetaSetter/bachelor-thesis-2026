import pytest

from demo.stream_queue import StreamQueueController
from demo.online_replay import consume_online_stream


def test_stream_queue_produces_items_and_stops() -> None:
    stream = StreamQueueController([1, 2, 3])
    stream.start()
    assert list(stream) == [1, 2, 3]
    stream.stop()


def test_stream_queue_pause_resume() -> None:
    stream = StreamQueueController(["a"])
    stream.start()
    stream.pause()
    stream.resume()
    assert stream.get(timeout=1.0) == "a"
    with pytest.raises(StopIteration):
        stream.get(timeout=1.0)
    stream.stop()


def test_online_consumer_waits_for_complete_window() -> None:
    stream = StreamQueueController(maxsize=4, delay_seconds=0.0)
    stream.start([{"x": value} for value in range(4)])
    outputs = consume_online_stream(stream, 3, lambda window: sum(window))
    stream.stop()
    assert outputs == [{"end_index": 3, "score": 3}, {"end_index": 4, "score": 6}]
