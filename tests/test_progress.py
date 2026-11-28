"""Check the terminal display without a live terminal."""

import io
import sys

import pytest


class _Terminal(io.StringIO):
    def isatty(self):
        return True


@pytest.mark.parametrize("error", [None, RuntimeError, KeyboardInterrupt])
def test_one_bar_closes_on_every_exit(monkeypatch, capsys, error):
    from src import progress

    terminal = _Terminal()
    monkeypatch.setattr(sys, "stderr", terminal)
    config = {"method_name": "mustard", "phase": "offline"}
    bars = []
    original = progress.tqdm

    def bar(**settings):
        result = original(**settings)
        bars.append(result)
        return result

    monkeypatch.setattr(progress, "tqdm", bar)

    def run():
        with progress.run_progress(config) as display:
            display(
                stage="stage_a",
                unit="epoch",
                completed=1,
                total=2,
                stats={"FPR": None, "loss": 0.0},
            )
            display(stage="evaluate", unit="step", completed=1, total=1, stats={})
            if error:
                raise error

    if error is None:
        run()
    else:
        with pytest.raises(error):
            run()
    assert len(bars) == 1 and bars[0].disable
    assert "FPR=—" in terminal.getvalue()
    assert "loss=0" in terminal.getvalue()
    assert "stage_a" in terminal.getvalue() and "evaluate" in terminal.getvalue()
    assert capsys.readouterr().out == ""


def test_piped_stderr_uses_no_bar(monkeypatch):
    from src import progress

    terminal = io.StringIO()
    monkeypatch.setattr(sys, "stderr", terminal)
    monkeypatch.setattr(progress, "tqdm", lambda **_: pytest.fail("bar on a pipe"))
    with progress.run_progress({}) as display:
        assert display is None
    assert terminal.getvalue() == ""


def test_narrow_terminal_keeps_all_epoch_statistics(monkeypatch):
    from src import progress

    terminal = _Terminal()
    monkeypatch.setattr(sys, "stderr", terminal)
    original = progress.tqdm
    monkeypatch.setattr(
        progress,
        "tqdm",
        lambda **settings: original(
            **(settings | {"dynamic_ncols": False, "ncols": 45})
        ),
    )
    config = {"method_name": "mustard", "phase": "offline"}
    with progress.run_progress(config) as display:
        display(
            stage="stage_a",
            unit="epoch",
            completed=1,
            total=2,
            stats={"train reconstruction": 0.25, "val classification": 0.5},
        )
    assert "train reconstruction=0.25 val classification=0.5" in terminal.getvalue()
