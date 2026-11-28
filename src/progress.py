"""Show one terminal progress bar for the current run."""

from contextlib import contextmanager
from numbers import Real
import sys

from tqdm import tqdm


def _format_value(value):
    if value is None:
        return "—"
    return f"{value:.3g}" if isinstance(value, Real) else str(value)


@contextmanager
def run_progress(config: dict):
    """Give a direct CLI run a callback when stderr is a terminal."""
    if not sys.stderr.isatty():
        yield None
        return
    label = f"{config['method_name']} {config['phase']}"
    with tqdm(
        desc=label,
        file=sys.stderr,
        dynamic_ncols=True,
        bar_format="{desc} {n_fmt}/{total_fmt} [{bar:8}]{postfix}",
    ) as bar:
        current = None

        def display(*, stage, unit, completed, total, stats):
            nonlocal current
            key = (stage, unit, total)
            if current != key:
                bar.set_description_str(f"{label} {stage} ({unit})", refresh=False)
                bar.set_postfix_str("", refresh=False)
                bar.total = total
                bar.reset(total=total)
                current = key
            if stats is not None:
                postfix = " ".join(
                    f"{name}={_format_value(value)}" for name, value in stats.items()
                )
                prefix_width = (
                    len(bar.desc) + len(str(completed)) + len(str(total)) + 14
                )
                if postfix and bar.ncols and prefix_width + len(postfix) > bar.ncols:
                    bar.write(
                        f"{bar.desc} {completed}/{total} {postfix}", file=sys.stderr
                    )
                    postfix = ""
                bar.set_postfix_str(postfix, refresh=False)
            refreshed = bar.update(completed - bar.n)
            if stats and not refreshed:
                bar.refresh()

        yield display


_METRIC_LABELS = {
    "segment_mean_vus_pr": "VUS-PR",
    "segment_mean_affiliation_f1": "Aff-F1",
    "fpr": "FPR",
    "segment_mean_vus_roc": "VUS-ROC",
    "recall": "recall",
}
_VARIANCE = "point_anomaly_score_variance_mean"


def _detection_stats(metrics):
    return {
        label: metrics[name]
        for name, label in _METRIC_LABELS.items()
        if name in metrics
    }


def epoch_stats(record):
    """Read losses and metrics from a saved fitter record."""
    stats = {}
    if monitor := record.get("monitor"):
        stats.update(space=monitor["score_space"], FPR=monitor["held_out_fpr"])
        stats.update(
            {
                f"syn {name}": value
                for name, value in _detection_stats(
                    monitor["synthetic_metrics"]
                ).items()
            }
        )
    for name, label in (("train", "train"), ("validation", "val")):
        split = record.get(name, {})
        total = split.get("total_loss", record.get(f"{name}_loss"))
        if total is not None:
            stats[f"{label} loss"] = total
        stats.update(
            {
                f"{label} {term}": value
                for term, value in split.get("loss_terms", {}).items()
            }
        )
    return stats


def evaluation_stats(test, validation=None):
    """Read final statistics from committed evaluation records."""
    metrics = test.get("metrics", {})
    stats = _detection_stats(metrics)
    if metrics.get("total_loss") is not None:
        stats["test loss"] = metrics["total_loss"]
    if validation is not None:
        stats["val var"] = validation.get("metrics", {}).get(_VARIANCE)
        stats["syn var"] = validation.get("synthetic_validation_uncertainty", {}).get(
            "mean"
        )
    stats["test var"] = metrics.get(_VARIANCE)
    return stats
