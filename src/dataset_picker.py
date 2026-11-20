"""Select prepared versions in a terminal."""

from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Label, Select


class DatasetPicker(App[dict[str, Path] | None]):
    CSS = """
    Screen { padding: 1 2; }
    #title { text-style: bold; color: $accent; }
    #choices { height: 1fr; }
    #actions { height: 3; }
    Button { min-width: 8; margin-right: 1; }
    """

    def __init__(self, choices: dict[str, list[tuple[Path, tuple[str, ...]]]]) -> None:
        super().__init__()
        self.choices = choices

    def compose(self) -> ComposeResult:
        yield Label("Select prepared versions", id="title")
        yield Label("Save updates the shared data profiles.", id="note")
        with VerticalScroll(id="choices"):
            for index, (dataset, versions) in enumerate(self.choices.items()):
                yield Label(dataset, markup=False)
                options = [(f"{path.name[:12]}  ({len(entities)} entities)", path) for path, entities in versions]
                yield Select(options, allow_blank=False, value=versions[0][0], id=f"dataset-{index}")
        with Horizontal(id="actions"):
            yield Button("Save", id="save", variant="primary")
            yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#dataset-0", Select).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.exit()
            return
        self.exit({
            dataset: self.query_one(f"#dataset-{index}", Select).value
            for index, dataset in enumerate(self.choices)
        })
