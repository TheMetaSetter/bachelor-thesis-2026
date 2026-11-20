"""Check version selection with only a keyboard."""

import asyncio
from pathlib import Path

import pytest


@pytest.mark.parametrize("size", [(80, 24), (40, 12)])
def test_select_and_save(size):
    from src.dataset_picker import DatasetPicker

    async def check():
        first = Path("/data/one")
        second = Path("/data/two")
        app = DatasetPicker({"smd": [(first, ("machine",)), (second, ("machine",))]})
        async with app.run_test(size=size) as pilot:
            await pilot.press("enter", "down", "esc")
            assert app.query_one("#dataset-0").value == first
            await pilot.press("enter", "down", "enter", "tab", "enter")
        assert app.return_value == {"smd": second}

    asyncio.run(check())


def test_cancel():
    from src.dataset_picker import DatasetPicker

    async def check():
        app = DatasetPicker({"smd": [(Path("/data/one"), ("machine",))]})
        async with app.run_test() as pilot:
            await pilot.press("tab", "tab", "enter")
        assert app.return_value is None

    asyncio.run(check())
