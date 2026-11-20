# Thesis runtime tests

Run the suite from the repository root.

```sh
.venv/bin/python -m pytest -q
```

1. `test_cli.py` and `test_dataset_picker.py` check the command and keyboard flow.
2. `test_data.py` and `test_protocols.py` check prepared data and runtime inputs.
3. The method tests check fitting, scoring, updates, and saved state.
4. The experiment and reporting tests check planning, resume, and result tables.
5. The compliance tests check the source size scanner.
