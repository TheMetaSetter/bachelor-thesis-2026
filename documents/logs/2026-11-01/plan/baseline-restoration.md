# Restore scientific baselines

The simplification wrongly removed scientific methods. Restore the original runtime together to avoid mixing incompatible contracts. No method names or semantics change.

- [x] 1. Inspect removed baseline paths using Git.
- [x] 2. Back up the current source, scripts, configs, tests and edited architecture documentation using Python.
- [x] 3. Restore those trees and documentation from the original Git revision using Git archive and Python.
- [x] 4. Verify restored tracked files exactly match that revision using Git.
- [x] 5. Run the original test suite using .venv/bin/python and pytest.
- [x] 6. Record the outcome and withdraw the prior reduction claim.

Verification: 508 passed, 2 skipped in 48.77 seconds. Tracked source/scripts/configs/tests/demo and restored documentation exactly match the original revision.
