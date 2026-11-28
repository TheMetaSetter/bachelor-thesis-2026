---
date: 2026-11-28T10:20:00+07:00
researcher: Antigravity
topic: "Overcomplicated design choices in src/methods/mustard and src/methods compared with documents/specs"
status: complete
---

# Research: Overcomplicated Design Choices in `src/methods`

## Summary

`src/methods/mustard/` and `src/methods/` contain five main sources of overcomplexity:

1. **Oversized and duplicated `fit()` loops:** Every neural method (`Mustard`, `CANDI`, `M2N2`, `RedLamp`, `Moment`) and `StumpyChannelAB` repeats a 60–91 line `fit()` method for optimizer setup, epoch iteration, resume handling, and initial/best checkpoint saving. These methods exceed the 50-line function limit in `codebase-specific-preferences.md`.
2. **Oversized `Mustard` entrypoint:** `src/methods/mustard/model.py` is 540 lines, which exceeds the 500-line file limit. `Mustard.__init__` also loads YAML files from disk instead of receiving resolved configuration from `src/config.py`.
3. **Python-loop online verification and contrastive loss:** `signature_verification.py`, `verification_buffer.py`, `contrastive.py`, and `online_update.py` convert PyTorch tensors into nested Python lists, dataclass wrappers, and per-token loops instead of using vectorized tensor operations.
4. **Heavy per-batch metadata in synthetic augmentation:** `augmentation.py` and `anomaly_families.py` build nested dictionaries and integer position lists for every channel of every window, even though `RedLamp` only needs a `[B, L, C]` boolean mask and `Mustard` ignores the metadata.
5. **Cross-file private imports and unused branches:** Methods import private functions across files (`_kmeans`, `_decoder`, `_endpoint_record`, `_fit_endpoint_estimator`, `_window_matrix`), and `SANA` / `Mustard._select_epoch` retain unused or superseded code paths.

## Research question

Which design choices in `src/methods/mustard/` and `src/methods/` are overcomplicated and can be re-implemented to align with `documents/specs/` and project preferences?

## Detailed findings

### 1. Duplicated `fit()` and checkpoint-resume boilerplate

Every neural method implements its own epoch loop, resume restore, best-state tracking, and `save_checkpoint` calls. All of them exceed the 50-line function limit:

| Symbol | File and lines | Line count | Current responsibility |
| --- | --- | ---: | --- |
| `Mustard._fit_stage` | `src/methods/mustard/model.py:442-502` | 61 | Stage resume, epoch loop, selection, checkpoint save |
| `CANDI.fit` | `src/methods/candi.py:288-378` | 91 | Adam setup, resume, cosine LR loop, reference fit, checkpoint save |
| `M2N2.fit` | `src/methods/m2n2.py:182-271` | 90 | AdamW setup, resume, epoch loop, threshold fit, checkpoint save |
| `RedLamp.fit` | `src/methods/redlamp.py:247-337` | 91 | Adam setup, resume, epoch loop, calibration, checkpoint save |
| `Moment.fit` | `src/methods/moment.py:149-239` | 91 | Adam setup, resume, epoch loop, checkpoint save |
| `_fit_endpoint_estimator` | `src/methods/kmeans_ad.py:43-119` | 77 | Shared fit/resume/validation quantile scaling for `KMeansAD` and `IsolationForestAD` |
| `StumpyChannelAB.fit` | `src/methods/stumpy_channel_ab.py:121-189` | 69 | Reference segment copy, validation calibration, checkpoint save |

**Alignment with specs:** `documents/logs/2026-11-16/detail/proposed-runtime-signatures.md` (Section 8) keeps `fit` as a method entrypoint and requires functions to stay within 50 lines. A small shared training/checkpoint helper (or splitting `fit` into focused private helpers like `_restore_or_init_stage` and `_run_epochs`) removes the 5x repetition without introducing an inheritance hierarchy.

### 2. `Mustard` size, config loading, and superseded monitoring path

- **File and method size:** `src/methods/mustard/model.py` is 540 lines (limit: 500). `Mustard.__init__` (`model.py:53-108`) is 56 lines (limit: 50).
- **Config loading inside `__init__`:** When `config["phase"] == "online"` and `source_method` is absent, `Mustard.__init__` (`model.py:64-73`) builds a path to `configs/method_profiles/mustard_<variant>_offline.yaml` and calls `load_yaml(path)`. Under `documents/specs/config-profiles-flow.md` and `proposed-runtime-signatures.md` (Section 5), `src/config.py` (`resolve_run`) owns profile loading and resolution.
- **Dual checkpoint-selection path:** `Mustard._select_epoch` (`model.py:425-440`) supports both `monitor_epoch` (held-out validation raw FPR) and `validation["total_loss"]` fallback. `documents/specs/protocol-validation-rules.md:72-76` states that raw FPR on held-out validation data is the agreed rule and replaces validation total loss.
- **Redundant mode save/restore:** `Mustard.score` (`model.py:215-223`) saves and restores `module.training` for all submodules when `generator is not None`, even though `_score` (`model.py:228`) immediately calls `self.network.eval()` under `@torch.no_grad()`.
- **Projector placement:** `NearIdentityMLPProjector` lives in `src/encoders.py:57-86`, but `Mustard` (`model.py:17,111`) is its only caller. `proposed-runtime-signatures.md` (Section 10) assigns `src/encoders.py` to the common time-preserving CNN (`create_encoder`).

### 3. Over-engineered MUSTARD online verification and contrastive loss

- **Tensor-to-Python-to-tensor round-trips in `signature_verification.py`:**
  - `ordered_continuous_signature` (`signature_verification.py:65-83`) computes cosine distances on tensor, calls `distances.tolist()`, sorts indices in Python loops, and returns `list[list[tuple[int, ...]]]`.
  - `find_recurrent_signatures` (`signature_verification.py:94-115`) normalizes each integer through `_normalize_signature_value` and `_normalize_signature_tuple` (`signature_verification.py:36-62`) and runs an $O(N^2)$ pair check over occurrences.
  - `build_pnn_token_mask` (`signature_verification.py:118-141`) converts the signatures back into a `torch.bool` tensor.
  - `online_update.py:127` converts `signatures` into a tensor again via `torch.as_tensor(signatures, device=x.device)`.
  - Keeping signatures as integer tensors (`torch.topk` on `[B, L, K]`) removes `SignatureWindow`, `_normalize_signature_*`, and all Python list conversions.
- **`VerificationBuffer` state coupling:**
  - `verification_buffer.py:43-48` accepts dual key names (`start_index` or `window_start`, `end_index` or `window_end`).
  - `state.py:25,57-61` serializes and restores `vars(method.verification_buffer)` directly, coupling checkpoints to private fields (`_items`, `_new_since_cycle`).
- **Per-token loop in `contrastive.py`:**
  - `compute_token_multi_positive_info_nce` (`contrastive.py:17-75`, 59 lines) calls `_mean_token_loss` (`contrastive.py:78-110`), which iterates over `anchor_mask.nonzero(as_tuple=False).tolist()` in Python and computes `torch.logsumexp` per token. Vectorized matrix multiplication with positive/negative masks replaces the per-token loop.

### 4. Cross-file private imports and duplicated helpers

- `src/methods/mustard/model.py:24` imports private `_kmeans` from `.retrieval`.
- `src/methods/m2n2.py:14` imports private `_decoder` from `.candi`.
- `src/methods/iforest.py:7` imports private `_endpoint_record`, `_fit_endpoint_estimator`, and `_window_matrix` from `.kmeans_ad`.
- `src/methods/mustard/losses.py:7-13` (`_classification_loss`) and `src/methods/redlamp.py:19-24` (`_corrected_targets`) duplicate the same label-correction cross-entropy target formula.

### 5. `SANA` per-channel `ModuleList` loops in `src/methods/candi_adapter.py`

- `SANA.__init__` (`candi_adapter.py:169-181`) creates `input_dim` separate `_TemporalEmbedding` modules and `input_dim` separate `nn.Linear(d_model, window_size)` layers inside `nn.ModuleList`.
- `SANA.forward` (`candi_adapter.py:199-213`) loops in Python over every channel and stacks the outputs. Grouped `nn.Conv1d(..., groups=input_dim)` and a batched linear/einsum projection express the exact same per-channel weights without Python loops.
- `SANA` also keeps a `sana_type == "Linear"` branch (`candi_adapter.py:148-167,190-198`), which `CANDI` (`candi.py:64-75`) never selects.

### 6. Verbose synthetic augmentation and unused metadata

- `src/methods/anomaly_families.py` (430 lines) repeats the same extract, boost, update, and metadata-attach steps across 11 `inject_*_family` functions.
- `src/methods/augmentation.py:137-148` (`_attach_change_metadata`) converts nonzero mask positions to Python `list[int]` for every affected channel in every training batch. `Mustard` ignores `augmentation_metadata`, and `RedLamp._cell_mask` (`redlamp.py:27-32`) only uses `changed_positions` to rebuild the `[B, L, C]` boolean mask. Returning the `[B, L, C]` boolean mask directly from `augment()` removes `_cell_mask` and per-channel position lists.
- `augmentation.py:195` contains a no-op `self._randint(0, 1, (1,), device=clean_window.device)` call kept only to preserve an older RNG step count.

## Evidence

- `src/methods/mustard/model.py:53-108` — 56-line `__init__` loading YAML from `configs/method_profiles/`.
- `src/methods/mustard/model.py:425-502` — 61-line `_fit_stage` and dual `monitor_epoch` vs `total_loss` selection.
- `src/methods/mustard/signature_verification.py:36-141` — Tensor-to-list signature sorting, type normalization, and $O(N^2)$ recurrence check.
- `src/methods/mustard/verification_buffer.py:18-81` — Dual start/end key names and cycle flags serialized via `vars()`.
- `src/methods/mustard/contrastive.py:17-110` — 59-line `compute_token_multi_positive_info_nce` and per-token Python loop in `_mean_token_loss`.
- `src/methods/candi.py:288-378`, `src/methods/m2n2.py:182-271`, `src/methods/redlamp.py:247-337`, `src/methods/moment.py:149-239` — Duplicated ~90-line `fit()` implementations.
- `src/methods/candi_adapter.py:148-214` — Unused `Linear` SANA branch and per-channel `ModuleList` Python loops.
- `src/methods/augmentation.py:137-244` and `src/methods/redlamp.py:27-32` — `changed_positions` list serialization immediately unpacked back into a boolean tensor mask.
- `documents/specs/protocol-validation-rules.md:72-76` — Confirms raw FPR on held-out validation data replaces validation total loss for MUSTARD monitoring.
- `documents/logs/2026-11-16/detail/proposed-runtime-signatures.md` — Defines the 50-line function limit, 500-line file limit, `config.py` ownership of profile resolution, and `encoders.py` scope.
