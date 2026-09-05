# Raw-input MSE training addendum

Approved scope: 2026-09-04; implementation: 2026-09-05.
This experiment extends `full-spec-v4.md` to reconstruction-based training
losses. The user approved O0/O1 × three entities × seeds 6, 8, 36, then requested
manual cloud CLI commands. GPU execution belongs to the user.

## Contract and terminology comparison

Compared with `full-spec-v1.md` (normal/anomalous reconstruction supervision),
v2 (input-target MSE), v3 (MC scoring and calibrated point scores), v4 (raw
operational MSE), and `two-stage-offline-pretraining-spec.md` (O0/O1 objectives):

| Earlier object | Current object | Status | Owner and boundary |
|---|---|---|---|
| reconstruction_loss | reconstruction_loss | unchanged name; raw units in this experiment | model loss; `reconstruction_loss_space: raw_input` |
| score_loss | score_loss | unchanged BCE formula; raw MSE supplies its logits | O1 Stage A loss |
| classification_loss / contrastive_loss | same | unchanged | model objectives |
| raw_input_point_mse / raw_input_window_mse | same | unchanged from v4 | scorer and evaluator |
| normalized reconstruction diagnostic | normalized_input_recon_mse_mean | new explicit diagnostic name | model logs |
| train scaler | same | unchanged | dataset and checkpoint scaler_state_dict |

Inverse-transform the post-injection input and each reconstruction sample.
Square their differences, average MC sample errors, then apply the existing
normal-position mask and average selected cells. Preserve the existing full-MSE
fallback when all positions are injected. Constant features retain the scaler's
existing passthrough semantics. O1 uses feature-mean raw errors with detached
normal-token mean/std and balanced BCE. Its internal logits are not anomaly scores.

Training remains deterministic where the existing forward is deterministic;
validation scores use mean(sample MSE). The trainer selects best checkpoints by
raw `val_synth_vus_pr` (max). Fit prediction thresholds only on clean validation.
Persist raw loss identity in config and restore its scaler through checkpoint load.
Use 25+5 epochs, window 20, MC 10, VUS buffer 20 and 200 thresholds.

## Sequential execution record

Each numbered operation is an atomic step; stages run in the listed order.

1. Contract phase — inspection stage (`rg`, Python, Markdown): read active
   sources; compare earlier specs; record the approved scope here.
2. Local phase — loss stage (PyTorch, pytest): add arithmetic tests; observe
   failures; attach train scaler; change reconstruction errors; restore checkpoint context.
3. Local phase — metric stage (PyTorch, pytest): test trainer score collection;
   use raw MC scores; fit clean-validation thresholds; test checkpoint selection.
4. Handoff phase — CLI stage (YAML, Python, Markdown): generate isolated rerun
   configs; verify dry-run commands; write one-cell and remaining-matrix commands.
5. Verification phase — local stage (pytest, CPU): run focused tests; run one
   CPU flow; record limitations. Cloud stage (user CLI): run one full GPU cell;
   inspect metrics and checkpoints; run the other 17 cells only after it passes.
