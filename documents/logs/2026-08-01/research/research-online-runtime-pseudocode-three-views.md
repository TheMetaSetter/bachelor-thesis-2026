---
date: 2026-08-01 10:00:00 +07:00
researcher: OpenAI Codex
topic: "So sánh pseudocode online runtime flow: bản gốc, terminology theo spec và code hiện tại"
status: complete
revision: 52e518e0b175a1ce6891e27a501322f91c9b0978
branch: dev
---

# Research: So sánh ba pseudocode của online runtime flow

## Summary

Tài liệu này đặt cạnh nhau ba phiên bản:

1. Pseudocode nguyên bản trong tài liệu debug.
2. Pseudocode giữ nguyên ý tưởng nhưng dùng tên theo full-spec-v3.
3. Pseudocode mô tả flow thật đang chạy trong codebase.

Kết luận chính:

- Bản theo spec đặt triage trước verification và chỉ cập nhật online_mlp_projector.
- Code hiện tại tạo signature và PNN sơ bộ trước triage. Sau đó code tính lại PNN cho các entry trong verification buffer.
- Bản gốc cần làm rõ ba điểm trước khi dùng để sửa code: hard-old chỉ dành cho A2; frozen-source forward khác adaptation forward; và pnn_verified phải được truyền rõ sang bước update.

## Câu hỏi nghiên cứu

So sánh pseudocode nguyên bản, pseudocode dùng tên theo full-spec-v3.md và pseudocode theo flow thật của codebase.

## Tài liệu và phạm vi kiểm tra

Pseudocode nguyên bản nằm trong:

documents/notes/online_runtime_flow_debug.md:1-287

Specification chuẩn:

documents/spec/full-spec-v3.md:781-940

Các file runtime được kiểm tra:

- src/engine/online_tta/online_engine_run.py:261-300
- src/engine/online_tta/online_engine_window_core.py:53-252
- src/engine/online_tta/online_engine_window_metrics.py:35-220
- src/engine/online_tta/online_engine_step.py:108-237
- src/engine/online_tta/verification_adapter.py:53-113
- src/engine/online_tta/verification_cycle.py:12-36

## Bảng thống nhất tên gọi

| Ý nghĩa | Tên trong bản gốc/code | Tên dùng để so sánh |
| --- | --- | --- |
| Low-score window | normal | normal |
| Old-normal latent region | hard_old_normality | hard_old_normality |
| Intermediate region | gray_zone | gray_zone |
| High-anomaly region | strong_anomaly | strong_anomaly |
| Buffer object | verification_buffer | verification_buffer / VerificationBuffer |
| Verified PNN decision | decision = pnn_verified | triage_decision = pnn_verified at adaptation boundary |
| Trainable object | online_mlp_projector | online_mlp_projector |
| A2 source-consistency term | SRC-ON loss | L_online_contrastive |
| PNN selection | pnn_mask | pnn_mask |

Spec v3 dùng hard_old trong bảng điều kiện. Code dùng hard_old_normality. Trong tài liệu này, hard_old_normality là tên dùng khi nói về code; hard_old là tên ngắn trong spec.

Code và spec v3 đều dùng strong_anomaly. Tài liệu không tự đổi cụm strong normal thành tên khác vì đây không phải tên đang có trong code hoặc spec.

## 1. Pseudocode nguyên bản của người dùng

Khối dưới đây được đồng bộ theo “Flow mình mong muốn” trong `documents/notes/online_runtime_flow_debug.md`. Khối này chỉ dùng để so sánh, không phải specification chuẩn.

~~~text
BEGIN ONLINE_STEP(W_t)

    RECEIVE W_t

    raw_point_scores
        <- FORWARD W_t THROUGH MODEL
           AND GET SCORE FOR EACH POINT

    IF previous_ewma_point_scores DOES NOT EXIST THEN
        previous_ewma_point_scores
            <- ARRAY OF ZEROS WITH LENGTH WINDOW_SIZE
    ENDIF

    IF online_time_step <= WINDOW_SIZE THEN
        PAD raw_point_scores WITH ZERO VALUES AT THE END
        PAD previous_ewma_point_scores WITH ZERO VALUES AT THE END
    ENDIF

    current_ewma_point_scores
        <- prev_weight * previous_ewma_point_scores
           + current_weight * raw_point_scores

    point_level_binary_predictions
        <- current_ewma_point_scores > B_point_high

    SAVE point_level_binary_predictions

    IF DEMO_UI_IS_ENABLED THEN
        SHOW point_level_binary_predictions
    ENDIF

    IF online_variant = "A0" THEN
        CONTINUE
    ENDIF

    input_window_score
        <- CALCULATE INPUT-SPACE WINDOW SCORE

    latent_window_score
        <- CALCULATE LATENT-SPACE WINDOW SCORE

    triage_decision
        <- CLASSIFY WINDOW USING
           input_window_score
           AND latent_window_score

    IF triage_decision = "normal" THEN
        DO NOT UPDATE online_mlp_projector

    ELSE IF triage_decision = "hard_old_normality" THEN
        COMPUTE HARD-OLD LOSS

        IF online_variant = "A2" THEN
            COMPUTE SRC-ON LOSS

        COMPUTE TOTAL HARD-OLD LOSS
        BACKPROPAGATE
        UPDATE online_mlp_projector ONLY

    ELSE IF triage_decision = "gray_zone" THEN
        verification_buffer.try_admit(W_t)

    ELSE IF triage_decision = "strong_anomaly" THEN
        DO NOT UPDATE online_mlp_projector
    ENDIF

    IF verification_buffer IS READY THEN
        entries <- verification_buffer.items()
        RUN FROZEN-SOURCE FORWARD ON entries

        CLASSIFY POINTS INTO:
            P_known_anomaly
            P_pseudo_new_normality
            P_other

        pnn_mask <- TRUE ONLY FOR P_pseudo_new_normality

        IF pnn_mask HAS AT LEAST ONE TRUE VALUE THEN
            COMPUTE PNN LOSS USING pnn_mask

            IF online_variant = "A2" THEN
                COMPUTE SRC-ON PNN LOSS

            COMPUTE TOTAL PNN LOSS
            BACKPROPAGATE
            UPDATE online_mlp_projector ONLY
            decision <- "pnn_verified"
        ENDIF
    ENDIF

    previous_ewma_point_scores <- current_ewma_point_scores

END ONLINE_STEP
~~~

### Nhận xét về pseudocode nguyên bản

- Bản này thể hiện đúng thứ tự lớn mà anh mong muốn: triage, update hard-old hoặc admission gray-zone, rồi verification và PNN adaptation.
- Nhánh hard-old chưa ghi rõ điều kiện online_variant == A2.
- SRC-ON loss không phải tên chuẩn trong v3.
- Frozen-source forward và adaptation forward đang viết thành một bước, dù hai bước này có vai trò khác nhau.
- Pseudocode gán decision = pnn_verified nhưng chưa nói rõ biến này được truyền sang bước update thế nào.
- Spec v3 không quy định chi tiết việc padding score bằng zero hoặc gộp B_point_high với các threshold khác.

## 2. Pseudocode dùng tên theo full-spec-v3

Bản này giữ ý tưởng người dùng muốn nhưng dùng tên và điều kiện của v3. Đây là workflow mục tiêu, không phải mô tả code hiện tại.

~~~text
BEGIN ONLINE_STEP(W_c, online_variant)

    RECEIVE causal window W_c

    source_and_query_outputs
        <- SCORE W_c USING THE ACTIVE ONLINE VARIANT
           A0: Z_query = Z_source
           A1/A2: Z_query = online_mlp_projector(Z_source)
           source encoder runs exactly once
           stochastic retrieval uses the configured inference count

    raw_point_score
        <- OFFICIAL MONTE CARLO MEAN POINT SCORE

    input_window_score
        <- INPUT-SPACE WINDOW RECONSTRUCTION SCORE

    latent_window_score
        <- DETERMINISTIC LATENT-MEMORY SCORE

    ewma_point_score
        <- AGGREGATE OVERLAPPING WINDOWS BY THE CONFIGURED EWMA RULE

    IF online_variant = "A0" THEN
        FINALIZE INFERENCE-ONLY OUTPUTS
        DO NOT CALL online_mlp_projector
        DO NOT CREATE AN OPTIMIZER
        END ONLINE_STEP
    ENDIF

    triage_decision
        <- CLASSIFY USING input_window_score AND latent_window_score

    IF triage_decision = "normal" THEN
        FINALIZE EVENT WITHOUT UPDATE OR BUFFER ADMISSION

    ELSE IF triage_decision = "hard_old_normality" THEN
        IF online_variant = "A2"
           AND hard_old_guard ACCEPTS W_c.interval THEN
            RUN ADAPTATION FOR THIS WINDOW
            COMPUTE L_hard
            ADD lambda_contrastive * L_online_contrastive
            BACKPROPAGATE
            UPDATE online_mlp_projector ONLY
            COMMIT hard_old_guard AFTER SUCCESS
        ELSE
            FINALIZE EVENT WITHOUT UPDATE
        ENDIF

    ELSE IF triage_decision = "gray_zone" THEN
        verification_buffer.try_admit(
            entry(
                entity_id,
                start_index,
                end_index,
                x,
                admitted_at_cursor
            )
        )

    ELSE IF triage_decision = "strong_anomaly" THEN
        FINALIZE EVENT WITHOUT UPDATE OR BUFFER ADMISSION
    ENDIF

    IF online_variant IN {"A1", "A2"}
       AND verification_buffer MEETS THE VERIFICATION TRIGGER THEN

        entries <- verification_buffer.items()

        verification_outputs
            <- RUN INDEPENDENT FROZEN-SOURCE FORWARD ON entries
               LABELS ARE ABSENT

        point_geometry
            <- COMPUTE hidden
               nearest_codeword_ids
               nearest_codeword_distances
               continuous_signature_ids

        P_known_anomaly
            <- POINTS INSIDE ANOMALOUS CODEWORD RADII

        P_pseudo_new_normality
            <- REMAINING POINTS WHOSE SIGNATURES OCCUR
               IN MORE THAN ONE NON-OVERLAPPING ADMITTED WINDOW

        P_other
            <- ALL REMAINING POINTS

        pnn_mask
            <- TRUE ONLY FOR P_pseudo_new_normality

        IF pnn_mask HAS AT LEAST ONE TRUE VALUE THEN

            adaptation_outputs
                <- RUN ONLINE MODEL FORWARD ON entries
                   USING reference_hidden
                   AND online_mlp_projector(reference_hidden)

            IF online_variant = "A1" THEN
                L_total <- L_pnn
            ELSE IF online_variant = "A2" THEN
                L_total
                    <- L_pnn
                       + lambda_contrastive * L_online_contrastive
            ENDIF

            COMPUTE L_total USING pnn_mask FOR PNN RECONSTRUCTION
            BACKPROPAGATE
            ASSERT ONLY online_mlp_projector IS TRAINABLE
            UPDATE online_mlp_projector ONLY
            COMMIT BUFFER STATE AFTER SUCCESS
            verified_decision <- "pnn_verified"
        ENDIF
    ENDIF

    FINALIZE FUTURE-ONLY EVENT RECORD
    SAVE RUNTIME STATE

END ONLINE_STEP
~~~

### Căn cứ của pseudocode theo spec

- Source-once and A0/A1/A2 query behavior: full-spec-v3.md:781-797.
- Point score and EWMA aggregation: full-spec-v3.md:799-807.
- Four-region triage and event order: full-spec-v3.md:809-828.
- Gray-zone admission and verification trigger: full-spec-v3.md:832-859.
- Frozen-source verification geometry: full-spec-v3.md:861-874.
- Projector-only update and A1/A2 losses: full-spec-v3.md:882-940.

## 3. Pseudocode theo code hiện tại

Bản này mô tả đúng thứ tự các hàm đang được gọi trong repository.

~~~text
BEGIN RUN_ONLINE_SEQUENCE

    FOR batch IN batcher:

        CALL PROCESS_ONLINE_WINDOW(batch)

            timing_logger.measure("prepare_event"):

                batch_on_device
                    <- MOVE BATCH TENSORS TO DEVICE

                scoring_outputs
                    <- FORWARD ONLINE MODEL
                       A0 uses forward_source when available
                       A1/A2 uses forward with online_mlp_projector

                raw_point_score
                    <- SCORE OF LAST POINT ONLY

                input_window_score
                    <- MEAN((recon - x)^2)

                latent_window_score
                    <- aux.latent_window_score
                       OR mean(window_scores)

                ewma_point_score
                    <- CURRENT/previous SCALAR EWMA

                IF online_variant != "A0" THEN

                    reference_hidden
                        <- scoring_outputs.aux.reference_hidden

                    known_anomaly_mask
                        <- FILTER KNOWN ANOMALY TOKENS
                           USING FROZEN CODEBOOK AND RADII

                    continuous_signature_ids
                        <- ORDERED TOP-3 CONTINUOUS SIGNATURE
                           FOR EACH TOKEN

                    current_signature_window
                        <- CREATE SignatureWindow FROM ENTITY/INTERVAL/SIGNATURES

                    recurrent_signatures
                        <- find_recurrent_signatures(
                               signature_history + current_signature_window
                           )

                    APPEND current_signature_window TO signature_history

                    pnn_mask
                        <- build_pnn_token_mask(
                               continuous_signature_ids,
                               recurrent_signatures,
                               known_anomaly_mask
                           )

                    ATTACH pnn_mask TO batch_on_device
                ENDIF

                triage_decision
                    <- classify_online_window(
                           input_window_score,
                           latent_window_score,
                           triage_thresholds
                       )

            timing_logger.measure("buffer_and_verification"):

                IF triage_decision == "gray_zone" THEN
                    verification_buffer.try_admit(entry)
                ENDIF

                verification_controller.maybe_run(
                    verify_and_adapt_entries
                )

                    verify_buffer_entries(entries):

                        FOR entry IN entries:
                            RUN frozen-source forward
                            COMPUTE known anomaly filtering
                            COMPUTE ordered continuous signatures
                            STORE scored entry
                        ENDFOR

                        recurrent_signatures
                            <- FIND RECURRENCE ACROSS BUFFER ENTRIES

                        FOR scored entry:
                            pnn_mask
                                <- BUILD FROM SIGNATURES AND
                                   known_anomaly_mask

                            IF pnn_mask.sum() > 0
                               AND online_variant != "A0" THEN
                                BUILD ENTRY BATCH
                                ATTACH pnn_mask
                                ATTACH signature IDs
                                ATTACH known_anomaly_mask
                                CALL execute_online_tta_step(
                                    triage_decision = "pnn_verified"
                                )
                            ENDIF
                        ENDFOR

            timing_logger.measure("adaptation_step"):

                IF online_variant != "A0" THEN
                    CREATE online optimizer
                ENDIF

                CALL execute_online_tta_step(
                    triage_decision = event["triage_decision"]
                )

                    IF optimizer IS NONE OR online_variant == "A0":
                        RETURN no update

                    IF triage_decision == "strong_anomaly":
                        RETURN no update

                    RUN model.forward(batch)

                    IF online_variant == "A1":
                        IF triage_decision != "pnn_verified":
                            RETURN no update
                        COMPUTE masked PNN reconstruction loss

                    ELSE IF online_variant == "A2":
                        IF triage_decision == "pnn_verified":
                            COMPUTE masked PNN reconstruction loss
                        ELSE IF triage_decision == "hard_old_normality":
                            COMPUTE hard-old hinge loss
                        ELSE:
                            RETURN no update
                        ENDIF

                        FOR hard_old_normality OR pnn_verified:
                            ADD source-consistency contrastive loss
                        ENDFOR
                    ENDIF

                    BACKPROPAGATE
                    CLIP PROJECTOR GRADIENTS
                    OPTIMIZER.STEP()
                    RETURN update result

            BUILD RECORD AND METRICS
            SYNC signature_history, buffer and guard TO runtime_state
            SAVE metric and record

        END PROCESS_ONLINE_WINDOW

    ENDFOR

END RUN_ONLINE_SEQUENCE
~~~

### Căn cứ từ runtime code

- Per-window loop: src/engine/online_tta/online_engine_run.py:261-300.
- prepare_event order: src/engine/online_tta/online_engine_window_core.py:141-194.
- PNN sơ bộ: src/engine/online_tta/online_engine_window_metrics.py:147-191.
- Gray-zone admission: src/engine/online_tta/online_engine_window_metrics.py:194-220.
- Tính lại PNN và truyền pnn_verified: src/engine/online_tta/verification_adapter.py:82-113 và src/engine/online_tta/online_engine_window_metrics.py:35-79.
- Điều kiện update A1/A2: src/engine/online_tta/online_engine_step.py:108-178.
- Điều kiện chạy verification: src/engine/online_tta/verification_cycle.py:12-36.

## So sánh trực tiếp

| Chủ đề | Pseudocode nguyên bản | Pseudocode theo v3 | Code hiện tại |
| --- | --- | --- | --- |
| Triage order | Before desired PNN verification | Before permitted update/admission/verification | After preliminary PNN/signature computation |
| A0 | Computes point output, then continues | Inference only; no projector/optimizer | A0 still passes through current event pipeline, but skips preliminary PNN and update |
| Hard-old update | Guarded, but variant gate is incomplete | A2 only | A2 only |
| Gray-zone admission | Yes | Only gray-zone windows | Yes |
| Verification trigger | Abstract READY condition | At least eight entries, new admission, no active cycle | Controller checks capacity and should_verify |
| PNN point classification | Known anomaly, PNN, other | Frozen-source geometry then recurrence | Recomputed for buffered entries; also preliminary path for every A1/A2 window |
| PNN mask | True only for PNN | [N,20] PNN mask | [B,L] mask, plus known_anomaly_mask |
| PNN adaptation forward | Written as frozen-source forward | Verification forward then online projector forward | Verification uses frozen source; adaptation uses model.forward |
| A1 | Not explicitly gated for hard-old | PNN verified only | pnn_verified only |
| A2 | PNN and hard-old paths | PNN or hard-old plus contrastive term | PNN or hard-old plus contrastive term |
| Projector ownership | Projector only | Projector only | Optimizer allowlist enforces projector only |
| Decision handoff | Local decision variable | Explicit verified decision | Verification adapter passes pnn_verified directly |

## Điểm lệch và điểm chưa rõ

1. Nhánh hard-old trong pseudocode nguyên bản cần điều kiện A2 rõ ràng.
2. Nhánh PNN verification cần điều kiện A1/A2 rõ ràng, dù A0 đã thoát sớm.
3. Frozen-source verification forward và online adaptation forward phải là hai bước riêng.
4. B_point_high, online_point_threshold_ewma và B_window có vai trò khác nhau trong v3. Không nên gộp chúng chỉ vì một config đang cho cùng một giá trị.
5. P_known_anomaly, P_pseudo_new_normality và P_other là các nhóm point để mô tả. Code hiện tại có known_anomaly_mask và pnn_mask; P_other chưa phải object runtime riêng.
6. Tên field trong v3 là start_index, end_index và x. Code hiện tại dùng window_start, window_end và window. Cần ghi rõ mapping trước khi sửa code.
7. Runtime hiện tại có hai PNN path: PNN sơ bộ trước triage và PNN verification sau khi buffer nhận window.

## Câu hỏi còn mở

- strong_normal trong trao đổi ban đầu có phải là strong_anomaly hiện tại không, hay là một trạng thái mới?
- Thiết kế cuối cùng có bỏ PNN sơ bộ và chỉ giữ PNN trong verification không?
- A2 có tiếp tục dùng latent của known anomaly làm negative không, khi PNN points là các anchor duy nhất cho adaptation?
- Code sẽ đổi field của buffer sang tên v3, hay sẽ ghi rõ mapping ở boundary?

Tài liệu này chỉ ghi nhận và so sánh. Em không sửa source code, test, config hoặc specification.
