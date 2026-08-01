Dưới đây là hai pseudocode song song: flow mình mong muốn và flow code hiện tại.

Trong phạm vi `online phase`, dùng mapping:

```text
B_point_high
= T_point_EWMA
= online_ewma_point_threshold
= threshold_value
```

<table>
<tr>
<td valign="top" width="50%">

### 1. Flow mình mong muốn

```text
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

        IF hard_old_guard ACCEPTS W_t THEN
            COMPUTE HARD-OLD LOSS

            IF online_variant = "A2" THEN
                COMPUTE SRC-ON LOSS
            
            COMPUTE TOTAL HARD-OLD LOSS
            BACKPROPAGATE
            UPDATE online_mlp_projector ONLY
        ENDIF

    ELSE IF triage_decision = "gray_zone" THEN

        verification_buffer.try_admit(W_t)

    ELSE IF triage_decision = "strong_anomaly" THEN

        DO NOT UPDATE online_mlp_projector

    ENDIF

    IF verification_buffer IS READY THEN

        entries
            <- verification_buffer.items()

        RUN FROZEN-SOURCE FORWARD ON entries

        CLASSIFY POINTS INTO:
            P_known_anomaly
            P_pseudo_new_normality
            P_other

        pnn_mask
            <- TRUE ONLY FOR P_pseudo_new_normality

        IF pnn_mask HAS AT LEAST ONE TRUE VALUE THEN

            COMPUTE PNN LOSS USING pnn_mask

            IF online_variant = "A2" THEN:
                COMPUTE SRC-ON PNN LOSS
            
            COMPUTE TOTAL PNN LOSS
            BACKPROPAGATE
            UPDATE online_mlp_projector ONLY
            decision <- "pnn_verified"
        ENDIF

    ENDIF

    previous_ewma_point_scores
        <- current_ewma_point_scores

END ONLINE_STEP
```

</td>
<td valign="top" width="50%">

### 2. Flow code hiện tại

```text
BEGIN PROCESS_ONLINE_WINDOW(batch)

    RECEIVE batch

    CALL prepare_event

        batch_on_device
            <- MOVE batch TENSORS TO DEVICE

        model_outputs
            <- FORWARD MODEL

        raw_point_score
            <- TAKE SCORE OF LAST POINT ONLY

        input_window_score
            <- MEAN SQUARED ERROR BETWEEN recon AND x

        latent_window_score
            <- GET latent_window_score
               OR FALL BACK TO window_scores

        IF previous_ewma_score DOES NOT EXIST THEN
            ewma_point_score <- raw_point_score
        ELSE
            ewma_point_score
                <- ewma_current_weight * raw_point_score
                   + ewma_previous_weight * previous_ewma_score
        ENDIF

        IF online_variant != "A0" THEN

            known_anomaly_mask
                <- FILTER KNOWN ANOMALY TOKENS

            continuous_signature_ids
                <- COMPUTE TOP-3 SIGNATURE
                   FOR EACH TOKEN

            recurrent_signatures
                <- FIND RECURRENT SIGNATURES

            APPEND CURRENT WINDOW TO signature_history

            pnn_mask
                <- BUILD PNN MASK

            ATTACH pnn_mask TO batch

        ENDIF

        triage_decision
            <- CLASSIFY WINDOW

    CALL buffer_and_verification

        IF triage_decision = "gray_zone" THEN
            admitted
                <- verification_buffer.try_admit(entry)
        ENDIF

        IF verification_buffer IS READY THEN

            entries
                <- verification_buffer.items()

            RECOMPUTE frozen-source HIDDEN STATES

            RECOMPUTE known anomaly filtering
            RECOMPUTE continuous signatures
            FIND recurrent signatures
            RECOMPUTE pnn_mask

            IF pnn_mask HAS AT LEAST ONE TRUE VALUE THEN
                decision <- "pnn_verified"
                RUN PNN ADAPTATION
            ENDIF

        ENDIF

    CALL adaptation_step

        CREATE online optimizer FOR A1/A2

        IF online_variant = "A0" THEN
            DO NOT UPDATE

        ELSE IF online_variant = "A1" THEN

            IF triage_decision = "pnn_verified" THEN
                COMPUTE MASKED PNN RECONSTRUCTION LOSS
                UPDATE online_mlp_projector
            ELSE
                DO NOT UPDATE
            ENDIF

        ELSE IF online_variant = "A2" THEN

            IF triage_decision = "pnn_verified" THEN
                COMPUTE PNN RECONSTRUCTION LOSS
                ADD CONTRASTIVE LOSS
                UPDATE online_mlp_projector

            ELSE IF triage_decision = "hard_old_normality" THEN
                COMPUTE HARD-OLD LOSS
                ADD CONTRASTIVE LOSS
                UPDATE online_mlp_projector

            ELSE
                DO NOT UPDATE
            ENDIF

        ENDIF

    BUILD RECORD AND METRICS

        SAVE ewma_point_score
        SAVE prediction
        SAVE triage_decision
        SAVE buffer metrics
        SAVE adaptation metrics

    previous_ewma_score
        <- ewma_point_score

END PROCESS_ONLINE_WINDOW
```

</td>
</tr>
</table>

## 3. Khác biệt chính

| Bước | Flow mong muốn | Code hiện tại |
|---|---|---|
| Point score | Vector `raw_point_scores` cho toàn bộ point | Chỉ lấy `raw_point_score` của point cuối |
| EWMA state | Vector `previous_ewma_point_scores` | Scalar `previous_ewma_score` |
| Point prediction | Tạo trước triage | Có prediction scalar, nhưng không có vector prediction |
| Threshold | `B_point_high` cho vector EWMA | `threshold_value` cho scalar EWMA |
| UI | Có thể hiển thị prediction mới ngay | Không nằm trong online engine flow hiện tại |
| Triage | Sau point prediction | Sau preliminary PNN computation |
| PNN | Chỉ sau gray-zone admission và verification trigger | Tạo preliminary PNN trước triage, sau đó verification tạo lại |
| `signature_history` | Nên thuộc các window được chọn theo protocol | Hiện được cập nhật trong preliminary PNN path |
| Gray-zone | Chỉ admission, chưa adaptation | Chỉ admission, chưa adaptation |
| Hard-old | Adapt ngay nếu guard cho phép | A2 có nhánh hard-old adaptation |
| PNN adaptation | Chỉ trên các point có `pnn_mask=True` | Verification path dùng `pnn_mask` để adaptation |

Các điểm code chính:

- Scoring hiện tại: [`online_engine_window_metrics.py#L82-L119`](</Users/conquerormikrokosmos/Downloads/LAP MAC/MYUNIVERSITY/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/online_tta/online_engine_window_metrics.py#L82-L119>)
- Preliminary PNN trước triage: [`online_engine_window_core.py#L155-L184`](</Users/conquerormikosmos/Downloads/LAP MAC/MYUNIVERSITY/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/online_tta/online_engine_window_core.py#L155-L184>)
- Gray-zone admission: [`online_engine_window_metrics.py#L194-L220`](</Users/conquerormikosmos/Downloads/LAP MAC/MYUNIVERSITY/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/online_tta/online_engine_window_metrics.py#L194-L220>)
- A1/A2 update branches: [`online_engine_step.py#L118-L170`](</Users/conquerormikosmos/Downloads/LAP MAC/MYUNIVERSITY/ĐH KHOA HỌC TỰ NHIÊN/Khoá luận tốt nghiệp/bachelor-thesis-2026/src/engine/online_tta/online_engine_step.py#L118-L170>)
