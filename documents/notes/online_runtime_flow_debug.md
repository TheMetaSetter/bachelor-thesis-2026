# THESIS Online TTA: Desired Flow and Current Runtime

Hai pseudocode dưới đây cố ý mô tả hai contracts khác nhau:

- “Flow người dùng mong muốn” dùng vector point scores cho cả causal window;
- “Flow code hiện tại” dùng scalar score của endpoint point.

Tên object tuân theo [`online_tta_terminology_ontology.md`](../spec/online_tta_terminology_ontology.md), và online ontology kế thừa [`offline_pretraining_terminology_ontology.md`](../spec/offline_pretraining_terminology_ontology.md).

Các mathematical aliases chỉ dùng khi đọc tài liệu cũ:

```text
online_point_ewma_threshold = B_point_high = T_point_EWMA
input_window_threshold = B_window
latent_window_low_threshold = A_low
latent_window_high_threshold = A_high
online_contrastive_loss = L_online_contrastive = SRC-ON loss
```

<table>
<tr>
<td valign="top" width="50%">

## 1. Flow người dùng mong muốn

```text
PROCEDURE RUN_ONLINE_TTA_PHASE(
    stage_b_best_checkpoint,
    threshold_artifact,
    online_variant,
    causal_stream
)
    frozen_source_model
        <- LOAD_AND_FREEZE_SOURCE_MODEL(stage_b_best_checkpoint)

    IF online_variant = A0 THEN
        online_mlp_projector <- NOT_CREATED
    ELSE
        online_mlp_projector <- INITIALIZE_NEAR_IDENTITY_PROJECTOR()
    ENDIF

    online_point_ewma_threshold
        <- threshold_artifact.online_point_ewma_threshold

    input_window_threshold
        <- threshold_artifact.input_window_threshold

    latent_window_low_threshold
        <- threshold_artifact.latent_window_low_threshold

    latent_window_high_threshold
        <- threshold_artifact.latent_window_high_threshold

    verification_buffer <- CREATE_VERIFICATION_BUFFER()
    hard_old_interval_guard <- CREATE_NON_OVERLAP_GUARD()
    recurrent_signature_set <- EMPTY_SET
    previous_window_ewma_point_scores <- NOT_CREATED

    FOR EACH causal_window IN causal_stream
        RECEIVE causal_window

        source_hidden
            <- frozen_source_model.shared_encoder(causal_window.x)

        IF online_variant = A0 THEN
            query_hidden <- source_hidden
        ELSE
            projected_hidden
                <- online_mlp_projector(source_hidden)
            query_hidden <- projected_hidden
        ENDIF

        online_model_outputs
            <- FORWARD query_hidden THROUGH FROZEN MEMORIES AND HEADS

        window_point_scores
            <- online_model_outputs.window_point_scores

        IF previous_window_ewma_point_scores DOES NOT EXIST THEN
            previous_window_ewma_point_scores
                <- ARRAY OF ZEROS WITH LENGTH window_size
        ENDIF

        aligned_window_point_scores
            <- ALIGN window_point_scores TO causal_window.absolute_indices
               WITH ZERO PADDING FOR UNAVAILABLE POSITIONS

        aligned_previous_window_ewma_point_scores
            <- ALIGN previous_window_ewma_point_scores
               TO causal_window.absolute_indices
               WITH ZERO PADDING FOR UNAVAILABLE POSITIONS

        current_window_ewma_point_scores
            <- ewma_previous_weight
               * aligned_previous_window_ewma_point_scores
               + ewma_current_weight
               * aligned_window_point_scores

        window_point_predictions
            <- current_window_ewma_point_scores
               > online_point_ewma_threshold

        SAVE window_point_predictions

        IF demo_ui_is_enabled THEN
            SHOW window_point_predictions
        ENDIF

        IF online_variant = A0 THEN
            previous_window_ewma_point_scores
                <- current_window_ewma_point_scores
            SAVE online_runtime_state
            CONTINUE TO NEXT causal_window
        ENDIF

        input_window_score
            <- COMPUTE_INPUT_WINDOW_SCORE(
                   online_model_outputs.reconstruction,
                   causal_window.x
               )

        latent_window_score
            <- COMPUTE_LATENT_WINDOW_SCORE(
                   source_hidden,
                   frozen_source_model
               )

        triage_region
            <- CLASSIFY_TRIAGE_REGION(
                   input_window_score,
                   latent_window_score,
                   input_window_threshold,
                   latent_window_low_threshold,
                   latent_window_high_threshold
               )

        IF triage_region = normal THEN
            DO NOT UPDATE online_mlp_projector

        ELSE IF triage_region = hard_old_normality THEN
            IF online_variant = A2
               AND hard_old_interval_guard ACCEPTS causal_window.interval THEN

                hard_old_reconstruction_loss
                    <- COMPUTE_HARD_OLD_RECONSTRUCTION_LOSS(
                           online_model_outputs.window_anomaly_scores,
                           input_window_threshold
                       )

                online_contrastive_loss
                    <- COMPUTE_ONLINE_CONTRASTIVE_LOSS(
                           source_hidden,
                           projected_hidden,
                           frozen_source_model.anomaly_verification_metadata
                       )

                online_total_loss
                    <- hard_old_reconstruction_loss
                       + lambda_online_contrastive
                       * online_contrastive_loss

                CALL UPDATE_ONLINE_MLP_PROJECTOR(
                    online_mlp_projector,
                    online_total_loss
                )

                ADD causal_window.interval TO hard_old_interval_guard
            ENDIF

        ELSE IF triage_region = gray_zone THEN
            verification_entry
                <- BUILD_VERIFICATION_ENTRY(causal_window)

            CALL verification_buffer.TRY_ADMIT(verification_entry)

        ELSE IF triage_region = strong_anomaly THEN
            DO NOT UPDATE online_mlp_projector
        ENDIF

        IF verification_buffer IS READY THEN
            verification_entries
                <- verification_buffer.ITEMS()

            stored_source_hidden
                <- RUN FROZEN SOURCE ENCODING
                   ON verification_entries

            known_anomaly_mask
                <- FILTER KNOWN ANOMALY TOKENS USING
                   frozen_source_model.discrete_codebook
                   AND frozen_source_model.anomaly_verification_metadata

            continuous_signature_ids
                <- COMPUTE ORDERED TOP-3 SIGNATURES USING
                   stored_source_hidden
                   AND frozen_source_model.continuous_prototype_bank

            recurrent_signature_set
                <- FIND SIGNATURES THAT OCCUR IN MORE THAN ONE
                   NON-OVERLAPPING verification_entry

            pnn_mask
                <- BUILD PNN MASK USING
                   continuous_signature_ids,
                   recurrent_signature_set,
                   known_anomaly_mask

            FOR EACH verification_entry IN verification_entries
                IF pnn_mask FOR verification_entry
                   HAS AT LEAST ONE TRUE VALUE THEN

                    pnn_reconstruction_loss
                        <- COMPUTE_PNN_RECONSTRUCTION_LOSS(
                               verification_entry.x,
                               pnn_mask FOR verification_entry
                           )

                    IF online_variant = A2 THEN
                        online_contrastive_loss
                            <- COMPUTE_ONLINE_CONTRASTIVE_LOSS(
                                   verification_entry,
                                   pnn_mask FOR verification_entry,
                                   frozen_source_model
                               )

                        online_total_loss
                            <- pnn_reconstruction_loss
                               + lambda_online_contrastive
                               * online_contrastive_loss
                    ELSE
                        online_total_loss <- pnn_reconstruction_loss
                    ENDIF

                    CALL UPDATE_ONLINE_MLP_PROJECTOR(
                        online_mlp_projector,
                        online_total_loss
                    )

                    MARK verification_entry AS adapted
                ELSE
                    KEEP verification_entry AS unresolved
                ENDIF
            NEXT verification_entry

            CALL verification_buffer.FINISH_VERIFICATION_CYCLE()
        ENDIF

        online_event_record
            <- BUILD_ONLINE_EVENT_RECORD(
                   causal_window,
                   window_point_scores,
                   current_window_ewma_point_scores,
                   window_point_predictions,
                   triage_region
               )

        SAVE online_event_record

        previous_window_ewma_point_scores
            <- current_window_ewma_point_scores

        SAVE online_runtime_state
    NEXT causal_window
ENDPROCEDURE
```

</td>
<td valign="top" width="50%">

## 2. Flow code hiện tại

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

        IF online_variant != A0 THEN
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

        IF triage_decision = gray_zone THEN
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
                decision <- pnn_verified
                RUN PNN ADAPTATION
            ENDIF
        ENDIF

    CALL adaptation_step

        CREATE online optimizer FOR A1 OR A2

        IF online_variant = A0 THEN
            DO NOT UPDATE

        ELSE IF online_variant = A1 THEN
            IF triage_decision = pnn_verified THEN
                COMPUTE MASKED PNN RECONSTRUCTION LOSS
                UPDATE online_mlp_projector
            ELSE
                DO NOT UPDATE
            ENDIF

        ELSE IF online_variant = A2 THEN
            IF triage_decision = pnn_verified THEN
                COMPUTE PNN RECONSTRUCTION LOSS
                ADD CONTRASTIVE LOSS
                UPDATE online_mlp_projector

            ELSE IF triage_decision = hard_old_normality THEN
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

    previous_ewma_score <- ewma_point_score

END PROCESS_ONLINE_WINDOW
```

</td>
</tr>
</table>

## 3. Khác biệt chính

| Concern | Flow người dùng mong muốn | Code hiện tại |
| --- | --- | --- |
| Point score | `window_point_scores` vector cho toàn bộ points | `endpoint_point_score` scalar; runtime field `raw_point_score` |
| EWMA state | `previous_window_ewma_point_scores` vector | `previous_endpoint_ewma_point_score` scalar; runtime field `previous_ewma_score` |
| Point prediction | `window_point_predictions` vector, cập nhật khi point còn nằm trong sliding window | `endpoint_point_prediction` scalar; runtime field `prediction` |
| Point threshold | `online_point_ewma_threshold` trên vector EWMA | Cùng threshold semantics trên scalar EWMA endpoint |
| Triage | Chạy trước gray-zone PNN work | Preliminary PNN computation chạy trước triage |
| PNN | Chỉ tính trên admitted `verification_entries` khi cycle sẵn sàng | Preliminary PNN trước triage, rồi verification recompute |
| Signature history | Chỉ dựa trên selected/admitted windows | Preliminary path append current window trước triage |
| PNN update gate | `pnn_mask` không rỗng | Runtime compatibility path dùng `triage_decision = pnn_verified` |
| Gray zone | Chỉ admission, chưa adaptation | Chỉ admission, chưa adaptation |
| Hard old | A2 update khi `hard_old_interval_guard` chấp nhận | A2 có hard-old branch và guard trong engine |
| Mutable module | Chỉ `online_mlp_projector` | Chỉ `online_mlp_projector` |

Các điểm code chính:

- Scoring và scalar endpoint EWMA: [`src/engine/online_tta/online_engine_window_metrics.py#L82-L119`](../../src/engine/online_tta/online_engine_window_metrics.py#L82-L119)
- Preliminary PNN trước triage: [`src/engine/online_tta/online_engine_window_core.py#L155-L211`](../../src/engine/online_tta/online_engine_window_core.py#L155-L211)
- Gray-zone admission: [`src/engine/online_tta/online_engine_window_metrics.py#L194-L220`](../../src/engine/online_tta/online_engine_window_metrics.py#L194-L220)
- A1/A2 update branches: [`src/engine/online_tta/online_engine_step.py#L118-L170`](../../src/engine/online_tta/online_engine_step.py#L118-L170)
- Verification buffer lifecycle: [`src/engine/online_tta/verification_buffer.py#L7-L82`](../../src/engine/online_tta/verification_buffer.py#L7-L82)
