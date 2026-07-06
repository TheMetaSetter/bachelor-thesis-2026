(✿◠‿◠)  BENCHMARK BRANCHES  ⸜(｡˃ ᵕ ˂ )⸝♡

                     benchmark run
                          |
        +-----------------+-----------------+
        |                 |                 |
        v                 v                 v
  [baseline]        [thesis base]   [thesis point-score]
  redlamp_baseline   two-stage base  two-stage + score loss
        |                 |                 |
        v                 v                 v
  3 losses          3 losses         3 losses + point-wise
                    + Stage A          balanced recon-score
                    + Stage B          loss
                                     
  configs/experiment/benchmark/baseline/
  configs/experiment/benchmark/thesis/
      smd__redlamp_baseline__...          smd__thesis_multitask__benchmark-two-stage-...           smd__thesis_multitask__benchmark-two-stage-point-score-...

(✿◠‿◠) BENCHMARK FLOW + CONFIG FILES ⸜(｡˃ ᵕ ˂ )⸝♡

[0] Chọn entrypoint
    scripts/launch_tmux_comparative_smd_experiment.sh
        |
        v
[1] Runner tổng
    scripts/run_comparative_smd_experiments.py
        |
        v
[2] Đọc config experiment
    load_experiment_config(...)
        |
        +--> config base cho benchmark
        |    - data:  configs/data/smd_benchmark_machine_1_6_window20.yaml
        |             configs/data/smd_benchmark_machine_3_4_window20.yaml
        |             configs/data/smd_benchmark_machine_3_9_window20.yaml
        |    - task:  configs/task/multitask_tsad_redlamp_multiclass_window20_benchmark_fixed_synth.yaml
        |    - model base:
        |             configs/model/thesis_multitask_two_stage_window20.yaml
        |    - model point-score:
        |             configs/model/thesis_multitask_two_stage_point_score_window20.yaml
        |
        +--> experiment benchmark baseline
        |    - configs/experiment/benchmark/baseline/smd__redlamp_baseline__benchmark-...__main.yaml
        |
        +--> experiment benchmark thesis base
        |    - configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-...__main.yaml
        |
        +--> experiment benchmark thesis point-score
             - configs/experiment/benchmark/thesis/smd__thesis_multitask__benchmark-two-stage-point-score-...__main.yaml
        |
        v
[3] Validate config + route runtime  (づ｡◕‿‿◕｡)づ
        |
        +--> baseline_single_stage
        |       train.py -> evaluate.py
        |
        +--> thesis_two_stage / thesis_point_score_supervised
                run_two_stage_offline_pretraining.py
                |
                +--> Stage A
                |    - reconstruction loss
                |    - classification loss
                |    - two-view contrastive loss
                |    - point-score loss only in point-score variant
                |
                +--> Stage B
                |    - fusion finetuning
                |
                +--> save best checkpoint
        |
        v
[4] Evaluate xong
    src/engine/evaluator.py
        |
        +--> point_scores
        +--> timeline merge
        +--> VUS-PR
        +--> VUS-ROC
        +--> Affiliation F1
        +--> confusion matrix
        +--> CKA / diagnostics / hard prediction ratio
        |
        v
[5] Ghi artifacts  ⸜(｡˃ ᵕ ˂ )⸝♡
    outputs/...
    +--> checkpoints/
    +--> evaluation_metrics.json
    +--> execution_report.json
    +--> comparative_manifest.json
    +--> wandb run
    +--> tmux logs