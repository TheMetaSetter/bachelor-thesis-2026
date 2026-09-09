#!/usr/bin/env bash

run_with_remaining_smd_resources() {
    local resource_class="$1"
    local cpu_mask="$2"
    local gpu_index="$3"
    shift 3

    if [[ "$#" -eq 0 ]]; then
        echo "resource wrapper requires a command" >&2
        return 2
    fi
    if [[ "$cpu_mask" != "0-7" && "$cpu_mask" != "8-15" \
        && "$cpu_mask" != "16-23" && "$cpu_mask" != "24-31" \
        && "$cpu_mask" != "32-37" && "$cpu_mask" != "38-43" ]]; then
        echo "unsupported CPU mask: $cpu_mask" >&2
        return 2
    fi

    case "$resource_class" in
        gpu)
            if [[ "$cpu_mask" != "0-7" && "$cpu_mask" != "8-15" \
                && "$cpu_mask" != "16-23" && "$cpu_mask" != "24-31" ]]; then
                echo "GPU workers require CPU masks 0-7, 8-15, 16-23, or 24-31" >&2
                return 2
            fi
            if [[ "$gpu_index" != "0" && "$gpu_index" != "1" \
                && "$gpu_index" != "2" && "$gpu_index" != "3" ]]; then
                echo "GPU workers require GPU index 0, 1, 2, or 3" >&2
                return 2
            fi
            export CUDA_VISIBLE_DEVICES="$gpu_index"
            export OMP_NUM_THREADS=1
            export MKL_NUM_THREADS=1
            export OPENBLAS_NUM_THREADS=1
            export NUMEXPR_NUM_THREADS=1
            ;;
        cpu)
            if [[ "$cpu_mask" != "32-37" && "$cpu_mask" != "38-43" ]]; then
                echo "CPU workers require CPU mask 32-37 or 38-43" >&2
                return 2
            fi
            if [[ -n "$gpu_index" ]]; then
                echo "CPU workers must not receive a GPU index" >&2
                return 2
            fi
            unset CUDA_VISIBLE_DEVICES
            export OMP_NUM_THREADS=6
            export MKL_NUM_THREADS=6
            export OPENBLAS_NUM_THREADS=6
            export NUMEXPR_NUM_THREADS=6
            ;;
        *)
            echo "resource class must be gpu or cpu" >&2
            return 2
            ;;
    esac

    echo "RESOURCE class=$resource_class cpu_mask=$cpu_mask gpu_index=${gpu_index:-none}"
    taskset -c "$cpu_mask" "$@"
}
