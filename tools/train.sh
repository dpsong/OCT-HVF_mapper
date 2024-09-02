#!/usr/bin/env bash

NODES=${NODES:-1}
GPUS=${GPUS:-1}
CPU_PER_TASK=${CPU_PER_TASK:-4}
MEM_PER_CPU=${MEM_PER_CPU:-"12G"}

TYPE=${TYPE:-"locale"}
PORT=${PORT:-29500}
SLURM_ARGS=${SLURM_ARGS:-""}
JOB_NAME=${JOB_NAME:-"drb"}

CONFIG=$1
PY_ARGS=${@:2}


if [[ $TYPE = "locale" ]]; then
    if [[ $GPUS = 1 ]]; then
        python $(dirname "$0")/train.py ${CONFIG} --launcher none ${PY_ARGS}
    else
        torchrun \
            --nnodes=1 \
            --master_port=${PORT} \
            --nproc_per_node=${GPUS} \
            $(dirname "$0")/train.py ${CONFIG} --launcher pytorch ${PY_ARGS}
    fi
elif [[ $TYPE = "run" ]]; then
    srun --job-name=${JOB_NAME} \
        --partition=batch \
        --nodes=${NODES} \
        --ntasks-per-node=${GPUS} \
        --cpus-per-task=${CPU_PER_TASK} \
        --mem-per-cpu=${MEM_PER_CPU} \
        --gres=gpu:${GPUS} \
        --kill-on-bad-exit=1 \
        ${SLURM_ARGS} \
        python -u $(dirname "$0")/train.py ${CONFIG} --launcher="slurm" ${PY_ARGS}
else
    echo "Unknow type: " $TYPE
fi