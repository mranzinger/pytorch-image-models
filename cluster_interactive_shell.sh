#!/bin/bash

# --partition 'adlr_interactive_32GB'
# --partition 'batch_32GB'
# --partition 'batch_dgx2_singlenode'

PARTITION=${1:-"interactive"}
# PARTITION=${1:-"batch_dgx1_m2"}
GPUS=${2:-2}
# GPUS=${2:-8}

# On aws, the correct drives are auto-mounted
MOUNT_CMD=""
if [[ $HOSTNAME != "draco-aws-login-01" ]]; then
    MOUNT_CMD="--mounts $MOUNTS"
fi

submit_job \
           --gpu $GPUS \
           --partition "$PARTITION" \
           $MOUNT_CMD \
           --workdir "/home/mranzinger/dev/scene-text" \
           --image "gitlab-master.nvidia.com/adlr/scene-text/scene-text:aws" \
           --coolname \
           --interactive \
           --duration 4 \
           -c "bash"
