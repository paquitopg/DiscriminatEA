#!/bin/bash
#SBATCH --job-name=predict-alignment
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

DISCRIMINATE_EA_CONTAINER=discriminate_ea_predict_container_$SLURM_JOBID
USER_ID=$(id -u ${USER})
USER_GID=$(id -g ${USER})
DOCKER_GID=$(getent group docker | cut -d: -f3)

# Create user-specific cache directory
mkdir -p /tmp/hf_cache_${USER_ID}

DISCRIMINATE_EA_CONTAINER=$DISCRIMINATE_EA_CONTAINER USER_UID=$USER_ID USER_GID=$USER_GID DOCKER_GID=$DOCKER_GID docker-compose run --rm discriminate_ea_predict_service