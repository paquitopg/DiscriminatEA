#!/bin/bash
#SBATCH --job-name=embedding-generator
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

SIMPLE_HHEA_CONTAINER=simple_hhea_embedding_container_$SLURM_JOBID
USER_ID=$(id -u ${USER})
USER_GID=$(id -g ${USER})
DOCKER_GID=$(getent group docker | cut -d: -f3)

# Create user-specific cache directory
mkdir -p /tmp/hf_cache_${USER_ID}

SIMPLE_HHEA_CONTAINER=$SIMPLE_HHEA_CONTAINER USER_UID=$USER_ID USER_GID=$USER_GID DOCKER_GID=$DOCKER_GID docker-compose run --rm simple_hhea_embedding_service