#!/bin/bash
#SBATCH --job-name=eval_val
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.out

set -x
set -e

module purge
pwd; hostname; date
module load singularity

cd $HOME/codes/robo3d

. $HOME/miniconda3/etc/profile.d/conda.sh
conda activate robo3d

sif_image=/scratch/shichen/singularity_images/nvcuda_v1.sif

python_bin=$HOME/miniconda3/envs/robo3d/bin/python

outdir=$1

for ckpt_step in {200000..100000..5000}
do
# srun --export=ALL,XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
    singularity exec \
    --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv \
    ${sif_image} \
    xvfb-run -a ${python_bin} scripts/python/eval_tst_split.py \
    --python_bin ${python_bin} \
    --exp_config  ${outdir}/logs/training_config.yaml \
    --seed 100 --num_demos 20 \
    --checkpoint ${outdir}/ckpts/model_step_${ckpt_step}.pt \
    --num_workers 5
done
