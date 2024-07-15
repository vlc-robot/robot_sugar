#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=10
#SBATCH --partition=willow
#SBATCH -A willow
#SBATCH --gres=gpu:a100:1
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --mem=30G
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

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_TASKS_PER_NODE))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

sif_image=/scratch/shichen/singularity_images/nvcuda_v1.sif

python_bin=$HOME/miniconda3/envs/robo3d/bin/python

output_dir=data3d/experiments/downstream/ocidref/pc10k-512x32-openclip-init.objaverse4_nolvis.multi.4096.mtgrasp-freeze.enc

# srun --export=ALL \
singularity exec \
    --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${sif_image} \
    $python_bin robo3d/run/downstream/train_ref_models.py \
    --exp-config robo3d/configs/downstream/pct_ocidref.yaml \
    output_dir ${output_dir} \
    TRAIN.num_epochs 20 TRAIN.warmup_steps 5000 \
    TRAIN.save_steps 2000 TRAIN.val_steps 2000 TRAIN.log_steps 1000 \
    TRAIN.learning_rate 1e-4 TRAIN.lr_sched cosine \
    TRAIN.train_batch_size 64 \
    TRAIN.freeze_params.encoder True \
    TRAIN.freeze_params.decoder False \
    TRAIN.freeze_params.all_except_head False \
    checkpoint data3d/experiments/pretrain/objaverse4-nolvis/multiobjrandcam1-pc4096.g256.s32-mae.color.0.05-csc.l1.txt.img-openclip-scene.mae.csc.obj.ref-multi.grasp-nodetach-init.objaverse_nolvis.single/ckpts/model_step_200000.pt
