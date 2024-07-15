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

output_dir=data3d/experiments/downstream/roborefit/pc10k-1536x32-bsz.16-openclip-init.objavers4nolvis.multi.4096.mtgrasp-lr4e-5

# srun --export=ALL \
singularity exec \
    --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${sif_image} \
    $python_bin robo3d/run/downstream/train_ref_models.py \
    --exp-config robo3d/configs/downstream/pct_roborefit.yaml \
    output_dir ${output_dir} \
    TRAIN.num_epochs 50 TRAIN.warmup_steps 2000 \
    TRAIN.save_steps 1000 TRAIN.val_steps 1000 TRAIN.log_steps 1000 \
    TRAIN.learning_rate 4e-5 TRAIN.lr_sched cosine \
    TRAIN.train_batch_size 16 \
    TRAIN.n_workers 8 \
    DATASET.num_points 10000 \
    DATASET.use_color True \
    DATASET.augment_color True \
    MODEL.transformer_config.num_groups 1536 \
    MODEL.transformer_config.group_size 32 \
    TRAIN.resume_encoder_only False \
    checkpoint data3d/experiments/pretrain/objaverse4-nolvis/multiobjrandcam1-pc4096.g256.s32-mae.color.0.05-csc.l1.txt.img-openclip-scene.mae.csc.obj.ref-multi.grasp-nodetach-init.objaverse_nolvis.single/ckpts/model_step_200000.pt