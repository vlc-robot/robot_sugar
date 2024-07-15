#!/bin/bash
#SBATCH --job-name=trainbc
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --mem=30G
#SBATCH -p willow
#SBATCH -A willow
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

seed=${1}
taskvars=10tasks
taskvars_file=$HOME/codes/robo3d/robo3d/assets/rlbench/${taskvars}.json

export TRN_NUM_POINTS=4096
export TRANS_NUM_GROUPS=256
export VOXEL_SIZE=0.005

output_dir=data_rlbench/experiments/${taskvars}/pc${TRN_NUM_POINTS}.voxel0.5cm-g${TRANS_NUM_GROUPS}.s32-heatmap.catft-init.shapenet.multi.4096.mtgrasp-lr2e-5
output_dir=${output_dir}/seed${seed}

# srun --export=ALL \
singularity exec \
    --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${sif_image} \
    $python_bin robo3d/run/rlbench/train_models.py \
    --exp-config robo3d/configs/rlbench/pct_manipulator.yaml \
    output_dir ${output_dir} \
    TRAIN.num_train_steps 200000 \
    TRAIN.save_steps 5000 TRAIN.warmup_steps 5000 \
    TRAIN.learning_rate 2e-5 TRAIN.lr_sched cosine \
    TRAIN.weight_decay 5e-2 \
    TRAIN.train_batch_size 8 \
    TRAIN.n_workers 0 DATASET.in_memory True \
    DATASET.taskvars ${taskvars_file} \
    DATASET.dataset_class pre_pcd_keystep_stepwise \
    DATASET.data_dir data_rlbench/train_dataset/10tasks/keysteps_pcd.6_point_norm/seed${seed} \
    DATASET.instr_embed_file data_rlbench/train_dataset/taskvar_instrs/openclip \
    DATASET.only_success False \
    DATASET.max_episode_steps 20 DATASET.sampled_episode_steps 15 \
    DATASET.pc_space workspace_on_table \
    DATASET.pc_center point \
    DATASET.pc_radius_norm True \
    DATASET.npoints ${TRN_NUM_POINTS} \
    DATASET.use_normal False DATASET.use_height False \
    DATASET.voxel_size ${VOXEL_SIZE} \
    MODEL.transformer_config.num_groups ${TRANS_NUM_GROUPS} \
    MODEL.transformer_config.group_size 32 \
    MODEL.action_head_config.pos_pred_type heatmap \
    MODEL.action_head_config.pc_gumbel_softmax False \
    MODEL.action_head_config.cat_pc_fts True \
    MODEL.action_head_config.pc_upsampling False \
    MODEL.action_head_config.heatmap_norm True \
    MODEL.action_head_config.multiscale_pc_layers 12 \
    TRAIN.freeze_params.encoder False \
    TRAIN.freeze_params.decoder False \
    TRAIN.resume_encoder_only False \
    checkpoint data3d/experiments/pretrain/shapenet/multiobjrandcam1-pc4096.g256.s32-mae.color.0.05-csc.l1.txt.img-openclip-scene.mae.csc.obj.ref-multi.grasp-nodetach-init.shapenet.single/ckpts/model_step_100000.pt

