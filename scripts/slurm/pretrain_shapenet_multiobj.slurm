#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=10
#SBATCH --partition=willow
#SBATCH -A willow
#SBATCH --gres=gpu:a100:1
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --mem=40G
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

ulimit -n 2048

sif_image=/scratch/shichen/singularity_images/nvcuda_v1.sif

python_bin=$HOME/miniconda3/envs/robo3d/bin/python

export TRN_NUM_POINTS=4096
export TRANS_NUM_GROUPS=256

output_dir=data3d/experiments/pretrain/shapenet/multiobjrandcam1-pc${TRN_NUM_POINTS}.g${TRANS_NUM_GROUPS}.s32-mae.color.0.05-csc.l1.txt.img-openclip-scene.mae.csc.obj.ref-multi.grasp-nodetach-init.shapenet.single

# srun --export=ALL \
singularity exec \
    --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${sif_image} \
    $python_bin robo3d/run/pretrain/train_models.py \
    --exp-config robo3d/configs/pretrain/pct_pretrain.yaml \
    output_dir ${output_dir} \
    TRAIN.num_train_steps 100000 \
    TRAIN.log_steps 1000 TRAIN.warmup_steps 5000 \
    TRAIN.val_steps 2000 TRAIN.save_steps 2000 \
    TRAIN.learning_rate 5e-5 TRAIN.lr_sched cosine \
    TRAIN.n_workers 8 \
    TRAIN.trn_task_ratios 2,1,1 \
    TRAIN.trn_task_batch_size 128,128,128 \
    TRAIN.val_batch_size 128 \
    SINGLEOBJ_DATASET.dataset_names "('shapenet', 'abo', '3dfuture', 'objaverse')" \
    SINGLEOBJ_DATASET.num_points ${TRN_NUM_POINTS} \
    GRASP_DATASET.num_points ${TRN_NUM_POINTS} \
    MULTIOBJ_DATASET.dataset_names "('shapenet_multiobj', )" \
    MULTIOBJ_DATASET.num_points ${TRN_NUM_POINTS} \
    MULTIOBJ_DATASET.return_scene_fts True \
    MULTIOBJ_DATASET.return_obj_fts True \
    MULTIOBJ_DATASET.return_ref_fts True \
    MULTIOBJ_DATASET.keep_background_ratio 0.5 \
    VAL_DATASET.dataset_cfgs.modelnet.output_num_points ${TRN_NUM_POINTS} \
    VAL_DATASET.dataset_cfgs.scanobjectnn.output_num_points ${TRN_NUM_POINTS} \
    VAL_DATASET.dataset_cfgs.objaverse_lvis.output_num_points ${TRN_NUM_POINTS} \
    MODEL.transformer_config.num_groups ${TRANS_NUM_GROUPS} \
    MODEL.transformer_config.group_size 32 \
    MODEL.transformer_config.input_size 6 \
    MODEL.transformer_config.num_heads 6 \
    MODEL.transformer_config.detach_enc_dec False \
    MODEL.transformer_config.csc_skip_dec_sa False \
    MODEL.transformer_config.mask_ratio 0.6 \
    MODEL.loss_config.mae_loss_weight 1.0 \
    MODEL.loss_config.mae_color_loss_weight 0.05 \
    MODEL.loss_config.csc_loss_type l1 \
    MODEL.loss_config.csc_txt_loss_weight 1.0 \
    MODEL.loss_config.csc_img_loss_weight 1.0 \
    MODEL.loss_config.scene_mae_csc_loss_weight 1.0 \
    MODEL.loss_config.obj_loss_weight 1.0 \
    MODEL.loss_config.ref_loss_weight 1.0 \
    MODEL.clip_model openclip \
    MODEL.cross_modal_config.img_ft_size 1280 \
    MODEL.cross_modal_config.txt_ft_size 1280 \
    checkpoint data3d/experiments/pretrain/shapenet/data.openshape-pc4096.g256.s32-mae.color.0.05-csc.l1.txt.img-openclip/ckpts/model_step_100000.pt

    