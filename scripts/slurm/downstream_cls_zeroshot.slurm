#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
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

sif_image=/scratch/shichen/singularity_images/nvcuda_v1.sif

python_bin=$HOME/miniconda3/envs/robo3d/bin/python

export TRN_NUM_POINTS=4096
export TRANS_NUM_GROUPS=256

COLOR_PAD_VALUE=-0.2
FUSION_RATIO=0.6 # adjust the weights for different datasets

# output_dir=data3d/experiments/pretrain/shapenet/data.openshape-pc4096.g256.s32-mae.color.0.05-csc.l1.txt.img-openclip
# output_dir=data3d/experiments/pretrain/shapenet/multiobjrandcam1-pc4096.g256.s32-mae.color.0.05-csc.l1.txt.img-openclip-scene.mae.csc.obj.ref-multi.grasp-nodetach-init.shapenet.single
# output_dir=data3d/experiments/pretrain/objaverse4-nolvis/data.openshape-pc4096.g256.s32-mae.color.0.05-csc.l1.txt.img-openclip
output_dir=data3d/experiments/pretrain/objaverse4-nolvis/multiobjrandcam1-pc4096.g256.s32-mae.color.0.05-csc.l1.txt.img-openclip-scene.mae.csc.obj.ref-multi.grasp-nodetach-init.objaverse_nolvis.single

ckpt_step=200000

# srun --export=ALL \
singularity exec \
    --bind $HOME:$HOME,$SCRATCH:$SCRATCH --nv ${sif_image} \
    $python_bin robo3d/run/downstream/eval_cls_zeroshot.py \
    --fusion_ratio $FUSION_RATIO \
    --exp-config robo3d/configs/downstream/pct_cls_zeroshot.yaml \
    MODEL.transformer_config.csc_skip_dec_sa False \
    output_dir ${output_dir} \
    checkpoint ${output_dir}/ckpts/model_step_${ckpt_step}.pt \
    VAL_DATASET.label_embed_dir data3d/pretrain_dataset/label_embeds \
    VAL_DATASET.dataset_cfgs.modelnet10.output_num_points ${TRN_NUM_POINTS} \
    VAL_DATASET.dataset_cfgs.modelnet40.output_num_points ${TRN_NUM_POINTS} \
    VAL_DATASET.dataset_cfgs.scanobjectnn.output_num_points ${TRN_NUM_POINTS} \
    VAL_DATASET.dataset_cfgs.scanobjectnn_nobg.output_num_points ${TRN_NUM_POINTS} \
    VAL_DATASET.dataset_cfgs.scanobjectnn_hardest.output_num_points ${TRN_NUM_POINTS} \
    VAL_DATASET.dataset_cfgs.objaverse_lvis.output_num_points ${TRN_NUM_POINTS} \
    VAL_DATASET.dataset_cfgs.modelnet10.color_pad_value ${COLOR_PAD_VALUE} \
    VAL_DATASET.dataset_cfgs.modelnet40.color_pad_value ${COLOR_PAD_VALUE} \
    VAL_DATASET.dataset_cfgs.scanobjectnn.color_pad_value ${COLOR_PAD_VALUE} \
    VAL_DATASET.dataset_cfgs.scanobjectnn_nobg.color_pad_value ${COLOR_PAD_VALUE} \
    VAL_DATASET.dataset_cfgs.scanobjectnn_hardest.color_pad_value ${COLOR_PAD_VALUE} \
    MODEL.transformer_config.num_groups ${TRANS_NUM_GROUPS}