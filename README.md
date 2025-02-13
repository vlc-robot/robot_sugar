# SUGAR: Pre-training 3D Visual Representation for Robotics

This repository is the official implementation of [SUGAR: Pre-training 3D Visual Representations for Robotics](https://cshizhe.github.io/projects/robot_sugar.html) (CVPR 2024).

<!-- ![teaser](files/teaser.png) -->
<img src="files/teaser.png" alt="teaser" width="600"/>


## Install
See [INSTALL.md](INSTALL.md) for detailed instruction in installation.

## Dataset
See [DATASET.md](DATASET.md) for detailed instruction in dataset.


## Pre-training
The pretrained checkpoints are available [here](https://www.dropbox.com/scl/fi/wyq9pku4gmpwu2n6en55q/pretrain.tar.gz?rlkey=ma6fyeiittl7bad1ho3vx4qsa&st=rpc1en7w&dl=0).

1. pre-training on single-object datasets
```bash
sbatch scripts/slurm/pretrain_shapenet_singleobj.slurm
sbatch scripts/slurm/pretrain_ensemble_singleobj.slurm
```

2. pre-training on multi-object datasets
```bash
sbatch scripts/slurm/pretrain_shapenet_multiobj.slurm
sbatch scripts/slurm/pretrain_ensemble_multiobj.slurm
```

## Zero-shot 3D object recognition

Evaluate on the modelnet, scanobjectnn and objaverse_lvis dataset with [pretrained checkpoints](https://www.dropbox.com/scl/fi/wyq9pku4gmpwu2n6en55q/pretrain.tar.gz?rlkey=ma6fyeiittl7bad1ho3vx4qsa&st=rpc1en7w&dl=0).
```bash
sbatch scripts/slurm/downstream_cls_zeroshot.slurm
```

## Robotic referring expression grounding

Train and evaluate on the ocidref and roborefit dataset. The trained models can be downloaded [here](https://www.dropbox.com/scl/fi/vv6wce2gvj5xhdpf4n99w/downstream_referit.tar.gz?rlkey=wbtgskdd0pjml3tnpo1t9ybk1&st=qcian2z2&dl=0).
```bash
sbatch scripts/slurm/downstream_ocidref.slurm
sbatch scripts/slurm/downstream_roborefit.slurm
```

## Language-guided robotic manipulation

Train and evaluate on the RLBench 10 tasks. The trained model can be downloaded [here](https://www.dropbox.com/scl/fi/6gshf5vij7wwwniko0zlz/rlbench.tar.gz?rlkey=6r7gy7fkmbj9q41bualphizc7&st=1wf7ml0z&dl=0).
```bash
sbatch scripts/slurm/rlbench_train_multitask_10tasks.slurm
sbatch scripts/slurm/rlbench_eval_val_split.slurm
sbatch scripts/slurm/rlbench_eval_tst_split.slurm
```

## Citation
If you find this work useful, please consider citing:
```bib
@InProceedings{Chen_2024_SUGAR,
    author    = {Chen, Shizhe and Garcia, Ricardo and Laptev, Ivan and Schmid, Cordelia},
    title     = {SUGAR: Pre-training 3D Visual Representations for Robotics},
    booktitle = {CVPR},
    year      = {2024}
}
```
