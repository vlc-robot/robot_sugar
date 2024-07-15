# Dataset

## Single-object dataset

We reuse the single object dataset used in [OpenShape](https://github.com/Colin97/OpenShape_code), but change it into the lmdb format.
The processed data can be download here:
[ShapeNet](https://www.dropbox.com/scl/fi/8yk5tfh4p110v4qhfwr6l/shapenet_openshape.tar.gz?rlkey=cz0kfj8smgxl66tm6yq3inen7&st=p9fcnviv&dl=0), [ABO](https://www.dropbox.com/scl/fi/q4w919jybh2yutvrbteor/ABO_openshape.tar.gz?rlkey=yeknt2z8ikodn43wc70i4bo8o&st=aal79ofi&dl=0), [3D-FUTURE](https://www.dropbox.com/scl/fi/pkdp1b3p9ewzu5z0vdwtl/3D-FUTURE_openshape.tar.gz?rlkey=1x498eukkg1vnzymo0r9tyfav&st=aiy51oxm&dl=0), [Objaverse]().

## Multi-object dataset

We use the [ACRONYM](https://sites.google.com/nvidia.com/graspdataset) to generate [grasping data](), and [blenderproc](https://github.com/DLR-RM/BlenderProc) to generate multi-object scenes [ShapeNet-multi]() and [Objaverse-multi]().

## Evaluation datasets

### Zero-shot object recognition
- [modelnet](https://modelnet.cs.princeton.edu/): The processed dataset can be downloaded [here]().
- [scanobjectnn](https://hkust-vgd.github.io/scanobjectnn/): We could send you the processed dataset if you get the permission to access the original data.
- [objaverse_lvis](https://objaverse.allenai.org/objaverse-1.0): The processed dataset can be downloaded [here]().

## Referring expression
- [ocidref](https://github.com/lluma/OCID-Ref): We could send you the processed dataset if you get the permission to access the original data.
- [roborefit](https://github.com/luyh20/VL-Grasp): We could send you the processed dataset if you get the permission to access the original data.

## Language-guided robotic manipulation
- RLBench data: follow [PolarNet](https://github.com/vlc-robot/polarnet/?tab=readme-ov-file) to download or generate the training data for the 10 robotic manipulation tasks.

## Data organization structure

- data3d/
    - pretrain_dataset/
        - shapenet_openshape/
        - ABO_openshape/
        - 3D-FUTURE_openshape/
        - objaverse_openshape/
    - recon_dataset/
        - modelnet/
        - scanobjectnn/
    - Objaverse
    - OCID
    - VLGrasp

    - experiments/
        - pretrain/
            - shapenet/
            - objaverse4-nolvis/
        - downstream/
            - roborefit/
            - ocidref/
        - rlbench/
        


    