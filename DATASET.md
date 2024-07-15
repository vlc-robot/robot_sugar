# Dataset

## Single-object dataset

We reuse the single object dataset used in [OpenShape](https://github.com/Colin97/OpenShape_code), but change it into the lmdb format.
The processed data can be download here:
[ShapeNet](https://www.dropbox.com/scl/fi/8yk5tfh4p110v4qhfwr6l/shapenet_openshape.tar.gz?rlkey=cz0kfj8smgxl66tm6yq3inen7&st=p9fcnviv&dl=0), [ABO](https://www.dropbox.com/scl/fi/q4w919jybh2yutvrbteor/ABO_openshape.tar.gz?rlkey=yeknt2z8ikodn43wc70i4bo8o&st=aal79ofi&dl=0), [3D-FUTURE](https://www.dropbox.com/scl/fi/pkdp1b3p9ewzu5z0vdwtl/3D-FUTURE_openshape.tar.gz?rlkey=1x498eukkg1vnzymo0r9tyfav&st=aiy51oxm&dl=0), [Objaverse](https://www.dropbox.com/scl/fo/5pow2g5sffixxvlr2bc5w/AMCBMhp1Sbag-JZliKYudDU?rlkey=nhrt7n7j4i94cns7m6pds0a6r&st=7ljnlpnd&dl=0).

## Multi-object dataset

We use the [ACRONYM](https://sites.google.com/nvidia.com/graspdataset) to generate [grasping data](https://www.dropbox.com/scl/fi/kam97krv938my5hanu4tt/acronym_grasp.tar.gz?rlkey=k32sia76caw9k4yx2qqqhdwfk&st=9tida7j2&dl=0), and [blenderproc](https://github.com/DLR-RM/BlenderProc) to generate multi-object scenes [ShapeNet-multi](https://www.dropbox.com/scl/fi/ca2jfirw4ai4mtqifb95x/shapenet_multiobj_randcam1.tar.gz?rlkey=143oetpu4ffvi2qpf087yojr9&st=hrl7fz0n&dl=0) and [Objaverse-multi](https://www.dropbox.com/scl/fi/qea8c4otrvh92wxc3jsn2/objaverse_multiobj_randcam3.tar.gz?rlkey=2iadd6yvp8618rf23yb2xy7y4&st=20y2qunx&dl=0).

## Evaluation datasets

### Zero-shot object recognition
- [modelnet](https://modelnet.cs.princeton.edu/): The processed dataset can be downloaded [here](https://www.dropbox.com/scl/fi/s6h34em6ny1d32evep5by/ModelNet.tar.gz?rlkey=4q7xzxw3dz48jlmlg30oph43p&st=bp4nl1eu&dl=0).
- [scanobjectnn](https://hkust-vgd.github.io/scanobjectnn/): We could send you the processed dataset if you get the permission to access the original data.
- [objaverse_lvis](https://objaverse.allenai.org/objaverse-1.0): The processed dataset can be downloaded [here](https://www.dropbox.com/scl/fi/amr393bno28sz1esdb0qw/OpenShape.tar.gz?rlkey=tmcme2q48bdrts57z3pzjte57&st=lki95zcd&dl=0).

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
        - acronym_grasp_single_object
        - acronym_grasp_multi_object
        - shapenet_multiobj_randcam1
        - objaverse_multiobj_randcam3
    - recon_dataset/
        - modelnet/
        - scanobjectnn/
    - OpenShape
        - objaverse-processed
        - meta_data
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
        


    