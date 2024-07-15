# Installation Instruction

Singularity is recommended to install all the packages in HPC.

1. Install basic packages
```bash
conda create -n robo3d python==3.10

conda activate robo3d

conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install tensorboardX lmdb msgpack msgpack_numpy yacs pyyaml tqdm jsonlines filelock
pip install typed-argument-parser easydict einops ninja absl-py h5py
pip install timm transformers
pip install open_clip_torch
pip install accelerate
pip install open3d==0.17.0
```

2. Install point cloud related packages
```bash
# install chamerdist, pointnet2_ops
git clone https://github.com/cshizhe/chamferdist.git
cd chamferdist
python setup.py develop
cd ..

git clone https://github.com/cshizhe/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/pointnet2_ops_lib
python setup.py install
```

3. Install RLBench
```bash
# download coppeliasim
export COPPELIASIM_ROOT=/home/shichen/codes/robo3d/dependencies/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# install pyrep
git clone https://github.com/stepjam/PyRep
cd PyRep
pip3 install -r requirements.txt
pip3 install .
cd ..

# install rlbench
git clone https://github.com/rjgpinel/RLBench
cd RLBench
pip install -r requirements.txt
pip install .
cd ..
```

4. Install the robot_sugar
```bash
pip install -e .
```