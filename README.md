# DFNet
# Installation and Dependencies
‘’‘conda create -n DFNet python=3.7 -y
conda activate DFNet
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install matplotlib
conda install tensorboardX
conda install scipy
conda install scikit-learn
\<path to your anaconda3\>/envs/DFNet/bin/pip install opencv-python
\<path to your anaconda3\>/envs/DFNet/bin/pip install opencv-contrib-python‘’‘

# Training
1. Download imagenet-vgg-m.mat in /models and ILSVRC2015 in /dateset.
2. Run pretrain_rtmdnet_DFNet.py
3. Download GTOT and RGBT234 in /dataset
4. Run train_rtmdnet_DFNet.py

# Testing
1. Select model_path
2. Run Run_rtmdnet_DFNet.py
