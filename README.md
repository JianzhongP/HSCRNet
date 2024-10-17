**HSCRNet-for-HRRS-Semantic-Segmentation**
====
Pre-training weights download:
----
Pre-training weights for the mobilenet_v2.pth and comparison models are available at the following link,
Files shared via Netflix: 

Comparison_wights
Link: https://pan.baidu.com/s/1HSunOBfseNLACArZlNHKkw 
Extraction Code: aqay

**Note:** After downloading stvit-base-384.pth, please place it in the module folder and the rest in the model_data folder.

The experiments were performed on two public datasets, and the respective training weights can be downloaded via the following links:
HSCRNet_wight
Link: https://pan.baidu.com/s/11YIqiWJRccwRiSNg56NK-w 
Extraction Code: seaq

Usage:
====
Installation
----
Please download the dataset and code support package by code yourself.

Training
----
```bash
#Training command
python train.py
#Distributed training command
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
