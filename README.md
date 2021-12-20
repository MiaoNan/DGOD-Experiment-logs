# Research experiment logs

My work is now focusing on combining Semantic Information with Domain Generation, this page is used to write my experiment log. 

### 2021.12.20

New Faster-R-CNN from: https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0

Modify the classification layer in 'faster-rcnn.pytorch-pytorch-1.0/lib/model/faster_rcnn/vgg16.py', line 50.

Batch size of classification layer could be changed in this project.

However, questions are listed as follow:

1. Is there any difference between projecting feature map into semantic space and mapping into same dimension?
2. Projection are same to referenced paper, mapping are to complicated?

Besides, RTX 2060 have not enough memory for Faster-R-CNN, servers are needed now.


### 2021.12.1

Raw word embedding tensor generation module finished.

Downloading 300d and 500d pretrained embeddings.

Interface on different dimision embeddings finished.

Modified three label words so it can be vectorized.

>diningtable -> table
>pottedplant -> plant
>tvmonitor -> monitor

~~Next step is debugging with voc2012 dataset. Remember that the classification layer was modified back to 512x1024, but input data are still 512x512.~~

Now the shape of each tensor have been reshaped properly. But the model is loading the whole weights.

Remove the weights, or modify the shape by adding several layers.


### 2021.11.29

Modify Faster-R-CNN module to add Self Attention module into classification subnet.

Downloading Word2Vec pretrained embeddings...English 100 dim.

### 2021.11.26

Self Attention module finished.

Do take care of the size of input and output.


### 2021.11.25

Giving up using Faster-R-CNN from torchvision because it is difficult to override Class.

Now using Faster-R-CNN from : https://github.com/lllsaaxx/Faster-RCNN

### 2021.11.22

Experiment environment are finished on PC, NOT server.

System: Ubuntu 20.04.3 LTS

NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2

Cuda compilation tools, release 11.2, V11.2.67   Build cuda_11.2.r11.2/compiler.29373293_0

Pytorch is instlled with this command:
>pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

See this page in detail: https://medium.com/analytics-vidhya/install-cuda-11-2-cudnn-8-1-0-and-python-3-9-on-rtx3090-for-deep-learning-fcf96c95f7a1
