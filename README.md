# Research experiment logs 实验研究记录
目前进行Semantic Embedding和Faster-R-CNN的结合的工作，以此page作为实验过程记录。

### 2021.12.1

Raw word embedding tensor generation finished.

Interface on different dimision embeddings finished.

Modified three label words so it can be vectorized.

>diningtable -> table
>pottedplant -> plant
>tvmonitor -> monitor

Next step is debugging with voc2012 dataset. Remember that the classification layer was modified back to 512x1024, but input data are still 512x512.

### 2021.11.29

Modify Faster-R-CNN module to add Self Attention module into classification subnet.

Downloading Word2Vec pretrained embeddings...

### 2021.11.26

Self Attention module finished.

Do take care of the size of input and output.


### 2021.11.25

Giving up using Faster-R-CNN from torchvision because it is difficult to override those Class.

Now using Faster-R-CNN from : https://github.com/lllsaaxx/Faster-RCNN

### 2021.11.22

Experiment environment are finished on PC, NOT server.

System: Ubuntu 20.04.3 LTS

NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2

Cuda compilation tools, release 11.2, V11.2.67   Build cuda_11.2.r11.2/compiler.29373293_0

Pytorch is instlled with this command:
>pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

See this page in detail: https://medium.com/analytics-vidhya/install-cuda-11-2-cudnn-8-1-0-and-python-3-9-on-rtx3090-for-deep-learning-fcf96c95f7a1
