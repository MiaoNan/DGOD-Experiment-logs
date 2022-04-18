# Research experiment logs

## My work is now focusing on combining Semantic Information with Domain Generation, so I write my experiment log here. 

Baseline Result, it will change according to experience conducting.

AP for aeroplane = 0.5207  
AP for bicycle = 0.7173  
AP for bird = 0.5351  
AP for boat = 0.3100  
AP for bottle = 0.3818  
AP for bus = 0.6089  
AP for car = 0.7326  
AP for cat = 0.6954  
AP for chair = 0.3188  
AP for cow = 0.6530  
AP for diningtable = 0.3305  
AP for dog = 0.6586  
AP for horse = 0.7295  
AP for motorbike = 0.7040  
AP for person = 0.6577  
AP for pottedplant = 0.2458  
AP for sheep = 0.6090  
AP for sofa = 0.5029  
AP for train = 0.5915  
AP for tvmonitor = 0.5937  
Mean AP = 0.5548  

### Add new dataset

* Modify __set in factory.py
* add xxx.py into lib/dataset 
* Modify lib/model/utils/parser_func.py

* And bdd100k.py modified for the different classes


### 2022.4.1

* Server environment installed, now the code can run on the server.
* Dataset I/O finished.
* Loss added, and the whole code can run now.

while testing, the label of test set should be the same with source domain.

Next step is to add ROI on base feature.

### 2022.3.18

* Add GRL to att_faster-R-CNN
* Add semantic information projection in both base and roi.

Next to add loss on these codes, and DA settings.

### 2022.1.18

* Reduce FC layer from 4 to 1 after ROI pooling.

Before: 0.1(logs lost yesterday)  
After: 0.0589  

It can be seen that reducing layers will decrease performance.  
However, according result, it may be necessary to use class prototype!

**Running on server with full VOC dataset!**

### 2022.1.16

* Consider using Metric Learning, which compare the distance between query image and class prototype.  
* So new type losses are need to be learned.
https://blog.csdn.net/xys430381_1/article/details/90705421

### 2022.1.14

在train阶段，会输出约2000个proposal，但只会抽取其中256个proposal来训练RPN的cls+reg结构（其中，128个前景proposal用来训练cls+reg。  
So inbalance problem is not a problem.  

### 2022.1.7

* Replace cosine similarity with L2 distrance. The problem still exists.

Tired.

### 2021.12.28

Plan B is to project feature map into semantic space, and semantic space are argumented by self attention module.

Codes are not completely to the reference paper, and the code are private. So I implemented by myself.

Result:

AP for aeroplane = 0.3687  
AP for bicycle = 0.6198  
AP for bird = 0.4262  
AP for boat = 0.2318  
AP for bottle = 0.3391  
AP for bus = 0.4852  
AP for car = 0.6773  
AP for cat = 0.5676  
AP for chair = 0.2627  
AP for cow = 0.5325  
AP for diningtable = 0.3137  
AP for dog = 0.5510  
AP for horse = 0.6349  
AP for motorbike = 0.6137  
AP for person = 0.6086  
AP for pottedplant = 0.2170  
AP for sheep = 0.5128  
AP for sofa = 0.3862  
AP for train = 0.4454  
AP for tvmonitor = 0.5008  
Mean AP = 0.4647  

It still has performance drop.

### 2021.12.27

* Change Loss from Smooth L1 to MSE.
* Add more layers after ROI Pooling.

Is nn functional must be initialized in __init__()? Maybe this is another bug before.

This time I use nn.MSELoss and initialized to be an object.

God bless this code...

AP for aeroplane = 0.0000  
AP for bicycle = 0.0047  
AP for bird = 0.0085  
AP for boat = 0.0001  
AP for bottle = 0.0000  
AP for bus = 0.0000  
AP for car = 0.0134  
AP for cat = 0.0111  
AP for chair = 0.0019  
AP for cow = 0.0296  
AP for diningtable = 0.0012  
AP for dog = 0.0215  
AP for horse = 0.0238  
AP for motorbike = 0.0119  
AP for person = 0.4322  
AP for pottedplant = 0.0000  
AP for sheep = 0.0114  
AP for sofa = 0.0066  
AP for train = 0.0000  
AP for tvmonitor = 0.0034  
Mean AP = 0.0291  

Here I guess I might change the method of getting acc. Cosine similarity may not be good enough.

After studying mAP calculation, the problem is that most ROIs have higgest acc on background so that these data will not be feed into mAP calculation.  
There are several possible problems:  
* High background/background rate(But why it didn't affect one hot encoding).  
* Cosine similarity.  
* Mapping layers from feature map to semantic vector are not deep enough(Now deepest is 4).  

Next step is to try Plan B, and ask tutor for help.

---

### 2021.12.25

Merry Christmas!

Result after debugging twice:  
AP for aeroplane = 0.0023  
AP for bicycle = 0.1045  
AP for bird = 0.0527  
AP for boat = 0.0021  
AP for bottle = 0.0161  
AP for bus = 0.0145  
AP for car = 0.3028  
AP for cat = 0.0893  
AP for chair = 0.1002  
AP for cow = 0.1342  
AP for diningtable = 0.0006  
AP for dog = 0.1770  
AP for horse = 0.0709  
AP for motorbike = 0.0476  
AP for person = 0.4382  
AP for pottedplant = 0.0026  
AP for sheep = 0.1459  
AP for sofa = 0.0108  
AP for train = 0.0136  
AP for tvmonitor = 0.0456  
Mean AP = 0.0886  

However, background it still have higgest probability.

---

### 2021.12.24

Christmas Eve, Starbucks chocolate drinks from zy, thx.

* A bug was found in faster_rcnn.py, line 68.
* Word vectors are generated from index of label words, not the word itself.
* Code in test_net.py also made this mistake, fixed.
* Bug fixed and retraining networks with 'true' semantic labels...

Result after debugging once:

AP for aeroplane = 0.0000  
AP for bicycle = 0.0002  
AP for bird = 0.0000  
AP for boat = 0.0000  
AP for bottle = 0.0000  
AP for bus = 0.0000  
AP for car = 0.0000  
AP for cat = 0.0007  
AP for chair = 0.0000  
AP for cow = 0.0000  
AP for diningtable = 0.0000  
AP for dog = 0.0303  
AP for horse = 0.0000  
AP for motorbike = 0.0002  
AP for person = 0.0014  
AP for pottedplant = 0.0000  
AP for sheep = 0.0000  
AP for sofa = 0.0001  
AP for train = 0.0000  
AP for tvmonitor = 0.0002  
Mean AP = 0.0017  

册那

The problem is column 0 have biggest value in each line after calculating cosine similarity, maybe some bugs are hidding in nets.

* More bugs fixed, a line of weight init code in faster-rnn.py was found and modified.
* Deleted reduce dim layer from 4 to 1 directly, and add tanh after that layer.
* Separated se and bl codes for easy training and testing.
* Typeset this page.

---

### 2021.12.23

BaseLine:

AP for aeroplane = 0.5207  
AP for bicycle = 0.7173  
AP for bird = 0.5351  
AP for boat = 0.3100  
AP for bottle = 0.3818  
AP for bus = 0.6089  
AP for car = 0.7326  
AP for cat = 0.6954  
AP for chair = 0.3188  
AP for cow = 0.6530  
AP for diningtable = 0.3305  
AP for dog = 0.6586  
AP for horse = 0.7295  
AP for motorbike = 0.7040  
AP for person = 0.6577  
AP for pottedplant = 0.2458  
AP for sheep = 0.6090  
AP for sofa = 0.5029  
AP for train = 0.5915  
AP for tvmonitor = 0.5937  
Mean AP = 0.5548  

With Semantic Information:

AP for aeroplane = 0.0017  
AP for bicycle = 0.1123  
AP for bird = 0.0058  
AP for boat = 0.0010  
AP for bottle = 0.0021  
AP for bus = 0.0033  
AP for car = 0.0183  
AP for cat = 0.0064  
AP for chair = 0.0182  
AP for cow = 0.0181  
AP for diningtable = 0.0000  
AP for dog = 0.0055  
AP for horse = 0.0285  
AP for motorbike = 0.0540  
AP for person = 0.1619  
AP for pottedplant = 0.0003  
AP for sheep = 0.0065  
AP for sofa = 0.0000  
AP for train = 0.0003  
AP for tvmonitor = 0.0000  
Mean AP = 0.0222  

Maybe some bugs in codes.

---

### 2021.12.22

* Convert semantic vector to acc completed.
* Before feeding data into mAP calculation module, calculate cosine similarity as acc directly. So we don't need modifiy mAP calculation module.

---

### 2021.12.21

* Modified RCNN-cls-loss to L2, adding semantic embedding query module.
* Mapping Feature Map to Semantic Vector, and calculate loss with true semantic vectors.

Now using small dataset with only 1000 images. Do remember **DELETE** pkl files after training in:

>'$Project_path/data/cache/'
>
>'$dataset_path/ImageSets/Main'

Next step is to overwrite mAP calculation in test phase.

---

### 2021.12.20

New Faster-R-CNN from: https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0

* Modified classification layer in 'faster-rcnn.pytorch-pytorch-1.0/lib/model/faster_rcnn/vgg16.py', line 50.
* Batch size of classification layer could be changed in this project, but we keep 1 as batch size.

Now, two questions are listed as follow:

1. Is there any difference between projecting feature map into semantic space and mapping into same dimension?

2. Projection are same to referenced paper, mapping are to complicated?

~~Besides, RTX 2060 have not enough memory for Faster-R-CNN, servers are needed now.~~

Size of input images can be reshaped (File ‘faster-rcnn.pytorch-pytorch-1.0/lib/model/utils/config.py’, line 63).

---

### 2021.12.1

* Raw word embedding tensor generation module finished.
* Interface on different dimision embeddings finished.

* Modified three label words so as to make sure that those words can be vectorized.

>diningtable -> table
>
>pottedplant -> plant
>
>tvmonitor -> monitor

Downloading 300d and 500d pretrained embeddings.

~~Next step is debugging with voc2012 dataset. Remember that the classification layer was modified back to 512x1024, but input data are still 512x512.~~

Now the shape of each tensor have been reshaped properly. But the model is loading the whole weights.  
Remove the weights, or modify the shape by adding several layers.

---

### 2021.11.29

* Modified Faster-R-CNN module and added Self Attention module into classification subnet.

Downloading Word2Vec pretrained embeddings...100 dims.

---

### 2021.11.26

* Self Attention module was built.

Do take care of the size of input and output tensors.

---

### 2021.11.25

Giving up using Faster-R-CNN from torchvision because it is difficult to override.

Now using Faster-R-CNN from : https://github.com/lllsaaxx/Faster-RCNN

---

### 2021.11.22

* Experiment environment have been built on laptop, NOT server.

System information:

>System: Ubuntu 20.04.3 LTS
>
>NVIDIA-SMI 460.91.03  Driver Version: 460.91.03  CUDA Version: 11.2
>
>Cuda compilation tools, release 11.2, V11.2.67
>
>Build cuda_11.2.r11.2/compiler.29373293_0

Pytorch was instlled by this command:  
`pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html`

See this page in detail: https://medium.com/analytics-vidhya/install-cuda-11-2-cudnn-8-1-0-and-python-3-9-on-rtx3090-for-deep-learning-fcf96c95f7a1
