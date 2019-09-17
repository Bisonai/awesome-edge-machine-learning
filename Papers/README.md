# Papers
[Back to awesome edge machine learning](https://github.com/bisonai/awesome-edge-machine-learning)

## Applications

There is a countless number of possible edge machine learning applications. Here, we collect papers that describe specific solutions.

- [Temporal Convolution for Real-time Keyword Spotting on Mobile Devices](https://arxiv.org/abs/1904.03814). Seungwoo Choi, Seokjun Seo, Beomjun Shin, Hyeongmin Byun, Martin Kersner, Beomsu Kim, Dongyoung Kim, Sungjoo Ha

- [Towards Real-Time Automatic Portrait Matting on Mobile Devices](https://arxiv.org/abs/1904.03816). Seokjun Seo, Seungwoo Choi, Martin Kersner, Beomjun Shin, Hyungsuk Yoon, Hyeongmin Byun, Sungjoo Ha

- [PFLD: A Practical Facial Landmark Detector](https://arxiv.org/abs/1902.10859). Xiaojie Guo, Siyuan Li, Jinke Yu, Jiawan Zhang, Jiayi Ma, Lin Ma, Wei Liu, Haibin Ling

## AutoML

Automated machine learning (AutoML) is the process of automating the end-to-end process of applying machine learning to real-world problems.<sup><a href="https://en.wikipedia.org/wiki/Automated_machine_learning" targe="_blank">Wikipedia</a></sup> AutoML is for example used to design new efficient neural architectures with a constraint on a computational budget (defined either as a number of FLOPS or as an inference time measured on real device) or a size of the architecture.

- [ChamNet: Towards Efficient Network Design through Platform-Aware Model Adaptation](https://arxiv.org/abs/1812.08934). Xiaoliang Dai, Peizhao Zhang, Bichen Wu, Hongxu Yin, Fei Sun, Yanghan Wang, Marat Dukhan, Yunqing Hu, Yiming Wu, Yangqing Jia, Peter Vajda, Matt Uyttendaele, Niraj K. Jha

- [FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search](https://arxiv.org/abs/1812.03443). Bichen Wu, Xiaoliang Dai, Peizhao Zhang, Yanghan Wang, Fei Sun, Yiming Wu, Yuandong Tian, Peter Vajda, Yangqing Jia, Kurt Keutzer

- [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/abs/1812.00332). Han Cai, Ligeng Zhu, Song Han

- [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626). Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, Quoc V. Le

- [NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications](https://arxiv.org/abs/1804.03230). Tien-Ju Yang, Andrew Howard, Bo Chen, Xiao Zhang, Alec Go, Mark Sandler, Vivienne Sze, Hartwig Adam

- [AMC: AutoML for Model Compression and Acceleration on Mobile Devices](https://arxiv.org/abs/1802.03494). Yihui He, Ji Lin, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han

- [MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks](https://arxiv.org/abs/1711.06798). Ariel Gordon, Elad Eban, Ofir Nachum, Bo Chen, Hao Wu, Tien-Ju Yang, Edward Choi

## Efficient Architectures

Efficient architectures represent neural networks with small memory footprint and fast inference time when measured on edge devices.

- [MixNet: Mixed Depthwise Convolutional Kernels](https://arxiv.org/abs/1907.09595). Mingxing Tan, Quoc V. Le

- [Efficient On-Device Models using Neural Projections](http://proceedings.mlr.press/v97/ravi19a/ravi19a.pdf). Sujith Ravi

- [Butterfly Transform: An Efficient FFT Based Neural Architecture Design](https://arxiv.org/abs/1906.02256). Keivan Alizadeh, Ali Farhadi, Mohammad Rastegari

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946). Mingxing Tan, Quoc V. Le

- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244). Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam

- [ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network](https://arxiv.org/abs/1811.11431). Sachin Mehta, Mohammad Rastegari, Linda Shapiro, Hannaneh Hajishirzi

- [ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation](https://arxiv.org/abs/1803.06815). Sachin Mehta, Mohammad Rastegari, Anat Caspi, Linda Shapiro, Hannaneh Hajishirzi

- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381). Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

- [CondenseNet: An Efficient DenseNet using Learned Group Convolutions](https://arxiv.org/abs/1711.09224). Gao Huang, Shichen Liu, Laurens van der Maaten, Kilian Q. Weinberger

- [DeepRebirth: Accelerating Deep Neural Network Execution on Mobile Devices](https://arxiv.org/abs/1708.04728). Dawei Li, Xiaolong Wang, Deguang Kong

- [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083). Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun

- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861). Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam

- [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360). Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer

## Federated Learning

Federated Learning enables mobile phones to collaboratively learn a shared prediction model while keeping all the training data on device, decoupling the ability to do machine learning from the need to store the data in the cloud.<sup><a href="https://ai.googleblog.com/2017/04/federated-learning-collaborative.html" target="_blank">Google AI blog: Federated Learning</a></sup>

- [Towards Federated Learning at Scale: System Design](https://arxiv.org/abs/1902.01046). Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex Ingerman, Vladimir Ivanov, Chloe Kiddon, Jakub Konečný, Stefano Mazzocchi, H. Brendan McMahan, Timon Van Overveldt, David Petrou, Daniel Ramage, Jason Roselander

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629). H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas

## Network Pruning

Pruning is a common method to derive a compact network – after training, some structural portion of the parameters is removed, along with its associated computations.<sup><a href="http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf" target="_blank">Importance Estimation for Neural Network Pruning</a></sup>

- [Importance Estimation for Neural Network Pruning](http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf). Pavlo Molchanov, Arun Mallya, Stephen Tyree, Iuri Frosio, Jan Kautz

- [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](https://arxiv.org/abs/1904.03837). Xiaohan Ding, Guiguang Ding, Yuchen Guo, Jungong Han

- [Towards Optimal Structured CNN Pruning via Generative Adversarial Learning](https://arxiv.org/abs/1903.09291). Shaohui Lin, Rongrong Ji, Chenqian Yan, Baochang Zhang, Liujuan Cao, Qixiang Ye, Feiyue Huang, David Doermann

- [Variational Convolutional Neural Network Pruning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Variational_Convolutional_Neural_Network_Pruning_CVPR_2019_paper.pdf). Chenglong Zhao, Bingbing Ni, Jian Zhang, Qiwei Zhao, Wenjun Zhang, Qi Tian

- [On Implicit Filter Level Sparsity in Convolutional Neural Networks](https://arxiv.org/abs/1811.12495). Dushyant Mehta, Kwang In Kim, Christian Theobalt

- [Structured Pruning of Neural Networks with Budget-Aware Regularization](https://arxiv.org/abs/1811.09332). Carl Lemaire, Andrew Achkar, Pierre-Marc Jodoin

- [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://arxiv.org/abs/1811.00250). Yang He, Ping Liu, Ziwei Wang, Zhilan Hu, Yi Yang

- [Discrimination-aware Channel Pruning for Deep Neural Networks](https://arxiv.org/abs/1810.11809). Zhuangwei Zhuang, Mingkui Tan, Bohan Zhuang, Jing Liu, Yong Guo, Qingyao Wu, Junzhou Huang, Jinhui Zhu

- [Rethinking the Value of Network Pruning](https://arxiv.org/abs/1810.05270). Zhuang Liu, Mingjie Sun, Tinghui Zhou, Gao Huang, Trevor Darrell

- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635). Jonathan Frankle, Michael Carbin

- [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878). Michael Zhu, Suyog Gupta

- [Channel Pruning for Accelerating Very Deep Neural Networks](https://arxiv.org/abs/1707.06168). Yihui He, Xiangyu Zhang, Jian Sun

## Others

This section contains papers that are related to edge machine learning but are not part of any major group. These papers often deal with deployment issues (i.e. optimizing inference on target platform).

- [On-Device Neural Net Inference with Mobile GPUs](https://arxiv.org/abs/1907.01989). Juhyun Lee, Nikolay Chirkov, Ekaterina Ignasheva, Yury Pisarchyk, Mogan Shieh, Fabio Riccardi, Raman Sarokin, Andrei Kulik, Matthias Grundmann

- [Deep Learning on Mobile Devices - A Review](https://arxiv.org/abs/1904.09274). Yunbin Deng

- [Machine Learning at Facebook:Understanding Inference at the Edge](https://research.fb.com/wp-content/uploads/2018/12/Machine-Learning-at-Facebook-Understanding-Inference-at-the-Edge.pdf). Carole-Jean Wu, David Brooks, Kevin Chen, Douglas Chen, Sy Choudhury, Marat Dukhan,Kim Hazelwood, Eldad Isaac, Yangqing Jia, Bill Jia, Tommer Leyvand, Hao Lu, Yang Lu, Lin Qiao,Brandon Reagen, Joe Spisak, Fei Sun, Andrew Tulloch, Peter Vajda, Xiaodong Wang,Yanghan Wang, Bram Wasti, Yiming Wu, Ran Xian, Sungjoo Yoo, Peizhao Zhang

## Quantization

Quantization is the process of reducing a precision (from 32 bit floating point into lower bit depth representations) of weights and/or activations in a neural network. The advantages of this method are reduced model size and faster model inference on hardware that support arithmetic operations in lower precision.

- [And the Bit Goes Down: Revisiting the Quantization of Neural Networks](https://arxiv.org/abs/1907.05686). Pierre Stock, Armand Joulin, Rémi Gribonval, Benjamin Graham, Hervé Jégou

- [Data-Free Quantization through Weight Equalization and Bias Correction](https://arxiv.org/abs/1906.04721). Markus Nagel, Mart van Baalen, Tijmen Blankevoort, Max Welling

- [Quantizing deep convolutional networks for efficient inference: A whitepaper](https://arxiv.org/abs/1806.08342). Raghuraman Krishnamoorthi

- [A Quantization-Friendly Separable Convolution for MobileNets](https://arxiv.org/abs/1803.08607). Tao Sheng, Chen Feng, Shaojie Zhuo, Xiaopeng Zhang, Liang Shen, Mickey Aleksic

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877). Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko

- [Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations](https://arxiv.org/abs/1609.07061). Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, Yoshua Bengio

- [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160). Shuchang Zhou, Yuxin Wu, Zekun Ni, Xinyu Zhou, He Wen, Yuheng Zou

- [Quantized Convolutional Neural Networks for Mobile Devices](https://arxiv.org/abs/1512.06473). Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, Jian Cheng

