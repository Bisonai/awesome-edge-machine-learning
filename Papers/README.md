# Papers
[Back to awesome edge machine learning](https://github.com/bisonai/awesome-edge-machine-learning)

## AutoML

Automated machine learning (AutoML) is the process of automating the end-to-end process of applying machine learning to real-world problems.<sup><a href="https://en.wikipedia.org/wiki/Automated_machine_learning" targe="_blank">Wikipedia</a></sup> AutoML is for example used to design new efficient neural architectures with a constraint on a computational budget (defined either as a number of FLOPS or as an inference time measured on real device) or a size of the architecture.

- [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/abs/1812.00332). Han Cai, Ligeng Zhu, Song Han

- [MnasNet: Platform-Aware Neural Architecture Search for Mobile](https://arxiv.org/abs/1807.11626). Mingxing Tan, Bo Chen, Ruoming Pang, Vijay Vasudevan, Mark Sandler, Andrew Howard, Quoc V. Le

- [NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications](https://arxiv.org/abs/1804.03230). Tien-Ju Yang, Andrew Howard, Bo Chen, Xiao Zhang, Alec Go, Mark Sandler, Vivienne Sze, Hartwig Adam

- [AMC: AutoML for Model Compression and Acceleration on Mobile Devices](https://arxiv.org/abs/1802.03494). Yihui He, Ji Lin, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han

- [MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks](https://arxiv.org/abs/1711.06798). Ariel Gordon, Elad Eban, Ofir Nachum, Bo Chen, Hao Wu, Tien-Ju Yang, Edward Choi

## Efficient Architectures

Efficient architectures represent neural networks with small memory footprint and fast inference time when measured on edge devices.

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946). Mingxing Tan, Quoc V. Le

- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244). Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam

- [ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network](https://arxiv.org/abs/1811.11431). Sachin Mehta, Mohammad Rastegari, Linda Shapiro, Hannaneh Hajishirzi

- [ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation](https://arxiv.org/abs/1803.06815). Sachin Mehta, Mohammad Rastegari, Anat Caspi, Linda Shapiro, Hannaneh Hajishirzi

- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381). Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

- [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083). Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun

- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861). Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam

- [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360). Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer

## Federated Learning

Federated Learning enables mobile phones to collaboratively learn a shared prediction model while keeping all the training data on device, decoupling the ability to do machine learning from the need to store the data in the cloud.<sup><a href="https://ai.googleblog.com/2017/04/federated-learning-collaborative.html" target="_blank">Google AI blog: Federated Learning</a></sup>

- [Towards Federated Learning at Scale: System Design](https://arxiv.org/abs/1902.01046). Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba, Alex Ingerman, Vladimir Ivanov, Chloe Kiddon, Jakub Konečný, Stefano Mazzocchi, H. Brendan McMahan, Timon Van Overveldt, David Petrou, Daniel Ramage, Jason Roselander

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629). H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas

## Pruning

Pruning is a common method to derive a compact network – after training, some structural portion of the parameters is removed, along with its associated computations.<sup><a href="http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf" target="_blank">Importance Estimation for Neural Network Pruning</a></sup>

- [Importance Estimation for Neural Network Pruning](http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf). Pavlo Molchanov, Arun Mallya, Stephen Tyree, Iuri Frosio, Jan Kautz

- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635). Jonathan Frankle, Michael Carbin

## Quantization

Quantization is the process of reducing a precision (from 32 bit floating point into lower bit depth representations) of weights and/or activations in a neural network. The advantages of this method are reduced model size and faster model inference on hardware that support arithmetic operations in lower precision.

- [Data-Free Quantization through Weight Equalization and Bias Correction](https://arxiv.org/abs/1906.04721). Markus Nagel, Mart van Baalen, Tijmen Blankevoort, Max Welling

- [Quantizing deep convolutional networks for efficient inference: A whitepaper](https://arxiv.org/abs/1806.08342). Raghuraman Krishnamoorthi

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877). Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko

- [Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations](https://arxiv.org/abs/1609.07061). Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, Yoshua Bengio

- [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160). Shuchang Zhou, Yuxin Wu, Zekun Ni, Xinyu Zhou, He Wen, Yuheng Zou

- [Quantized Convolutional Neural Networks for Mobile Devices](https://arxiv.org/abs/1512.06473). Jiaxiang Wu, Cong Leng, Yuhang Wang, Qinghao Hu, Jian Cheng

