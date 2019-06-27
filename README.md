# Awesome Edge Machine Learning
[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)

A curated list of awesome edge machine learning resources, including research papers, inference engines, challenges, books, meetups and others.

## Table of Contents
- [Papers](https://github.com/bisonai/awesome-edge-machine-learning#papers)
	- [AutoML](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/AutoML)
	- [Efficient Architectures](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/Efficient_Architectures)
	- [Pruning](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/Pruning)
	- [Quantization](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/Quantization)
- [Inference Engines](https://github.com/bisonai/awesome-edge-machine-learning#inference-engines)
- [Books](https://github.com/bisonai/awesome-edge-machine-learning#books)
- [Challenges](https://github.com/bisonai/awesome-edge-machine-learning#challenges)
- [Other Resources](https://github.com/bisonai/awesome-edge-machine-learning#other-resources)
- [Contribute](https://github.com/bisonai/awesome-edge-machine-learning#contribute)
- [License](https://github.com/bisonai/awesome-edge-machine-learning#license)

## Papers
### [AutoML](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/AutoML)
Automated machine learning (AutoML) is the process of automating the end-to-end process of applying machine learning to real-world problems<sup><a href="https://en.wikipedia.org/wiki/Automated_machine_learning" targe="_blank">Wikipedia</a></sup>. AutoML is for example used to design new efficient neural architectures with a constraint on a computational budget (defined either as a number of FLOPS or as an inference time measured on real device) or a size of the architecture.


### [Efficient Architectures](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/Efficient_Architectures)
Efficient architectures represent neural networks with small memory footprint and fast inference time when measured on edge devices.


### [Pruning](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/Pruning)
Pruning is a common method to derive a compact network – after training, some structural portion of the parameters is removed, along with its associated computations<sup><a href="http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf" target="_blank">Importance Estimation for Neural Network Pruning</a></sup>.


### [Quantization](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/Quantization)
Quantization is the process of reducing a precision (from 32 bit floating point into lower bit depth representations) of weights and/or activations in a neural network. The advantages of this method are reduced model size and faster model inference on hardware that support arithmetic operations in lower precision.


## Inference Engines
List of machine learning inference engines and APIs that are optimized for execution and/or training on edge devices.

### Caffe 2
- Source code: [https://github.com/pytorch/pytorch/tree/master/caffe2](https://github.com/pytorch/pytorch/tree/master/caffe2)
- Documentation: [https://caffe2.ai/](https://caffe2.ai/)
- Facebook

### CoreML
- Documentation: [https://developer.apple.com/documentation/coreml](https://developer.apple.com/documentation/coreml)
- Apple

### Deeplearning4j
- Documentation: [https://deeplearning4j.org/docs/latest/deeplearning4j-android](https://deeplearning4j.org/docs/latest/deeplearning4j-android)
- Skymind

### Feather CNN
- Source code: [https://github.com/Tencent/FeatherCNN](https://github.com/Tencent/FeatherCNN)
- Tencent

### MNN
- Source code: [https://github.com/alibaba/MNN](https://github.com/alibaba/MNN)
- Alibaba

### MXNet
- Documentation: [https://mxnet.incubator.apache.org/versions/master/faq/smart_device.html](https://mxnet.incubator.apache.org/versions/master/faq/smart_device.html)
- Amazon

### NCNN
- Source code: [https://github.com/tencent/ncnn](https://github.com/tencent/ncnn)
- Tencent

### Neural Networks API
- Documentation: [https://developer.android.com/ndk/guides/neuralnetworks/](https://developer.android.com/ndk/guides/neuralnetworks/)
- Google

### Paddle Mobile
- Source code: [https://github.com/PaddlePaddle/paddle-mobile](https://github.com/PaddlePaddle/paddle-mobile)
- Baidu

### TensorFlow Lite
- Source code: [https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite)
- Documentation: [https://www.tensorflow.org/lite/](https://www.tensorflow.org/lite/)
- Google

## Books
List of books with focus on on-device (e.g., edge or mobile) machine learning.

### [TinyML: Machine Learning with TensorFlow on Arduino, and Ultra-Low Power Micro-Controllers](http://shop.oreilly.com/product/0636920254508.do)
- Authors: Pete Warden, Daniel Situnayake
- Published: 2020

### [Machine Learning by Tutorials: Beginning machine learning for Apple and iOS](https://store.raywenderlich.com/products/machine-learning-by-tutorials)
- Author: Matthijs Hollemans
- Published: 2019

### [Core ML Survival Guide](https://leanpub.com/coreml-survival-guide)
- Author: Matthijs Hollemans
- Published: 2018

## Challenges
### [Low Power Recognition Challenge (LPIRC)](https://rebootingcomputing.ieee.org/lpirc)
Competition with focus on the best vision solutions that can simultaneously achieve high accuracy in computer vision and energy efficiency. LPIRC is regularly held during computer vision conferences (CVPR, ICCV and others) since 2015 and the winners’ solutions have already improved 24 times in the ratio of accuracy divided by energy.

- [Online Track](https://rebootingcomputing.ieee.org/lpirc/online-track)

- [Onsite Track](https://rebootingcomputing.ieee.org/lpirc/onsite-track)


## Other Resources
### [Awesome Mobile Machine Learning](https://github.com/fritzlabs/Awesome-Mobile-Machine-Learning)

A curated list of awesome mobile machine learning resources for iOS, Android, and edge devices.

### [Machine Think](https://machinethink.net/)

Machine learning tutorials targeted for iOS devices.

### [Pete Warden's blog](https://petewarden.com/)



## Contribute
Unlike other awesome list, we are storing data in <a href="https://en.wikipedia.org/wiki/YAML">YAML</a> format and markdown files are generated with `awesome.py` script.

Every directory contains `data.yaml` which stores data we want to display and `config.yaml` which stores its metadata (e.g. way of sorting data). The way how data will be presented is defined in `renderer.py`.


## License
[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Bisonai](https://bisonai.com/) has waived all copyright and related or neighboring rights to this work.