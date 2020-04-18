# Awesome Edge Machine Learning
[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)

A curated list of awesome edge machine learning resources, including research papers, inference engines, challenges, books, meetups and others.

## Table of Contents
- [Papers](https://github.com/bisonai/awesome-edge-machine-learning#papers)
	- [Applications](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/Applications)
	- [AutoML](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/AutoML)
	- [Efficient Architectures](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/Efficient_Architectures)
	- [Federated Learning](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/Federated_Learning)
	- [ML Algorithms For Edge](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/ML_Algorithms_For_Edge)
	- [Network Pruning](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/Network_Pruning)
	- [Others](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/Others)
	- [Quantization](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/Quantization)
- [Datasets](https://github.com/bisonai/awesome-edge-machine-learning#datasets)
- [Inference Engines](https://github.com/bisonai/awesome-edge-machine-learning#inference-engines)
- [MCU and MPU Software Packages](https://github.com/bisonai/awesome-edge-machine-learning#mcu-and-mpu-software-packages)
- [Books](https://github.com/bisonai/awesome-edge-machine-learning#books)
- [Challenges](https://github.com/bisonai/awesome-edge-machine-learning#challenges)
- [Other Resources](https://github.com/bisonai/awesome-edge-machine-learning#other-resources)
- [Contribute](https://github.com/bisonai/awesome-edge-machine-learning#contribute)
- [License](https://github.com/bisonai/awesome-edge-machine-learning#license)

## Papers
### [Applications](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/Applications)
There is a countless number of possible edge machine learning applications. Here, we collect papers that describe specific solutions.


### [AutoML](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/AutoML)
Automated machine learning (AutoML) is the process of automating the end-to-end process of applying machine learning to real-world problems.<sup><a href="https://en.wikipedia.org/wiki/Automated_machine_learning" targe="_blank">Wikipedia</a></sup> AutoML is for example used to design new efficient neural architectures with a constraint on a computational budget (defined either as a number of FLOPS or as an inference time measured on real device) or a size of the architecture.


### [Efficient Architectures](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/Efficient_Architectures)
Efficient architectures represent neural networks with small memory footprint and fast inference time when measured on edge devices.


### [Federated Learning](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/Federated_Learning)
Federated Learning enables mobile phones to collaboratively learn a shared prediction model while keeping all the training data on device, decoupling the ability to do machine learning from the need to store the data in the cloud.<sup><a href="https://ai.googleblog.com/2017/04/federated-learning-collaborative.html" target="_blank">Google AI blog: Federated Learning</a></sup>


### [ML Algorithms For Edge](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/ML_Algorithms_For_Edge)
Standard machine learning algorithms are not always able to run on edge devices due to large computational requirements and space complexity. This section introduces optimized machine learning algorithms.


### [Network Pruning](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/Network_Pruning)
Pruning is a common method to derive a compact network – after training, some structural portion of the parameters is removed, along with its associated computations.<sup><a href="http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf" target="_blank">Importance Estimation for Neural Network Pruning</a></sup>


### [Others](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/Others)
This section contains papers that are related to edge machine learning but are not part of any major group. These papers often deal with deployment issues (i.e. optimizing inference on target platform).


### [Quantization](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers/Quantization)
Quantization is the process of reducing a precision (from 32 bit floating point into lower bit depth representations) of weights and/or activations in a neural network. The advantages of this method are reduced model size and faster model inference on hardware that support arithmetic operations in lower precision.


## Datasets
### [Visual Wake Words Dataset](https://arxiv.org/abs/1906.05721)
Visual Wake Words represents a common microcontroller vision use-case of identifying whether a person is present in the image or not, and provides a realistic benchmark for tiny vision models. Within a limited memory footprint of 250 KB, several state-of-the-art mobile models achieve accuracy of 85-90% on the Visual Wake Words dataset.


## Inference Engines
List of machine learning inference engines and APIs that are optimized for execution and/or training on edge devices.

### Arm Compute Library
- Source code: [https://github.com/ARM-software/ComputeLibrary](https://github.com/ARM-software/ComputeLibrary)
- Arm

### Bender
- Source code: [https://github.com/xmartlabs/Bender](https://github.com/xmartlabs/Bender)
- Documentation: [https://xmartlabs.github.io/Bender/](https://xmartlabs.github.io/Bender/)
- Xmartlabs

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

### Embedded Learning Library
- Source code: [https://github.com/Microsoft/ELL](https://github.com/Microsoft/ELL)
- Documentation: [https://microsoft.github.io/ELL](https://microsoft.github.io/ELL)
- Microsoft

### Feather CNN
- Source code: [https://github.com/Tencent/FeatherCNN](https://github.com/Tencent/FeatherCNN)
- Tencent

### MACE
- Source code: [https://github.com/XiaoMi/mace](https://github.com/XiaoMi/mace)
- Documentation: [https://mace.readthedocs.io/](https://mace.readthedocs.io/)
- XiaoMi

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

### Qualcomm Neural Processing SDK for AI
- Source code: [https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)
- Qualcomm

### Tengine
- Source code: [https://github.com/OAID/Tengine](https://github.com/OAID/Tengine)
- OAID

### TensorFlow Lite
- Source code: [https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite)
- Documentation: [https://www.tensorflow.org/lite/](https://www.tensorflow.org/lite/)
- Google

### dabnn
- Source code: [https://github.com/JDAI-CV/dabnn](https://github.com/JDAI-CV/dabnn)
- JDAI Computer Vision

## MCU and MPU Software Packages
List of software packages for AI development on MCU and MPU

### FP-AI-Sensing
- Company: STMicroelectronics
- [https://www.st.com/content/st_com/en/products/embedded-software/mcu-mpu-embedded-software/stm32-embedded-software/stm32-ode-function-pack-sw/fp-ai-sensing1.html](https://www.st.com/content/st_com/en/products/embedded-software/mcu-mpu-embedded-software/stm32-embedded-software/stm32-ode-function-pack-sw/fp-ai-sensing1.html)
STM32Cube function pack for ultra-low power IoT node with artificial intelligence (AI) application based on audio and motion sensing

### FP-AI-VISION1
- Company: STMicroelectronics
- [https://www.st.com/content/st_com/en/products/embedded-software/mcu-mpu-embedded-software/stm32-embedded-software/stm32cube-expansion-packages/fp-ai-vision1.html](https://www.st.com/content/st_com/en/products/embedded-software/mcu-mpu-embedded-software/stm32-embedded-software/stm32cube-expansion-packages/fp-ai-vision1.html)
FP-AI-VISION1 is an STM32Cube function pack featuring examples of computer vision applications based on Convolutional Neural Network (CNN)

### Processor SDK Linux for AM57x
- Company: Texas Instruments
- [www.ti.com/tool/SITARA-MACHINE-LEARNING](www.ti.com/tool/SITARA-MACHINE-LEARNING)
TIDL software framework leverages a highly optimized neural network implementation on TI’s Sitara AM57x processors, making use of hardware acceleration on the device

### X-LINUX-AI-CV
- Company: STMicroelectronics
- [https://www.st.com/content/st_com/en/products/embedded-software/mcu-mpu-embedded-software/stm32-embedded-software/stm32-mpu-openstlinux-expansion-packages/x-linux-ai-cv.html](https://www.st.com/content/st_com/en/products/embedded-software/mcu-mpu-embedded-software/stm32-embedded-software/stm32-mpu-openstlinux-expansion-packages/x-linux-ai-cv.html)
X-LINUX-AI-CV is an STM32 MPU OpenSTLinux Expansion Package that targets Artificial Intelligence for computer vision applications based on Convolutional Neural Network (CNN)

### e-AI Checker
- Company: Renesas
- [https://www.renesas.com/jp/en/solutions/key-technology/e-ai/tool.html](https://www.renesas.com/jp/en/solutions/key-technology/e-ai/tool.html)
Based on the output result from the translator, the ROM/RAM mounting size and the inference execution processing time are calculated while referring to the information of the selected MCU/MPU

### e-AI Translator
- Company: Renesas
- [https://www.renesas.com/jp/en/solutions/key-technology/e-ai/tool.html](https://www.renesas.com/jp/en/solutions/key-technology/e-ai/tool.html)
Tool for converting  Caffe and TensorFlow models to MCU/MPU development environment

### eIQ Auto deep learning (DL) toolkit
- Company: NXP
- [https://www.nxp.com/design/software/development-software/eiq-auto-dl-toolkit:EIQ-AUTO](https://www.nxp.com/design/software/development-software/eiq-auto-dl-toolkit:EIQ-AUTO)
The NXP eIQ™ Auto deep learning (DL) toolkit enables developers to introduce DL algorithms into their applications and to continue satisfying automotive standards

### eIQ ML Software Development Environment
- Company: NXP
- [https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ](https://www.nxp.com/design/software/development-software/eiq-ml-development-environment:EIQ)
The NXP® eIQ™ machine learning software development environment enables the use of ML algorithms on NXP MCUs, i.MX RT crossover MCUs, and i.MX family SoCs. eIQ software includes inference engines, neural network compilers and optimized libraries

### eIQ™ Software for Arm® NN Inference Engine
- Company: NXP
- [https://www.nxp.com/design/software/development-software/eiq-ml-development-environment/eiq-software-for-arm-nn-inference-engine:eIQArmNN](https://www.nxp.com/design/software/development-software/eiq-ml-development-environment/eiq-software-for-arm-nn-inference-engine:eIQArmNN)

### eIQ™ for Arm® CMSIS-NN
- Company: NXP
- [https://www.nxp.com/design/software/development-software/eiq-ml-development-environment/eiq-for-arm-cmsis-nn:eIQArmCMSISNN](https://www.nxp.com/design/software/development-software/eiq-ml-development-environment/eiq-for-arm-cmsis-nn:eIQArmCMSISNN)

### eIQ™ for Glow Neural Network Compiler
- Company: NXP
- [https://www.nxp.com/design/software/development-software/eiq-ml-development-environment/eiq-for-glow-neural-network-compiler:eIQ-Glow](https://www.nxp.com/design/software/development-software/eiq-ml-development-environment/eiq-for-glow-neural-network-compiler:eIQ-Glow)

### eIQ™ for TensorFlow Lite
- Company: NXP
- [https://www.nxp.com/design/software/development-software/eiq-ml-development-environment/eiq-for-tensorflow-lite:eIQTensorFlowLite](https://www.nxp.com/design/software/development-software/eiq-ml-development-environment/eiq-for-tensorflow-lite:eIQTensorFlowLite)

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

### [Building Mobile Applications with TensorFlow](https://www.oreilly.com/library/view/building-mobile-applications/9781491988435/)
- Author: Pete Warden
- Published: 2017

## Challenges
### [Low Power Recognition Challenge (LPIRC)](https://rebootingcomputing.ieee.org/lpirc)
Competition with focus on the best vision solutions that can simultaneously achieve high accuracy in computer vision and energy efficiency. LPIRC is regularly held during computer vision conferences (CVPR, ICCV and others) since 2015 and the winners’ solutions have already improved 24 times in the ratio of accuracy divided by energy.

- [Online Track](https://rebootingcomputing.ieee.org/lpirc/online-track)

- [Onsite Track](https://rebootingcomputing.ieee.org/lpirc/onsite-track)


## Other Resources
### [Awesome EMDL](https://github.com/EMDL/awesome-emdl)

Embedded and mobile deep learning research resources

### [Awesome Mobile Machine Learning](https://github.com/fritzlabs/Awesome-Mobile-Machine-Learning)

A curated list of awesome mobile machine learning resources for iOS, Android, and edge devices

### [Awesome Pruning](https://github.com/he-y/Awesome-Pruning)

A curated list of neural network pruning resources

### [Efficient DNNs](https://github.com/MingSun-Tse/EfficientDNNs)

Collection of recent methods on DNN compression and acceleration

### [Machine Think](https://machinethink.net/)

Machine learning tutorials targeted for iOS devices

### [Pete Warden's blog](https://petewarden.com/)



## Contribute
Unlike other awesome list, we are storing data in <a href="https://en.wikipedia.org/wiki/YAML">YAML</a> format and markdown files are generated with `awesome.py` script.

Every directory contains `data.yaml` which stores data we want to display and `config.yaml` which stores its metadata (e.g. way of sorting data). The way how data will be presented is defined in `renderer.py`.


## License
[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, [Bisonai](https://bisonai.com/) has waived all copyright and related or neighboring rights to this work.