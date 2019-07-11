# Others
[Back to awesome edge machine learning](https://github.com/bisonai/awesome-edge-machine-learning)

[Back to Papers](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers)

This section contains papers that are related to edge machine learning but are not part of any major group. These papers often deal with deployment issues (i.e. optimizing inference on target platform).


## [On-Device Neural Net Inference with Mobile GPUs](https://arxiv.org/abs/1907.01989), 2019/07
Juhyun Lee, Nikolay Chirkov, Ekaterina Ignasheva, Yury Pisarchyk, Mogan Shieh, Fabio Riccardi, Raman Sarokin, Andrei Kulik, Matthias Grundmann

On-device inference of machine learning models for mobile phones is desirable due to its lower latency and increased privacy. Running such a compute-intensive task solely on the mobile CPU, however, can be difficult due to limited computing power, thermal constraints, and energy consumption. App developers and researchers have begun exploiting hardware accelerators to overcome these challenges. Recently, device manufacturers are adding neural processing units into high-end phones for on-device inference, but these account for only a small fraction of hand-held devices. In this paper, we present how we leverage the mobile GPU, a ubiquitous hardware accelerator on virtually every phone, to run inference of deep neural networks in real-time for both Android and iOS devices. By describing our architecture, we also discuss how to design networks that are mobile GPU-friendly. Our state-of-the-art mobile GPU inference engine is integrated into the open-source project TensorFlow Lite and publicly available at this [https://tensorflow.org/lite](URL).


