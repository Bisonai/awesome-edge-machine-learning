# ML Algorithms For Edge
[Back to awesome edge machine learning](https://github.com/bisonai/awesome-edge-machine-learning)

[Back to Papers](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers)

Standard machine learning algorithms are not always able to run on edge devices due to large computational requirements and space complexity. This section introduces optimized machine learning algorithms.


## [Shallow RNNs: A Method for Accurate Time-series Classification on Tiny Devices](https://dkdennis.xyz/static/sharnn-neurips19-paper.pdf), 2019/12
Don Dennis, Durmus Alp Emre Acar, Vikram Mandikal, Vinu Sankar Sadasivan, Harsha Vardhan Simhadri, Venkatesh Saligrama, Prateek Jain

Recurrent Neural Networks (RNNs) capture long dependencies and context, and hence are the key component of typical sequential data based tasks. However, the sequential nature of RNNs dictates a large inference cost for long sequences even if the hardware supports parallelization. To induce long-term dependencies, and yet admit parallelization, we introduce novel shallow RNNs. In this architecture, the first layer splits the input sequence and runs several independent RNNs. The second layer consumes the output of the first layer using a second RNN thus capturing long dependencies. We provide theoretical justification for our architecture under weak assumptions that we verify on real-world benchmarks. Furthermore, we show that for time-series classification, our technique leads to substantially improved inference time over standard RNNs without compromising accuracy. For example, we can deploy audio-keyword classification on tiny Cortex M4 devices (100MHz processor, 256KB RAM, no DSP available) which was not possible using standard RNN models. Similarly, using ShaRNN in the popular Listen-Attend-Spell (LAS) architecture for phoneme classification [4], we can reduce the lag in phoneme classification by 10-12x while maintaining state-of-the-art accuracy.


## [ProtoNN: Compressed and Accurate kNN for Resource-scarce Devices](http://proceedings.mlr.press/v70/gupta17a.html), 2017/08
Chirag Gupta, Arun Sai Suggala, Ankit Goyal, Harsha Vardhan Simhadri, Bhargavi Paranjape, Ashish Kumar, Saurabh Goyal, Raghavendra Udupa, Manik Varma, Prateek Jain

Several real-world applications require real-time prediction on resource-scarce devices such as an Internet of Things (IoT) sensor. Such applications demand prediction models with small storage and computational complexity that do not compromise significantly on accuracy. In this work, we propose ProtoNN, a novel algorithm that addresses the problem of real-time and accurate prediction on resource-scarce devices. ProtoNN is inspired by k-Nearest Neighbor (KNN) but has several orders lower storage and prediction complexity. ProtoNN models can be deployed even on devices with puny storage and computational power (e.g. an Arduino UNO with 2kB RAM) to get excellent prediction accuracy. ProtoNN derives its strength from three key ideas: a) learning a small number of prototypes to represent the entire training set, b) sparse low dimensional projection of data, c) joint discriminative learning of the projection and prototypes with explicit model size constraint. We conduct systematic empirical evaluation of ProtoNN on a variety of supervised learning tasks (binary, multi-class, multi-label classification) and show that it gives nearly state-of-the-art prediction accuracy on resource-scarce devices while consuming several orders lower storage, and using minimal working memory.


## [Resource-efficient Machine Learning in 2 KB RAM for the Internet of Things](http://proceedings.mlr.press/v70/kumar17a.html), 2017/08
Ashish Kumar, Saurabh Goyal, Manik Varma

This paper develops a novel tree-based algorithm, called Bonsai, for efficient prediction on IoT devices – such as those based on the Arduino Uno board having an 8 bit ATmega328P microcontroller operating at 16 MHz with no native floating point support, 2 KB RAM and 32 KB read-only flash. Bonsai maintains prediction accuracy while minimizing model size and prediction costs by: (a) developing a tree model which learns a single, shallow, sparse tree with powerful nodes; (b) sparsely projecting all data into a low-dimensional space in which the tree is learnt; and (c) jointly learning all tree and projection parameters. Experimental results on multiple benchmark datasets demonstrate that Bonsai can make predictions in milliseconds even on slow microcontrollers, can fit in KB of memory, has lower battery consumption than all other algorithms while achieving prediction accuracies that can be as much as 30\% higher than state-of-the-art methods for resource-efficient machine learning. Bonsai is also shown to generalize to other resource constrained settings beyond IoT by generating significantly better search results as compared to Bing’s L3 ranker when the model size is restricted to 300 bytes. Bonsai’s code can be downloaded from [http://www.manikvarma.org/code/Bonsai/download.html](http://www.manikvarma.org/code/Bonsai/download.html).


