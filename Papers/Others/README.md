# Others
[Back to awesome edge machine learning](https://github.com/bisonai/awesome-edge-machine-learning)

[Back to Papers](https://github.com/bisonai/awesome-edge-machine-learning/tree/master/Papers)

This section contains papers that are related to edge machine learning but are not part of any major group. These papers often deal with deployment issues (i.e. optimizing inference on target platform).


## [On-Device Neural Net Inference with Mobile GPUs](https://arxiv.org/abs/1907.01989), 2019/07
Juhyun Lee, Nikolay Chirkov, Ekaterina Ignasheva, Yury Pisarchyk, Mogan Shieh, Fabio Riccardi, Raman Sarokin, Andrei Kulik, Matthias Grundmann

On-device inference of machine learning models for mobile phones is desirable due to its lower latency and increased privacy. Running such a compute-intensive task solely on the mobile CPU, however, can be difficult due to limited computing power, thermal constraints, and energy consumption. App developers and researchers have begun exploiting hardware accelerators to overcome these challenges. Recently, device manufacturers are adding neural processing units into high-end phones for on-device inference, but these account for only a small fraction of hand-held devices. In this paper, we present how we leverage the mobile GPU, a ubiquitous hardware accelerator on virtually every phone, to run inference of deep neural networks in real-time for both Android and iOS devices. By describing our architecture, we also discuss how to design networks that are mobile GPU-friendly. Our state-of-the-art mobile GPU inference engine is integrated into the open-source project TensorFlow Lite and publicly available at [https://tensorflow.org/lite](https://tensorflow.org/lite).


## [Deep Learning on Mobile Devices - A Review](https://arxiv.org/abs/1904.09274), 2019/03
Yunbin Deng

Recent breakthroughs in deep learning and artificial intelligence technologies have enabled numerous mobile applications. While traditional computation paradigms rely on mobile sensing and cloud computing, deep learning implemented on mobile devices provides several advantages. These advantages include low communication bandwidth, small cloud computing resource cost, quick response time, and improved data privacy. Research and development of deep learning on mobile and embedded devices has recently attracted much attention. This paper provides a timely review of this fast-paced field to give the researcher, engineer, practitioner, and graduate student a quick grasp on the recent advancements of deep learning on mobile devices. In this paper, we discuss hardware architectures for mobile deep learning, including Field Programmable Gate Arrays, Application Specific Integrated Circuit, and recent mobile Graphic Processing Units. We present Size, Weight, Area and Power considerations and their relation to algorithm optimizations, such as quantization, pruning, compression, and approximations that simplify computation while retaining performance accuracy. We cover existing systems and give a state-of-the-industry review of TensorFlow, MXNet, Mobile AI Compute Engine, and Paddle-mobile deep learning platform. We discuss resources for mobile deep learning practitioners, including tools, libraries, models, and performance benchmarks. We present applications of various mobile sensing modalities to industries, ranging from robotics, healthcare and multi-media, biometrics to autonomous drive and defense. We address the key deep learning challenges to overcome, including low quality data, and small training/adaptation data sets. In addition, the review provides numerous citations and links to existing code bases implementing various technologies.


## [Machine Learning at Facebook:Understanding Inference at the Edge](https://research.fb.com/wp-content/uploads/2018/12/Machine-Learning-at-Facebook-Understanding-Inference-at-the-Edge.pdf), 2018/12
Carole-Jean Wu, David Brooks, Kevin Chen, Douglas Chen, Sy Choudhury, Marat Dukhan,Kim Hazelwood, Eldad Isaac, Yangqing Jia, Bill Jia, Tommer Leyvand, Hao Lu, Yang Lu, Lin Qiao,Brandon Reagen, Joe Spisak, Fei Sun, Andrew Tulloch, Peter Vajda, Xiaodong Wang,Yanghan Wang, Bram Wasti, Yiming Wu, Ran Xian, Sungjoo Yoo, Peizhao Zhang

At Facebook, machine learning provides a wide range ofcapabilities that drive many aspects of user experienceincluding ranking posts, content understanding, objectdetection and tracking for augmented and virtual real-ity, speech and text translations.  While machine learn-ing  models  are  currently  trained  on  customized  data-center infrastructure, Facebook is working to bring ma-chine learning inference to the edge.  By doing so, userexperience is improved with reduced latency (inferencetime) and becomes less dependent on network connec-tivity.  Furthermore, this also enables many more appli-cations  of  deep  learning  with  important  features  onlymade available at the edge.  This paper takes a data-driven  approach  to  present  the  opportunities  and  de-sign  challenges  faced  by  Facebook  in  order  to  enablemachine learning inference locally on smartphones andother edge platforms.


