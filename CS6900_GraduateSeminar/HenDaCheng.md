---
title: 'Dr. Heng-Da Cheng'
author: Philip Nelson
date: 9th September 2019
---

Dr. Heng-Da Cheng's major research areas are computer vision, pattern recognition, image processing, artificial intelligence, uncertainty, and fuzzy logic

Dr. Heng-Da Cheng mentioned two research projects I believe he and his group are working on are the following projects:

* Neutrosophic Logic to CVPRIP
* Short-tracking speed skating
* Biomedical Image processing
* Automatic vehicle classification
* Pavement defects detection and classification

Dr. Cheng also spoke about two goals in research. He said that you need to develop something **new** and **better**. He elaborated on these two topics in the following way:

* new
  1. theory
  2. algorithms
  3. applications
  4. architectures

* better
  1. mathematically
  2. experiment
  3. simulation


Dr. Cheng did not discuss his current projects more than a brief mention, however, I thought his work with fuzzy logic sounded interesting. I have never worked with this field and Dr. Cheng made fuzzy logic sound like it has applications in many areas of computer science as a tool to take a new approach to current problems.

Two researchers in the field of fuzzy logic and image processing are Dr. Claudia I. Gonzalez at Instituto Tecnologico de Tijuana and Dr. Phillip Isola at MIT.

Dr. Claudia Gonzalez published a paper entitled "Optimization of interval type-2 fuzzy systems for image edge detection" [https://www-sciencedirect-com.dist.lib.usu.edu/science/article/pii/S1568494614006425](https://www-sciencedirect-com.dist.lib.usu.edu/science/article/pii/S1568494614006425). This paper presents optimization of a fuzzy-logic edge detector which is based on a traditional Sobel edge detection technique in conjunction with interval type-2 fuzzy logic. Dr. Gonzalez' goal of using the interval type-2 fuzzy logic in her edge detection algorithm is to provide it with the ability to deal with uncertainty while processing real images. The optimal design of fuzzy logic system, however, is a difficult task and that is why using meta-heuristic optimization techniques is also considered in the paper. The Cuckoo Search and Genetic Algorithms are applied for the optimization of the fuzzy inference system. Dr. Gonzalez' simulations results show that the using an optimized interval type-2 fuzzy logic system in conjunction with the Sobel technique provide a powerful edge detection technique that outperforms its type-1 fuzzy logic cousin and pure Sobel methods.

Dr. Gonzalez and her team found that in several simulations, results for the optimization of the fuzzy logic edge detectors were similar for Cuckoo Search and Genetic Algorithms. Although they performed similarly, they out performed results achieved by non optimized interval type-2 fuzzy logic systems and optimized type-1 fuzzy logic systems as well as non-optimized type-1 fuzzy logic systems and traditional the Sobel technique. For future work, Dr. Gonzalez would like to test this technique on real images as opposed to synthetic images used in this study.

Dr. Claudia Gonzalez published a paper entitled "An improved Sobel edge detection method based on generalized
type-2 fuzzy logic" [https://link.springer.com/content/pdf/10.1007/s00500-014-1541-0.pdf](https://link.springer.com/content/pdf/10.1007/s00500-014-1541-0.pdf). In this paper, Dr. Gonzalez proposes an improved method for edge detection over Sobel edge detection using a generalized type-2 fuzzy logic system. Inorder to limit the complexity of handling a generalized type-2 fuzzy logic system, $\alpha$-planes were used. The simulation results were gathered with the Sobel operator without fuzzy logic, then with a type-1 fuzzy logic system, an interval type-2 fuzzy logic system and finally with a generalized type-2 fuzzy logic system. The method in question, a generalized type-2 fuzzy logic system, is tested with synthetically generated images and showed promising results. In order to illustrate the advantages that a generalized type-2 fuzzy logic system has over a traditional Sobel approach, the figure of merit of Pratt measure is applied to measure the accuracy of the edge detecting process. Dr. Gonzalez and her team found that the use of generalized type-2 fuzzy inference systems can improve the performance in edge detection with respect to interval type-2 fuzzy systems and traditional Sobel methods.

Dr. Phillip Isola published a paper entitled "What makes an image memorable?" [https://ieeexplore-ieee-org.dist.lib.usu.edu/document/5995721](https://ieeexplore-ieee-org.dist.lib.usu.edu/document/5995721). Dr. Isola writes a fascinating paper about the elements of an image that make it memorable. He begins by talking about the incredible capacity of the human mind to remember images but cites a lack of research into what makes certain images memorable. Opposed to most studies on human memory which focus on evaluation of how good human memory can be at remembering images, Dr. Isola's study focuses instead on the images and what makes them more memorable than others. He theorizes that like beauty, quality and other properties of an image, memorability is attached to the viewer's context and is probably subject to variability based on the viewer. Contrary to other photographic properties, there are no computer vision systems that attempt to predict image memorability. There are therefore no databases of "memorable" images.

By crow sourcing a dataset of memorable images through Amazon Mechanical Turk, Dr. Isola was able to gather data composed of 'targets' (2222 images) and 'fillers' (8220 images). Target and filler images represented a random sampling of the scene categories from the SUN dataset. After data collection, they assigned a memorability score to each image. They then investigated color, simple image features, object statistics, object semantics, and scene semantics as reasons an image may be memorable. Their analysis of object semantics demonstrated that if a system knows which objects an image contains, it is able to predict memorability with a performance not too far from human consistency. They concluded that image memorability is a task that can be addressed with modern computer vision techniques. Through their preliminary findings, they determined that some images are more memorable than others. In the future, Dr. Isola would like to to instigate the relationship between image memorability and other measure such as object importance, saliency and photo quality.

Dr. Phillip Isola published a paper entitled "Colorful Image Colorization" [https://arxiv.org/pdf/1603.08511.pdf](https://arxiv.org/pdf/1603.08511.pdf). In this paper, Dr, Isola tackles the problem of taking a gray scale image as input, "hallucinating" a believable color version of the original image. Dr. Isola and his team chose to embrace the underlying uncertainty of the problem by approaching it as a classification task. They used class re-balancing at the training time in order to increase the diversity of colors in the resulting image. The system was implemented as a feed-forward pass in a CNN at test time and was trained on over a million color images. They evaluated their algorithm using a “colorization Turing test”. Dr. Isola asked human participants to choose between a generated image and an original true color image. Their method successfully fooled humans on 32% of the trials which is significantly higher than previous colorization methods. They also showed that colorization can be a powerful pretext task for self-supervised feature learning, acting as a cross-channel encoder. This approach results in state-of-the-art performance on several feature learning benchmarks. Additionally they showed examples of applying their model to legacy black and white photographs including work of renowned photographers, such as Ansel Adams and Henri Cartier-Bresson, photographs of politicians and celebrities, and old family photos. You can see that their model is able to produce good colorizations, even though the old legacy photographs are quite different from those of modern-day photos.
