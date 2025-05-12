# DeiT: Training ViTs Using Distillation through Attention

## INTRODUCTION
-----------------------------------------------------------------------------------------------------

The motivation for this paper is that in order to achieve comparable performance with CNNs on image classification tasks, ViTs require a huge amount of training data to understand local patterns and don’t have built in biases like in CNNs. However, due to better scalability and understanding of global context, ViTs are still desirable. Thus, we and the authors of this paper seek to find a potential solution for the data-hungriness of the ViTs.

The authors of the paper hypothesize that by incorporating distillation (aka teacher student training) through the addition of a class token and distillation token to the ViT architecture, we can achieve comparable performance to CNNs on image classification benchmarks without training it on such extraordinary amounts of data. Our goal is to test the hypothesis and achieve the results outlined in the original DeiT paper, with much more limited computational resources and training time. 

## Obtaining our Dataset
-----------------------------------------------------------------------------------------------------
To train and test our models, we use the [Cifar-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html ), and implement it into our model using the PyTorch built-in functionality

## PROJECT SETUP INSTRUCTIONS
-----------------------------------------------------------------------------------------------------

1) 

## FILE INFO
-----------------------------------------------------------------------------------------------------

Explanations of the different files used in the project
* `code/`: A directory containing our implementation code for each model 
* `results/`: A directory containing the results of our re-implementation
* `poster/`: A directory containing a PDF of the poster related to the paper which we presented in the course
* `report/`: A PDF of a paper style final report that complements our poster
* `LICENSE`: A file specifying the license under which your code is released (e.g., MIT,
Apache 2.0).
* `.gitignore`: A file specifying files or directories that should be ignored by Git

    
## INSIGHTS
-----------------------------------------------------------------------------------------------------
Each iteration of the hyperparameter tuning job focused on improving the accuracy(as defined in the optimization of `objective_metric_name`)

![hpo](images/Status%20of%20hyperparameter%20Tuning%20Jobs.png)

The CPU utilization metric as captured by the debugging hook:

![de](images/CPU%20Utilization.png)


**NOTE:** An effort was made previously to accomplish the classification task using a `resnet18` model. However, both the training and testing inferences by this model were inaccurate. This indicated that the underlying model is not able to handle the complexity of the classification problem.



## REFERENCES
-----------------------------------------------------------------------------------------------------

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xi-
aohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg
Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for
image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Alex Krizhevsky. Learning Multiple Layers of Features from Tiny Images. Technical report, University of Toronto, 2009. Available at: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf 

Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in PyTorch. In Proceedings of the NIPS 2017 Workshop, 2017.

Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. 2020. Training data-efficient image transformers & distillation through attention. arXiv [cs.CV]. Retrieved from http://arxiv.org/abs/2012.12877  

Ilija Radosavovic, Raj Prateek Kosaraju, Ross B. Girshick, Kaiming He, and Piotr
Doll´ ar. Designing network design spaces. Conference on Computer Vision and Pattern
Recognition, 2020

J. D. Hunter. Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3):90–95, 2007. doi:10.1109/MCSE.2007.55. 


## ACKNOWLEDGEMENTS
-----------------------------------------------------------------------------------------------------

Based off an final assignment for Cornell Project Course CS4782 - Intro to Deep Learning, taught by Jennifer Sun and Kilian Weinberger 
