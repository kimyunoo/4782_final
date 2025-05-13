# DeiT: Training ViTs Using Distillation through Attention

## INTRODUCTION
-----------------------------------------------------------------------------------------------------

The motivation for this repository is to test the hypothesis and achieve the results outlined in the original DeiT paper, DeiTs are the names of models with architecture for a data-efficient ViT with comparable performances to CNNs. 

## DATASET
-----------------------------------------------------------------------------------------------------
To train and test our models, we use the [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html ), and implement it into our model using the PyTorch built-in functionality

## PROJECT SETUP INSTRUCTIONS
-----------------------------------------------------------------------------------------------------

1) Clone the repository to your local machine
2) Navigate into the correct folder where the respository is stored
3) Install the required packages `matplotlib` and  `pytorch` using the following <br/>
   `pip install matplotlib` <br/>
   `pip install torch` 
4) Navigate to the `code/` folder and decide on the model you would like to run
5) Change the `distill_mode` variable of type string to one of the following desired model types
   * none (DeiT-Ti - no distillation)
   * soft (DeiT-Ti - usual distillation)
   * hard (DeiT-Ti - hard distillation)
   * class (DeiT-Ti ⚗️ - class embedding)
   * distill (DeiT-Ti ⚗️ - distil embedding)
   * both (DeiT-Ti ⚗️ - class+distillation with hard teacher model)
6) Run the notebook 

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
Interestingly, **the model using both class and distillation tokens with hard distillation loss did not perform as well as the model with no distillation loss**. We trained the same architecture with soft distillation loss instead, which achieved a better validation accuracy (69.19%, the highest out of all the distillation modes). **We suspect that this may be because our teacher is trained on ImageNet rather than CIFAR-10, so the domain mismatch may cause differences with the paper’s observations** (CIFAR-10 has 10 classes, while ImageNet has 1000).

We achieved 88.46% validation accuracy when we trained a ResNet-18, a CNN, for 20 epochs on the same task. **Thus, our DeiT model is unfortunately not comparable to CNN performance.** This may be due to the teacher domain mismatch as well as having less data augmentation and compute than in the original paper. **We can also observe that after 20 epochs the test accuracy for the model with class and distillation tokens and soft distillation loss still has a positive slope and does not seem to have plateaued yet.** So, we can assume if we trained the full number of epochs it would have achieved an even higher validation accuracy, perhaps comparable to that achieved in the paper. 

Additional areas of research could also understand what teacher model to train with, and continue comparisons from there. We could also retrain the models on ImageNet given more resources. 

## REFERENCES
-----------------------------------------------------------------------------------------------------
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, 2021. https://arxiv.org/abs/2010.11929

Learning Multiple Layers of Features from Tiny Images, 2009. https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf 

Automatic differentiation in PyTorch, 2017. https://openreview.net/pdf?id=BJJsrmfCZ

Training data-efficient image transformers & distillation through attention, 2020. http://arxiv.org/abs/2012.12877  

Designing network design spaces. Conference on Computer Vision and Pattern Recognition, 2020. https://arxiv.org/pdf/2003.13678

Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3):90–95, 2007. https://ieeexplore.ieee.org/document/4160265


## ACKNOWLEDGEMENTS
-----------------------------------------------------------------------------------------------------

Final assignment for Cornell Project Course CS4782 - Intro to Deep Learning, taught by Jennifer Sun and Kilian Weinberger 
ViT implementation inspired by CS4782 Assignment 3
Reimplementation of "Training data-efficient image transformers & distillation through attention" by Touvron et al. at Facebook
