# DeiT: Training ViTs Using Distillation through Attention

## INTRODUCTION
-----------------------------------------------------------------------------------------------------

The motivation for this repository is to test the hypothesis and achieve the results outlined in the original DeiT paper. DeiTs are the names of models with architecture for a data-efficient ViT with comparable performances to CNNs, and the primary purpose of the paper **Training data-efficient image transformers & distillation through attention** cited below.

## CHOSEN RESULT
-----------------------------------------------------------------------------------------------------
![Table of Model Accuracies from Original Paper](https://drive.google.com/file/d/1_v0tIZxVJVuxrC9ZUx3q2pUmYXHhi377/view?usp=sharing "Relevant Section of Paper with their Findings")

We aim to reproduce a subsection of the models tested in the paper. We specifically want to test their core findings related to their DeiT-Ti models with their distillation architecture and without their distillation architecture listed above. 

## DATASET
-----------------------------------------------------------------------------------------------------
To train and test our models, we use the [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html ), and implement it into our model using the PyTorch built-in functionality

## PROJECT SETUP INSTRUCTIONS
-----------------------------------------------------------------------------------------------------

1) Clone the repository to your local machine
2) Navigate into the correct folder where the respository is stored
3) Install the required packages `matplotlib` and  `pytorch` using the following <br/>
   `pip install matplotlib` <br/>
   `pip install torch` <br/>
   `pip install torchvision` <br/>
   `pip install numpy` <br/>
4) Navigate to the `code/` folder and open the notebook labeled `deit_model_code.ipynb`
5) Change the `distill_mode` variable of type string to one of the following desired model types
   * none (DeiT-Ti - no distillation)
   * soft (DeiT-Ti - usual distillation)
   * hard (DeiT-Ti - hard distillation)
   * class (DeiT-Ti ⚗️ - class embedding)
   * distill (DeiT-Ti ⚗️ - distil embedding)
   * both (DeiT-Ti ⚗️ - class+distillation with hard teacher model)
6) Run the notebook
7) If you would like to run the CNN model baseline, open the notebook labeled `cnn_baseline.ipynb`
8) Ensure you have required packages, and then run the notebook


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
![Table of Model Accuracies from Our Implementation](https://github.com/kimyunoo/4782_final/blob/main/results/table-of-accuracies-all-models.png "Our results from implementation trained for 20 epochs")

* Our model with both class+distillation tokens and model with hard distillation did not perform as well as the model with no distillation loss.
   * This could be becuase our teacher is trained on ImageNet rather than CIFAR-10, leading to a possible domain mismatch
   * This hypothesis is supported by the increase in accuracy to 69.19% when trained with soft distillation. 
* Our model accuracies did not plateau
   * Shows we could train for longer than 20 epochs to achieve a higher validation accuracy
* Further research includes retraining on ImageNet or looking into different teacher models

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
