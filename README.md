# PAU - Padé Activation Units
Padé Activation Units: End-to-end Learning of Activation Functions in Deep Neural Network 
Arxiv link:

## 1. About adé Activation Units

Padé Activation Units (PAU) are a novel learnable activation function. PAUs encode activation functions as rational functions, trainable in an end-to-end fashion using backpropagation and can be seemingless integrated into any neural network in the same way as common activation functions (e.g. ReLU).

![alt text](https://github.com/ml-research/pau/images/logs_mean.pdf)
![alt text](https://github.com/ml-research/pau/images/activations_approx.pdf)

PAU matches or outperforms common activations in terms of predictive performance and training time. 
And, therefore relieves the network designer of having to commit to a potentially underperforming choice.

## 2. Using PAU in Neural Networks

PAU can be integrated in the same way as any other common activation function.

~~~~
import torch
from pau import PAU

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.PAU(), # e.g. instead of torch.nn.ReLU() 
    torch.nn.Linear(H, D_out),
)

## 3. Reproducing Results

To reproduce the reported results of the paper execute:

~~~~
$ export PYTHONPATH="./"
$ python main.py --dataset mnist --optimizer adam --lr 2e-3

# DATASET is the name of the dataset, for MNIST use mnist and for FashionMNISt use fmnist
# OPTIMIZER 
# LR
~~~~

Note: Throught the implementation of PAU in CUDA the behavior is not determenistic