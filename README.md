# CoBA (Conjugate gradient based Adam) PyTorch sample

# CAUTION
This implementation must be incorrect.

# PAPER
https://arxiv.org/pdf/2003.00231.pdf

# BASE code
this code is based on pytorch's adam.py 
https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py

# required
pytorch
torchvision
numpy

# Usage
put coba.py and
```
from coba import CoBA
net=(your network)
optimizer=CoBA(net.parameters(), lr=0.001, betas=(0.9,0.999),amsgrad=True, gammatype="FR")
```
