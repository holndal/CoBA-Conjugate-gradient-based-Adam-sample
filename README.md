# CoBA (Conjugate gradient based Adam) PyTorch sample

# CAUTION
holndal(author) started learning pytorch this month. there must be some errors.

CoBA's paper is here.
https://arxiv.org/pdf/2003.00231.pdf

# required
pytorch
torchvision
numpy

# Usage
put coba.py and
```
from coba import CoBA
(network name such as "net" "generator")=(your network)
optimizer=CoBA((network name).parameters(), lr=0.001, betas=(0.9,0.999),amsgrad=True, gammatype="FR")
```
