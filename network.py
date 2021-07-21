import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fa=nn.Sequential(
            nn.Conv2d(1,32,3),
            nn.ReLU(),
            nn.Conv2d(32,64,3),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(),
            nn.Flatten(),
            nn.Linear(12*12*64,128,bias=False),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Linear(128,10,bias=False),
        )
    def forward(self, x):
        x=self.fa(x)
        return x
