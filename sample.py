# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
from network import Net
from coba import CoBA
torch.backends.cudnn.benchmark = True

BATCH_SIZE=512

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))])
trainset = torchvision.datasets.MNIST(root='./data', 
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', 
                                        train=False, 
                                        download=True, 
                                        transform=transform)
testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=BATCH_SIZE,
                                            shuffle=False, 
                                            num_workers=2)

classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))

def acc(net):
  correct = 0
  total = 0
  loss = 0
  criterion = torch.nn.CrossEntropyLoss()
  with torch.no_grad():
      for (images, labels) in testloader:
          images=images.cuda()
          labels=labels.cuda()
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          loss += criterion(outputs, labels) / len(testloader)
  return correct/total ,loss.item()

def run(epochs,op):
  net=Net().cuda()
  criterion = torch.nn.CrossEntropyLoss()
  if op=="CoBA FR":
    optimizer=CoBA(net.parameters(), lr=0.001, betas=(0.9,0.999),amsgrad=True, gammatype="FR")
  elif op=="CoBA PRP":
    optimizer=CoBA(net.parameters(), lr=0.001, betas=(0.9,0.999),amsgrad=True, gammatype="PRP")
  elif op=="ADAM":
    optimizer=torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9,0.999))
  elif op=="AdaGrad":
    optimizer=torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9,0.999),amsgrad=True)
  results=[]
  for epoch in range(epochs):
      net.train()
      for i, (inputs, labels) in enumerate(trainloader, 0):
          inputs=inputs.cuda()
          labels=labels.cuda()
          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

      net.eval()
      temp=acc(net)
      print("EPOCH {:02} / {:02} val acc:{}% loss(train):{} loss(val):{}".format(epoch,epochs,temp[0],loss.item(), temp[1]))
      results.append(temp)
  print('Finished Training')

ops=["CoBA FR", "CoBA PRP","ADAM","AdaGrad"]
for op in ops:
  run(30,op)

