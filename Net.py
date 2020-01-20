import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


class double(nn.Module):
    def __init__(self):
        super(double, self).__init__()
        self.net1 = Net()
        self.net2 = Net()
        self.fc1 = nn.Linear(8*8*16*2, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x, y):
        x = self.net1(x)
        # y = self.net2(x[1])
        y = self.net2(y)
        #y = net1(y)
        z = torch.cat([x,y], dim = 1)
        z = self.fc1(z)
        z = self.fc2(z)
        return z

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = BnReluConv(1,4,padding = 0)
    self.conv2 = BnReluConv(4,8,padding = 0)
    self.conv3 = BnReluConv(8,16,padding = 0)
    self.pool = nn.MaxPool2d(2, 2)

  def forward(self, x):
      x = self.conv1(x)
      x = self.pool(self.conv2(x))
      x = self.pool(self.conv3(x))
      x = x.view(-1,8*8*16)
      return x

class BnReluConv(nn.Module):
		"""docstring for BnReluConv"""
		def __init__(self, inChannels, outChannels, kernelSize = 1, stride = 1, padding = 0):
				super(BnReluConv, self).__init__()
				self.inChannels = inChannels
				self.outChannels = outChannels
				self.kernelSize = kernelSize
				self.stride = stride
				self.padding = padding

				self.bn = nn.BatchNorm2d(self.inChannels)
				self.conv = nn.Conv2d(self.inChannels, self.outChannels, self.kernelSize, self.stride, self.padding)
				self.relu = nn.ReLU()

		def forward(self, x):
				x = self.bn(x)
				x = self.relu(x)
				x = self.conv(x)
				return x