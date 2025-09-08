import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(512, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x)) #26*26*4*1
        x = self.conv2(x) #24*24*8
        x = torch.max_pool2d(x, kernel_size=2) #12*12*8
        x = torch.relu(self.conv3(x)) # 10*10*16
        x = torch.relu(self.conv4(x)) # 8*8*32
        x = torch.max_pool2d(x, kernel_size=2) #4*4*32
        x = x.view(-1, 512)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x