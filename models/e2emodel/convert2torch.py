
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

#all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

class CarModel(nn.Module):
    def __init__(self):
        super(CarModel, self).__init__()
        input_dims = (88, 200, 3)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_dims[2], out_channels=32, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.Dropout(),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.Dropout(),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.Dropout(),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.Dropout(),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.Dropout(),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.Dropout(),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.Dropout(),
            nn.ReLU(),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=8192, out_features=512, bias=False),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512, bias=False),
            nn.Dropout(),
            nn.ReLU(),
        )

        self.speed_linear = nn.Sequential(
            nn.Linear(in_features=1, out_features=128, bias=False),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128, bias=False),
            nn.Dropout(),
            nn.ReLU(),
        )

        self.joint_linear = nn.Sequential(
            nn.Linear(in_features=128 + 512, out_features=512, bias=False),
            nn.Dropout(),
            nn.ReLU(),
        )

        self.branches = []
        branch_config = [["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], \
                         ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], ["Speed"]]

        for i in range(0, len(branch_config)):
            branch_output = nn.Sequential(
                nn.Linear(in_features=512, out_features=256, bias=False),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=256, bias=False),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=len(branch_config[i]), bias=False),
                nn.Dropout(),
                nn.ReLU(),
            )
            self.branches.append(branch_output)

    def load_from_tf_model(self, all_vars):
        dict_tf = {}
        for var in all_vars:
            dict_tf[var.name] = agent._sess.run(var)

        dict_conversion = {}
        dict_pytorch = {}

        # for each

        for k, v in dict_conversion.items():
            dict_pytorch[v] = dict_tf[k]

        self.conv_layers.load(dict_pytorch)

    def forward(self, input):
        output = self.conv_layers(input)
        output = output.view(output.size(0), 64*1*18)
        output = self.linear_layers(output)
        return output



"""
#If you want to load and play manually in the console run the following

dict_tf
dict_conversion
dict_pytorch
for k, v in dict_conversion.items():
    dict_pytorch[v] = dict_tf[k]
d
from agents.imitation.imitation_learning import ImitationLearning
from agents.imitation.convert2torch import CarModel
agent = ImitationLearning("", True)
cm = CarModel()
import os
import scipy
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

"""
