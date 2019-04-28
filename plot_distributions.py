from data_loading import bddv_img_loader
from utils import config
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
from models import xception_backbone, cloning_nvidia_model
import imageio

f = open("/home/andrei/workspace/nemodrive/steeringnetwork/configs/bddv_img.yaml", 'r')
di = yaml.load(f)
ns = config.dict_to_namespace(di)
loader = bddv_img_loader.BDDVImageLoader(ns)
loader.load_data()
train_loader = loader.get_train_loader()
test_loader = loader.get_test_loader()
stop = False

def close_event():
    plt.close()

class MyModel(nn.Module):
    def __init__(self, out_size):
        super(MyModel, self).__init__()
        self.conv_layers = nn.Sequential(

            # input is batch_size x 3 x 66 x 200
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Dropout(p=0.2),

            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Dropout(p=0.2),
            # nn.BatchNorm2d(36),

            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Dropout(p=0.2),
            # nn.BatchNorm2d(48),

            nn.Conv2d(48, 64, 3, stride=1),
            nn.ELU(),
            nn.Dropout(p=0.2),
            # nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, stride=1),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(4, stride=4)
        )
        self.linear_layers = nn.Sequential(
            # input from sequential conv layers
            nn.Linear(in_features=4224, out_features=500),
            nn.ELU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=500, out_features=out_size))
        self._initialize_weights()

    # custom weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.xavier_uniform(m.weight)
                init.constant(m.bias, 0.001)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, mean=1, std=0.02)
                init.constant(m.bias, 0.001)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform(m.weight)
                init.constant(m.bias, 0.001)

    def forward(self, input):
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output

model = MyModel(181)
model = model.cuda()

model = MyModel(181)
model = model.cuda()
model.load_state_dict(torch.load('/home/andrei/storage/nemodrive/logs/train_checkpoints/nll_12'))
model.eval()

ims = []

images = []
plots = []
nr = 0

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

for _, data in enumerate(test_loader):
    with torch.no_grad():
        eval_inputs, eval_speeds, eval_labels = data[0].cuda(), data[1].cuda(), data[2].cuda()
        eval_outputs = model(eval_inputs.float(), eval_speeds.float())
        eval_outputs = nn.functional.softmax(eval_outputs, dim=1)
        for it, image in enumerate(data[0]):
            outputs = moving_average(eval_outputs[it].cpu().numpy(), 5)
            nr += 1
            print(nr)
            image = image[0:1]
            copy = image.numpy()
            copy = copy.reshape((90, 320)) / 255.0
            #cv2.imshow('frame', copy)
            images.append(copy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop = True
                break
            fig = plt.figure()
            timer = fig.canvas.new_timer(interval=2)
            timer.add_callback(close_event)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(data[2][it].numpy())
            ax.plot(outputs)
            ax.grid()
            '''
            ax.set_ylim(-1.5, 3)
            ax.set_xlim(-10, 190)
            ax.set_xlim(-10, 190)
            '''
            #plt.plot(data[2][it].numpy())
            #plt.plot(eval_outputs[it].cpu().numpy())
            fig.canvas.draw()  # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            plots.append(image)
        break

imageio.mimsave('/home/andrei/storage/nemodrive/upb_data/logs/distr.gif', plots, fps=30)
imageio.mimsave('/home/andrei/storage/nemodrive/upb_data/logs/mov.gif', images, fps=30)