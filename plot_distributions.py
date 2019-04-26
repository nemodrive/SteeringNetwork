from data_loading import bddv_img_loader
from utils import config
import yaml
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from models import xception_backbone
import imageio

f = open("/home/nemodrive3/workspace/andreim/SteeringNetwork/configs/bddv_img.yaml", 'r')
di = yaml.load(f)
ns = config.dict_to_namespace(di)
loader = bddv_img_loader.BDDVImageLoader(ns)
loader.load_data()
train_loader = loader.get_train_loader()
test_loader = loader.get_test_loader()
stop = False


def close_event():
    plt.close()


model = xception_backbone.xception(None, None, 1000)
model.last_linear = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 181)
        )
model = model.cuda()
model.load_state_dict(torch.load('/home/nemodrive3/workspace/andreim/upb_data/logs/eval_checkpoints/cross_entropy_sgd_1_366'))
model.eval()

ims = []

images = []
plots = []
nr = 0
for _, data in enumerate(test_loader):
    with torch.no_grad():
        eval_inputs, eval_labels = data[0].cuda(), data[2].cuda()
        eval_outputs = model(eval_inputs.float())
        eval_outputs = nn.functional.softmax(eval_outputs, dim=1)
        for it, image in enumerate(data[0]):
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
            ax.plot(eval_outputs[it].cpu().numpy())
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

imageio.mimsave('/home/nemodrive3/workspace/andreim/upb_data/logs/distr.gif', plots, fps=30)
imageio.mimsave('/home/nemodrive3/workspace/andreim/upb_data/logs/mov.gif', images, fps=30)