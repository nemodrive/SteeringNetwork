from data_loading import bddv_img_loader
from utils import config
import yaml
import cv2
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from models import xception_backbone

f = open("/home/nemodrive3/workspace/andreim/SteeringNetwork/configs/bddv_img.yaml", 'r')
di = yaml.load(f)
ns = config.dict_to_namespace(di)

loader = bddv_img_loader.BDDVImageLoader(ns)
loader.load_data()
train_loader = loader.get_train_loader()

stop = False

def close_event():
    plt.close()

net = xception_backbone.xception(None, None, 180)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.paramseters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

"""
# data load test
for i, data in enumerate(train_loader):
    for it, image in enumerate(data[0]):
        image = image[0:1]
        copy = image.numpy()
        copy = copy.reshape((360, 640)) / 255.0
        cv2.imshow('frame', copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop = True
            break
        fig = plt.figure()
        timer = fig.canvas.new_timer(interval=2000)
        timer.add_callback(close_event)
        plt.plot(data[2][it].numpy())
        timer.start()
        plt.show()
    if stop:
        break
"""

for epoch in range(1000):
    for i, data in enumerate(train_loader):
        print(data.size())