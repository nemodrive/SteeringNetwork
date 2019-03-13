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
model.last_linear = nn.Linear(2048, 180)
model = model.cuda()
model.load_state_dict(torch.load('/home/nemodrive3/workspace/andreim/upb_data/logs/eval_checkpoints/best_eval_model_kl_div_sgd_52_1000'))
model.eval()

with torch.no_grad():
    for j, data in enumerate(train_loader):
        eval_inputs, eval_labels = data[0].cuda(), data[2].cuda()
        eval_outputs = model(eval_inputs.float())

        for it, image in enumerate(data[0]):
            image = image[0:1]
            copy = image.numpy()
            copy = copy.reshape((180, 320)) / 255.0
            cv2.imshow('frame', copy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop = True
                break
            fig = plt.figure()
            timer = fig.canvas.new_timer(interval=2000)
            timer.add_callback(close_event)
            plt.plot(data[2][it].numpy())
            plt.plot(eval_outputs[it].cpu().numpy())
            plt.show()
        if stop:
            break