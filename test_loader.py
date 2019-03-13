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
from tensorboardX import SummaryWriter
import sys
sys.path.append('utils/emd')
from utils.emd.modules.emd import EMDModule
import ot


def normalize(vector):
    min_v = torch.min(vector)
    range_v = torch.max(vector) - min_v
    if range_v > 0:
        normalized = (vector - min_v) / range_v
    else:
        normalized = torch.zeros(vector.size())
    return normalized

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

n = 180
x = np.arange(n, dtype=np.float32)
M = (x[:,np.newaxis] - x[np.newaxis,:]) ** 2
M /= M.max()
#criterion = WassersteinLossStab(torch.from_numpy(M), lam=0.1)
#criterion = nn.modules.loss.BCELoss().cuda()
#criterion = nn.modules.loss.BCEWithLogitsLoss().cuda()
#criterion = EMDModule().cuda()
criterion = nn.modules.loss.KLDivLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience = 8, eps=0.00001, min_lr=0.000001)
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
base_dir = '/home/nemodrive3/workspace/andreim/upb_data/logs/'
writer = SummaryWriter(base_dir + '/runs/kl_div_sgd')

print_each = 50
step = print_each
eval_each = 500

best_eval_loss = float('inf')

for epoch in range(5000):

    running_loss = 0.0

    for i, data in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        inputs, labels = data[0].cuda(), data[2].cuda()
        outputs = model(inputs.float())
        #outputs = outputs.unsqueeze(2)
        #labels = labels.unsqueeze(2)
        outputs = nn.functional.log_softmax(outputs)
        labels = nn.functional.softmax(labels)
        loss = criterion(outputs.double(), labels.double())

        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()

        if i % print_each == print_each - 1:  # print every 50 mini-batches
            print('[%d, %5d] Training loss: %.9f' %
                  (epoch + 1, i + 1, running_loss / print_each))
            with open(base_dir + 'epochs_kl_div_sgd.txt', 'a+') as f:
                f.write('[%d, %5d] Training loss: %.9f\n' %
                        (epoch + 1, i + 1, running_loss / print_each))
            running_loss = 0.0

            writer.add_image('Input', data[0], step + 1)

            writer.add_scalar('Loss', loss.item(), step + 1)

            writer.add_histogram('Output', outputs, step + 1)
            writer.add_histogram('Labels', labels, step + 1)

            avg_weights = 0
            min_weights = 10e9
            max_weights = 10e-9
            avg_grad = 0
            min_grad = 10e9
            max_grad = 10e-9

            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_scalar(tag + '/mean', value.data.cpu().numpy().mean(), step + 1)
                writer.add_scalar(tag + '/min', value.data.cpu().numpy().min(), step + 1)
                writer.add_scalar(tag + '/max', value.data.cpu().numpy().max(), step + 1)
                writer.add_scalar(tag + '/grad/mean', value.grad.data.cpu().numpy().mean(), step + 1)
                writer.add_scalar(tag + '/grad/min', value.grad.cpu().numpy().min(), step + 1)
                writer.add_scalar(tag + '/grad/max', value.grad.cpu().numpy().max(), step + 1)

                avg_weights += value.data.cpu().numpy().mean()
                avg_grad += value.grad.data.cpu().numpy().mean()
                min_weights = min(min_weights, value.data.cpu().numpy().min())
                max_weights = max(max_weights, value.data.cpu().numpy().max())
                min_grad = min(min_grad, value.grad.cpu().numpy().min())
                max_grad = max(max_grad, value.grad.cpu().numpy().max())

            writer.add_scalar('avg_weight', avg_weights, step + 1)
            writer.add_scalar('min_weight', min_weights, step + 1)
            writer.add_scalar('max_weight', max_weights, step + 1)

            writer.add_scalar('avg_grad', avg_grad, step + 1)
            writer.add_scalar('min_grad', min_grad, step + 1)
            writer.add_scalar('max_grad', max_grad, step + 1)

            step += print_each

        if i % eval_each == eval_each - 1:
            model.eval()
            with torch.no_grad():
                total_eval_loss = 0.0
                num_eval = 0
                for j, eval_data in enumerate(test_loader):
                    eval_inputs, eval_labels = eval_data[0].cuda(), eval_data[2].cuda()
                    eval_outputs = model(eval_inputs.float())
                    #eval_outputs = eval_outputs.unsqueeze(2)
                    #eval_labels = eval_labels.unsqueeze(2)
                    eval_outputs = nn.functional.log_softmax(eval_outputs)
                    eval_labels = nn.functional.softmax(eval_labels)
                    eval_loss = criterion(eval_outputs, eval_labels.float())
                    total_eval_loss += eval_loss.item()
                    num_eval += 1

                    '''
                    for it, image in enumerate(eval_data[0]):
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
                        plt.plot(nn.Softmax()(data[2][it]).numpy())
                        plt.plot(eval_outputs[it].squeeze(-1).cpu().numpy())
                        plt.show()
                    if stop:
                        break
                    '''
            total_eval_loss /= num_eval

            writer.add_scalar('Validation loss', total_eval_loss, step + 1)

            print('[%d, %5d] Eval loss: %.9f' % (epoch + 1, i + 1, total_eval_loss))
            with open(base_dir + 'eval_epochs_kl_div_sgd.txt', 'a+') as f:
                f.write('[%d, %5d] Eval loss: %.9f\n' % (epoch + 1, i + 1, total_eval_loss))
            if total_eval_loss < best_eval_loss:
                best_eval_loss = total_eval_loss
                torch.save(model.state_dict(), (base_dir + 'eval_checkpoints/best_eval_model_kl_div_sgd_%d_%d') % (epoch + 1, i + 1))
            scheduler.step(total_eval_loss)


    torch.save(model.state_dict(), (base_dir + 'train_checkpoints/baseline_model_kl_div_sgd%d') % (epoch))

writer.export_scalars_to_json(base_dir + 'all_scalars.json')
writer.close()