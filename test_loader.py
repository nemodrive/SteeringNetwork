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
import torch.nn.functional as F

from models import xception_backbone
from tensorboardX import SummaryWriter
import sys
sys.path.append('utils/emd')


f = open("./configs/bddv_img.yaml", 'r')
di = yaml.load(f)
ns = config.dict_to_namespace(di)
loader = bddv_img_loader.BDDVImageLoader(ns)
loader.load_data()
train_loader = loader.get_train_loader()
test_loader = loader.get_test_loader()
stop = False


def close_event():
    plt.close()


# define model
class MyModel(nn.Module):
    def __init__(self, no_outputs):
        super(MyModel, self).__init__()
        self.features = xception_backbone.xception(None, None, 1000)
        self.last_linear = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1000, no_outputs)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.last_linear(x)
        return x


class MySecondModel(nn.Module):
    def __init__(self, no_outputs):
        super(MySecondModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, (5, 5), padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(24, 36, (5, 5), padding=2),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(36, 48, (5, 5), padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(48, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(64 * 5 * 20 + 1, no_outputs)


    def forward(self, x, speeds):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.cat((x, speeds), dim=1)
        x = self.classifier(x)
        return x

model = MySecondModel(181)
model = model.cuda()
print(model)

#criterion = WassersteinLossStab(torch.from_numpy(M), lam=0.1)
#criterion = nn.modules.loss.BCELoss().cuda()
#criterion = nn.modules.loss.BCEWithLogitsLoss(weight=weights).cuda() -- OK
#criterion = EMDModule().cuda()
#criterion = nn.modules.loss.KLDivLoss()
criterion = nn.modules.loss.KLDivLoss(reduction='batchmean')

# optimizer = optim.Adam(model.parameters(), lr=1e-4)
optimizer = optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=2, eps=0.9, min_lr=0.000001)

base_dir = './logs/'
f = open(base_dir + 'cross_entropy_sgd.txt', 'a+')
g = open(base_dir + 'cross_entropy_sgd.txt', 'w')

print_each = 50
step = print_each
eval_each = 1

best_eval_loss = float('inf')
writer = SummaryWriter(base_dir + '/runs/cross_entropy_sgd_scs')
train_loader = loader.get_train_loader()

for epoch in range(5000):
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()

        # pass inputs through model
        inputs, speed, labels = data[0].cuda(), data[1].cuda(), data[2].cuda()
        outputs = model(inputs.float(), speed.float())

        # #show image & plot distributions
        # image = inputs.cpu().numpy()[0]
        # image = image.transpose(1, 2, 0)
        # cv2.imshow("IMG", image + 0.5)
        # cv2.waitKey(1)
        #
        # print("Course:", labels.cpu().numpy()[0].argmax())
        #
        # plt.plot(labels.cpu().numpy()[0], 'b')
        # plt.plot(F.softmax(outputs, dim=1).cpu().detach().numpy()[0], 'r')
        # plt.show()

        # compute loss
        outputs = F.log_softmax(outputs, dim=1)
        loss = criterion(outputs, labels.float())

        # gradient step
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % print_each == print_each - 1:  # print every 50 mini-batches
            print('[%d, %5d] Training loss: %.9f' %
                  (epoch + 1, i + 1, running_loss / print_each))
            f.write('[%d, %5d] Training loss: %.9f\n' %
                    (epoch + 1, i + 1, running_loss / print_each))
            running_loss = 0.0

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

    if epoch % eval_each == eval_each - 1:
        model.eval()
        with torch.no_grad():
            total_eval_loss = 0.0

            for j, eval_data in enumerate(test_loader):
                # pass through modle
                eval_inputs, eval_speeds, eval_labels = eval_data[0].cuda(), eval_data[1].cuda(), eval_data[2].cuda()
                eval_outputs = model(eval_inputs.float(), eval_speeds.float())

                # compute loss
                eval_outputs = F.log_softmax(eval_outputs, dim=1)
                eval_labels = eval_labels.float()
                eval_loss = criterion(eval_outputs, eval_labels)
                total_eval_loss += eval_loss.item()

        total_eval_loss /= len(test_loader)

        # log & print
        writer.add_scalar('Validation loss', total_eval_loss, step + 1)
        print('[%d] Eval loss: %.9f' % (epoch + 1, total_eval_loss))
        g.write('[%d] Eval loss: %.9f\n' % (epoch + 1, total_eval_loss))

        # save best model
        if total_eval_loss < best_eval_loss:
            best_eval_loss = total_eval_loss
            torch.save(model.state_dict(), (base_dir + 'eval_checkpoints/DKL_%d_%d') % (epoch + 1, i + 1))
            print("Best model saved")

        # scheduler step
        scheduler.step(total_eval_loss)

    # save model
    torch.save(model.state_dict(), (base_dir + 'train_checkpoints/DKL_%d') % (epoch))
    print("Model saved")

f.close()
g.close()

writer.export_scalars_to_json(base_dir + 'all_scalars.json')
writer.close()