from data_loading import bddv_img_loader
from utils import config
import yaml
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from models import xception_backbone, cloning_nvidia_model
from tensorboardX import SummaryWriter
import sys
sys.path.append('utils/emd')

n = 181

def normalize(vector):
    min_v = torch.min(vector)
    range_v = torch.max(vector) - min_v
    if range_v > 0:
        normalized = (vector - min_v) / range_v
    else:
        normalized = torch.zeros(vector.size())
    return normalized

f = open("/home/andrei/workspace/nemodrive/steeringnetwork/configs/bddv_img.yaml", 'r')
di = yaml.load(f)
ns = config.dict_to_namespace(di)
loader = bddv_img_loader.BDDVImageLoader(ns)
loader.load_data()
train_loader = loader.get_train_loader()
test_loader = loader.get_test_loader()
stop = False

weights = torch.ones(1, n).long()
'''
for _, data in enumerate(train_loader):
    a = data[2]
    view = a.transpose(1, 0)
    m = view==view.max(0)[0]
    m = m.transpose(1, 0)
    weights = torch.add(weights, m.sum(0))
weights = torch.div(torch.ones(1, n), weights.float())
weights = weights[0].cuda()
#print(weights)

#plt.plot(weights.cpu().numpy())
#plt.show()
'''

def close_event():
    plt.close()

# define model

class MyModel(nn.Module):
    def __init__(self, no_outputs):
        super(MyModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, (5, 5), padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, (5, 5), padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, (5, 5), padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 32, (3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Linear(32 * 5 * 20 + 1, no_outputs)


    def forward(self, x, speeds):
        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.cat((x, speeds), dim=1)
        x = self.classifier(x)
        return x


model = MyModel(181)
model = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=8, eps=0.9, min_lr=0.000001)

criterion = nn.KLDivLoss().cuda()

base_dir = '/home/andrei/storage/nemodrive/logs/'
f = open(base_dir + 'nvidia_bce_logits_train.txt', 'w')
g = open(base_dir + 'nvidia_bce_logits_eval.txt', 'w')

print_each = 50
step = print_each
eval_each = 1

best_eval_loss = float('inf')

writer = SummaryWriter(base_dir + '/runs/nvidia_bce_logits')

train_loader = loader.get_train_loader()

for epoch in range(5000):

    running_loss = 0.0

    for i, data in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        inputs, speeds, labels = data[0].cuda(), data[1].cuda(), data[2].cuda()
        outputs = model(inputs.float(), speeds.float())
        outputs = nn.functional.log_softmax(outputs, dim=1)
        loss = criterion(outputs.double(), labels.double())
        #loss = torch.mean(-torch.sum(labels.float() * outputs, dim=1))

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

            #writer.add_image('Input', data, step + 1)

            writer.add_scalar('Loss', loss.item(), step + 1)

            writer.add_histogram('Output', outputs, step + 1)
            writer.add_histogram('Labels', labels, step + 1)

            avg_weights = 0
            min_weights = 10e9
            max_weights = 10e-9
            avg_grad = 0
            min_grad = 10e9
            max_grad = 10e-9

            step += print_each

    if epoch % eval_each == eval_each - 1:
        model.eval()
        with torch.no_grad():
            total_eval_loss = 0.0
            num_eval = 0
            for j, eval_data in enumerate(test_loader):
                eval_inputs, eval_speeds, eval_labels = eval_data[0].cuda(), eval_data[1].cuda(), eval_data[2].cuda()
                eval_outputs = model(eval_inputs.float(), eval_speeds.float())
                eval_outputs = nn.functional.log_softmax(eval_outputs, dim=1)
                #eval_loss = torch.mean(-torch.sum(eval_labels.float() * eval_outputs, dim=1))
                eval_loss = criterion(eval_outputs.double(), eval_labels.double())
                total_eval_loss += eval_loss.item()
                num_eval += 1

        total_eval_loss /= num_eval

        writer.add_scalar('Validation loss', total_eval_loss, step + 1)

        print('[%d] Eval loss: %.9f' % (epoch + 1, total_eval_loss))
        g.write('[%d] Eval loss: %.9f\n' % (epoch + 1, total_eval_loss))
        if total_eval_loss < best_eval_loss:
            best_eval_loss = total_eval_loss
            torch.save(model.state_dict(), (base_dir + 'eval_checkpoints/nvidia_bce_logits_%d') % (epoch + 1))
        scheduler.step(total_eval_loss)


    torch.save(model.state_dict(), (base_dir + 'train_checkpoints/nvidia_bce_logits_%d') % (epoch))

f.close()
g.close()

writer.export_scalars_to_json(base_dir + 'all_scalars.json')
writer.close()