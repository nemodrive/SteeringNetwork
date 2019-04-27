import torch.nn as nn
import torch.nn.init as init


def get_models():
    return [CloningSimpleModel]


class CloningSimpleModel(nn.Module):
    def __init__(self, cfg, in_size, out_size):
        super(CloningSimpleModel, self).__init__()

        self.cfg = cfg
        self.in_size = in_size
        self.out_size = out_size

        self.conv_layers = nn.Sequential(
            # input is batch_size x 3 x 16 x 32
            nn.Conv2d(3, 24, 3, stride=2, bias=False),
            nn.ELU(),
            nn.Conv2d(24, 48, 3, stride=2, bias=False),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.25)
        )

        self.linear_layers = nn.Sequential(
            # input from sequential conv layers
            nn.Linear(in_features=48*4*19, out_features=50, bias=False),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10, bias=False),
            nn.Linear(in_features=10, out_features=1, bias=False),
        )

        self._initialize_weights()

    # custom weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, mean=1, std=0.02)
                init.constant(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), 3, 75, 320)
        output = self.conv_layers(x)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output
