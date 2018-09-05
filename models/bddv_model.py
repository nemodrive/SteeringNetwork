from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init as init


def get_models():
    return [BDDVModel]


class BDDVModel(nn.Module):
    def __init__(self, cfg, in_size, out_size):
        super(BDDVModel, self).__init__()
        batch_norm_eps = cfg.train.batch_norm_eps
        self._dropout_conv = cfg.train.dropout_conv
        self._dropout_liniar = cfg.train.dropout_liniar
        self._in_size = in_size
        self._out_size = out_size
        self.freeze_bn_gamma = cfg.train.freeze_bn_gamma
        paper_init = cfg.train.paper_init

        self.conv_block0 = nn.Sequential(
            nn.Conv2d(
                in_channels=self._in_size[0],
                out_channels=32,
                kernel_size=5,
                stride=2,
                padding=0),
            nn.BatchNorm2d(32, eps=batch_norm_eps),
            nn.Dropout(p=self._dropout_conv),
            nn.ReLU())

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32, eps=batch_norm_eps),
            nn.Dropout(p=self._dropout_conv),
            nn.ReLU())

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=0),
            nn.BatchNorm2d(64, eps=batch_norm_eps),
            nn.Dropout(p=self._dropout_conv),
            nn.ReLU())

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0),
            nn.BatchNorm2d(64, eps=batch_norm_eps),
            nn.Dropout(p=self._dropout_conv),
            nn.ReLU())

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=0),
            nn.BatchNorm2d(64, eps=batch_norm_eps),
            nn.Dropout(p=self._dropout_conv),
            nn.ReLU())

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=0),
            nn.BatchNorm2d(128, eps=batch_norm_eps),
            nn.Dropout(p=self._dropout_conv),
            nn.ReLU())

        self.conv_block6 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=0),
            nn.BatchNorm2d(128, eps=batch_norm_eps),
            nn.Dropout(p=self._dropout_conv),
            nn.ReLU())

        self.conv_block7 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=0),
            nn.BatchNorm2d(128, eps=batch_norm_eps),
            nn.Dropout(p=self._dropout_conv),
            nn.ReLU(),
        )

        self.conv_block8 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0),
            nn.BatchNorm2d(256, eps=batch_norm_eps),
            nn.Dropout(p=self._dropout_conv),
            nn.ReLU(),
        )

        self.conv_block9 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0),
            nn.BatchNorm2d(256, eps=batch_norm_eps),
            nn.Dropout(p=self._dropout_conv),
            nn.ReLU(),
        )

        self.fc0 = nn.Sequential(
            nn.Linear(in_features=5632, out_features=512, bias=True),
            nn.Dropout(p=self._dropout_liniar),
            nn.ReLU())

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512, bias=True),
            nn.Dropout(p=self._dropout_liniar),
            nn.ReLU())

        self.speed_linear = nn.Sequential(
            nn.Linear(in_features=1, out_features=128, bias=True),
            nn.Dropout(p=self._dropout_liniar),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128, bias=True),
            nn.Dropout(p=self._dropout_liniar),
            nn.ReLU(),
        )

        self.joint_linear = nn.Sequential(
            nn.Linear(in_features=128 + 512, out_features=512, bias=True),
            nn.Dropout(p=self._dropout_liniar),
            nn.ReLU(),
        )

        self.branches = nn.ModuleList()
        branch_config = [["Steer"], ["Steer"], ["Steer"],
                        ["Steer"], ["Steer"], ["Steer"]]

        #speed branch
        branch_output = nn.Sequential(
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.Dropout(p=self._dropout_liniar),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256, bias=True),
            nn.Dropout(p=self._dropout_liniar),
            nn.ReLU(),
            nn.Linear(
                in_features=256,
                out_features=1,
                bias=True),
        )
        self.branches.append(branch_output)

        #steer branches
        for i in range(0, len(branch_config)):
            branch_output = nn.Sequential(
                nn.Linear(in_features=512, out_features=256, bias=True),
                nn.Dropout(p=self._dropout_liniar),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=256, bias=True),
                nn.Dropout(p=self._dropout_liniar),
                nn.ReLU(),
                nn.Linear(
                    in_features=256,
                    out_features= self._out_size,
                    bias=True),
            )
            self.branches.append(branch_output)

        if paper_init:
            self._initialize_weights()

    def build_deconv_network(self):

        self.deconv0 = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=1,
            kernel_size=5,
            stride=2,
            padding=0,
            output_padding=1)

        self.deconv1 = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)

        self.deconv2 = nn.ConvTranspose2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            stride=2,
            output_padding=1)

        self.deconv3 = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)

        self.deconv4 = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=0)

        self.deconv5 = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)

        self.deconv6 = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=0)

        self.deconv7 = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=3, stride=1)

        self.deconv8 = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=3, stride=1)

        self.deconv9 = nn.ConvTranspose2d(
            in_channels=1, out_channels=1, kernel_size=3, stride=1)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                init.constant_(m.weight, 1.0)
                init.constant_(m.bias, 0.0)

    def set_forward(self, func_name):

        print(func_name)
        if func_name is "forward_simple":
            self.forward = self._forward_simple
        else:
            self.build_deconv_network()
            self.forward = self._forward_deconv

    def forward(self, image, speed):
        return self._forward_simple(image, speed)

    def _forward_simple(self, image, speed):

        x = self.conv_block0(image)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = self.conv_block7(x)
        x = self.conv_block8(x)
        x = self.conv_block9(x)

        #liniarize the features
        x = x.view(x.size(0), -1)

        x = self.fc0(x)
        x = self.fc1(x)

        speed_out = self.branches[0](x)

        x_speed = self.speed_linear(speed)
        x = torch.cat((x, x_speed), 1)

        output = self.joint_linear(x)
        return output, speed_out, None

    def _forward_deconv(self, image, speed):

        x1 = self.conv_block0(image)
        x2 = self.conv_block1(x1)
        x3 = self.conv_block2(x2)
        x4 = self.conv_block3(x3)
        x5 = self.conv_block4(x4)
        x6 = self.conv_block5(x5)
        x7 = self.conv_block6(x6)
        x8 = self.conv_block7(x7)
        x9 = self.conv_block8(x8)
        x10 = self.conv_block9(x9)

        #liniarize the features
        x = x10.view(x10.size(0), -1)

        x = self.fc0(x)
        x = self.fc1(x)

        speed_out = self.branches[0](x)

        x_speed = self.speed_linear(speed)
        x = torch.cat((x, x_speed), 1)

        output = self.joint_linear(x)

        #compute deconvolution mask

        activation_map = torch.mean(x10, 1).unsqueeze(1)

        activation_map = self.deconv9(activation_map)
        activation_map = torch.mul(activation_map,
                                   torch.mean(x9, 1).unsqueeze(1))

        activation_map = self.deconv8(activation_map)
        activation_map = torch.mul(activation_map,
                                   torch.mean(x8, 1).unsqueeze(1))

        activation_map = self.deconv7(activation_map)
        activation_map = torch.mul(activation_map,
                                   torch.mean(x7, 1).unsqueeze(1))

        activation_map = self.deconv6(activation_map)
        activation_map = torch.mul(activation_map,
                                   torch.mean(x6, 1).unsqueeze(1))

        activation_map = self.deconv5(activation_map)
        activation_map = torch.mul(activation_map,
                                   torch.mean(x5, 1).unsqueeze(1))

        activation_map = self.deconv4(activation_map)
        activation_map = torch.mul(activation_map,
                                   torch.mean(x4, 1).unsqueeze(1))

        activation_map = self.deconv3(activation_map)
        activation_map = torch.mul(activation_map,
                                   torch.mean(x3, 1).unsqueeze(1))

        activation_map = self.deconv2(activation_map)
        activation_map = torch.mul(activation_map,
                                   torch.mean(x2, 1).unsqueeze(1))

        activation_map = self.deconv1(activation_map)
        activation_map = torch.mul(activation_map,
                                   torch.mean(x1, 1).unsqueeze(1))

        activation_map = self.deconv0(activation_map)

        return output, speed_out, activation_map

    # custom weight initialization
    def _initialize_weights(self):
        freeze_bn_gamma = self.freeze_bn_gamma

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm2d) and freeze_bn_gamma:
                init.constant_(m.weight, 1)
                m.weight.requires_grad = False

    def get_branches(self, use_cuda):
        return self.branches

