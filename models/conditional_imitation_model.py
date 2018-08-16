from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.init as init


def get_models():
    return [ConditionalImitationModel]


class ConditionalImitationModel(nn.Module):
    def __init__(self, cfg, in_size, out_size):
        super(ConditionalImitationModel, self).__init__()
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
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=0),
            nn.BatchNorm2d(128, eps=batch_norm_eps),
            nn.Dropout(p=self._dropout_conv),
            nn.ReLU())

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
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
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0),
            nn.BatchNorm2d(256, eps=batch_norm_eps),
            nn.Dropout(p=self._dropout_conv),
            nn.ReLU())

        self.conv_block7 = nn.Sequential(
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
            nn.Linear(in_features=8192, out_features=512, bias=True),
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
        branch_config = [["Speed"], ["Steer", "Gas",
                                     "Brake"], ["Steer", "Gas", "Brake"],
                         ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"]]

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
                    out_features=len(branch_config[i]),
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
            in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)

        self.deconv7 = nn.ConvTranspose2d(
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

        #liniarize the features
        x = x8.view(x8.size(0), -1)

        x = self.fc0(x)
        x = self.fc1(x)

        speed_out = self.branches[0](x)

        x_speed = self.speed_linear(speed)
        x = torch.cat((x, x_speed), 1)

        output = self.joint_linear(x)

        #compute deconvolution mask

        activation_map = torch.mean(x8, 1).unsqueeze(1)

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

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(ConditionalImitationModel, self).train(mode)
        if self.freeze_bn_gamma:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.requires_grad = False

    def resume_tensorflow(self, path):
        import re
        model = self

        data = torch.load(path)

        conv_w_match = "W_c_(\d+):0"
        conv_bias_match = "Network/conv_block(\d+)/Variable"
        bn = "bn(\d+)/*."
        # bn_beta = "bn(\d+)/beta:0"
        # bn_moving_mean = "bn(\d+)/moving_mean:0"
        # bn_moving_variance = "bn(\d+)/moving_variance:0"
        ln_w = "W_f_(\d+):0"
        ln_b = [
            "Network/fc(\d+)/Variable:0", "Network/Speed/fc(\d+)/Variable:0",
            "Network/Branch_(\d+)/fc(\d+)/Variable:0"
        ]
        ln_b_final = "Network/Branch_(\d+)/Variable:0"

        map_linear = {
            1: model.fc1[0],
            2: model.fc2[0],
            3: model.speed_linear[0],
            4: model.speed_linear[3],
            5: model.joint_linear[0],
            6: model.branches[0][0],
            7: model.branches[0][3],
            8: model.branches[0][6],
            9: model.branches[1][0],
            10: model.branches[1][3],
            11: model.branches[1][6],
            12: model.branches[2][0],
            13: model.branches[2][3],
            14: model.branches[2][6],
            15: model.branches[3][0],
            16: model.branches[3][3],
            17: model.branches[3][6],
            18: model.branches[4][0],
            19: model.branches[4][3],
            20: model.branches[4][6],
        }

        def test_lnb(k):
            res = False
            ans = None
            for x, i in enumerate(ln_b):
                res = res or re.match(i, k)
                if re.match(i, k):
                    if x == 2:
                        ans = re.match(i, k).group(2)
                    else:
                        ans = re.match(i, k).group(1)
                    ans = int(ans)

            return res, ans

        not_loaded = []
        # Print data parameters:
        for k, v in data.items():
            # print(f"{k}___{v.shape}")

            # continue

            if re.match(conv_w_match, k):
                no = int(re.match(conv_w_match, k).group(1)) - 1
                conv_layer = getattr(model, f"conv_block{no}")
                # TODO CHECK if kernel is WxH or HxW
                new_w = torch.FloatTensor(v).permute(3, 2, 0, 1)
                conv_layer[0].weight.data.copy_(new_w)
                print(f"Done: {k}")
            elif re.match(conv_bias_match, k):
                no = int(re.match(conv_bias_match, k).group(1))
                conv_layer = getattr(model, f"conv_block{no}")
                new_w = torch.FloatTensor(v)
                conv_layer[0].bias.data.copy_(new_w)
                print(f"Done: {k}")
            elif re.match(bn, k):
                no = int(re.match(bn, k).group(1)) - 1
                conv_layer = getattr(model, f"conv_block{no}")
                bn_layer = conv_layer[1]
                if k.endswith("/beta:0"):
                    new_w = torch.FloatTensor(v)
                    bn_layer.bias.data.copy_(new_w)
                elif k.endswith("/moving_mean:0"):
                    new_w = torch.FloatTensor(v)
                    bn_layer.running_mean.data.copy_(new_w)
                elif k.endswith("/moving_variance:0"):
                    new_w = torch.FloatTensor(v)
                    bn_layer.running_var.data.copy_(new_w)
                print(f"Done: {k}")
            elif re.match(ln_w, k):
                no = int(re.match(ln_w, k).group(1))
                new_w = torch.FloatTensor(v).permute(1, 0)
                map_linear[no].weight.data.copy_(new_w)
                print(f"Done: {k}")
            elif test_lnb(k)[0]:
                no = test_lnb(k)[1]
                new_w = torch.FloatTensor(v)
                map_linear[no].bias.data.copy_(new_w)
                print(f"Done: {k}")
            elif re.match(ln_b_final, k):
                no = int(re.match(ln_b_final, k).group(1))
                new_w = torch.FloatTensor(v)
                model.branches[no][6].bias.data.copy_(new_w)
                print(f"Done: {k}")
            else:
                print(f"NOT DONE: {k}")

                not_loaded.append(k)

        if len(not_loaded) > 0:
            print(f"SOME LAYERS NOT LOADED: \n {not_loaded}")


# for k, v in data.items():
#     print(f"{k}___{v.shape}")

if __name__ == "__main__":
    import argparse
    import numpy as np
    """
    CONFIG: 
    model: &model
      name: "CloningSimpleModel"
      dropout: [0.0] * 8 + [0.3] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 0.] * 5
      batch_norm_eps: 1e-3
      freeze_bn_gamma: True
      paper_init: True
      resume_tensorflow: "/media/andrei/CE04D7C504D7AF291/nemodrive/imitation-learning/weights"
    """

    batch_size = 128

    cfg = argparse.Namespace()
    cfg.dropout = [0.0] * 8 + [0.3] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 0.] * 5
    cfg.batch_norm_eps = 1e-3
    cfg.freeze_bn_gamma = True
    cfg.paper_init = True
    cfg.resume_tensorflow = "/media/andrei/CE04D7C504D7AF291/nemodrive/imitation-learning/weights"
    in_size = torch.Size([3, 88, 200])
    out_size = (torch.Size([3]), torch.Size([1]))
    model = ConditionalImitationModel(cfg, in_size, out_size)

    in_image = torch.rand(torch.Size([batch_size]) + in_size)
    in_speed = torch.rand([batch_size, 1])
    in_branch_no = torch.FloatTensor(np.random.randint(0, 4, batch_size))

    out_branch, out_speed = model(in_image, in_speed, in_branch_no)
    target_branch = Variable(torch.rand(batch_size, 3))
    target_speed = Variable(torch.rand(batch_size, 1))

    loss = torch.sum(target_branch - out_branch)
    loss += torch.sum(target_speed - out_speed)

    print(loss)
    loss.backward()
