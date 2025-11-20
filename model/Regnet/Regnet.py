import torch
import torch.nn as nn
import torch.nn.functional as F

class Regressor(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.homography = opt.homography
        self.input_direction = 4
        self.first_layer = self.input_direction * opt.get('first_layer', 256)  # 4 * 256 = 1024

        self.conv45 = self.conv_block(self.first_layer, 512, kernel_size=3, stride=2)
        self.conv5a = self.conv_block(512, 512, kernel_size=3, stride=1)
        self.conv5b = self.conv_block(512, 512, kernel_size=3, stride=1)
        self.conv56 = self.conv_block(512, 512, kernel_size=1, stride=2)
        self.conv6a = self.conv_block(512, 512, kernel_size=1, stride=1)
        self.conv6b = self.conv_block(512, 512, kernel_size=1, stride=1)

        self.fc = nn.Linear(512, 6 * self.homography * self.input_direction)
        self.fc.weight.data.zero_()  # initialize zero
        bias = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32).repeat(self.homography * self.input_direction)
        self.fc.bias.data.copy_(bias)

    @staticmethod
    def conv_block(in_dim, out_dim, kernel_size=3, stride=1):
        """
        Build Convolutional Block [Conv, BatchNorm, ELU]
        """
        conv_up_block = []
        conv_up_block += [nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=1),
                          nn.BatchNorm2d(out_dim, momentum=0.1),
                          nn.ELU()]

        return nn.Sequential(*conv_up_block)
    
    def forward(self, feat):
        """
        feat: [B x 512 x 2 x 4]
        Get Parameters of Homography
        """
        x = feat
        # x: [Batch, 512, 2, 4]
        x = self.conv45(x)
        # x: [Batch, 512, 1, 2]
        x = self.conv5a(x)
        # x: [Batch, 512, 1, 2]
        x = self.conv5b(x)
        # x: [Batch, 512, 1, 2]
        x = self.conv56(x)
        # x: [Batch, 512, 1, 1]
        x = self.conv6a(x)
        # x: [Batch, 512, 1, 1]
        x = self.conv6b(x)
        # x: [Batch, 512, 1, 1]
        x = F.avg_pool2d(x, (x.shape[2], x.shape[3]))

        # x: [Batch, 512]
        x = x.view(-1, x.shape[1])

        # theta: [Batch, 6 * homography * input_direction]
        theta = self.fc(x)
        theta = theta.view(-1, self.homography * self.input_direction, 2, 3) # 2*3 affine

        return theta