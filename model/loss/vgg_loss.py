import torch
import torch.nn as nn
from torchvision import models
from typing import List, Union, cast
import torch.nn.functional as F


class VGG16(nn.Module):
    """
    VGG-16 feature extractor returning outputs from each MaxPool layer.
    - Uses official ImageNet pretrained weights
    - Frozen (non-trainable)
    - Returns (p1..p5) after each pooling layer
    - Optionally applies ImageNet normalization
    """

    def __init__(self, normalize_input=True):
        super(VGG16, self).__init__()

        # 从 torchvision 加载官方预训练权重
        vgg_pretrained = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features

        # 按池化层切分模块
        self.cnn_module1 = vgg_pretrained[:5]    # conv1_1, relu, conv1_2, relu, pool1
        self.cnn_module2 = vgg_pretrained[5:10]  # conv2_1, relu, conv2_2, relu, pool2
        self.cnn_module3 = vgg_pretrained[10:17] # conv3_1, relu, conv3_2, relu, conv3_3, relu, pool3
        self.cnn_module4 = vgg_pretrained[17:24] # conv4_1, relu, conv4_2, relu, conv4_3, relu, pool4
        self.cnn_module5 = vgg_pretrained[24:31] # conv5_1, relu, conv5_2, relu, conv5_3, relu, pool5

        # 冻结参数
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

        # 可选：输入标准化 (ImageNet mean/std)
        self.normalize_input = normalize_input
        if normalize_input:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

    def forward(self, x):
        """Return features after each MaxPool"""
        if self.normalize_input:
            x = (x - self.mean) / self.std

        p1 = self.cnn_module1(x)
        p2 = self.cnn_module2(p1)
        p3 = self.cnn_module3(p2)
        p4 = self.cnn_module4(p3)
        p5 = self.cnn_module5(p4)

        return p1, p2, p3, p4, p5

class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=(2, 3, 4), weights=(1.0, 0.5, 0.25)):
        super().__init__()
        self.vgg = VGG16().eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.layers = layers
        self.weights = weights

    def forward(self, pred, target):
        f_pred = self.vgg(pred)
        f_tgt = self.vgg(target)
        loss = 0.0
        for li, w in zip(self.layers, self.weights):
            loss += w * F.l1_loss(f_pred[li], f_tgt[li])
        return loss
