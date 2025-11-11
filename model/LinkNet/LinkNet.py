import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights


# =================== LinkNet with ResNet-18 Encoder ===================
class LinkNet(nn.Module):
    def __init__(self, N=4):
        super(LinkNet, self).__init__()
        self.N = N  # Number of views
        # 使用 ResNet-18 作为编码器
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # 编码器部分: 采用 ResNet-18 的前几个层
        self.encoder = nn.ModuleList([
            resnet.conv1,  # 初始卷积层
            resnet.bn1,    # BatchNorm
            resnet.relu,   # ReLU
            resnet.maxpool,  # 最大池化层
            resnet.layer1,  # 第一个残差块
            resnet.layer2,  # 第二个残差块
            resnet.layer3,  # 第三个残差块
            resnet.layer4   # 第四个残差块
        ])

        # 解码器部分
        self.decoder4 = self._decoder_block(512 * N, 256 * N)
        self.decoder3 = self._decoder_block(256 * N, 128 * N)
        self.decoder2 = self._decoder_block(128 * N, 64 * N)
        self.decoder1 = self._decoder_block(64 * N, 64 * N)

        self.conv_after_decoder = nn.Conv2d(64 * N, 64 * N, kernel_size=1)

        # 最后一个卷积层，用于输出
        self.final_conv = nn.Conv2d(64 * N, 16, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        """解码器中的上采样模块"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),  # 上采样
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def encode_single(self, x):
        enc1 = self.encoder[0](x)  # Initial Conv
        enc2 = self.encoder[1](enc1)
        enc3 = self.encoder[2](enc2)
        enc4 = self.encoder[3](enc3)
        enc5 = self.encoder[4](enc4)
        enc6 = self.encoder[5](enc5)
        enc7 = self.encoder[6](enc6)
        enc8 = self.encoder[7](enc7)

        return [enc1, enc2, enc3, enc4, enc5, enc6, enc7, enc8]

    def forward(self, inputs):
        # 编码器部分
        feats = []
        for i in range(self.N):
            single_view = inputs[:, i]  # 获取每个视图
            feats.append(self.encode_single(single_view))
        
        # 对每一层的特征进行拼接
        cat_features = []
        for layer in range(8):
            cat_layer = torch.cat([f[layer] for f in feats], dim=1)
            cat_features.append(cat_layer)

        # 解码器部分：逐层解码
        d1, d2, d3, d4, d5, d6, d7, d8 = cat_features  # for example d8 [B, 512 * N]

        dec4 = self.decoder4(d8)
        dec3 = self.decoder3(dec4 + d7)  # 跳跃连接
        dec2 = self.decoder2(dec3 + d6)  # 跳跃连接
        dec1 = self.decoder1(dec2 + d5)  # 跳跃连接

        # 额外卷积层，用于调整通道数
        dec1 = self.conv_after_decoder(dec1)

        # 新增的上采样操作
        dec1 = F.interpolate(dec1, scale_factor=2, mode='bilinear', align_corners=False)

        # 最后输出
        out = self.final_conv(dec1)

        return out, d8  # 返回输出结果和编码器最后一层的特征


# =================== 测试 ===================
if __name__ == "__main__":
    model = LinkNet()  
    imgs = torch.randn(2, 4, 3, 256, 512)  # [B, N, C, H, W]
    output, encoding_map = model(imgs)
    print("Input shape:", imgs.shape)
    print("Output shape:", output.shape)
    print("Encoding map shape:", encoding_map.shape)
