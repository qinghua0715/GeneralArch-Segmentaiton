import torch.nn as nn
import torch
from torch.nn import functional as F
from timm.models.layers import DropPath, trunc_normal_

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


# 定义CConv类，将Unet内多次用到的几个相同步骤组合在一起成一个网络，避免重复代码太多
class CConv(nn.Module):
    # 定义网络结构
    def __init__(self, in_ch, out_ch):
        super(CConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    # 重写父类forward方法
    def forward(self, t):
        t = self.conv1(t)
        t = self.bn1(t)
        t = F.relu(t)
        t = self.conv2(t)
        t = self.bn2(t)
        output = F.relu(t)
        return output


class Attention_Gate(nn.Module):

    def __init__(self, channel, h, w):
        super(Attention_Gate, self).__init__()
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.W_x = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, bias=True),
            nn.GELU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, g, x):
        shortcut = x
        g_hw_avg = torch.mean(g, 1, True)
        g_ch_avg = self.avg_pool_x(g)
        g_cw_avg = self.avg_pool_y(g)

        x = self.W_x(x)
        x = x * g_hw_avg * g_ch_avg * g_cw_avg

        x = self.conv(x)
        out = shortcut + self.gamma*x

        return out


class Unet(nn.Module):
    # 定义网络结构共9层，四次下采样，四次上采样
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        self.conv1 = CConv(in_ch=in_channels, out_ch=64)
        self.down1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = CConv(in_ch=64, out_ch=128)
        self.down2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = CConv(in_ch=128, out_ch=256)
        self.down3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = CConv(in_ch=256, out_ch=512)
        self.down4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = CConv(in_ch=512, out_ch=1024)

        self.bn1 = nn.BatchNorm2d(num_features=1024)
        self.up1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.conv6 = CConv(in_ch=1024, out_ch=512)

        self.bn2 = nn.BatchNorm2d(num_features=512)
        self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv7 = CConv(in_ch=512, out_ch=256)

        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv8 = CConv(in_ch=256, out_ch=128)

        self.up4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv9 = CConv(in_ch=128, out_ch=64)

        self.conv10 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    # 重写父类forward方法
    def forward(self, t):
        # layer1
        c1 = self.conv1(t)
        t = self.down1(c1)

        # layer2
        c2 = self.conv2(t)
        t = self.down2(c2)

        # layer3
        c3 = self.conv3(t)
        t = self.down3(c3)

        # layer4
        c4 = self.conv4(t)
        t = self.down4(c4)

        # layer5
        c5 = self.conv5(t)

        d1 = self.up1(c5)

        # layer6
        t = torch.cat([d1, c4], dim=1)
        t = self.conv6(t)
        d2 = self.up2(t)

        # layer7
        t = torch.cat([d2, c3], dim=1)
        t = self.conv7(t)
        d3 = self.up3(t)

        # layer8
        t = torch.cat([d3, c2], dim=1)
        t = self.conv8(t)
        d4 = self.up4(t)

        # layer9
        t = torch.cat([d4, c1], dim=1)
        t = self.conv9(t)
        t = self.conv10(t)
        out = t
        return out


# 极简Unet
class Unet_S(nn.Module):
    def __init__(self, input_channels, output_channels, c_list=[8, 16, 32, 64, 128, 256]):
        super(Unet_S, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[0])
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[1])
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[2])
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[3])
        )


        self.encoder5 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[4])
        )


        self.bottle_neck = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[5], kernel_size=3, padding=1),
            nn.Conv2d(c_list[5], c_list[5], kernel_size=3, padding=1),
            nn.Conv2d(c_list[5], c_list[4], kernel_size=3, padding=1)
        )

        # self.ca = Cross_Attention(channel=c_list[4], h=14, w=14)
        self.skip1 = nn.Conv2d(c_list[0], c_list[0], kernel_size=3, padding=1)
        self.skip2 = nn.Conv2d(c_list[1], c_list[1], kernel_size=3, padding=1)
        self.skip3 = nn.Conv2d(c_list[2], c_list[2], kernel_size=3, padding=1)
        self.skip4 = nn.Conv2d(c_list[3], c_list[3], kernel_size=3, padding=1)
        self.skip5 = nn.Conv2d(c_list[4], c_list[4], kernel_size=3, padding=1)

        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[3])
            )

        self.decoder4 = nn.Sequential(
            nn.Conv2d(2*c_list[3], c_list[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[2])
            )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(2*c_list[2], c_list[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[1])
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(2*c_list[1], c_list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[0])
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(2*c_list[0], c_list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[0])
        )

        # self.skip_aug4 = Attention_Gate(64, 14, 14)
        # self.skip_aug3 = Attention_Gate(32, 28, 28)
        # self.skip_aug2 = Attention_Gate(16, 56, 56)
        # self.skip_aug1 = Attention_Gate(8, 112, 112)

        self.final = nn.Conv2d(in_channels=c_list[0], out_channels=output_channels, kernel_size=1)


    # 重写父类forward方法
    def forward(self, x):
        # layer1
        e1 = F.gelu(F.max_pool2d(self.encoder1(x), 2, 2))     # B 8 H/2 W/2
        s1 = self.skip1(e1)

        # layer2
        e2 = F.gelu(F.max_pool2d(self.encoder2(e1), 2, 2))     # B 16 H/4 W/4
        s2 = self.skip2(e2)

        # layer3
        e3 = F.gelu(F.max_pool2d(self.encoder3(e2), 2, 2))    # B 32 H/8 W/8
        s3 = self.skip3(e3)

        # layer4
        e4 = F.gelu(F.max_pool2d(self.encoder4(e3), 2, 2))    # B 64 H/16 W/16
        s4 = self.skip4(e4)

        # layer5
        e5 = F.gelu(self.encoder5(e4))  # B 128 H/16 W/16
        s5 = self.skip5(e5)

        d5 = F.gelu(self.bottle_neck(e5))    # B 128 H/16 W/16
        # d5 = self.ca(d5)

        # layer6
        # d5_ = torch.cat([d5, e5], dim=1)    # B 256 H/16 W/16
        d4 = F.gelu(self.decoder5(d5))   # B 64 H/16 W/16

        # layer7
        d4_ = torch.cat([d4, s4], dim=1)    # B 128 H/16 W/16
        d3 = F.gelu(F.interpolate(self.decoder4(d4_), scale_factor=(2, 2), mode='bilinear',
                                  align_corners=True))   # B 32 H/8 W/8

        # layer8
        d3_ = torch.cat([d3, s3], dim=1)    # B 64 H/8 W/8
        d2 = F.gelu(F.interpolate(self.decoder3(d3_), scale_factor=(2, 2), mode='bilinear',
                                  align_corners=True))  # B 16 H/4 W/4

        # layer9
        d2_ = torch.cat([d2, s2], dim=1)    # B 32 H/4 W/4
        d1 = F.gelu(F.interpolate(self.decoder2(d2_), scale_factor=(2, 2), mode='bilinear',
                                  align_corners=True))  # B 8 H/2 W/2

        d1_ = torch.cat([d1, s1], dim=1)  # B 16 H/2 W/2
        out = F.gelu(F.interpolate(self.decoder1(d1_), scale_factor=(2, 2), mode='bilinear',
                                  align_corners=True))  # B 8 H W
        out = self.final(out)    # B 1 H W
        # out = torch.sigmoid(out)
        return out


# 测试SE模块代码
if __name__ == '__main__':
    input = torch.randn(1, 3, 224, 224).to('cuda')
    model = Unet(3, 1).to('cuda')
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", total_params)

    output = model(input)
    print(output.shape)
