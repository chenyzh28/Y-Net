import torch.nn as nn
from torch.nn import functional as F
import torch
affine_par = True
import functools

from libs import InPlaceABN, InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual      
        out = self.relu_inplace(out)

        return out

class ASPPModule(nn.Module):
    """
    Reference: 
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """
    def __init__(self, features, inner_features=256, out_features=512, dilations=(6, 12, 18)):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv2 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv3 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv4 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv5 = nn.Sequential(nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
                                   InPlaceABNSync(inner_features))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
            )
        
    def forward(self, x):

        _, _, h, w = x.size()

        feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle


class ASPPModule_HFF(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=512, dilations=(6, 12, 18)):
        super(ASPPModule_HFF, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False),
                                   InPlaceABNSync(inner_features))
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(inner_features))
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            InPlaceABNSync(inner_features))
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            InPlaceABNSync(inner_features))
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            InPlaceABNSync(inner_features))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(out_features),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)

        feat2 = self.conv2(x)
        temp1 = feat1 + feat2
        feat3 = self.conv3(x)
        temp2 = temp1 + feat3
        feat4 = self.conv4(x)
        temp3 = temp2 + feat4
        feat5 = self.conv5(x)
        temp4 = temp3 + feat5
        out = torch.cat((feat1, temp1, temp2, temp3, temp4), 1)

        bottle = self.bottleneck(out)
        return bottle

class AttentiveMerge(nn.Module):
    def __init__(self, features, feature_num=2, ratio=2):
        super(AttentiveMerge, self).__init__()
        l_v = int(features / ratio)
        self.features = features
        self.fc = nn.Linear(features, l_v)
        self.fcs = nn.ModuleList([])
        for _ in range(feature_num):
            self.fcs.append(nn.Linear(l_v, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        fea_U = torch.sum(x, dim=1)
        fea_g = fea_U.mean(-1).mean(-1)
        fea_v = self.fc(fea_g)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_v).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)

        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (x * attention_vectors).sum(dim=1)
        return fea_v

class Guidance_CSD(nn.Module):
    def __init__(self, num_classes):
        super(Guidance_CSD, self).__init__()
        channel_blocks = []
        dim_list = list((2048, 1024, 512, 256))
        inner_features = 512
        for i, dim in enumerate(dim_list):
            channel_blocks.append(dim)
        self.head0 = ASPPModule_HFF(channel_blocks[0])
        self.head1 = ASPPModule_HFF(channel_blocks[1])
        self.SKComb0 = nn.Sequential(AttentiveMerge(inner_features, feature_num=2, ratio=2))
        self.gamma = torch.FloatTensor(1).cuda()
        self.gamma = nn.Parameter(self.gamma, requires_grad=True)
        self.gamma.data.fill_(0.1)
        #TODO:Dual Branch
        self.conv_op = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            InPlaceABNSync(512),
            nn.ReLU(inplace=False)
            )
        self.conv = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, f0, f1):

        fms_0 = f0
        fms_0 = self.head0(fms_0)
        fms_0 = fms_0.unsqueeze_(dim=1)

        fms_1 = f1
        fm_high_1 = torch.cat([self.head1(fms_1).unsqueeze_(dim=1), fms_0], dim=1)
        fm_high_1 = self.SKComb0(fm_high_1)

        fm_high_1_op = self.conv_op(fm_high_1)
        fm_high_1_op = F.upsample(fm_high_1_op, size=x.size()[2:], mode="bilinear")
        x = x + self.gamma * x * fm_high_1_op

        fm_high_1 = self.conv(fm_high_1)

        return x, fm_high_1

class Optimization_CSD(nn.Module):
    def __init__(self, num_classes):
        super(Optimization_CSD, self).__init__()
        channel_blocks = []
        dim_list = list((2048, 1024, 512, 256))
        inner_features = 512
        for i, dim in enumerate(dim_list):
            channel_blocks.append(dim)
        self.head0 = ASPPModule_HFF(channel_blocks[0])
        self.head1 = ASPPModule_HFF(channel_blocks[1])
        self.SKComb0 = nn.Sequential(AttentiveMerge(inner_features, feature_num=2, ratio=2))
        self.conv = nn.Conv2d(inner_features, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, f0, f1):
        fms_0 = f0
        fms_0 = self.head0(fms_0)
        fms_0 = fms_0.unsqueeze_(dim=1)

        fms_1 = f1
        fm_high_1 = torch.cat([self.head1(fms_1).unsqueeze_(dim=1), fms_0], dim=1)
        fm_high_1 = self.SKComb0(fm_high_1)

        fm_high_1 = self.conv(fm_high_1)

        return fm_high_1

class YNet(nn.Module):
    def __init__(self, block, layers, num_classes, os=16):
        self.inplanes = 128
        super(YNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if os == 16:
            dilation = [1, 2]
            stride = [2, 1]
        elif os == 8:
            dilation = [2, 4]
            stride = [1, 1]

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_1 = self._make_layer(block, 256, layers[2], stride=stride[0], dilation=dilation[0])
        self.layer4_1 = self._make_layer(block, 512, layers[3], stride=stride[1], dilation=dilation[1], multi_grid=(1,2,4))

        self.inplanes = 512
        self.layer3_2 = self._make_layer(block, 256, layers[2], stride=stride[0], dilation=dilation[0])
        self.layer4_2 = self._make_layer(block, 512, layers[3], stride=stride[1], dilation=dilation[1], multi_grid=(1,2,4))

        self.decoder1 = Guidance_CSD(num_classes)
        self.decoder2 = Optimization_CSD(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion,affine = affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x1_3 = self.layer3_1(x)
        x1_4 = self.layer4_1(x1_3)
        x, x1 = self.decoder1(x, x1_4, x1_3)

        #TODO:Dual Branch
        x2_3 = self.layer3_2(x)
        x2_4 = self.layer4_2(x2_3)
        result = self.decoder2(x2_4, x2_3)

        return [result, x1]

def Res_YNet(num_classes=21, os=8):
    model = YNet(Bottleneck,[3, 4, 23, 3], num_classes, os=os)
    return model
