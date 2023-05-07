import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x


class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(policy))
        self.policy = policy

    def forward(self, x):
        if self.policy == "add":
            return sum(x)
        elif self.policy == "cat":
            return torch.cat(x, dim=1)
        else:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy))


class FPNDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        encoder_depth=5,
        pyramid_channels=256,
        segmentation_channels=128,
        dropout=0.2,
        merge_policy="add",
    ):
        super().__init__()

        self.out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[: encoder_depth + 1]
        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.seg_blocks = nn.ModuleList(
            [
                SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
                for n_upsamples in [3, 2, 1, 0]
            ]
        )

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, *features):
        c2, c3, c4, c5 = features[-4:]
        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)

        return x

class Tissue2Cell(nn.Module):
    def __init__(self, feature_channels,feature_depth,scale_factor):
        super(Tissue2Cell, self).__init__()
        self.feature_depth = feature_depth + 1
        self.conv_layers = nn.ModuleList()
        self.origin_shape = 1024
        if(self.feature_depth==1):
            feature_channels = [feature_channels]
        for i in range(self.feature_depth):
            self.conv_layers.append(nn.Conv2d(feature_channels[i], feature_channels[i], kernel_size=3, padding=1))

        self.up_sampling = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

    def forward(self, tissue_features, cell_features, loc):
        out = []

        x = loc[:,1]
        y = loc[:,0]

        if(self.feature_depth==1):
            tissue_features = [tissue_features]
            cell_features = [cell_features]

        for i in range(self.feature_depth):
            conv = self.conv_layers[i]
            origin = conv(tissue_features[i])
            upsample = self.up_sampling(origin)

            trans_x = (x * upsample.shape[-1] / self.origin_shape).long()
            trans_y = (y * upsample.shape[-1] / self.origin_shape).long()
            lenth = int(128 * upsample.shape[-1] / self.origin_shape)

            roi_list = []
            for j in range(upsample.shape[0]): # 遍历batch_size
                roi = upsample[j, :, trans_x[j]-lenth:trans_x[j]+lenth, trans_y[j]-lenth:trans_y[j]+lenth]
                roi_list.append(roi)
            roi = torch.stack(roi_list, dim=0)
            out.append(roi+cell_features[i])
        if(self.feature_depth==1):
            out = out[0]
        return out

class Cell2Tissue(nn.Module):
    def __init__(self, feature_channels,feature_depth,scale_factor):
        super(Cell2Tissue, self).__init__()
        self.feature_depth = feature_depth + 1
        self.conv_layers = nn.ModuleList()
        self.origin_shape = 1024
        if(self.feature_depth==1):
            feature_channels = [feature_channels]
        for i in range(self.feature_depth):
            self.conv_layers.append(nn.Conv2d(feature_channels[i], feature_channels[i], kernel_size=3, padding=1))

        self.avg_pool = nn.AvgPool2d(kernel_size=(scale_factor, scale_factor))

    def forward(self, tissue_features, cell_features, loc):
        out = []

        x = loc[:,1]
        y = loc[:,0]

        if(self.feature_depth==1):
            tissue_features = [tissue_features]
            cell_features = [cell_features]

        for i in range(self.feature_depth):
            conv = self.conv_layers[i]
            tissue_feature = tissue_features[i]
            origin = conv(cell_features[i])
            avgpool = self.avg_pool(origin)
            trans_x = (x * tissue_feature.shape[-1] / self.origin_shape).long()
            trans_y = (y * tissue_feature.shape[-1] / self.origin_shape).long()
            lenth = int(128 * tissue_feature.shape[-1] / self.origin_shape)

            roi_list = []
            for j in range(tissue_features[i].shape[0]): # 遍历batch_size
                roi = tissue_feature[j, :, trans_x[j]-lenth:trans_x[j]+lenth, trans_y[j]-lenth:trans_y[j]+lenth].unsqueeze(0)
                roi += avgpool
                tissue_feature[j, :, trans_x[j]-lenth:trans_x[j]+lenth, trans_y[j]-lenth:trans_y[j]+lenth] = roi
                
                roi_list.append(tissue_feature)
            roi = torch.stack(roi_list, dim=0)
            out.append(roi)
        if(self.feature_depth==1):
            out = out[0].squeeze(0)
        return out