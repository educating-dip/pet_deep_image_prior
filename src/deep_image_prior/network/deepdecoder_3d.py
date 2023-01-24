import torch
import torch.nn as nn


def DeepDecoder3D(
        output_channels=1,
        output_size=(47,128,128),
        num_channels_up=256
        ):
    model = nn.Sequential()
    model.add_module("B_0", nn.Conv3d(num_channels_up, num_channels_up,  1, stride = 1))
    model.add_module("U_0", nn.Upsample(scale_factor=(2,2,2), mode='trilinear'))
    model.add_module("relu_0", nn.ReLU())
    model.add_module("cn_0", nn.BatchNorm3d( num_channels_up, affine=True))
    model.add_module("B_1", nn.Conv3d(num_channels_up, num_channels_up,  1, stride = 1))
    model.add_module("U_1", nn.Upsample(scale_factor=(2,2,2), mode='trilinear'))
    model.add_module("relu_1", nn.ReLU())
    model.add_module("cn_1", nn.BatchNorm3d( num_channels_up, affine=True))
    model.add_module("B_2", nn.Conv3d(num_channels_up, num_channels_up,  1, stride = 1))
    model.add_module("U_2", nn.Upsample(scale_factor=(2,2,2), mode='trilinear'))
    model.add_module("relu_2", nn.ReLU())
    model.add_module("cn_2", nn.BatchNorm3d( num_channels_up, affine=True))
    model.add_module("B_3", nn.Conv3d(num_channels_up, num_channels_up,  1, stride = 1))
    model.add_module("U_3", nn.Upsample(size=output_size, mode='trilinear'))
    model.add_module("relu_3", nn.ReLU())
    model.add_module("cn_3", nn.BatchNorm3d( num_channels_up, affine=True))
    model.add_module("conv_final", nn.Conv3d(num_channels_up, 1,  1, stride = 1))
    model.add_module("relu_final", nn.ReLU())
    return model