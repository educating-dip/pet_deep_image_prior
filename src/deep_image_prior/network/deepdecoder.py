import torch
import torch.nn as nn

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)

torch.nn.Module.add = add_module

def DeepDecoder(
        num_output_channels=1, 
        num_channels_up=[128]*5,
        upsample_first = True,
        ):
    
    num_channels_up = num_channels_up + [num_channels_up[-1],num_channels_up[-1]]
    model = nn.Sequential()
    for i in range(len(num_channels_up)-1):

        if upsample_first:
            model.add(nn.Conv2d(num_channels_up[i], num_channels_up[i+1],  1, stride = 1))
            model.add(nn.Upsample(scale_factor=2, mode='bilinear'))

        else:
            model.add(nn.Upsample(scale_factor=2, mode='bilinear'))
            model.add(nn.Conv2d(num_channels_up[i], num_channels_up[i+1],  1, stride = 1))      

        if i != len(num_channels_up)-1:	
            model.add(nn.PReLU())
            model.add(nn.BatchNorm2d( num_channels_up[i+1], affine=True))
    
    
    model.add(nn.Conv2d( num_channels_up[-1], num_output_channels, 1))
    model.add(nn.ReflectionPad2d([-3, -2, -3, -2]))
    model.add(nn.ReLU())
    return model