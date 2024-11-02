from torch import nn, Tensor
from utils.conf import Configuration


class ResNet(nn.Module):
    
    def __init__(self, conf:Configuration, to_encode:bool):
        super(ResNet, self).__init__()
        
        num_resblock = conf.getEntry('num_en_resblock') if to_encode else conf.getEntry('num_de_resblock')
        
        if conf.getEntry('dilation_type') == 'exponential':
            assert num_resblock > 1 and 2 ** (num_resblock + 1) <= conf.getEntry('dim_series') + 1
            
        inner_channel = conf.getEntry('num_en_channel') if to_encode else conf.getEntry('num_de_channel')
        out_channel = conf.getEntry('dim_en_latent') if to_encode else conf.getEntry('dim_de_latent')
        
        layers = [PreActivatedResBlock(conf, 1, inner_channel, conf.getDilatoin(1, to_encode), first=True)]
        layers += [PreActivatedResBlock(conf, inner_channel, inner_channel, conf.getDilatoin(depth, to_encode)) for depth in range(2, num_resblock)]
        layers += [PreActivatedResBlock(conf, inner_channel, out_channel, conf.getDilatoin(num_resblock, to_encode), last=True)]
        
        self.model = nn.Sequential(*layers)
        
        
    def forward(self, input:Tensor):
        return self.model(input)
    

class PreActivatedResBlock(nn.Module):
    
    def __init__(self, conf: Configuration, in_channels, out_channels, dilation, first = False, last = False):
        super(PreActivatedResBlock, self).__init__()

        dim_series = conf.getEntry('dim_series')
        kernel_size = conf.getEntry('size_kernel')
        padding = int(kernel_size / 2) * dilation
        activation_name = "relu"
        bias = "layernorm"
        
        if first:
            self.first_block = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias)
            in_channels = out_channels
        else:
            self.first_block = nn.Identity()
        
        self.residual_linked = nn.Sequential(nn.ReLU(),
                                             nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias),
                                             nn.ReLU(),
                                             nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias))
        if in_channels != out_channels:
            self.identity_linked = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        else:
            self.identity_linked = nn.Identity()
            
        if last:
            self.after_addition = nn.Sequential(nn.ReLU())
        else:
            self.after_addition = nn.Identity()
    
    
    def forward(self, input: Tensor) -> Tensor:
        input = self.first_block(input)
        residual = self.residual_linked(input)
        identity = self.identity_linked(input)
        return self.after_addition(identity + residual)