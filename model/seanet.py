from torch import Tensor, nn
from utils.conf import Configuration
from model.resnet import ResNet
from utils.activation import LeCunTanh
from utils.commons import Squeeze, Reshape


class SEAnetEncoder(nn.Module):
    
    def __init__(self, conf:Configuration):
        super(SEAnetEncoder, self).__init__()
        
        dim_embedding = conf.getEntry('dim_embedding')
        num_channel = conf.getEntry('num_en_channel')
        dim_latent = conf.getEntry('dim_en_latent')
        
        self.model = nn.Sequential(ResNet(conf, to_encode=True),
                                     nn.AdaptiveMaxPool1d(1),
                                     Squeeze(),
                                     nn.Linear(num_channel, dim_latent),
                                     LeCunTanh(),
                                     nn.Linear(dim_latent, dim_embedding, bias=False),
                                     nn.LayerNorm(dim_embedding, elementwise_affine=False))
        self.model.to(conf.getEntry('device'))
        
    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)
        
        
class SEAnetDecoder(nn.Module):
    
    def __init__(self, conf:Configuration):
        super(SEAnetDecoder, self).__init__()
        
        dim_series = conf.getEntry('dim_series')
        dim_embedding = conf.getEntry('dim_embedding')
        num_channels = conf.getEntry('num_de_channel')
        dim_latent = conf.getEntry('dim_de_latent')
        
        self.model = nn.Sequential(Reshape([-1, 1, dim_embedding]),
                                     nn.Linear(dim_embedding, dim_series),
                                     LeCunTanh(),
                                     ResNet(conf, to_encode=False),
                                     nn.AdaptiveMaxPool1d(1),
                                     Reshape([-1, 1, num_channels]),
                                     nn.Linear(num_channels, dim_latent),
                                     LeCunTanh(),
                                     nn.Linear(dim_latent, dim_series, bias=False),
                                     nn.LayerNorm(dim_series, elementwise_affine=False))

        self.model.to(conf.getEntry('device'))
    

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)
        

class SEAnet(nn.Module):
    
    def __init__(self, conf:Configuration):
        super(SEAnet, self).__init__()
        
        self.encoder = SEAnetEncoder(conf)
        self.decoder = SEAnetDecoder(conf)
        
    def forward(self, input: Tensor) -> Tensor:
        embedding = self.encoder(input)
        return self.decoder(embedding)