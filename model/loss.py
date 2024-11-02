from numpy import sqrt
from torch import nn, mean, squeeze
from torch.nn.modules.distance import PairwiseDistance

class ScaledTransLoss(nn.Module):
    
    def __init__(self, original_dimension:int = 256, embedding_dimension: int = 16, to_scale: bool = False):
        super(ScaledTransLoss, self).__init__()

        self.l2 = PairwiseDistance(p=2).cuda()
        self.l1 = PairwiseDistance(p=1).cuda()
        if to_scale:
            self.scale_factor_original = sqrt(original_dimension)
            self.scale_factor_embedding = sqrt(embedding_dimension)
        else:
            self.scale_factor_original = 1
            self.scale_factor_embedding = 1


    def forward(self, database, query, db_embedding, query_embedding):
        original_l2 = self.l2(squeeze(database), squeeze(query)) / self.scale_factor_original
        embedding_l2 = self.l2(squeeze(db_embedding), squeeze(query_embedding)) / self.scale_factor_embedding 
        return self.l1(original_l2.view([1, -1]), embedding_l2.view([1, -1]))[0] / database.shape[0]


class ScaledReconsLoss(nn.Module):
    
    def __init__(self, original_dimension: int = 256, to_scale: bool = False):
        super(ScaledReconsLoss, self).__init__()
        self.l2 = PairwiseDistance(p=2).cuda()
        if to_scale:
            self.scale_factor = sqrt(original_dimension)
        else:
            self.scale_factor = 1
            
    
    def forward(self, database, reconstructed):
        return mean(self.l2(squeeze(database), squeeze(reconstructed))) / self.scale_factor