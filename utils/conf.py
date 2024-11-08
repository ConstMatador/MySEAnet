import json


class Configuration:
    
    def __init__(self, confPath: str = ""):
        self.confPath = confPath
        self.defaultConf = {
            "data_path": "/mnt/data/user_liangzhiyu/wangzhongzheng/Data/Mine/gist/gist.data",
            "log_path": "/mnt/data/user_liangzhiyu/wangzhongzheng/MySEAnet/example/example.log",
            "model_path": "/mnt/data/user_liangzhiyu/wangzhongzheng/MySEAnet/example/example.pth",
            "dim_series": 960,
            "dim_embedding": 60,
            "batch_size": 256,
            "epoch_max": 100,
            "train_size": 200000,
            "val_size": 10000,
            "data_size": 1000000,
            "device": "cuda:0",
            "train_path": "/mnt/data/user_liangzhiyu/wangzhongzheng/MySEAnet/example/data/train.data",
            "val_path": "/mnt/data/user_liangzhiyu/wangzhongzheng/MySEAnet/example/data/val.data",
            "train_indices_path": "/mnt/data/user_liangzhiyu/wangzhongzheng/MySEAnet/example/data/train_indices.data",
            "val_indices_path": "/mnt/data/user_liangzhiyu/wangzhongzheng/MySEAnet/example/data/val_indices.data",
            "num_en_resblock": 3,
            "num_de_resblock": 2,
            "num_en_channel": 256,
            "num_de_channel": 256,
            "dim_en_latent": 256,
            "dim_de_latent": 256,
            "dilation_type": "exponential",
            "size_kernel": 3,
            "model_init": "lsuv'",
            "optim_type": "sgd",
            "momentum": 0.9,
            "lr_mode": "linear", 
            "lr_max": 1e-2,
            "lr_min": 1e-5,
            "wd_mode": "linear", 
            "wd_max": 1e-4,
            "wd_min": 1e-8,
            "orth_regularizer": "srip",
            "srip_mode": "linear",
            "srip_max": 5e-4,
            "srip_min": 0,
            "reconstruct_weight": 0.25,
            "lsuv_size": 2000,
            "lsuv_ortho": True,
            "lsuv_maxiter": 10,
            "lsuv_mean": 0,
            "lsuv_std": 1.0,
            "lsuv_std_tol": 0.1
        }
        self.loadConf()
    
    
    def loadConf(self):
        with open(self.confPath, 'r') as fin:
            self.confLoaded = json.load(fin)
            
            
    def getEntry(self, key: str):
        if key in self.confLoaded:
            return self.confLoaded[key]
        elif key in self.defaultConf:
            return self.defaultConf[key]
        else:
            raise Exception(f"Key {key} not found in configuration.")
        
    
    def getDilatoin(self, depth: int, to_encode: bool = True) -> int:
        if not to_encode:
            depth = self.getEntry('num_de_resblock') + 1 - depth
        return int(2 ** (depth - 1))