import torch
import sys
import argparse
from PIL import Image
import numpy as np

sys.path.append('./modules/vietocr')
from .tool.predictor import Predictor
from .tool.config import Cfg


class VietOCR:
    def __init__(self, config_):
        config = Cfg.load_config_from_file(config_['config'])
        print(config_)
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        config['weights'] = config_['weight_path']
        self.model = Predictor(config)
        self.batch_size = config_['batch_size']

    # def __call__(self, img):
    #     print(np.array(img).shape)
    #     if len(img) == 1:
    #         text = self.model.predict(img)
    #         return text
    #     else:
    #         texts = self.model.predict_batch(img, self.batch_size)
    #         return texts

    def predict(self, img):
        text = self.model.predict(img)
        return text

    def predict_batch(self, imgs):
        texts = self.model.predict_batch(imgs, self.batch_size)
        return texts


        
