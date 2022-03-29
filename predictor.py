import yaml

from modules import Predictor

# sys.path.append(0, './modules/PICK')
# sys.path.insert(0, './modules/PaddleOCR')
# sys.path.insert(0, './modules')
# sys.path.insert(0, './modules/vietocr')

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

predictor = Predictor(config)