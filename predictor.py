import yaml
import json

from modules import Predictor
from PaddleOCR.ppocr.utils.utility import get_image_file_list

# sys.path.append(0, './modules/PICK')
# sys.path.insert(0, './modules/PaddleOCR')
# sys.path.insert(0, './modules')
# sys.path.insert(0, './modules/vietocr')

with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

predictor = Predictor(config)
imgs = get_image_file_list(config['Yolo']['infer_img'])
img, final_dict = predictor(imgs)
img.save('result.jpg')
with open('./result.json', 'w', encoding='utf8') as f:
    json.dump(final_dict, f, sort_keys=True, ensure_ascii=False, indent=4)