import sys
from pathlib import Path
import shutil
import os
import yaml
import torch

sys.path.insert(0, './PICK')
sys.path.insert(0, './PaddleOCR')
sys.path.insert(0, './modules')
sys.path.insert(0, './modules/vietocr')

# print(os.getcwd())

from PaddleOCR.OCRSystem import OCRSystem
from PICK.PICKSystem import PICKSystem
from PaddleOCR.ppocr.utils.utility import get_image_file_list
import PaddleOCR.tools.my_program as program
from KIE import get_kie
from yolov5 import Yolo
from vietocr import VietOCR

def main():
    ocr = OCRSystem(config, device, logger, vdl_writer)
    picker = PICKSystem(logger)
    img_list = get_image_file_list(config['Det']['Global']['infer_img'])
    ocr(img_list)
    picker('result_imgs', 'result_trans')

    with open('./config.yml') as f:
        config = yaml.safe_load(f)
    model_yolo = Yolo(config['Yolo'])
    model_vietocr = VietOCR(config['VietOCR'])
    
if __name__ == '__main__':
    if os.path.exists('./result_imgs'):
        shutil.rmtree('./result_imgs')
    if os.path.exists('./result_trans'):
        shutil.rmtree('./result_trans')
    if os.path.exists('./result_json'):
        shutil.rmtree('./result_json')
    # config, device, logger, vdl_writer = program.preprocess()
    main()
