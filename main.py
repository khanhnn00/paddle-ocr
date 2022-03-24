import sys
from pathlib import Path

sys.path.insert(0, './PICK')
sys.path.insert(0, './PaddleOCR')

from PaddleOCR.OCRSystem import OCRSystem
from PICK.PICKSystem import PICKSystem
from PaddleOCR.ppocr.utils.utility import get_image_file_list
import PaddleOCR.tools.my_program as program

def main():
    ocr = OCRSystem(config, device, logger, vdl_writer)
    picker = PICKSystem(logger)
    img_list = get_image_file_list(config['Det']['Global']['infer_img'])
    ocr(img_list)
    picker('result_imgs', 'result_trans')
    
if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main()
