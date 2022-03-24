from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from infer_full import OCRSystem

import tools.my_program as program

class ABoringOCR(object):
    def __init__(self, config):
        ocr = OCRSystem(config, device, logger, vdl_writer)
        img_list = get_image_file_list(config['Det']['Global']['infer_img'])
        ocr(img_list)
        

def main():
    config, device, logger, vdl_writer = program.preprocess()
