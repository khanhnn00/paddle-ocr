import numpy as np
from PIL import Image
import time
import yaml
import os

from yolov5 import Yolo
from vietocr import VietOCR
import utils.preprocess as preprocess
import utils.custom_plots as custom_plots

class Predictor:
    def __init__(self, config):
        self.config = config
        self.text_line_recognition = VietOCR(config['VietOCR'])
        self.text_line_detection = Yolo(config['Yolo'])

    def predict(self, image):
	# Preprocessing: Corner Detection and Stretch
        start_text_line_detection = time.time()
        text_line_bboxes = self.text_line_detection.predict(image)
        end_text_line_detection = time.time()
        # draw_image = custom_plots.display_yolo(preprocessed_image, text_line_bboxes)

        text_line_images = []
        for idx, text_line_bbox in enumerate(text_line_bboxes):
            tlbr = preprocess.xyxy2tlbr(text_line_bbox[:4])
            text_line_image = preprocess.four_point_transform(preprocessed_image, tlbr)
            text_line_images.append(text_line_image)
        text_line_texts = []

        start_text_line_recognition = time.time()
        text_line_texts = self.text_line_recognition.predict_batch(text_line_images)
        end_text_line_recognition = time.time()
        print(text_line_texts)

        draw_image = custom_plots.display(preprocessed_image, text_line_bboxes, text_line_texts)

        print('text_line_detection time: ', end_text_line_detection - start_text_line_detection)
        print('text_line_recognition time: ', end_text_line_recognition - start_text_line_recognition)
        return draw_image

# print(os.getcwd())
# with open('../config.yml') as f:
#     config = yaml.safe_load(f)
# predictor = Predictor(config)
