import numpy as np
from PIL import Image
import time
import yaml
import os
import paddle
import cv2
import shutil

from .yolov5 import Yolo
from .vietocr import VietOCR
from .PaddleOCR.OCRSystem import OCRSystem
from .PICK.PICKSystem import PICKSystem
import modules.utils.preprocess as preprocess
import modules.utils.custom_plots as custom_plots
from modules.PaddleOCR.ppocr.data import create_operators, transform

class Predictor:
    def __init__(self, config):
        self.config = config
        self.vietocr = VietOCR(config['VietOCR'])
        self.yolo = Yolo(config['Yolo'])
        self.ocr = OCRSystem(config)
        self.paddle_det = self.ocr.model_det
        self.paddle_rec = self.ocr.model_rec
        self.pick = PICKSystem()

        self.detect_algorithm = 'paddle'
        self.recog_algorithm = 'paddle'
        self.text_line_detection = self.yolo
        self.text_line_recognition = self.vietocr

    def __call__(self, img_list):
	# Preprocessing: Corner Detection and Stretch
        if os.path.exists('./result'):
            shutil.rmtree('./result')
        image = cv2.cvtColor(cv2.imread(img_list[0]), cv2.COLOR_BGR2RGB)
        start_text_line_detection = time.time()
        if self.detect_algorithm == 'paddle':
            # with open(image[0], 'rb') as f:
            #     img = f.read()
            #     data = {'image': img}
            # image = cv2.cvtColor(cv2.imread(image[0]), cv2.COLOR_BGR2RGB)
            # image = np.expand_dims(image, axis=0)
            # print(np.array(data['image']).shape)
            # batch = transform(data, self.ocr.ops_det)
            # img = np.expand_dims(batch[0], axis=0)
            # shape_list = np.expand_dims(batch[1], axis=0)
            # input = paddle.to_tensor(img)
            
            # text_line_bboxes = self.text_line_detection(input)
            # text_line_bboxes = self.ocr.post_process_class_det(text_line_bboxes, shape_list)
            text_line_bboxes, img_list, image, dt_boxes = self.ocr.predict_det(img_list)
        else:
            text_line_bboxes = self.text_line_detection(img_list)
            # print(text_line_bboxes)
        end_text_line_detection = time.time()

        # preprocessed_image = np.squeeze(np.array(image), axis=0)
        # print(preprocessed_image.shape)

        text_line_images = []
        if self.detect_algorithm == 'paddle':
            new_text_line = []
            for box in text_line_bboxes:
                tmp = []
                tmp.append(box[0][0])
                tmp.append(box[0][1])
                tmp.append(box[2][0])
                tmp.append(box[2][1])
                new_text_line.append(tmp)
            # text_line_bboxes =new_text_line

        for idx, text_line_bbox in enumerate(text_line_bboxes):
            if self.detect_algorithm != 'paddle':
                tlbr = preprocess.xyxy2tlbr(text_line_bbox[:4])
            else:
                tlbr = text_line_bbox
            # print(tlbr)
            text_line_image = preprocess.four_point_transform(image, tlbr)
            # print('----------------------')
            # print(text_line_image.shape)
            text_line_images.append(text_line_image)
            cv2.imwrite('./{}.jpg'.format(str(idx)), text_line_image)
        text_line_texts = []

        start_text_line_recognition = time.time()
        text_line_texts = self.ocr.predict_rec(img_list, image, dt_boxes)
        end_text_line_recognition = time.time()
        # print(text_line_texts)

        if self.detect_algorithm == 'paddle':
            draw_image = custom_plots.display(image, new_text_line, text_line_texts)
        else:
            draw_image = custom_plots.display(image, text_line_bboxes, text_line_texts)

        print('text_line_detection time: ', end_text_line_detection - start_text_line_detection)
        print('text_line_recognition time: ', end_text_line_recognition - start_text_line_recognition)
        return draw_image