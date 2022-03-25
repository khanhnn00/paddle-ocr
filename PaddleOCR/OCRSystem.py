from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import json
import cv2
import paddle
import time
import math
from PIL import Image
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    print(ROOT)
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
from paddle import inference
from tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image, draw_ocr
import tools.my_program as program

class OCRSystem(object):
    def __init__(self, config, device, logger, vdl_writer):
        ##load config and post process it ()
        self.logger = logger
        self.config = config
        self.global_config_det = config['Det']['Global']
        self.global_config_rec = config['Rec']['Global']

        self.post_process_class_det = build_post_process(config['Det']['PostProcess'],
                                                self.global_config_det)

        self.post_process_class_rec = build_post_process(config['Rec']['PostProcess'],
                                                self.global_config_rec)     

        self.rec_algorithm = config['Rec']['Architecture']['algorithm']                                                                           

        ##create transforms
        self.transforms_det = []
        for op in config['Det']['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                continue
            elif op_name == 'KeepKeys':
                op[op_name]['keep_keys'] = ['image', 'shape']
            self.transforms_det.append(op)
        self.ops_det = create_operators(self.transforms_det, self.global_config_det)

        self.transforms_rec = []
        for op in config['Rec']['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                continue
            elif op_name in ['RecResizeImg']:
                op[op_name]['infer_mode'] = True
            elif op_name == 'KeepKeys':
                if config['Rec']['Architecture']['algorithm'] == "SRN":
                    op[op_name]['keep_keys'] = [
                        'image', 'encoder_word_pos', 'gsrm_word_pos',
                        'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'
                    ]
                elif config['Rec']['Architecture']['algorithm'] == "SAR":
                    op[op_name]['keep_keys'] = ['image', 'valid_ratio']
                else:
                    op[op_name]['keep_keys'] = ['image']
            self.transforms_rec.append(op)
        self.global_config_rec['infer_mode'] = True
        self.ops_rec = create_operators(self.transforms_rec, self.global_config_rec)

        ##create model and load checkpoint
        self.model_det = build_model(config['Det']['Architecture'])
        load_model(config['Det'], self.model_det)

        if hasattr(self.post_process_class_rec, 'character'):
            char_num = len(getattr(self.post_process_class_rec, 'character'))
            if config['Rec']['Architecture']["algorithm"] in ["Distillation",
                                                    ]:  # distillation model
                for key in config['Architecture']["Models"]:
                    config['Rec']['Architecture']["Models"][key]["Head"][
                        'out_channels'] = char_num
            else:  # base rec model
                config['Rec']['Architecture']["Head"]['out_channels'] = char_num
        self.model_rec = build_model(config['Rec']['Architecture'])
        load_model(config['Rec'], self.model_rec)

        self.model_det.eval()
        self.model_rec.eval()

    def getImgFromBbox(self, src_im, bbox):
        box = bbox.astype(np.int32).reshape((-1, 1, 2))
        x1, y1 = box[0][0][0], box[0][0][1]
        x2, y2 = box[1][0][0], box[1][0][1]
        x3, y3 = box[2][0][0], box[2][0][1]
        x4, y4 = box[3][0][0], box[3][0][1]
        # print('end')
        left = x1 if x1<x4 else x4
        right = x2 if x2>x3 else x3
        top = y1 if y1 < y2 else y2
        btm = y3 if y3 > y4 else y4
        this_img = src_im[top:btm, left:right, :]
        # this_img = np.expand_dims(this_img, axis=0)
        # print(this_img.shape)
        # this_img = this
        return this_img

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def draw_det_res(self, dt_boxes, config, img, img_name, save_path):
        if len(dt_boxes) > 0:
            import cv2
            src_im = img
            for box in dt_boxes:
                box = box.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, os.path.basename(img_name))
            cv2.imwrite(save_path, src_im)
            self.logger.info("The detected Image saved in {}".format(save_path))

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = [3, 32, 320]
        
        assert imgC == img.shape[2]
        imgW = int((32 * max_wh_ratio))
        # if use_onnx:
        #     w = input_tensor.shape[3:][0]
        #     if w is not None and w > 0:
        #         imgW = w
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def get_infer_gpuid(self):
        if os.name == 'nt':
            try:
                return int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
            except KeyError:
                return 0
        if not paddle.fluid.core.is_compiled_with_rocm():
            cmd = "env | grep CUDA_VISIBLE_DEVICES"
        else:
            cmd = "env | grep HIP_VISIBLE_DEVICES"
        env_cuda = os.popen(cmd).readlines()
        if len(env_cuda) == 0:
            return 0
        else:
            gpu_id = env_cuda[0].strip().split("=")[1]
            return int(gpu_id[0])

    def write_output(self, rec_det_res, result_file_path='result_trans/result.tsv', prob_thres=0.7):
        if not os.path.exists('result_trans'):
            os.mkdir('result_trans')
        result = ''
        for res in rec_det_res:
            bbox, value = res
            des, prob = value[0], value[1]
            s = [str(1)]
            for i, box in enumerate(bbox):
                xx, yy = box
                s.append(str(xx))
                s.append(str(yy))
            if prob > prob_thres:
                line = ','.join(s) + ',' + des
            else:
                line = ','.join(s) + ','
            result += line + '\n'
        result = result.rstrip('\n')
        with open(result_file_path, 'w', encoding='utf8') as res:
            res.write(result)

    def __call__(self, img_list):
        start = time.time()
        for file in img_list:
            self.logger.info("infer_img: {}".format(file))
            with open(file, 'rb') as f:
                img = f.read()
                data = {'image': img}
            
            batch = transform(data, self.ops_det)

            images = np.expand_dims(batch[0], axis=0)
            shape_list = np.expand_dims(batch[1], axis=0)
            images = paddle.to_tensor(images)
            preds = self.model_det(images)
            post_result = self.post_process_class_det(preds, shape_list)

            src_img = cv2.imread(file)
            ori_im = src_img.copy()

            ##postprocessing the result
            img_list = []
            start_det = time.time()
            # parser boxes if post_result is dict
            if isinstance(post_result, dict):
                for k in post_result.keys():
                    boxes = post_result[k][0]['points']
                    dt_boxes = self.filter_tag_det_res(boxes, ori_im.shape)
                    for box in boxes:
                        this_img = self.getImgFromBbox(src_img, box)
                        img_list.append(this_img)
            else:
                boxes = post_result[0]['points']
                dt_boxes = self.filter_tag_det_res(boxes, ori_im.shape)
                for box in boxes:
                    this_img = self.getImgFromBbox(src_img, box)
                    img_list.append(this_img)
            stop_det = time.time()

            #rec goes after
            # predictor, input_tensor, output_tensors, this_config = \
            #     self.create_predictor(config['Rec'], 'rec', self.logger)
            img_num = len(img_list)
            # Calculate the aspect ratio of all text bars
            width_list = []
            for img in img_list:
                width_list.append(img.shape[1] / float(img.shape[0]))
            # Sorting can speed up the recognition process
            indices = np.argsort(np.array(width_list))
            rec_res = [['', 0.0]] * img_num
            batch_num = 8

            #inference time :)
            start_rec = time.time()
            for beg_img_no in range(0, img_num, batch_num):
                end_img_no = min(img_num, beg_img_no + batch_num)
                norm_img_batch = []
                max_wh_ratio = 0
                for ino in range(beg_img_no, end_img_no):
                    h, w = img_list[indices[ino]].shape[0:2]
                    wh_ratio = w * 1.0 / h
                    max_wh_ratio = max(max_wh_ratio, wh_ratio)
                for ino in range(beg_img_no, end_img_no):
                    norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                        max_wh_ratio)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                norm_img_batch = np.concatenate(norm_img_batch)
                norm_img_batch = norm_img_batch.copy()
                            
                input_tensor = paddle.to_tensor(norm_img_batch)
                preds = self.model_rec(input_tensor)
                rec_result = self.post_process_class_rec(preds)
                for rno in range(len(rec_result)):
                    rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            stop_rec = time.time()
            for text, score in rec_res:
                self.logger.debug("{}, {:.3f}".format(text, score))
            
            stop = time.time()
            self.logger.info(
                "Predict time of detection phase: %.3fs" % (stop_det - start_det))
            self.logger.info(
                "Predict time of recognition phase: %.3fs" % (stop_rec - start_rec))
            self.logger.info(
                "Total time to run for 1 image: %.3fs" % (stop - start))
            
            #visualize result
            image = Image.fromarray(cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB))
            if not os.path.exists('result_imgs'):
                os.mkdir('result_imgs')
            image.save('result_imgs/result.jpg')
            image.save('result/original.jpg')
            # boxes = dt_boxes
            # txts = [rec_res[i][0] for i in range(len(rec_res))]
            # scores = [rec_res[i][1] for i in range(len(rec_res))]

            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]
            im_show = draw_ocr(image, boxes, txts, scores, font_path='./PaddleOCR/StyleText/fonts/en_standard.ttf')
            im_show = Image.fromarray(im_show)
            im_show.save('result/result.jpg')

            #write_result for the next step
            final_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
            for line in final_res:
                print(line)
            self.write_output(final_res)