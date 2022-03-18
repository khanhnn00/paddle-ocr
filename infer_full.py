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

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import get_image_file_list
from paddle import inference
import tools.my_program as program

def getImgFromBbox(src_im, bbox):
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

def draw_det_res(dt_boxes, config, img, img_name, save_path):
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
        logger.info("The detected Image saved in {}".format(save_path))

def resize_norm_img(img, max_wh_ratio):
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

def process_image_srn(img, image_shape, num_heads, max_text_length):
        print(img.shape)
        norm_img = resize_norm_img_srn(img, image_shape)
        print(norm_img.shape)
        norm_img = norm_img[np.newaxis, :]

        [encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1, gsrm_slf_attn_bias2] = \
            srn_other_inputs(image_shape, num_heads, max_text_length)

        gsrm_slf_attn_bias1 = gsrm_slf_attn_bias1.astype(np.float32)
        gsrm_slf_attn_bias2 = gsrm_slf_attn_bias2.astype(np.float32)
        encoder_word_pos = encoder_word_pos.astype(np.int64)
        gsrm_word_pos = gsrm_word_pos.astype(np.int64)

        return (norm_img, encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
                gsrm_slf_attn_bias2)

def get_infer_gpuid():
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

def srn_other_inputs(image_shape, num_heads, max_text_length):

        imgC, imgH, imgW = image_shape
        feature_dim = int((imgH / 8) * (imgW / 8))

        encoder_word_pos = np.array(range(0, feature_dim)).reshape(
            (feature_dim, 1)).astype('int64')
        gsrm_word_pos = np.array(range(0, max_text_length)).reshape(
            (max_text_length, 1)).astype('int64')

        gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
        gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias1 = np.tile(
            gsrm_slf_attn_bias1,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
            [-1, 1, max_text_length, max_text_length])
        gsrm_slf_attn_bias2 = np.tile(
            gsrm_slf_attn_bias2,
            [1, num_heads, 1, 1]).astype('float32') * [-1e9]

        encoder_word_pos = encoder_word_pos[np.newaxis, :]
        gsrm_word_pos = gsrm_word_pos[np.newaxis, :]

        return [
            encoder_word_pos, gsrm_word_pos, gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2
        ]

def create_predictor(cfg, mode, logger):

    model_dir = './inference/rec_en_number_lite/inference'

    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)

    model_file_path = model_dir + "/inference.pdmodel"
    params_file_path = model_dir + "/inference.pdiparams"
    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(
            model_file_path))
    if not os.path.exists(params_file_path):
        raise ValueError("not find params file path {}".format(
            params_file_path))

    config = inference.Config(model_file_path, params_file_path)

    precision = inference.PrecisionType.Float32

    if cfg['Global']['use_gpu']:
        gpu_id = get_infer_gpuid()
        if gpu_id is None:
            logger.warning(
                "GPU is not found in current device by nvidia-smi. Please check your device or ignore it if run on jeston."
            )
        config.enable_use_gpu(500, 0)

        use_dynamic_shape = True
   
        if cfg['Architecture']['algorithm'] != "CRNN":
            use_dynamic_shape = False
        min_input_shape = {"x": [1, 3, 32, 10]}
        max_input_shape = {"x": [8, 3, 32, 1536]}
        opt_input_shape = {"x": [8, 3, 32, 320]}

        if use_dynamic_shape:
            config.set_trt_dynamic_shape_info(
                min_input_shape, max_input_shape, opt_input_shape)

    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(10)
        
    # enable memory optim
    config.enable_memory_optim()
    config.disable_glog_info()

    config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
    if mode == 'table':
        config.delete_pass("fc_fuse_pass")  # not supported for table
    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)

    # create predictor
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()
    for name in input_names:
        input_tensor = predictor.get_input_handle(name)
    output_names = predictor.get_output_names()
    output_tensors = []
    for output_name in output_names:
        output_tensor = predictor.get_output_handle(output_name)
        output_tensors.append(output_tensor)
    return predictor, input_tensor, output_tensors, config

def main():

    ##load config and post process it ()
    global_config_det = config['Det']['Global']
    global_config_rec = config['Rec']['Global']

    post_process_class_det = build_post_process(config['Det']['PostProcess'],
                                            global_config_det)

    post_process_class_rec = build_post_process(config['Rec']['PostProcess'],
                                            global_config_rec)     

    rec_algorithm = config['Rec']['Architecture']['algorithm']                                                                           

    ##create transforms
    transforms_det = []
    for op in config['Det']['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name == 'KeepKeys':
            op[op_name]['keep_keys'] = ['image', 'shape']
        transforms_det.append(op)
    ops_det = create_operators(transforms_det, global_config_det)

    transforms_rec = []
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
        transforms_rec.append(op)
    global_config_rec['infer_mode'] = True
    ops_rec = create_operators(transforms_rec, global_config_rec)

    ##create model and load checkpoint
    model_det = build_model(config['Det']['Architecture'])
    load_model(config['Det'], model_det)

    if hasattr(post_process_class_rec, 'character'):
        char_num = len(getattr(post_process_class_rec, 'character'))
        if config['Rec']['Architecture']["algorithm"] in ["Distillation",
                                                   ]:  # distillation model
            for key in config['Architecture']["Models"]:
                config['Rec']['Architecture']["Models"][key]["Head"][
                    'out_channels'] = char_num
        else:  # base rec model
            config['Rec']['Architecture']["Head"]['out_channels'] = char_num
    model_rec = build_model(config['Rec']['Architecture'])
    load_model(config['Rec'], model_rec)

    ##now pass image and infer :)
    #det go first
    model_det.eval()
    
    for file in get_image_file_list(config['Det']['Global']['infer_img']):
        logger.info("infer_img: {}".format(file))
        with open(file, 'rb') as f:
            img = f.read()
            data = {'image': img}
        batch = transform(data, ops_det)

        images = np.expand_dims(batch[0], axis=0)
        shape_list = np.expand_dims(batch[1], axis=0)
        images = paddle.to_tensor(images)
        preds = model_det(images)
        post_result = post_process_class_det(preds, shape_list)

        src_img = cv2.imread(file)

        ##postprocessing the result
        img_list = []
        # parser boxes if post_result is dict
        if isinstance(post_result, dict):
            for k in post_result.keys():
                boxes = post_result[k][0]['points']
                for box in boxes:
                    this_img = getImgFromBbox(src_img, box)
                    img_list.append(this_img)
        else:
            boxes = post_result[0]['points']
            for box in boxes:
                this_img = getImgFromBbox(src_img, box)
                img_list.append(this_img)
        
        #rec goes after
        predictor, input_tensor, output_tensors, this_config = \
            create_predictor(config['Rec'], 'rec', logger)
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        batch_num = 8
        st = time.time()

        model_rec.eval()
        
        start = time.time()
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = resize_norm_img(img_list[indices[ino]],
                                                    max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
                # norm_img = process_image_srn(
                #     img=img_list[indices[ino]], image_shape=[3, 32, 320], num_heads=8, max_text_length=40)
                # # print(norm_img[1].shape)
                # encoder_word_pos_list = []
                # gsrm_word_pos_list = []
                # gsrm_slf_attn_bias1_list = []
                # gsrm_slf_attn_bias2_list = []
                # encoder_word_pos_list.append(norm_img[1])
                # gsrm_word_pos_list.append(norm_img[2])
                # gsrm_slf_attn_bias1_list.append(norm_img[3])
                # gsrm_slf_attn_bias2_list.append(norm_img[4])
                # norm_img_batch.append(norm_img[0])
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
                        
            # input_tensor = paddle.to_tensor(norm_img_batch)
            # preds = model_rec(input_tensor)
            # rec_result = post_process_class_rec(preds)
            # for rno in range(len(rec_result)):
            #     rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        # stop = time.time()
        # print('Total inference time: {}s'.format(str(stop-start)))

            input_tensor.copy_from_cpu(norm_img_batch)
            predictor.run()
            outputs = []
            for output_tensor in output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)

            if len(outputs) != 1:
                preds = outputs
            else:
                preds = outputs[0]
            rec_result = post_process_class_rec(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        print(rec_result) 



if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main()


                                        