import sys
from pathlib import Path
import os
# sys.path.insert(0, './PICK')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import math
import argparse
import collections
import time
import json

from tqdm import tqdm
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from modules.PICK.model.graph import GLCN
from parse_config import ConfigParser
import modules.PICK.model.pick as pick_arch_module
from .data_utils import pick_dataset

from .model import resnet
from .data_utils.pick_dataset import PICKDataset, BatchCollateFn
from .parse_config import *
from .utils.util import iob_index_to_str, text_index_to_str
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

class PICKSystem(object):
    def __init__(self):
        #self.logger = logger
        self.device = torch.device('cuda')
        checkpoint = torch.load('./PICK/pretrained_models/model_best.pth')
        config = checkpoint['config']
        state_dict = checkpoint['state_dict']
        self.pick_model = config.init_obj('model_arch', pick_arch_module)
        self.pick_model = self.pick_model.to(self.device)
        self.pick_model.load_state_dict(state_dict)
        self.pick_model.eval()

    def __call__(self, img_pth, transcript_pth):
        os.mkdir('result_json')
        start = time.time()
        self.img_pth = img_pth
        self.transcript_pth = transcript_pth
        test_dataset = PICKDataset(boxes_and_transcripts_folder=self.transcript_pth,
                               images_folder=self.img_pth,
                               resized_image_size=(480, 960),
                               ignore_error=False,
                               training=False)
        test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                  num_workers=2, collate_fn=BatchCollateFn(training=False))
        with torch.no_grad():
            for step_idx, input_data_item in tqdm(enumerate(test_data_loader)):
                for key, input_value in input_data_item.items():
                    if input_value is not None and isinstance(input_value, torch.Tensor):
                        input_data_item[key] = input_value.to(self.device)

                # For easier debug.
                image_name = input_data_item["filenames"][0]
                # print('image names la: {}'.format(image_names))

                output = self.pick_model(**input_data_item)
                logits = output['logits']  # (B, N*T, out_dim)
                new_mask = output['new_mask']
                image_indexs = input_data_item['image_indexs']  # (B,)
                text_segments = input_data_item['text_segments']  # (B, num_boxes, T)
                mask = input_data_item['mask']
                # List[(List[int], torch.Tensor)]
                best_paths = self.pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
                predicted_tags = []
                for path, score in best_paths:
                    predicted_tags.append(path)

                # convert iob index to iob string
                decoded_tags_list = iob_index_to_str(predicted_tags)
                # union text as a sequence and convert index to string
                decoded_texts_list = text_index_to_str(text_segments, mask)

                for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
                    # print('text', decoded_texts)
                    # List[ Tuple[str, Tuple[int, int]] ]
                    spans = bio_tags_to_spans(decoded_tags, [])
                    spans = sorted(spans, key=lambda x: x[1][0])

                    entities = []  # exists one to many case
                    for entity_name, range_tuple in spans:
                        entity = dict(entity_name=entity_name,
                                    text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                        entities.append(entity)

                    result_file = './result_json/{}.json'.format(os.path.basename(image_name).split('.')[0])
                    final_dict = {}
                    with open(result_file, 'w') as f:
                        for item in entities:
                            final_dict[item['entity_name']] = item['text']
                            # f.write('{}\t{}\n'.format(item['entity_name'], item['text']))
                    # json.dump()
                    with open(result_file, 'w', encoding='utf8') as json_file:
                        json.dump(final_dict, json_file, sort_keys=True, ensure_ascii=False, indent=4)
                    json_file.close()
        stop = time.time()
        #self.logger.info("PICK takes {}s to finish 1 image".format(stop-start))