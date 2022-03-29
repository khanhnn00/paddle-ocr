# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import Levenshtein
import string

import argparse
# from vietocr.tool.utils import compute_accuracy
import os
from jiwer import wer,cer
import numpy as np


class RecMetric(object):
    def __init__(self, main_indicator='acc', is_filter=False, **kwargs):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.reset()

    def _normalize_text(self, text):
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text))
        return text.lower()

    def _compute_accuracy(self, ground_truth, predictions, mode='full_sequence'):
        """
        Computes accuracy
        :param ground_truth:
        :param predictions:
        :param display: Whether to print values to stdout
        :param mode: if 'per_char' is selected then
                    single_label_accuracy = correct_predicted_char_nums_of_single_sample / single_label_char_nums
                    avg_label_accuracy = sum(single_label_accuracy) / label_nums
                    if 'full_sequence' is selected then
                    single_label_accuracy = 1 if the prediction result is exactly the same as label else 0
                    avg_label_accuracy = sum(single_label_accuracy) / label_nums
        :return: avg_label_accuracy
        """
        if mode == 'per_char':

            accuracy = []

            for index, label in enumerate(ground_truth):
                prediction = predictions[index]
                total_count = len(label)
                correct_count = 0
                try:
                    for i, tmp in enumerate(label):
                        if tmp == prediction[i]:
                            correct_count += 1
                except IndexError:
                    continue
                finally:
                    try:
                        accuracy.append(correct_count / total_count)
                    except ZeroDivisionError:
                        if len(prediction) == 0:
                            accuracy.append(1)
                        else:
                            accuracy.append(0)
            avg_accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
        elif mode == 'full_sequence':
            try:
                correct_count = 0
                for index, label in enumerate(ground_truth):
                    prediction = predictions[index]
                    if prediction == label:
                        correct_count += 1
                avg_accuracy = correct_count / len(ground_truth)
            except ZeroDivisionError:
                if not predictions:
                    avg_accuracy = 1
                else:
                    avg_accuracy = 0
        else:
            raise NotImplementedError('Other accuracy compute mode has not been implemented')

        return avg_accuracy

    def _precision(self, preds, labels):

        acc_full_seq = self._compute_accuracy(preds, labels, mode='full_sequence')
        acc_per_char = self._compute_accuracy(preds, labels, mode='per_char')

        return acc_full_seq, acc_per_char

    def _get_cer_wer(self, preds, labels):
        total_wer = 0
        total_cer = 0
        for (labels, preds) in zip(labels, preds):
            labels = labels.lower()
            preds = preds.lower()
            actual_list = labels.split(' ')
            total_cer += cer(labels, preds)# / len(labels)
            # print('{} {}, CER: {}'.format(labels, preds, cer(labels, preds)))
            total_wer += wer(labels, preds)# / len(actual_list)
            if wer(labels, preds) >=1:
                print('{} --- {}, WER: {}'.format(labels, preds, wer(labels, preds)))
        return total_cer/len(labels), total_wer/len(labels)

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0

        predss = []
        labelss = []

        for (pred, pred_conf), (target, _) in zip(preds, labels):
            predss.append(pred)
            labelss.append(target)
            # pred = pred.replace(" ", "")
            # target = target.replace(" ", "")
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            norm_edit_dis += Levenshtein.distance(pred, target) / max(
                len(pred), len(target), 1)
            if pred == target:
                correct_num += 1
            all_num += 1
            # predss.append(pred)
            # labelss.append(target)
        # print('Total fucking number: {}'.format(len(labelss)))
        self.full_seq, self.per_char = self._precision(predss, labelss)
        self.cer, self.wer = self._get_cer_wer(predss, labelss)

        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        return {
            'acc': correct_num / all_num,
            'norm_edit_dis': 1 - norm_edit_dis / (all_num + 1e-3),
            'full_seq': self.full_seq,
            'per_char': self.per_char,
            'cer': self.cer,
            'wer': self.wer
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + 1e-3)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + 1e-3)
        full_seq = self.full_seq
        per_char = self.per_char
        cer = self.cer
        wer = self.wer
        self.reset()
        return {
            'acc': acc,
            'norm_edit_dis': norm_edit_dis,
            'full_seq': full_seq,
            'per_char': per_char,
            'cer': cer,
            'wer': wer
        }

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
        self.wer = 0
        self.cer = 0
        self.per_char = 0
        self.full_seq = 0
