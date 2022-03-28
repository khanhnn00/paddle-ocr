import argparse
from vietocr.tool.utils import compute_accuracy
import os
from jiwer import wer,cer

def precision(actual_sents, pred_sents):

    acc_full_seq = compute_accuracy(actual_sents, pred_sents, mode='full_sequence')
    acc_per_char = compute_accuracy(actual_sents, pred_sents, mode='per_char')

    return acc_full_seq, acc_per_char

def get_cer_wer(actual_sents, pred_sents):
    total_wer = 0
    total_cer = 0
    for (actual_sent, pred_sent) in zip(actual_sents, pred_sents):
        actual_sent = actual_sent.lower()
        pred_sent = pred_sent.lower()
        actual_list = actual_sent.split(' ')
        total_cer += cer(actual_sent, pred_sent)# / len(actual_sent)
        # print('{} {}, CER: {}'.format(actual_sent, pred_sent, cer(actual_sent, pred_sent)))
        total_wer += wer(actual_sent, pred_sent)# / len(actual_list)
        # print('{} {}, WER: {}'.format(actual_sent, pred_sent, wer(actual_sent, pred_sent)))
    return total_cer/len(actual_sents), total_wer/len(actual_sents)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='input file predict', default='predict.txt') #format line: file_name /t gt /t predict
    args = parser.parse_args()

    gts = []
    preds = []

    lines = open(args.input, "r")
    for line in lines:
        line = line.split('\t')
        print(line)
        gts.append(line[1])
        preds.append(line[2].replace('\n', ''))

    acc_full_seq, acc_per_char = precision(gts,preds)
    print('acc_full_seq: ', acc_full_seq)
    print('acc_per_char: ', acc_per_char)

    cer, wer = get_cer_wer(gts,preds)
    print('cer: ', cer)
    print('wer: ', wer)


    print('done')

if __name__ == '__main__':
    main()
