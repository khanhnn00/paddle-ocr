import argparse
import os
import json
from jiwer import wer,cer


def get_cer_wer(actual_sents, pred_sents):
    total_wer = 0
    total_cer = 0
    total_ser = 0
    for (actual_sent, pred_sent) in zip(actual_sents, pred_sents):
        actual_sent = actual_sent.lower()
        pred_sent = pred_sent.lower()
        # print((actual_sent, pred_sent))
        # total_cer += cer(actual_sent, pred_sent)
        # total_wer += wer(actual_sent, pred_sent)
        # total_ser += actual_sent != pred_sent
        total_cer += min(1,cer(actual_sent, pred_sent))
        total_wer += min(1,wer(actual_sent, pred_sent))
        total_ser += actual_sent != pred_sent
    return total_cer/len(actual_sents), total_wer/len(actual_sents), total_ser/len(actual_sents)

def main():
    from jiwer import wer,cer
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', default='../dataset/public_test_100/public_test_100_gt', help='gt folder') 
    parser.add_argument('--pred', default='./result_json', help='pred folder') 
    args = parser.parse_args()

    gt_path = args.gt
    pred_path = args.pred
    print(gt_path)
    print(pred_path)

    gts = {}
    preds = {}  

    class_names = {}
    meta_path = os.path.join(gt_path, 'meta.json')
    print(os.path.exists(meta_path))
    assert os.path.exists(meta_path), "Can not find meta.json in gt"
    with open(meta_path, "r") as f:
        mapping_names = json.load(f)  
    for mapping_name in mapping_names:
        class_names[str(mapping_name['id'])] = mapping_name['name']
    print(class_names)
    gt_path = os.path.join(gt_path, 'all')

    # for key in class_names:
    #     gts[key] = []
    #     preds[key] = []
    file_names = [i for i in os.listdir(gt_path) if 'meta' not in i and os.path.splitext(i)[1] != '.jpg']
    file_names.sort()
    all_cer = 0
    all_wer = 0
    all_ser = 0

    all_cer_dict = {}
    all_wer_dict = {}
    all_ser_dict = {}
    total_text_dict = {}

    total_text = 0
    for file_name in file_names:
        for key in class_names:
            gts[key] = []
            preds[key] = []
        print('---------------------------------------------------')
        print(file_name)
        with open(os.path.join(gt_path, file_name), "r") as f:
            gt = json.load(f)  
        assert os.path.exists(os.path.join(pred_path, file_name)), \
                "Can not find {} in predict folder".format(os.path.join(pred_path, file_name))
        with open(os.path.join(pred_path, file_name), "r") as f:
            pred = json.load(f)  
        
        for key in class_names:
            if key in gt:
                gts[key].append(gt[key])
                preds[key].append("" if key not in pred else pred[key])
            # if key in gt or key in pred:
            #     print('key gt pred',key,gt,pred)
            #     if key not in gt:
            #         if pred[key] != "":
            #             gts[key].append(pred[key])
            #             preds[key].append("")
            #     else:
            #         gts[key].append(gt[key])
            #         preds[key].append("" if key not in pred else pred[key])
        file_cer = 0
        file_wer = 0
        file_ser = 0
        total_file_text = 0
        for key in class_names:
            if key == '26':
            # if key == '26' or key in ['11','12','13','14','15','16','24', '9']:
            # if key not in ['0','1','2','3','4','5','6', '7', '8', '9', '10', '22', '23', '25']:
                continue
            print(class_names[key])
            # print('gts[{}]: {}'.format(key, gts[key]))
            # print('preds[{}]: {}'.format(key, preds[key]))
            if len(gts[key]) != 0:
                cer, wer, ser = get_cer_wer(gts[key],preds[key])
                file_cer += cer*len(gts[key])
                file_wer += wer*len(gts[key])
                file_ser += ser*len(gts[key])
                
                total_file_text += len(gts[key])
                print('----- cer : {}'.format(cer))
                print('----- wer: {}'.format(wer))
                print('----- ser: {}'.format(ser))

                if key in all_cer_dict:
                    all_cer_dict[key] += cer*len(gts[key])
                    all_wer_dict[key] += wer*len(gts[key])
                    all_ser_dict[key] += ser*len(gts[key])
                    total_text_dict[key] += len(gts[key])
                else:
                    all_cer_dict[key] = cer*len(gts[key])
                    all_wer_dict[key] = wer*len(gts[key])
                    all_ser_dict[key] = ser*len(gts[key])
                    total_text_dict[key] = len(gts[key])


        if (total_file_text != 0):
            print('----- file cer : {}'.format(file_cer/total_file_text))
            print('----- file wer: {}'.format(file_wer/total_file_text))
            print('----- file ser: {}'.format(file_ser/total_file_text))
        all_cer += file_cer
        all_wer += file_wer
        all_ser += file_ser
        total_text += total_file_text
    print('-------------------------------\n ERROR ON KEY:')
    for key in class_names:
        if key in all_cer_dict:
            print(key)
            print('----- number text: ', total_text_dict[key])
            print('----- total cer : {}'.format(all_cer_dict[key]/total_text_dict[key]))
            print('----- total wer: {}'.format(all_wer_dict[key]/total_text_dict[key]))
            print('----- total ser: {}'.format(all_ser_dict[key]/total_text_dict[key]))

    print('All')
    print('Number file: ', len(file_names))
    print('Number text: ', total_text)
    if (total_text != 0):
        print('----- total cer : {}'.format(all_cer/total_text))
        print('----- total wer: {}'.format(all_wer/total_text))
        print('----- total ser: {}'.format(all_ser/total_text))

if __name__ == '__main__':
    main()