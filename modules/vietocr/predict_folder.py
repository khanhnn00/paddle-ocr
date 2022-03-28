import argparse
from PIL import Image
import os

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='foo help')
    parser.add_argument('--annotation', required=True, help='foo help')
    parser.add_argument('--dest', required=True, help='foo help', default='predict.txt')
    parser.add_argument('--config', required=True, help='foo help')

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)

    detector = Predictor(config)

    save_file = args.dest
    
    file_names = []
    gts = []
    lines = open(os.path.join(args.source, args.annotation), "r")
    for line in lines:
        line = line.split('\t')
        print(line)
        file_names.append(line[0])
        gts.append(line[1].replace('\n', ''))
    # file_names = os.listdir(args.source)
    file_number = len(file_names)
    with open(save_file, 'w') as f:
        
        for idx, (file_name, gt) in enumerate(zip(file_names, gts)):
            print('{} / {} {}'.format(idx, file_number, file_name))
            img = Image.open(os.path.join(args.source, file_name))
            p = detector.predict(img)
            f.write(file_name + '\t' + gt + '\t' + p + '\n')
    
            print(p)
    print('done')

if __name__ == '__main__':
    main()
