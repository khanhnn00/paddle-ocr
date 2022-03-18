import os

CUR_PATH = os.getcwd()

BASE_DATA_PATH = '/data/khanhnn/dataset/ocr_dataset/vi_00'

CUR_DIR = os.getcwd()

RECOG_GT_FILE = open('gt.txt', 'w')

new_dict = {}

def createDict(path, dict):
    for file in os.listdir(BASE_DATA_PATH):
        # print(file)
        if file.split('.')[-1] == 'txt':
            with open(os.path.join(BASE_DATA_PATH, file), 'r') as f:
                name = file.replace('txt', 'jpg')
                # name = os.path.join(BASE_DATA_PATH, name)
                content = str(f.read())
                dict[name] = content
            f.close()
        else:
            continue
    return dict

if __name__ == '__main__':

    dict = createDict(BASE_DATA_PATH, new_dict)
    # print(dict)
    for key in dict.keys():
        RECOG_GT_FILE.write('{}\t{}\n'.format(key, dict[key]))
    RECOG_GT_FILE.close()