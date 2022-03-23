import os

RECOG_GT_FILE = open('vi_dict.txt', 'w')

with open('noob_dict.txt') as f:
    raw_text = f.read()
f.close()

for char in raw_text:
    RECOG_GT_FILE.write('{}\n'.format(char))
RECOG_GT_FILE.close()