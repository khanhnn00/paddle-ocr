# recommended paddle.__version__ == 2.0.0
python3.8 -m paddle.distributed.launch --gpus '0,1' ./PaddleOCR/tools/train.py\
            -c ./PaddleOCR/configs/rec/multi_language/rec_en_number_lite_finetune.yml