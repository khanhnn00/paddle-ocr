# recommended paddle.__version__ == 2.0.0
python3.8 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1' tools/train.py -c configs/rec/multi_language/rec_en_number_lite_finetune.yml
