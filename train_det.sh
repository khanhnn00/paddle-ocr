# recommended paddle.__version__ == 2.0.0
python3.8 -m paddle.distributed.launch --log_dir=./debug2/ --gpus '1' tools/train.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml