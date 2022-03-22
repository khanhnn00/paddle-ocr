# recommended paddle.__version__ == 2.0.0
python -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1' tools/train.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml