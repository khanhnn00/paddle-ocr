# Predict English results
python3.8 main.py -c PaddleOCR/configs/full.yml -o Rec.Global.pretrained_model=PaddleOCR/output/rec_en_number_lite_final/best_accuracy Det.Global.pretrained_model=PaddleOCR/output/det_mobile_v2.0/best_accuracy Det.Global.load_static_weights=false Rec.Global.load_static_weights=false Det.Global.infer_img=../dataset/corner_dataset/images/train/crawl_00028.jpg
