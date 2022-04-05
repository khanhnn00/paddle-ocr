# Predict English results
python3.8 PaddleOCR/tools/infer_rec.py -c ./PaddleOCR/configs/rec/multi_language/rec_en_number_lite_finetune.yml -o \
Global.pretrained_model=./PaddleOCR/output/rec_en_number_lite_final/latest \
Global.load_static_weights=false \
Global.infer_img=../dataset/public_test_100/public_test_100_recog/images/preprocessed_cavet_fb_2_0000_crop_20.jpg