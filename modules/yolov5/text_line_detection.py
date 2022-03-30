import torch

class Yolo:
    def __init__(self, config):
        self.device = ('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('modules/yolov5', 'custom', path=config['weight_path'], device=self.device, source='local', force_reload=True)
        self.model.conf = config['conf']
        self.model.iou = config['iou']

    def __call__(self, img, size=640):
        if len(img) == 1:
            bboxes = self.model(img, size = size)
            bboxes = bboxes.pandas().xyxy[0].values.tolist()
            return bboxes
        else:
            bboxes = self.model(img, size = size)
            bboxes = bboxes.pandas().xyxy
            batch_bboxes = [i.values.tolist() for i in bboxes]
            return batch_bboxes

        
if __name__ == '__main__':
    config = {}
    config['weight_path'] = './weights/yolov5s_b32_epoch300.pt'
    config['device'] = 'cuda:0'
    config['conf'] = 0.4
    config['iou'] = 0.2
    text_line_detection = Yolo(config)
    print(text_line_detection)
