from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def plot_one_box(x, label, im, color=(128, 128, 128), line_thickness=1):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    lw = max(round(0.0009 * sum(im.shape) / 2, 2), 1)  # line width
    # tl = round(0.0002 * sum(im.shape) / 2) + 1  # line/font thickness
    tl = max(lw - 1, 1)
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # cv2.rectangle(im, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA, )
    # cv2.putText(im, label, (int(x[0]), int(x[1] - 2)), 0, lw/3,  color, thickness=line_thickness, lineType=cv2.LINE_AA)
    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("./utils/arial-unicode-ms.ttf", 14)
    draw.text((int(x[0]), int(x[1] - 15)), label, font=font, fill=color)
    draw.rectangle([c1, c2], outline ="red")
    im = np.array(im)
    return im


def display_yolo(im, pred):
    if pred is not None:
        for *box, conf, cls, name_cls in pred:  # xyxy, confidence, class
            label = f'{name_cls} {conf:.2f}'
            plot_one_box(box, label, im, color=colors(cls))

    im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
    return im

def display(im, preds, labels):
    if preds is not None:
        for (pred, label) in zip (preds,labels):  # xyxy, confidence, class
            label = f'{label}'
            im = plot_one_box(pred, label, im, color=colors(0))

    im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
    return im
