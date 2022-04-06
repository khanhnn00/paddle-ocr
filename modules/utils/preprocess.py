import cv2
import numpy as np
import os
import math
from PIL import Image, ImageDraw

def get_bbox_center(bboxes_yolo):
    bboxes = {}
    center = None
    for bbox in bboxes_yolo:
        c = bbox[5]
        xyxy = bbox[:4]
        if c in [0, 1, 2, 3]:
            bboxes[c] = xyxy
        elif c in [4, 5]:
            center = xyxy
    return bboxes, center
        

def xyxy2tlbr(bbox):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    top_left = [x1, y1]
    top_right = [x2, y1]
    bottom_right = [x2, y2]
    bottom_left = [x1, y2]
    return [top_left, top_right, bottom_right, bottom_left]

def xywh2tlbr(bbox, image_width, image_height):
    x, y, w, h = bbox
    x *= image_width
    y *= image_height
    w *= image_width
    h *= image_height
    top_left = (x - w / 2, y - h / 2)
    top_right = (x + w / 2, y - h / 2)
    bottom_right = (x + w / 2, y + h / 2)
    bottom_left = (x - w / 2, y + h / 2)
    return (top_left, top_right, bottom_right, bottom_left)

def xywh2centerpoint(bbox, image_width, image_height):
    x, y, w, h = bbox
    x *= image_width
    y *= image_height
    return x, y

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def get_farthest_bbox(bbox, center):
    points = xyxy2tlbr(bbox)
    max_dist = -1
    max_point = None
    for point in points:
        d = dist(point, center)
        if d > max_dist:
            max_dist = d
            max_point = point
    return max_point

def get_farthest_points(bboxes, center):
    center_x1, center_y1, center_x2, center_y2 = center
    center_x, center_y = center_x1 + (center_x2 - center_x1) / 2, center_y1 + (center_y2 - center_y1) / 2
    farthest = {}
    for k in bboxes:
        bbox = bboxes[k]
        farthest_x, farthest_y = get_farthest_bbox(bbox, (center_x, center_y))
        farthest[k] = (farthest_x, farthest_y)
    return farthest

def four_point_transform(image, pts):
    if type(image) == str:
        image = cv2.imread(image)
    (tl, tr, br, bl) = pts
    # print(np.array(image).shape)
    
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(np.float32(pts), dst)
    warped = cv2.warpPerspective(np.float32(image), M, (max_width, max_height))
    return warped

def stretch(img, bboxes, center):
    farthest = get_farthest_points(bboxes, center)
    top_left = farthest[0]
    top_right = farthest[1]
    bottom_right = farthest[2]
    bottom_left = farthest[3]
    pts = (top_left, top_right, bottom_right, bottom_left)
    out_img = four_point_transform(img, pts)
    return out_img
