import json
import os
import cv2
import io
import re
import math
import collections

keys_pattern = {
    0: "Tên chủ|chủ xe|Họ tên|Owner|full name", 
    1: "Số máy|Engine N", 
    2: "Địa chỉ|Address|thường trú", 
    3: "CMND|Hộ chiếu|Identity Card|Passport", 
    4: "Số khung|Chassis N", 
    5: "Nhãn hiệu|Brand", 
    6: "Loại xe|Type", 
    7: "Số loại|Model code", 
    8: "Màu sơn|Color", 
    9: "Số người được phép chở|Seat capacity",
    10: "Dung tích|Capacity|xi lanh", 
    11: "Công suất|Horsepower", 
    12: "Năm sản xuất|Year of manufacture", 
    13: "Tự trọng|Empty weight", 
    14: "Dài|Length", 
    15: "Rộng|Width", 
    16: "Cao|Heigh|Height", 
    17: "Số chỗ ngồi|Sit", 
    18: "đứng|Đứng|Stand", 
    19: "nằm|Nằm|Lie", 
    20: "Hàng hóa|Goods", 
    21: "Nguồn gốc|Origin", 
    22: "Biển số đăng ký|Biển số|số đăng ký|N Plate", 
    23: "Đăng ký lần đầu ngày|Date of first registration|First registration date|Đăng ký lần đầu|first registration", 
    24: "Đăng ký xe có giá trị đến ngày|Giá trị đến ngày|Date of expiry|Valid until date|ó giá trị đến ngày|date of expiry", 
    25: "Số (Number)|Số:|Number"
}

ignore_pattern = "ngày*(date)|(date)*tháng|tháng*năm|trưởng |thiếu tá|trung tá|thượng tá|đại tá|thiếu tướng|trung tướng|thượng tướng"

mapping = {
    0: "Tên chủ xe (Owner’s full name)",
    1: "Số máy (Engine N)",
    2: "Địa chỉ (Address)",
    3: "Số CMND / Hộ chiếu (Identity Card N / Passport)",
    4: "Số khung (Chassis N)",
    5: "Nhãn hiệu (Brand)",
    6: "Loại xe (Type)",
    7: "Số loại (Model code)",
    8: "Màu sơn (Color)",
    9: "Số người được phép chở (Seat capacity)",
    10: "Dung tích (Capacity) / Dung tích xi lanh",
    11: "Công suất (Powerhorse)",
    12: "Năm sản xuất (Year of manufacture)",
    13: "Tự trọng (Empty weight)",
    14: "Dài (Length)",
    15: "Rộng (Width)",
    16: "Cao (Height)",
    17: "Số chỗ ngồi (Sit)",
    18: "Đứng (Stand)",
    19: "Nằm (Lie)",
    20: "Hàng hóa (Goods)",
    21: "Nguồn gốc (Origin)",
    22: "Biển số đăng ký (N Plate) / Biển số",
    23: "Đăng ký lần đầu ngày First registration date",
    24: "Đăng ký xe có giá trị đến ngày (date of expiry)",
    25: "Số (number)"
}

def check_keys(query, key_pattern):
    pattern = re.compile(key_pattern)
    r = re.search(pattern, query)
    if r != None:
        return True
    return False

def get_center(bbox):
    return [int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)]

def dis_2point(a,b):
    return math.sqrt( (a[0] - b[0])**2 + (a[1] - b[1])**2 )

def dis_min2bbox(bboxa,bboxb):
    list_dis = []
    for pointa in [[bboxa[0], bboxa[1]], [bboxa[0], bboxa[3]], [bboxa[2], bboxa[1]], [bboxa[2], bboxa[3]], get_center(bboxa)]:
        for pointb in [[bboxb[0], bboxb[1]], [bboxb[0], bboxb[3]], [bboxb[2], bboxb[1]], [bboxb[2], bboxb[3]], get_center(bboxb)]:
            list_dis.append(dis_2point(pointa, pointb))
    return min(list_dis)

def dis_y(bbox,point):
    return abs(bbox[1] - point[1])/abs(bbox[1] - bbox[3])

def check_point_in_bbox(point, bbox):
    return bbox[0] <= point[0] and point[0] <= bbox[2] and bbox[1] <= point[1] and point[1] <= bbox[3]

def check_intersection_over_searchbbox(search_bbox, area):

    xA = max(search_bbox[0], area[0])
    yA = max(search_bbox[1], area[1])
    xB = min(search_bbox[2], area[2])
    yB = min(search_bbox[3], area[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    search_bboxArea = (search_bbox[2] - search_bbox[0] + 1) * (search_bbox[3] - search_bbox[1] + 1)
    ios = interArea / float(search_bboxArea)
    return ios 

def check_same_col(key_bbox, check_bbox):
    max_x1 = max(key_bbox[0], check_bbox[0])
    min_x2 = min(key_bbox[2], check_bbox[2])
    if min_x2-max_x1 > 0:
        return True
    return False

def check_same_row(key_bbox, check_bbox):
    max_y1 = max(key_bbox[1], check_bbox[1])
    min_y2 = min(key_bbox[3], check_bbox[3])
    if min_y2-max_y1 > 1/2*(key_bbox[3] - key_bbox[1]):
        return True
    return False

def get_bbox_from_key(key_bbox, bboxes, classes, height, width, labels):
    #search min bottom bbox have same col
    match_y_idx = -1
    for idx in range(len(classes)):
        if 0 <= classes[idx] < 26:
            if check_same_col(key_bbox, bboxes[idx]) and key_bbox[1] < bboxes[idx][1]:
                if match_y_idx == -1:             
                    match_y_idx = idx
                else:         
                    if bboxes[idx][1] < bboxes[match_y_idx][1]:
                        match_y_idx = idx
    #search min right bbox have same row
    match_x_idx = -1
    for idx in range(len(classes)):
        if 0 <= classes[idx] < 26:
            if match_y_idx == -1:
                search_bbox = [key_bbox[0], key_bbox[1], key_bbox[2], height-1]
            else:
                search_bbox = [key_bbox[0], key_bbox[1], bboxes[match_y_idx][2], bboxes[match_y_idx][1]] 
#             if not check_same_col(search_bbox, bboxes[idx]) and search_bbox[0] < bboxes[idx][0]:
            if check_same_row(key_bbox, bboxes[idx]) and not check_same_col(key_bbox, bboxes[idx])\
                and search_bbox[0] < bboxes[idx][0]:
                if match_x_idx == -1:             
                    match_x_idx = idx
                else:         
                    if bboxes[idx][0] < bboxes[match_x_idx][0]:
                        match_x_idx = idx
    if match_x_idx == -1:
        search_bbox = [search_bbox[0], search_bbox[1], width-1, search_bbox[3]]
    else:
        search_bbox = [search_bbox[0], search_bbox[1], bboxes[match_x_idx][0], search_bbox[3]]
    # if match_y_idx != -1:
    #     print('---',labels[match_y_idx])
    # if match_x_idx != -1:
    #     print('---',labels[match_x_idx])
    return search_bbox

def get_kie(image, img_name, bboxes, labels, ios=0.5, max_dis_y=3):
    """ Key Information Extraction.
    # Arguments
        bboxes:  bboxes of text in image (N,4)
        labels:  labels of bboxes (N,1)
    # Returns
        kie in json format
    """
    # print('bboxes: ', bboxes)
    # print('labels: ', labels)
    if len(bboxes) != len(labels):
        return None
    # width, height = image.size
    height, width = image.shape[:2]
    print(height, width)


    # classify key and value in predict results
    num_bbox = len(bboxes)
    value_class = len(keys_pattern)
    classes = [value_class]*num_bbox
    for key_idx in keys_pattern:
        match_key_idx = -1
        # for idx, (bbox, label) in enumerate(zip(bboxes, labels)):
        for idx in range(num_bbox):
            bbox = bboxes[idx]
            label = labels[idx]
            # print(label)
            # print(type(label))
            # print(keys_pattern[key_idx])
            # print(type(keys_pattern[key_idx]))
            if check_keys(label, keys_pattern[key_idx]):
                if match_key_idx == -1:
                    classes[idx] = key_idx
                    match_key_idx = idx
                else:
                    if bbox[1] < bboxes[match_key_idx][1]:
                        classes[match_key_idx] = -1
                        classes[idx] = key_idx
                        match_key_idx = idx
                    else:
                        classes[idx] = -1
    # remove ignore pattern
    for idx in range(len(classes)):
        if check_keys(labels[idx], ignore_pattern):
            classes[idx] = -1

    # kie
    dict_file = {}
    for key_idx in keys_pattern:
        try:
            idx = classes.index(key_idx)
        except ValueError:
            continue
        if idx != -1:
            search_area = get_bbox_from_key(bboxes[idx], bboxes, classes, height, width, labels)
            match_value_idx = -1
            for idx_value in range(len(classes)):
                if classes[idx_value] == value_class:
                    if check_intersection_over_searchbbox(bboxes[idx_value], search_area) > ios \
                        and dis_y(bboxes[idx],bboxes[idx_value][:2]) < max_dis_y:
                        if match_value_idx == -1:
                            if key_idx == 22 or key_idx == 23:
                                if bboxes[idx_value][0] < bboxes[idx][3]:
                                    match_value_idx = idx_value
                            else:
                                match_value_idx = idx_value
                        else:
                            if key_idx == 22 or key_idx == 23: #bien so dang ky va ngay dau dang ky
                                if bboxes[idx_value][0] < bboxes[match_value_idx][0]:
                                    match_value_idx = idx_value
                            else:
                                check_dis = dis_min2bbox(bboxes[idx_value],bboxes[idx])
                                match_dis = dis_min2bbox(bboxes[match_value_idx],bboxes[idx])
                                if check_dis < match_dis:
                                    match_value_idx = idx_value
            if match_value_idx != -1:
                classes[match_value_idx] = -1
                # dict_file[mapping[key_idx]] = labels[match_value_idx]
                dict_file["{}".format(key_idx)] = labels[match_value_idx]
            else:
                # dict_file[mapping[key_idx]] = ''
                dict_file["{}".format(key_idx)] = ''
    
    res = collections.OrderedDict(sorted(dict_file.items()))
    if not os.path.exists('result_json'):
        os.mkdir('result_json')
    with open('result_json/{}.json'.format(img_name.split('.')[0]), 'w', encoding='utf8') as json_file:
        json.dump(res, json_file, sort_keys=True, ensure_ascii=False, indent=4)
    json_file.close()
    return collections.OrderedDict(sorted(dict_file.items()))


class KIE:
    def __init__(self, config):
        self.ios = config['ios']
        self.max_dis_y = config['max_dis_y']
    def predict(self, image, bboxes, labels):
        return get_kie(image, bboxes, labels, self.ios, self.max_dis_y)
